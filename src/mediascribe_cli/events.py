from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from typing import Any, TextIO


# ANSI escape sequences: CSI (ESC [ ... final byte) and OSC (ESC ] ... BEL/ST).
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;:]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
# Remaining C0/C1 control characters (incl. DEL).
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
# Secret-looking tokens that must never reach human-readable logs.
_SECRET_RE = re.compile(
    r"(?i)\b(bearer|api[-_]?key|token|authorization)([\s:=]+)[A-Za-z0-9._\-]{8,}"
    r"|\b(?:sk|hf|xoxb|ghp|gho|github_pat|or)[-_][A-Za-z0-9_\-]{12,}"
)


def _sanitize(value: Any) -> str:
    """Make a value safe for a terminal: collapse whitespace, strip ANSI
    escapes and control characters, and mask secret-looking tokens."""
    text = " ".join(str(value).split())
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)

    def _mask(match: re.Match) -> str:
        if match.group(1):
            return f"{match.group(1)}{match.group(2)}***"
        return "***"

    return _SECRET_RE.sub(_mask, text)


def _redacted_text(value: Any) -> dict[str, Any]:
    text = str(value)
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "chars": len(text)}


class EventEmitter:
    """Emits run events either as human-readable stderr lines or JSONL on stdout.

    Human output is sanitized (control characters, ANSI escapes, secrets).
    JSONL payloads keep raw values — json.dumps escapes them — except that
    with redact=True (the default) the `text` field of transcript_segment
    and progress events is replaced by a digest.
    """
    def __init__(
        self,
        json_lines: bool = False,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        redact: bool = True,
    ):
        self.json_lines = json_lines
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.redact = bool(redact)
        self._progress_active = False
        self._interactive = self._supports_carriage_return(self.stderr)

    @staticmethod
    def _supports_carriage_return(stream: TextIO) -> bool:
        try:
            return bool(stream.isatty())
        except (AttributeError, OSError):
            return False

    def emit(self, event_type: str, **payload: Any) -> None:
        if self.json_lines:
            event = {
                "type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
            if self.redact and event_type in {"transcript_segment", "progress"} and "text" in event:
                event["text"] = _redacted_text(event["text"])
            # Error/message strings reach JSONL verbatim today; apply the
            # same secret/control-character masking as human output so the
            # "keys never appear in logs" invariant holds on both paths.
            for field in ("error", "message", "reason"):
                if isinstance(event.get(field), str):
                    event[field] = _sanitize(event[field])
            print(json.dumps(event, ensure_ascii=False), file=self.stdout, flush=True)
            return

        if event_type == "progress":
            self._emit_progress(payload)
            return
        self._finish_progress()
        message = self._human_message(event_type, payload)
        if message:
            print(message, file=self.stderr, flush=True)

    def error(self, message: str, **payload: Any) -> None:
        self.emit("error", error=message, **payload)

    def _emit_progress(self, payload: dict[str, Any]) -> None:
        stage = {
            "model": "모델",
            "stt": "STT",
            "diarization": "화자 분리",
            "summary": "요약",
            "commit": "저장",
        }.get(str(payload.get("stage", "progress")), str(payload.get("stage", "진행")))
        progress = payload.get("progress")
        progress_text = f"{float(progress):5.1f}%" if progress is not None else "   --.-%"
        elapsed = self._format_duration(payload.get("elapsed_seconds"))
        eta = self._format_duration(payload.get("eta_seconds"))
        file_name = _sanitize(payload.get("file_name", ""))
        text = _sanitize(payload.get("text", ""))
        if len(text) > 46:
            text = f"{text[:45]}…"
        prefix = f"[{file_name}] " if file_name else ""
        message = f"{prefix}{stage} {progress_text} | 경과 {elapsed} | 예상 잔여 {eta}"
        if text:
            message = f"{message} | {text}"
        if self._interactive:
            self.stderr.write(f"\r\033[2K{message}")
            self.stderr.flush()
            self._progress_active = True
        else:
            print(message, file=self.stderr, flush=True)

    def _finish_progress(self) -> None:
        if self._progress_active:
            self.stderr.write("\r\033[2K\n")
            self.stderr.flush()
            self._progress_active = False

    @staticmethod
    def _format_duration(seconds: Any) -> str:
        if seconds is None:
            return "--:--"
        try:
            total = max(0, int(round(float(seconds))))
        except (TypeError, ValueError):
            return "--:--"
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def _human_message(event_type: str, payload: dict[str, Any]) -> str:
        file_name = _sanitize(payload.get("file_name", ""))
        prefix = f"[{file_name}] " if file_name else ""
        if event_type == "run_started":
            return f"MediaScribe-CLI 시작: {payload.get('files', 0)}개 파일"
        if event_type == "file_started":
            return f"{prefix}모델과 STT 준비 중..."
        if event_type == "file_skipped":
            return f"{prefix}건너뜀: {_sanitize(payload.get('reason', '이미 처리됨'))}"
        if event_type == "checkpoint_saved":
            return f"{prefix}체크포인트 저장: {_sanitize(payload.get('kind', ''))}"
        if event_type == "diarization_started":
            return f"{prefix}화자 분리 준비 중..."
        if event_type == "diarization_complete":
            return f"{prefix}화자 분리 완료: {payload.get('speakers', 0)}명"
        if event_type == "summary_started":
            return f"{prefix}본문 요약 생성 중..."
        if event_type == "file_complete":
            outputs = payload.get("outputs", {})
            return f"{prefix}완료: {', '.join(_sanitize(value) for value in outputs.values())}"
        if event_type == "file_failed":
            return f"{prefix}실패: {_sanitize(payload.get('error', '알 수 없는 오류'))}"
        if event_type == "artifacts_removed":
            key = _sanitize(payload.get("key", ""))
            key_prefix = f"[{key}] " if key else ""
            return f"{key_prefix}이전 실행의 산출물 {payload.get('count', 0)}개를 정리했습니다"
        if event_type == "provider_failed":
            return f"공급자 실패 ({_sanitize(payload.get('provider', '?'))}): {_sanitize(payload.get('error', '알 수 없는 오류'))}"
        if event_type == "config_warning":
            return f"설정 경고: {_sanitize(payload.get('message', ''))}"
        if event_type == "minutes_complete":
            return f"{prefix}본문 요약 완료: {_sanitize(payload.get('output', ''))} ({_sanitize(payload.get('provider', ''))})"
        if event_type == "run_complete":
            return f"작업 종료: 완료 {payload.get('completed', 0)}, 실패 {payload.get('failed', 0)}"
        if event_type == "error":
            return f"오류: {_sanitize(payload.get('error', '알 수 없는 오류'))}"
        return _sanitize(payload.get("message", ""))
