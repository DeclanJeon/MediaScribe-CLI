from __future__ import annotations

import hashlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mediascribe_cli.events import EventEmitter  # noqa: E402


def _emitter(json_lines=False, redact=True):
    stdout = io.StringIO()
    stderr = io.StringIO()
    return EventEmitter(json_lines=json_lines, stdout=stdout, stderr=stderr, redact=redact), stdout, stderr


class SanitizationTests(unittest.TestCase):
    def test_control_characters_and_ansi_stripped_from_human_output(self) -> None:
        emitter, _stdout, stderr = _emitter()
        emitter.emit("file_failed", file_name="evil\x1b[31m\x07name.m4a", error="boom\x1b[2J\x00")
        output = stderr.getvalue()
        self.assertNotIn("\x1b", output)
        self.assertNotIn("\x00", output)
        self.assertNotIn("\x07", output)
        self.assertNotIn("[31m", output)
        self.assertIn("evilname.m4a", output)

    def test_progress_text_is_sanitized(self) -> None:
        emitter, _stdout, stderr = _emitter()
        emitter.emit(
            "progress",
            stage="stt",
            file_name="a\x1b]8;;http://evil\x07b.m4a",
            progress=10.0,
            elapsed_seconds=1.0,
            text="seg\x1b[1mment",
        )
        output = stderr.getvalue()
        self.assertNotIn("\x1b", output)
        self.assertNotIn("http://evil", output)
        self.assertIn("segment", output)

    def test_secret_like_tokens_masked_in_human_output(self) -> None:
        emitter, _stdout, stderr = _emitter()
        emitter.emit(
            "error",
            error="request failed: Bearer sk-abcdef0123456789abcd token=hf_0123456789abcdef",
        )
        output = stderr.getvalue()
        self.assertNotIn("sk-abcdef0123456789abcd", output)
        self.assertNotIn("hf_0123456789abcdef", output)
        self.assertIn("***", output)

    def test_hf_underscore_token_masked(self) -> None:
        emitter, _stdout, stderr = _emitter()
        emitter.emit("error", error="auth failed for hf_Ab3dEf9012345xYz")
        output = stderr.getvalue()
        self.assertNotIn("hf_Ab3dEf9012345xYz", output)
        self.assertIn("***", output)

    def test_jsonl_error_fields_are_sanitized(self) -> None:
        # Error/message/reason strings get the same secret+control-char
        # masking as human output; other payload fields stay raw.
        emitter, stdout, _stderr = _emitter(json_lines=True, redact=False)
        emitter.emit("file_failed", file_name="n\x1b[31mame.m4a", error="raw\x00error")
        event = json.loads(stdout.getvalue().strip())
        self.assertEqual(event["type"], "file_failed")
        self.assertEqual(event["file_name"], "n\x1b[31mame.m4a")
        self.assertEqual(event["error"], "rawerror")


class RedactionTests(unittest.TestCase):
    def test_jsonl_redacts_transcript_text_by_default(self) -> None:
        emitter, stdout, _stderr = _emitter(json_lines=True)
        emitter.emit("transcript_segment", file_name="m.m4a", text="민감한 발화 내용", start=0.0)
        event = json.loads(stdout.getvalue().strip())
        self.assertIsInstance(event["text"], dict)
        self.assertIn("sha256", event["text"])
        self.assertNotIn("민감한 발화 내용", stdout.getvalue())

    def test_opt_out_keeps_full_text(self) -> None:
        emitter, stdout, _stderr = _emitter(json_lines=True, redact=False)
        emitter.emit("transcript_segment", file_name="m.m4a", text="민감한 발화 내용", start=0.0)
        event = json.loads(stdout.getvalue().strip())
        self.assertEqual(event["text"], "민감한 발화 내용")

    def test_redacted_jsonl_replaces_text_with_digest(self) -> None:
        emitter, stdout, _stderr = _emitter(json_lines=True, redact=True)
        emitter.emit("transcript_segment", file_name="m.m4a", text="민감한 발화 내용", start=0.0)
        emitter.emit("progress", stage="stt", file_name="m.m4a", progress=5.0, text="진행 텍스트")
        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        for event, original in zip(events, ("민감한 발화 내용", "진행 텍스트")):
            self.assertIsInstance(event["text"], dict)
            self.assertEqual(event["text"]["sha256"], hashlib.sha256(original.encode("utf-8")).hexdigest())
            self.assertEqual(event["text"]["chars"], len(original))
        self.assertNotIn("민감한 발화 내용", stdout.getvalue())
        self.assertNotIn("진행 텍스트", stdout.getvalue())

    def test_redaction_leaves_other_fields_and_events_alone(self) -> None:
        emitter, stdout, _stderr = _emitter(json_lines=True, redact=True)
        emitter.emit("file_failed", file_name="m.m4a", error="plain error")
        emitter.emit("progress", stage="stt", file_name="m.m4a", progress=5.0, text="t")
        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(events[0]["error"], "plain error")
        self.assertEqual(events[1]["file_name"], "m.m4a")

    def test_human_output_unaffected_by_redaction(self) -> None:
        emitter, _stdout, stderr = _emitter(redact=True)
        emitter.emit("progress", stage="stt", file_name="m.m4a", progress=5.0, elapsed_seconds=2.0, text="표시 텍스트")
        self.assertIn("표시 텍스트", stderr.getvalue())


class ProgressFormatTests(unittest.TestCase):
    def test_normal_progress_output_is_stable(self) -> None:
        emitter, _stdout, stderr = _emitter()
        emitter.emit(
            "progress",
            stage="stt",
            file_name="meeting.m4a",
            progress=42.0,
            elapsed_seconds=18.0,
            eta_seconds=24.0,
            text="현재 세그먼트",
        )
        output = stderr.getvalue()
        self.assertIn("[meeting.m4a] STT  42.0%", output)
        self.assertIn("경과 00:18", output)
        self.assertIn("예상 잔여 00:24", output)
        self.assertIn("현재 세그먼트", output)

    def test_progress_without_values_uses_placeholders(self) -> None:
        emitter, _stdout, stderr = _emitter()
        emitter.emit("progress", stage="summary", file_name="m.m4a", progress=None, elapsed_seconds=0.0)
        output = stderr.getvalue()
        self.assertIn("요약", output)
        self.assertIn("--.-%", output)
        self.assertIn("경과 00:00", output)

    def test_stage_labels_cover_pipeline_stages(self) -> None:
        emitter, _stdout, stderr = _emitter()
        for stage in ("model", "stt", "diarization", "summary", "commit"):
            emitter.emit("progress", stage=stage, progress=1.0, elapsed_seconds=0.0)
        output = stderr.getvalue()
        for label in ("모델", "STT", "화자 분리", "요약", "저장"):
            self.assertIn(label, output)


class NewEventTypeTests(unittest.TestCase):
    def test_new_event_types_have_human_messages(self) -> None:
        emitter, _stdout, stderr = _emitter()
        emitter.emit("file_skipped", file_name="m.m4a", reason="이미 커밋됨")
        emitter.emit("checkpoint_saved", file_name="m.m4a", kind="transcript")
        emitter.emit("provider_failed", provider="openrouter", error="HTTP 503")
        emitter.emit("config_warning", message="custom endpoint allowed")
        output = stderr.getvalue()
        self.assertIn("건너뜀", output)
        self.assertIn("체크포인트", output)
        self.assertIn("openrouter", output)
        self.assertIn("HTTP 503", output)
        self.assertIn("custom endpoint allowed", output)

    def test_new_event_types_emit_jsonl(self) -> None:
        emitter, stdout, _stderr = _emitter(json_lines=True)
        emitter.emit("file_skipped", file_name="m.m4a", reason="committed")
        emitter.emit("provider_failed", provider="opencode", error="timeout")
        events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual([e["type"] for e in events], ["file_skipped", "provider_failed"])
        self.assertEqual(events[1]["provider"], "opencode")
    def test_artifacts_removed_has_human_message(self) -> None:
        emitter, _stdout, stderr = _emitter()
        emitter.emit("artifacts_removed", key="meeting", count=2)
        output = stderr.getvalue()
        self.assertIn("meeting", output)
        self.assertIn("2", output)

    def test_artifacts_removed_emits_jsonl(self) -> None:
        emitter, stdout, _stderr = _emitter(json_lines=True)
        emitter.emit("artifacts_removed", key="meeting", count=3)
        event = json.loads(stdout.getvalue().strip())
        self.assertEqual(event["type"], "artifacts_removed")
        self.assertEqual(event["key"], "meeting")
        self.assertEqual(event["count"], 3)



if __name__ == "__main__":
    unittest.main()
