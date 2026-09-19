from __future__ import annotations

import io
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import RuntimeConfig
from .errors import DependencyError, UsageError
from .safeio import open_verified
from .transcript import (
    load_glossary,
    load_speaker_map,
    split_segments_by_speaker,
)



SUPPORTED_EXTENSIONS = {
    ".aac",
    ".avi",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogg",
    ".wav",
    ".webm",
    ".wma",
}


def default_compute_type(device: str) -> str:
    return "float16" if str(device or "auto").strip().lower() == "cuda" else "int8"


def _resolve_media_entry(entry: Path, root: Path, follow_symlinks: bool) -> Path | None:
    """Return the resolved media path for ``entry`` or ``None`` when it is not
    an acceptable regular file.

    Symlinked entries are rejected unless ``follow_symlinks`` is set; when
    followed, the link must resolve inside ``root`` or a UsageError is raised
    so discovery can never escape the input directory.
    """
    if entry.is_symlink():
        if not follow_symlinks:
            return None
        resolved = entry.resolve()
        if not resolved.is_relative_to(root):
            raise UsageError(f"심볼릭 링크가 입력 디렉터리를 벗어납니다: {entry} -> {resolved}")
        if not resolved.is_file():
            return None
        return resolved
    if not entry.is_file():
        return None
    return entry


def collect_media_files(input_path: str | Path, follow_symlinks: bool = False) -> list[Path]:
    raw_source = Path(input_path).expanduser()
    if raw_source.is_symlink():
        if not follow_symlinks:
            raise UsageError(f"심볼릭 링크 입력은 지원하지 않습니다: {raw_source}")
        resolved_source = raw_source.resolve()
        anchor = raw_source.parent.resolve()
        if not resolved_source.is_relative_to(anchor):
            raise UsageError(f"심볼릭 링크가 입력 디렉터리를 벗어납니다: {raw_source} -> {resolved_source}")
        source = resolved_source
    else:
        source = raw_source.resolve()
    if not source.exists():
        raise UsageError(f"입력 경로가 없습니다: {source}")
    root = source if source.is_dir() else source.parent
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UsageError(f"지원하지 않는 미디어 확장자입니다: {source.suffix}")
        return [source]
    seen: set[Path] = set()
    files: list[Path] = []
    for item in source.iterdir():
        if item.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        resolved = _resolve_media_entry(item, root, follow_symlinks)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        files.append(resolved)
    files.sort(key=lambda item: item.name.lower())
    if not files:
        raise UsageError(f"입력 디렉터리에 지원되는 미디어 파일이 없습니다: {source}")
    return files


def _materialize_segments(raw_segments: Iterable[Any], on_segment: Callable[[dict[str, Any]], None] | None = None, include_words: bool = False) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for segment in raw_segments:
        raw_text = str(getattr(segment, "text", "") or "").strip()
        if not raw_text:
            continue
        normalized: dict[str, Any] = {
            "start": float(getattr(segment, "start", 0.0)),
            "end": float(getattr(segment, "end", 0.0)),
            "text": raw_text,
            "raw_text": raw_text,
        }
        if include_words:
            words = []
            for word in getattr(segment, "words", None) or []:
                words.append(
                    {
                        "start": getattr(word, "start", None),
                        "end": getattr(word, "end", None),
                        "word": str(getattr(word, "word", "")),
                        "probability": getattr(word, "probability", None),
                    }
                )
            normalized["words"] = words
            probabilities = [float(word["probability"]) for word in words if word["probability"] is not None]
            if probabilities:
                normalized["confidence"] = sum(probabilities) / len(probabilities)
        segments.append(normalized)
        if on_segment:
            on_segment(normalized)
    return segments


class _NamedAudio(io.RawIOBase):
    """Binary file object whose ``.name`` is the source path.

    faster-whisper accepts ``BinaryIO``; pyannote accepts ``{"audio": file}``.
    ``os.fdopen`` leaves ``.name`` as the fd int, so wrap the raw file and
    expose the real path for downstream diagnostics/fakes.
    """

    def __init__(self, raw, name: str):
        self._raw = raw
        self.name = name

    def readable(self) -> bool:
        return True

    def read(self, size=-1):
        return self._raw.read(size)

    def readinto(self, b):
        return self._raw.readinto(b)

    def seek(self, offset, whence=0):
        return self._raw.seek(offset, whence)

    def tell(self):
        return self._raw.tell()

    def seekable(self) -> bool:
        return True

    def fileno(self):
        return self._raw.fileno()

    def close(self):
        try:
            self._raw.close()
        finally:
            super().close()


def _open_media(media_path: Path):
    """Open ``media_path`` proving the opened file is the collected file.

    ``collect_media_files`` validates the directory entry once; the decoder
    must not re-trust the pathname afterward. ``safeio.open_verified``
    rejects a final-component symlink, walks the resolved path with
    ``O_NOFOLLOW`` per component, and requires the opened file's
    device+inode to match the checked entry — a swap to a symlink or a
    different file between check and open fails instead of being read.
    The returned binary file object is what faster-whisper and pyannote
    consume — the path is never reopened by name.
    """
    try:
        fd = open_verified(media_path)
    except OSError as exc:
        raise UsageError(f"미디어 파일을 안전하게 열 수 없습니다(심볼릭 링크이거나 접근 불가): {media_path}") from exc
    return _NamedAudio(os.fdopen(fd, "rb"), str(media_path))


def _transcribe_once(
    model: Any,
    media_path: Path,
    config: RuntimeConfig,
    on_segment: Callable[[dict[str, Any]], None] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    options: dict[str, Any] = {
        # The configured language applies to both transcribe and translate;
        # faster-whisper only auto-detects when no language is configured.
        "language": config.language or None,
        "task": config.task,
        "beam_size": config.beam_size,
        "vad_filter": config.vad_filter,
    }
    if config.diarize:
        options["word_timestamps"] = True
    with _open_media(media_path) as audio:
        raw_segments, info = model.transcribe(audio, **options)
        duration = getattr(info, "duration", None) or getattr(info, "duration_after_vad", None)
        started_at = time.monotonic()

        def handle_segment(segment: dict[str, Any]) -> None:
            if on_segment:
                on_segment(segment)
            if on_progress:
                elapsed = time.monotonic() - started_at
                progress = None
                eta = None
                if duration:
                    progress = min(99.0, max(0.0, float(segment["end"]) / float(duration) * 100.0))
                    if progress > 0:
                        eta = elapsed * (100.0 / progress - 1.0)
                on_progress({"segment": segment, "progress": progress, "eta_seconds": eta, "elapsed_seconds": elapsed})

        return _materialize_segments(raw_segments, on_segment=handle_segment, include_words=config.diarize), info


def _should_retry_vad(media_path: Path, config: RuntimeConfig, segments: Sequence[dict[str, Any]]) -> bool:
    return bool(
        config.vad_filter
        and not segments
        and config.task == "transcribe"
        and media_path.suffix.lower() in {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav", ".wma"}
    )


def _diarize_file(media_path: Path, config: RuntimeConfig, environment: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    # The merged environment (--env-file / user env / process env) is the
    # documented secret source; process env alone would ignore env files.
    env = dict(os.environ if environment is None else environment)
    token = str(env.get("HF_TOKEN", "") or "")
    if not token:
        raise DependencyError("화자 분리를 사용하려면 HF_TOKEN 환경 변수가 필요합니다.")
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DependencyError("화자 분리 의존성이 없습니다. `pip install 'mediascribe-cli[diarization]'` 후 다시 시도하세요.") from exc
    try:
        pipeline = Pipeline.from_pretrained(config.diarization_model, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(config.diarization_model, use_auth_token=token)
    if pipeline is None:
        raise DependencyError(f"화자 분리 모델을 불러오지 못했습니다: {config.diarization_model}")

    options: dict[str, int] = {}
    if config.num_speakers > 0:
        options["num_speakers"] = config.num_speakers
    else:
        if config.min_speakers > 0:
            options["min_speakers"] = config.min_speakers
        if config.max_speakers > 0:
            options["max_speakers"] = config.max_speakers
    with _open_media(media_path) as audio:
        diarization = pipeline({"audio": audio}, **options)
    return [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]


def validate_runtime_dependencies(config: RuntimeConfig, environment: Mapping[str, str] | None = None) -> None:
    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        raise DependencyError("faster-whisper가 설치되지 않았습니다. `pip install -e .` 후 다시 시도하세요.") from exc
    if config.diarize:
        env = dict(os.environ if environment is None else environment)
        if not str(env.get("HF_TOKEN", "") or ""):
            raise DependencyError("화자 분리를 사용하려면 HF_TOKEN 환경 변수가 필요합니다.")
        try:
            import pyannote.audio  # noqa: F401
        except ImportError as exc:
            raise DependencyError("화자 분리 의존성이 없습니다. `pip install 'mediascribe-cli[diarization]'` 후 다시 시도하세요.") from exc




def transcribe_file(
    media_path: Path,
    config: RuntimeConfig,
    emitter: Any,
    model: Any,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Transcribe one media file and return the transcript result.

    Emits transcript_segment/progress/diarization events via ``emitter``.
    Writes no artifacts — checkpointing and artifact staging belong to the
    pipeline caller.
    """
    glossary = load_glossary(config.glossary_path)
    speaker_map = load_speaker_map(config.speaker_map_path)

    def on_segment(segment: dict[str, Any]) -> None:
        emitter.emit(
            "transcript_segment",
            file_name=media_path.name,
            start=segment["start"],
            end=segment["end"],
            text=segment["text"],
        )

    def on_progress(update: dict[str, Any]) -> None:
        segment = update["segment"]
        emitter.emit(
            "progress",
            stage="stt",
            file_name=media_path.name,
            progress=update["progress"],
            elapsed_seconds=update["elapsed_seconds"],
            eta_seconds=update["eta_seconds"],
            text=segment["text"],
        )

    segments, info = _transcribe_once(model, media_path, config, on_segment=on_segment, on_progress=on_progress)
    vad_retry_used = False
    vad_filter_used = config.vad_filter
    if _should_retry_vad(media_path, config, segments):
        retry_config = replace(config, vad_filter=False)
        retry_segments, retry_info = _transcribe_once(model, media_path, retry_config, on_segment=on_segment, on_progress=on_progress)
        if retry_segments:
            segments, info = retry_segments, retry_info
            vad_retry_used = True
            vad_filter_used = False
            emitter.emit("vad_retry", file_name=media_path.name)

    turns: list[dict[str, Any]] = []
    if config.diarize:
        emitter.emit("diarization_started", file_name=media_path.name, model=config.diarization_model)
        emitter.emit("progress", stage="diarization", file_name=media_path.name, progress=None, elapsed_seconds=0.0, eta_seconds=None, text="화자 분리 중")
        diarization_started_at = time.monotonic()
        turns = _diarize_file(media_path, config, environment)
        diarization_elapsed = time.monotonic() - diarization_started_at
        emitter.emit("progress", stage="diarization", file_name=media_path.name, progress=None, elapsed_seconds=diarization_elapsed, eta_seconds=None, text="화자 분리 완료")
        emitter.emit("diarization_complete", file_name=media_path.name, speakers=len({turn["speaker"] for turn in turns}), elapsed_seconds=diarization_elapsed)

    enriched, speakers = split_segments_by_speaker(
        segments,
        diarization_turns=turns,
        glossary=glossary,
        requested_labels=config.speaker_labels,
        speaker_map=speaker_map,
    )
    detected_language = str(getattr(info, "language", "") or config.language or "unknown")
    language_probability = getattr(info, "language_probability", None)
    selected_compute_type = config.compute_type or default_compute_type(config.device)
    metadata = {
        "model": config.model,
        "device": config.device,
        "compute_type": selected_compute_type,
        "detected_language": detected_language,
        "language_probability": language_probability,
        "task": config.task,
        "diarization": config.diarize,
        "diarization_model": config.diarization_model if config.diarize else "",
        "vad_retry_used": vad_retry_used,
        "vad_filter_used": vad_filter_used,
        "glossary_path": str(Path(config.glossary_path).expanduser().resolve()) if config.glossary_path else "",
    }
    return {
        "media_path": str(media_path),
        "detected_language": detected_language,
        "language_probability": language_probability,
        "segments": enriched,
        "speakers": speakers,
        "metadata": metadata,
    }
