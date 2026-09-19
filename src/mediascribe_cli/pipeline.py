"""Per-file pipeline state machine: STT → checkpoint → summarize → commit.

Implements production-readiness §4.4 and the §6 data flow:

    collect_media_files ──► output_key ──► manifest.should_skip? ──yes──► file_skipped
            │no
            ▼
    transcribe_file (STT events) ──► save_transcript_checkpoint ──► state=transcribed
            ▼
    transcript_lines ──► chunked generate_body_summary (per-digest checkpoints)
            ▼
    ArtifactStager.stage ×3 ──► commit (rename → fsync → stale cleanup)
            ▼
    state=committed ──► file_complete

Any failure aborts staged temps, removes final artifacts for the key, marks
the manifest entry ``failed``, and — for per-file errors — lets the run
continue with the remaining inputs.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import (
    ARTIFACT_KINDS,
    ARTIFACT_SUFFIX,
    ARTIFACT_EXTENSIONS,
    ArtifactStager,
    disambiguated_key,
    plan_artifacts,
    plan_output_keys,
)
from .config import RuntimeConfig
from .errors import CLIError, ConfigError, DependencyError, ProviderError, UsageError
from .events import EventEmitter
from .manifest import Manifest, config_fingerprint, file_fingerprint
from .meeting_minutes import generate_body_summary, summary_config_fingerprint
from .transcriber import (
    default_compute_type,
    transcribe_file,
    validate_runtime_dependencies,
)
from .transcript import (
    render_full_body,
    render_speaker_script,
    render_summary,
    transcript_lines,
)


class StageError(CLIError):
    """A per-file failure at a known pipeline stage. Run continues; exit 5."""

    exit_code = 5

    def __init__(self, stage: str, file_name: str, cause: Exception):
        self.stage = stage
        self.file_name = file_name
        self.cause = cause
        super().__init__(f"{file_name}: {stage} 단계 실패: {cause}")


@dataclass
class PipelineContext:
    config: RuntimeConfig
    emitter: EventEmitter
    output_dir: Path
    environment: Mapping[str, str]
    manifest: Manifest | None  # None when --no-resume
    model: Any = None  # lazy faster-whisper model
    meeting_title: str = ""
    minutes_provider: str = ""
    minutes_timeout: float = 120.0
    allow_custom_endpoint: bool = False


def ensure_model(ctx: PipelineContext) -> Any:
    """Build the faster-whisper model on first use and cache it on the context."""
    if ctx.model is not None:
        return ctx.model
    config = ctx.config
    validate_runtime_dependencies(config, ctx.environment)
    ctx.emitter.emit(
        "progress",
        stage="model",
        progress=None,
        elapsed_seconds=0.0,
        eta_seconds=None,
        text=f"모델 로딩 중: {config.model}",
    )
    started_at = time.perf_counter()
    from faster_whisper import WhisperModel

    ctx.model = WhisperModel(
        config.model,
        device=config.device,
        compute_type=config.compute_type or default_compute_type(config.device),
    )
    ctx.emitter.emit(
        "progress",
        stage="model",
        progress=None,
        elapsed_seconds=time.perf_counter() - started_at,
        eta_seconds=None,
        text="모델 로딩 완료",
    )
    return ctx.model


def transcript_config_fingerprint(config: RuntimeConfig) -> str:
    """Config fingerprint restricted to fields that change transcript output.

    ``artifact_format`` only changes rendering, so a format-only rerun must
    still reuse the transcript checkpoint.
    """
    return config_fingerprint(replace(config, artifact_format=""))

def _remove_final_artifacts(output_dir: Path, key: str, keep: frozenset[Path] = frozenset()) -> int:
    """Delete final artifacts for ``key`` across all formats, except paths in
    ``keep`` (a prior run's still-complete committed set).

    A failed or interrupted input leaves zero final artifacts *of its own*:
    staged temps are removed by ``ArtifactStager.abort`` and this clears
    partial finals — but never a complete triplet a previous run committed,
    since an aborted rerun must not destroy still-valid deliverables.
    """
    removed = 0
    for kind in ARTIFACT_KINDS:
        for extension in ARTIFACT_EXTENSIONS:
            candidate = Path(output_dir) / f"{key}_{ARTIFACT_SUFFIX[kind]}.{extension}"
            try:
                if candidate.resolve() in keep:
                    continue
                if os.path.lexists(candidate):
                    candidate.unlink()
                    removed += 1
            except OSError:
                pass
    return removed


def _recorded_complete_set(output_dir: Path, entry: dict | None, manifest: Manifest | None = None) -> frozenset[Path]:
    """Resolved recorded output paths when they still form a complete,
    untampered in-output-dir triplet — i.e. a prior run's committed set that
    must be preserved. Empty set when missing, incomplete, foreign, or
    partially replaced (digest mismatch)."""
    if manifest is None or not manifest.recorded_outputs_intact(entry, ARTIFACT_KINDS):
        return frozenset()
    outputs = entry.get("outputs") if isinstance(entry, dict) else {}
    keep: set[Path] = set()
    for recorded in outputs.values():
        try:
            keep.add(Path(str(recorded)).resolve())
        except OSError:
            return frozenset()
    return frozenset(keep)


def _on_disk_complete_set(output_dir: Path, key: str) -> frozenset[Path]:
    """Complete triplet found on disk for ``key`` — the ``--no-resume``
    fallback when no manifest can vouch for a prior run's commit. Union over
    all formats; a complete-looking set is kept rather than destroyed, even
    though without recorded digests a mixed set cannot be distinguished."""
    keep: set[Path] = set()
    for extension in ARTIFACT_EXTENSIONS:
        paths = [Path(output_dir) / f"{key}_{ARTIFACT_SUFFIX[kind]}.{extension}" for kind in ARTIFACT_KINDS]
        try:
            if all(path.is_file() and not path.is_symlink() for path in paths):
                keep.update(path.resolve() for path in paths)
        except OSError:
            continue
    return frozenset(keep)


def _bound_key(media_path: Path, key: str, manifest: Manifest | None, emitter: EventEmitter) -> str:
    """Bind ``key`` to this input's identity. A prior entry whose recorded
    ``media_path`` resolves to a different file — or a legacy entry with no
    recorded path at all — must not reuse or overwrite that key's committed
    artifacts: re-key with the shared path-digest scheme and warn."""
    if manifest is None:
        return key
    prior = manifest.entry(key)
    if prior is None:
        return key
    recorded_media = str(prior.get("media_path") or "")
    same_input = False
    if recorded_media:
        try:
            same_input = Path(recorded_media).resolve() == media_path.resolve()
        except OSError:
            same_input = False  # unprovable → treat as a different input
    if same_input:
        return key
    new_key = disambiguated_key(media_path)
    emitter.emit(
        "config_warning",
        file_name=media_path.name,
        message=f"다른 입력이 '{key}' 키를 사용 중입니다. 출력 키를 '{new_key}'로 구분합니다.",
    )
    return new_key


def _reconcile_uncommitted(ctx: PipelineContext) -> None:
    """Startup self-heal for crashed commits.

    A crash between the three ``os.replace`` calls and ``set_state``
    leaves a partial triplet behind a non-``committed`` entry. At run start
    (manifest lock held), delete partial finals for each non-committed key —
    but preserve a recorded output set that is still complete: that is a
    prior run's valid committed triplet, not this run's debris.

    Recorded output paths are never unlinked — a tampered manifest could
    point them outside ``output_dir``. ``_remove_final_artifacts`` only ever
    touches deterministic ``{key}_{suffix}.{ext}`` names inside the output
    directory, which covers every artifact this pipeline can write.
    """
    manifest = ctx.manifest
    if manifest is None:
        return
    for key, entry in manifest.entries().items():
        if entry.get("state") == "committed":
            continue
        if not key or key in (".", "..") or Path(key).name != key:
            continue  # tampered manifest key — never build paths from it
        keep = _recorded_complete_set(ctx.output_dir, entry, manifest)
        removed = _remove_final_artifacts(ctx.output_dir, key, keep=keep)
        if removed:
            ctx.emitter.emit("artifacts_removed", key=key, count=removed)


class _StageTracker:
    """Emitter proxy that records whether diarization began, so a failure
    inside ``transcribe_file`` can be attributed to the right stage."""

    def __init__(self, emitter: EventEmitter):
        self._emitter = emitter
        self.diarization_started = False

    def emit(self, event_type: str, **payload: Any) -> None:
        if event_type == "diarization_started":
            self.diarization_started = True
        self._emitter.emit(event_type, **payload)


def _abort_failed(ctx: PipelineContext, key: str, stager: ArtifactStager | None, error: BaseException) -> None:
    """Common failure path: drop temps, drop this run's partial finals, mark
    failed. A prior run's still-complete committed set is preserved — an
    aborted rerun must not convert valid deliverables into zero outputs."""
    if stager is not None:
        stager.abort()
    entry = ctx.manifest.entry(key) if ctx.manifest is not None else None
    if ctx.manifest is not None:
        keep = _recorded_complete_set(ctx.output_dir, entry, ctx.manifest)
    else:
        # --no-resume: no manifest can vouch for a prior commit, but a
        # complete on-disk triplet is still better preserved than destroyed.
        keep = _on_disk_complete_set(ctx.output_dir, key)
    removed = _remove_final_artifacts(ctx.output_dir, key, keep=keep)
    if removed:
        ctx.emitter.emit("artifacts_removed", key=key, count=removed)
    if ctx.manifest is not None:
        try:
            ctx.manifest.set_state(key, "failed", error=str(error))
        except Exception:
            pass  # never mask the real failure with a manifest write error

def process_file(media_path: Path, key: str, ctx: PipelineContext) -> dict[str, str]:
    """Run the full per-file state machine. Returns ``{kind: final_path}``.

    Raises ``StageError(stage='stt'|'diarization'|'summary'|'commit')`` for
    per-file failures. Fatal errors (``DependencyError``, ``ConfigError``,
    ``UsageError``, ``KeyboardInterrupt``) are re-raised after the same
    abort/cleanup so the caller can abort the run with the right exit code.
    """


    config = ctx.config
    emitter = ctx.emitter
    manifest = ctx.manifest
    stager: ArtifactStager | None = None
    try:
        title = ctx.meeting_title or media_path.stem
        key = _bound_key(media_path, key, manifest, emitter)
        targets = plan_artifacts(ctx.output_dir, key, config.artifact_format)
        summary_fp = summary_config_fingerprint(
            ctx.environment, ctx.minutes_provider, title, config.summary_chunk_chars
        )
        file_fp = config_fp = tx_fp = ""
        if manifest is not None:
            try:
                file_fp = file_fingerprint(media_path)
            except OSError as exc:
                raise StageError("stt", media_path.name, exc) from exc
            config_fp = config_fingerprint(config)
            tx_fp = transcript_config_fingerprint(config)
            combined = f"{file_fp}|{config_fp}"
            expected_outputs = {kind: target.path for kind, target in targets.items()}
            if manifest.should_skip(
                key, combined, ARTIFACT_KINDS,
                expected_outputs=expected_outputs, summary_fingerprint=summary_fp,
            ):
                prior = manifest.entry(key) or {}
                outputs = {str(kind): str(path) for kind, path in (prior.get("outputs") or {}).items()}
                emitter.emit("file_skipped", file_name=media_path.name, reason="이미 처리됨")
                return outputs
            manifest.begin(key, combined, "transcribing", media_path=str(media_path))

        # -- STT (or transcript-checkpoint reuse) --------------------------
        # The checkpoint self-validates: stored media_path must resolve to
        # this file, and both the file and transcript-config fingerprints
        # must match — no separate manifest gate needed.
        result: dict | None = None
        if manifest is not None:
            result = manifest.transcript_checkpoint(key, media_path, file_fp, tx_fp)
        if result is None:
            tracker = _StageTracker(emitter)
            try:
                result = transcribe_file(media_path, config, tracker, ensure_model(ctx), environment=ctx.environment)
            except (DependencyError, ConfigError, UsageError):
                raise
            except Exception as exc:
                stage = "diarization" if tracker.diarization_started else "stt"
                raise StageError(stage, media_path.name, exc) from exc
            if manifest is not None:
                manifest.save_transcript_checkpoint(key, result, file_fp, tx_fp)
                emitter.emit("checkpoint_saved", file_name=media_path.name, kind="transcript")
        else:
            emitter.emit(
                "progress",
                stage="stt",
                file_name=media_path.name,
                progress=None,
                elapsed_seconds=0.0,
                eta_seconds=None,
                text="저장된 전사 체크포인트 사용",
            )
        if manifest is not None:
            manifest.set_state(key, "transcribed", transcript_fingerprint=tx_fp, config_fingerprint=config_fp)

        # -- Summary (before any final artifact) ---------------------------
        if manifest is not None:
            manifest.set_state(key, "summarizing")
        emitter.emit("summary_started", file_name=media_path.name)
        emitter.emit(
            "progress",
            stage="summary",
            file_name=media_path.name,
            progress=None,
            elapsed_seconds=0.0,
            eta_seconds=None,
            text="본문 요약 API 요청 중",
        )
        summary_started_at = time.perf_counter()
        try:
            generated = generate_body_summary(
                "\n".join(transcript_lines(result.get("segments") or [])),
                title=title,
                environment=ctx.environment,
                provider=ctx.minutes_provider,
                timeout=ctx.minutes_timeout,
                max_chunk_chars=config.summary_chunk_chars,
                emitter=emitter,
                allow_custom_endpoint=ctx.allow_custom_endpoint,
                checkpoint=manifest.summary_store(key) if manifest is not None else None,
            )
        except (DependencyError, ConfigError, UsageError):
            raise
        except Exception as exc:
            raise StageError("summary", media_path.name, exc) from exc
        emitter.emit(
            "minutes_complete",
            file_name=media_path.name,
            output=str(targets["body_summary"].path),
            provider=generated.provider,
            model=generated.model,
            elapsed_seconds=time.perf_counter() - summary_started_at,
        )

        # -- Stage all three rendered artifacts, then commit atomically ----
        stager = ArtifactStager(targets)
        try:
            stager.stage("speaker_script", render_speaker_script(media_path, result.get("segments") or [], config.artifact_format))
            stager.stage("body_summary", render_summary(generated.text, title, config.artifact_format))
            stager.stage("full_body", render_full_body(media_path, result.get("segments") or [], config.artifact_format))
            outputs = stager.commit()
            output_digests = stager.digests
        except (DependencyError, ConfigError, UsageError):
            raise
        except Exception as exc:
            raise StageError("commit", media_path.name, exc) from exc
        stager = None  # committed; nothing left to abort

        if manifest is not None:
            manifest.set_state(key, "committed", outputs=outputs, output_digests=output_digests, summary_fingerprint=summary_fp, error=None)
        emitter.emit("file_complete", file_name=media_path.name, outputs=outputs)
        return outputs
    except StageError as exc:
        _abort_failed(ctx, key, stager, exc)
        raise
    except (DependencyError, ConfigError, UsageError, KeyboardInterrupt) as exc:
        _abort_failed(ctx, key, stager, exc)
        raise
    except Exception as exc:
        _abort_failed(ctx, key, stager, exc)
        raise StageError("commit", media_path.name, exc) from exc


def run_pipeline(files: Sequence[Path], ctx: PipelineContext) -> tuple[int, list[StageError]]:
    """Per-file loop: emit ``file_started``, run ``process_file``, count
    committed files, collect ``StageError``s. Never raises for per-file
    failures; fatal errors (dependency/config/usage/keyboard) propagate."""
    emitter = ctx.emitter
    emitter.emit("run_started", files=len(files), output_dir=str(ctx.output_dir))
    _reconcile_uncommitted(ctx)
    keys = plan_output_keys(files)
    committed = 0
    errors: list[StageError] = []
    for media_path in files:
        key = keys[Path(media_path)]
        emitter.emit("file_started", file_name=media_path.name)
        try:
            process_file(media_path, key, ctx)
            committed += 1
        except StageError as exc:
            errors.append(exc)
            emitter.emit("file_failed", file_name=exc.file_name, stage=exc.stage, error=str(exc.cause))
    emitter.emit("run_complete", completed=committed, failed=len(errors), output_dir=str(ctx.output_dir))
    return committed, errors
