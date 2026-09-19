from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from . import __version__
from .artifacts import ARTIFACT_KINDS, ARTIFACT_SUFFIX, ArtifactStager, ArtifactTarget, cleanup_staging, plan_artifacts, plan_output_keys, reconcile_commits
from .chunking import chunk_transcript
from .config import build_runtime_config, load_environment, provider_candidates
from .errors import CLIError, PartialFailure, ProviderError, UsageError
from .events import EventEmitter
from .manifest import Manifest, config_fingerprint, file_fingerprint
from .meeting_minutes import generate_body_summary, summary_config_fingerprint
from .pipeline import PipelineContext, _bound_key, run_pipeline, transcript_config_fingerprint
from .transcriber import collect_media_files
from .transcript import (
    load_glossary,
    load_speaker_map,
    normalize_artifact_format,
    read_transcript_source,
    render_summary,
    transcript_lines,
)


def _add_common_transcription_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="전사할 미디어 파일 또는 디렉터리")
    parser.add_argument("--output-dir", default="outputs", help="출력 디렉터리 (기본값: ./outputs)")
    parser.add_argument("--format", dest="output_format", choices=("html", "md", "txt"), default=None, help="세 산출물 포맷 (기본값: md)")
    parser.add_argument("--model", default=None, help="faster-whisper 모델 (기본값: small)")
    parser.add_argument("--language", default=None, help="언어 코드, 예: ko")
    parser.add_argument("--task", choices=("transcribe", "translate"), default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda")
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--no-vad-filter", action="store_true")
    parser.add_argument("--diarize", action="store_true", help="pyannote.audio 화자 분리 사용 (HF_TOKEN 환경 변수 필요)")
    parser.add_argument("--diarization-model", default=None)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--glossary", default=None, help="용어 사전 JSON")
    parser.add_argument("--speaker-map", default=None, help="화자 ID → 표시명 JSON")
    parser.add_argument("--speaker-labels", default=None, help="최초 등장 순서 레이블, 예: A,B,C")
    parser.add_argument("--meeting-title", default="", help="본문 요약 제목")
    parser.add_argument("--minutes-provider", choices=("openrouter", "opencode"), default="", help="요약 공급자 강제 선택")
    parser.add_argument("--minutes-timeout", type=float, default=120.0, help="요약 API timeout 초")
    parser.add_argument("--json-lines", action="store_true", help="사람용 로그 대신 stdout JSONL 이벤트 출력")
    parser.add_argument("--dry-run", action="store_true", help="입력과 설정만 검증하고 모델을 로드하지 않음")
    _add_environment_options(parser)
    parser.add_argument("--no-resume", action="store_true", default=None, help="매니페스트·체크포인트 재개를 끄고 항상 새로 처리")
    parser.add_argument("--follow-symlinks", action="store_true", help="입력 디렉터리 안의 심볼릭 링크를 따라감")
    parser.add_argument("--summary-chunk-chars", type=int, default=None, help="요약 요청 한 번의 최대 전사 문자 수 (기본값: 24000)")


def _add_environment_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", default=None, help="환경 변수 파일 경로 (MEDIASCRIBE_ENV_FILE과 동일)")
    parser.add_argument("--allow-custom-endpoint", action="store_true", help="허용 목록에 없는 https 공급자 엔드포인트 허용")
    parser.add_argument("--events-full-text", action="store_true", help="JSONL 이벤트에 전사 텍스트를 그대로 포함(기본은 해시로 대체)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mediascribe", description="Standalone MediaScribe transcription and meeting-minutes CLI")
    parser.add_argument("--version", action="version", version=f"MediaScribe-CLI {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser("transcribe", help="음성·영상 파일을 전사하고 3개 산출물을 저장합니다.")
    _add_common_transcription_options(transcribe)

    minutes = subparsers.add_parser("minutes", help="기존 전사 파일에서 본문 요약을 생성합니다.")
    minutes.add_argument("input", help="TXT, Markdown 또는 구조화 JSON 전사 파일")
    minutes.add_argument("--output", default="", help="요약 출력 경로")
    minutes.add_argument("--format", dest="output_format", choices=("html", "md", "txt"), default=None, help="요약 포맷 (기본값: md)")
    minutes.add_argument("--title", default="", help="요약 제목")
    minutes.add_argument("--provider", choices=("openrouter", "opencode"), default="", help="공급자 강제 선택")
    minutes.add_argument("--timeout", type=float, default=120.0)
    minutes.add_argument("--summary-chunk-chars", type=int, default=None, help="요약 요청 한 번의 최대 전사 문자 수 (기본값: 24000)")
    minutes.add_argument("--json-lines", action="store_true")
    _add_environment_options(minutes)
    return parser


def _transcription_config(args, environment):
    config = build_runtime_config(args, environment=environment)
    return replace(config, artifact_format=normalize_artifact_format(config.artifact_format))


def _validate_optional_inputs(config) -> None:
    if config.glossary_path:
        load_glossary(config.glossary_path)
    if config.speaker_map_path:
        load_speaker_map(config.speaker_map_path)


def _dry_run_report(args, config, emitter, environment, files, output_dir, manifest) -> int:
    """Read-only run report: resolved keys, manifest skip state, chunk
    estimate, and validated provider endpoints. No lock, no writes."""
    config_fp = config_fingerprint(config)
    file_states: list[dict] = []
    key_map = plan_output_keys(files)
    for media_path in files:
        key = key_map[media_path]
        title = getattr(args, "meeting_title", None) or media_path.stem
        # Report the key the real run would use: the manifest media_path
        # binding can re-key a same-stem input from a different path.
        bound = _bound_key(media_path, key, manifest, emitter)
        combined = f"{file_fingerprint(media_path)}|{config_fp}"
        entry = manifest.entry(bound) if manifest is not None else None
        checkpoint = (
            manifest.transcript_checkpoint(bound, media_path, file_fingerprint(media_path), transcript_config_fingerprint(config))
            if manifest is not None
            else None
        )
        estimated_chunks = None
        if checkpoint is not None:
            lines = transcript_lines(checkpoint.get("segments") or [])
            estimated_chunks = len(chunk_transcript(lines, config.summary_chunk_chars))
        expected_outputs = {
            kind: target.path
            for kind, target in plan_artifacts(output_dir, bound, config.artifact_format).items()
        }
        summary_fp = summary_config_fingerprint(environment, args.minutes_provider, title, config.summary_chunk_chars)
        file_states.append(
            {
                "file": str(media_path),
                "key": bound,
                "manifest_state": (entry or {}).get("state") or "new",
                "will_skip": bool(
                    manifest is not None
                    and manifest.should_skip(
                        bound, combined, ARTIFACT_KINDS,
                        expected_outputs=expected_outputs, summary_fingerprint=summary_fp,
                    )
                ),
                "estimated_summary_chunks": estimated_chunks,
            }
        )
    providers = provider_candidates(
        environment,
        provider=args.minutes_provider,
        allow_custom_endpoint=config.allow_custom_endpoint,
        emitter=emitter,
    )
    emitter.emit(
        "dry_run",
        files=file_states,
        output_dir=str(output_dir),
        model=config.model,
        artifact_format=config.artifact_format,
        artifact_names=list(ARTIFACT_SUFFIX.values()),
        diarize=config.diarize,
        resume=config.resume,
        summary_chunk_chars=config.summary_chunk_chars,
        minutes_providers=[{"id": provider.id, "base_url": provider.base_url, "model": provider.model} for provider in providers],
        message=f"dry-run: {len(files)}개 파일, 출력 {output_dir}, 공급자 {len(providers)}개",
    )
    return 0


def _run_transcribe(args) -> int:
    environment = load_environment(args.env_file)
    config = _transcription_config(args, environment)
    emitter = EventEmitter(json_lines=config.json_lines, redact=not config.events_full_text)
    files = collect_media_files(args.input, follow_symlinks=config.follow_symlinks)
    output_dir = Path(args.output_dir).expanduser().resolve()
    _validate_optional_inputs(config)

    if args.dry_run:
        manifest = Manifest.open(output_dir, read_only=True) if config.resume else None
        try:
            return _dry_run_report(args, config, emitter, environment, files, output_dir, manifest)
        finally:
            if manifest is not None:
                manifest.close()

    # The managed output dir is private: transcripts and artifacts live here.
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(output_dir, 0o700)
    except OSError:
        pass

    manifest = Manifest.open(output_dir) if config.resume else None
    try:
        # PID-aware: safe without a lock — live concurrent temps/markers are
        # skipped, dead-pid leftovers (incl. --no-resume crashes) are removed.
        cleanup_staging(output_dir)
        reconcile_commits(output_dir)
        ctx = PipelineContext(
            config=config,
            emitter=emitter,
            output_dir=output_dir,
            environment=environment,
            manifest=manifest,
            meeting_title=args.meeting_title,
            minutes_provider=args.minutes_provider,
            minutes_timeout=args.minutes_timeout,
            allow_custom_endpoint=config.allow_custom_endpoint,
        )
        _committed, errors = run_pipeline(files, ctx)
    finally:
        if manifest is not None:
            manifest.close()

    if not errors:
        return 0
    summary_errors = [error for error in errors if error.stage == "summary"]
    if summary_errors:
        raise ProviderError("; ".join(str(error) for error in errors))
    raise PartialFailure("; ".join(str(error) for error in errors))


def _run_minutes(args) -> int:
    environment = load_environment(args.env_file)
    config = build_runtime_config(args, environment=environment)
    emitter = EventEmitter(json_lines=config.json_lines, redact=not config.events_full_text)
    raw_source = Path(args.input).expanduser()
    if raw_source.is_symlink():
        raise UsageError(f"심볼릭 링크 입력은 지원하지 않습니다: {raw_source}")
    source = raw_source.resolve()
    if not source.is_file():
        raise UsageError(f"전사 파일이 없습니다: {source}")
    artifact_format = normalize_artifact_format(config.artifact_format)
    transcript = read_transcript_source(source)
    title = args.title or source.stem
    emitter.emit("summary_started", file_name=source.name)
    emitter.emit("progress", stage="summary", file_name=source.name, progress=None, elapsed_seconds=0.0, eta_seconds=None, text="본문 요약 API 요청 중")
    started_at = time.monotonic()
    generated = generate_body_summary(
        transcript,
        title=title,
        environment=environment,
        provider=args.provider,
        timeout=args.timeout,
        max_chunk_chars=config.summary_chunk_chars,
        emitter=emitter,
        allow_custom_endpoint=config.allow_custom_endpoint,
    )
    if args.output:
        destination = Path(args.output).expanduser().resolve()
    else:
        destination = source.parent / f"{source.stem}_본문요약.{artifact_format}"
    stager = ArtifactStager({"body_summary": ArtifactTarget(kind="body_summary", path=destination)})
    try:
        stager.stage("body_summary", render_summary(generated.text, title, artifact_format))
        stager.commit()
    except Exception:
        stager.abort()
        raise
    emitter.emit(
        "minutes_complete",
        output=str(destination),
        provider=generated.provider,
        model=generated.model,
        elapsed_seconds=time.monotonic() - started_at,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    emitter = EventEmitter(json_lines=bool(getattr(args, "json_lines", False)))
    try:
        if args.command == "transcribe":
            return _run_transcribe(args)
        if args.command == "minutes":
            return _run_minutes(args)
        parser.error("명령을 선택하세요.")
    except CLIError as exc:
        emitter.error(str(exc))
        return exc.exit_code
    except (OSError, ValueError, RecursionError) as exc:
        emitter.error(str(exc))
        return 2
    except KeyboardInterrupt:
        print("중단되었습니다.", file=sys.stderr)
        return 130
    return 2
