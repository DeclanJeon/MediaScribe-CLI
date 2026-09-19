"""Focused integration tests for pipeline.py + cli.py (production §4.4/§4.7/§6).

All tests run with fake whisper/provider — no ML packages, no network.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mediascribe_cli.cli import main  # noqa: E402
from mediascribe_cli.config import RuntimeConfig  # noqa: E402
from mediascribe_cli.errors import ProviderError  # noqa: E402
from mediascribe_cli.events import EventEmitter  # noqa: E402
from mediascribe_cli.manifest import Manifest  # noqa: E402
from mediascribe_cli.meeting_minutes import GeneratedMinutes  # noqa: E402
from mediascribe_cli.pipeline import PipelineContext, run_pipeline  # noqa: E402


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _FakeSegment:
    def __init__(self, text="발화 내용", start=0.0, end=1.5):
        self.text = text
        self.start = start
        self.end = end


class _FakeInfo:
    language = "ko"
    language_probability = 0.99
    duration = 2.0


class _FakeModel:
    """Records transcribe() calls; can be told to fail for specific files."""

    def __init__(self, fail_names=()):
        self.calls = []
        self.fail_names = set(fail_names)

    def transcribe(self, audio, **_options):
        # ``audio`` is a binary file object (O_NOFOLLOW open); ``.name``
        # carries the source path for fail-name matching.
        name = Path(getattr(audio, "name", audio)).name
        self.calls.append(name)
        if name in self.fail_names:
            raise RuntimeError("STT exploded")
        return iter([_FakeSegment()]), _FakeInfo()


class _RecordingEmitter:
    def __init__(self):
        self.events = []

    def emit(self, event_type, **payload):
        self.events.append((event_type, payload))

    def types(self):
        return [kind for kind, _ in self.events]

    def of(self, event_type):
        return [payload for kind, payload in self.events if kind == event_type]


class _FakeResponse:
    status = 200

    def __init__(self, text="## 핵심 요약\n- 일정 논의"):
        self._raw = json.dumps({"choices": [{"message": {"content": text}}]}, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size=-1):
        return self._raw if size is None or size < 0 else self._raw[:size]


def _provider_env():
    return {
        "OPENROUTER_API_KEY": "or-test",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "OPENCODE_API_KEY": "oc-test",
        "OPENCODE_API_BASE_URL": "https://opencode.ai/zen/go/v1",
    }


def _media(directory: Path, name: str) -> Path:
    media = directory / name
    media.write_bytes(b"fake-media")
    return media


def _ctx(root: Path, config=None, model=None, manifest=None, emitter=None, environment=None) -> PipelineContext:
    output = root / "output"
    return PipelineContext(
        config=config or RuntimeConfig(artifact_format="md"),
        emitter=emitter or _RecordingEmitter(),
        output_dir=output,
        environment=environment if environment is not None else _provider_env(),
        manifest=manifest,
        model=model,
    )


def _artifact_names(output: Path) -> list[str]:
    return sorted(path.name for path in output.iterdir() if path.is_file()) if output.is_dir() else []


def _summary_ok(*_args, **_kwargs):
    return GeneratedMinutes("# 요약\n\n- 일정 논의\n", "openrouter", "test-model")


# --------------------------------------------------------------------------
# Pipeline-level tests (process_file / run_pipeline with injected fakes)
# --------------------------------------------------------------------------


class PipelineRunTests(unittest.TestCase):
    def test_fresh_run_creates_exactly_three_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(), manifest=manifest)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    committed, errors = run_pipeline([media], ctx)
            finally:
                manifest.close()

            self.assertEqual(errors, [])
            self.assertEqual(committed, 1)
            self.assertEqual(
                _artifact_names(output),
                ["meeting_본문요약.md", "meeting_본문전체.md", "meeting_화자스크립트.md"],
            )
            entry = manifest_entry(output, "meeting")
            self.assertEqual(entry["state"], "committed")
            self.assertEqual(set(entry["outputs"]), {"speaker_script", "body_summary", "full_body"})

    def test_summary_failure_leaves_zero_artifacts_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(), manifest=manifest)
                with patch(
                    "mediascribe_cli.pipeline.generate_body_summary",
                    side_effect=ProviderError("모든 공급자 요청에 실패했습니다."),
                ):
                    committed, errors = run_pipeline([media], ctx)
            finally:
                manifest.close()

            self.assertEqual(committed, 0)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0].stage, "summary")
            self.assertEqual(_artifact_names(output), [])
            # Transcript checkpoint survives → next run resumes at summary.
            checkpoint = output / ".mediascribe" / "checkpoints" / "meeting.transcript.json"
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(manifest_entry(output, "meeting")["state"], "failed")

    def test_committed_rerun_skips_without_stt_or_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            model = _FakeModel()
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=model, manifest=manifest)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    run_pipeline([media], ctx)
            finally:
                manifest.close()
            self.assertEqual(len(model.calls), 1)

            # Second run: committed → file_skipped, zero STT, zero summary calls.
            emitter = _RecordingEmitter()
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=model, manifest=manifest, emitter=emitter)
                with patch("mediascribe_cli.pipeline.generate_body_summary") as summary:
                    committed, errors = run_pipeline([media], ctx)
            finally:
                manifest.close()

            self.assertEqual(errors, [])
            self.assertEqual(committed, 1)
            self.assertEqual(len(model.calls), 1)  # no new STT
            summary.assert_not_called()
            self.assertEqual(emitter.of("file_skipped")[0]["file_name"], "meeting.m4a")

    def test_failed_rerun_reuses_transcript_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            model = _FakeModel()
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=model, manifest=manifest)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=ProviderError("down")):
                    run_pipeline([media], ctx)
            finally:
                manifest.close()
            self.assertEqual(len(model.calls), 1)

            # Retry: summary now succeeds; STT must not run again.
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=model, manifest=manifest)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok) as summary:
                    committed, errors = run_pipeline([media], ctx)
            finally:
                manifest.close()

            self.assertEqual(errors, [])
            self.assertEqual(committed, 1)
            self.assertEqual(len(model.calls), 1)  # checkpoint reused, no duplicate STT
            summary.assert_called_once()
            self.assertEqual(len(_artifact_names(output)), 3)

    def test_same_stem_inputs_produce_six_distinct_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m4a = _media(root, "meeting.m4a")
            wav = _media(root, "meeting.wav")
            output = root / "output"
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(), manifest=manifest)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    committed, errors = run_pipeline([m4a, wav], ctx)
            finally:
                manifest.close()

            self.assertEqual(errors, [])
            self.assertEqual(committed, 2)
            names = _artifact_names(output)
            self.assertEqual(len(names), 6)
            self.assertTrue(any(name.startswith("meeting.m4a_") for name in names))
            self.assertTrue(any(name.startswith("meeting.wav_") for name in names))

    def test_format_rerun_replaces_old_format_without_stt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            model = _FakeModel()
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=model, manifest=manifest)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    run_pipeline([media], ctx)
            finally:
                manifest.close()
            self.assertEqual(len(model.calls), 1)

            # Rerun with --format html: transcript checkpoint reused, md files
            # replaced by exactly three html files.
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, config=RuntimeConfig(artifact_format="html"), model=model, manifest=manifest)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    committed, errors = run_pipeline([media], ctx)
            finally:
                manifest.close()

            self.assertEqual(errors, [])
            self.assertEqual(committed, 1)
            self.assertEqual(len(model.calls), 1)  # no duplicate STT
            names = _artifact_names(output)
            self.assertEqual(len(names), 3)
            self.assertTrue(all(name.endswith(".html") for name in names))
            self.assertFalse(list(output.glob("*.md")))

    def test_per_file_failure_does_not_abort_remaining_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = _media(root, "bad.m4a")
            good = _media(root, "good.m4a")
            output = root / "output"
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(fail_names={"bad.m4a"}), manifest=manifest)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    committed, errors = run_pipeline([bad, good], ctx)
            finally:
                manifest.close()

            self.assertEqual(committed, 1)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0].stage, "stt")
            self.assertEqual(errors[0].file_name, "bad.m4a")
            names = _artifact_names(output)
            self.assertEqual(len(names), 3)
            self.assertTrue(all(name.startswith("good_") for name in names))

    def test_no_resume_writes_no_manifest_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            ctx = _ctx(root, model=_FakeModel(), manifest=None)
            with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                committed, errors = run_pipeline([media], ctx)

            self.assertEqual(errors, [])
            self.assertEqual(committed, 1)
            self.assertEqual(len(_artifact_names(output)), 3)
            self.assertFalse((output / ".mediascribe").exists())

    def test_summary_progress_has_real_elapsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            emitter = _RecordingEmitter()
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(), manifest=manifest, emitter=emitter)
                # Real generate path with fake HTTP → real elapsed measurement.
                with patch("mediascribe_cli.meeting_minutes.urlopen", return_value=_FakeResponse()):
                    committed, errors = run_pipeline([media], ctx)
            finally:
                manifest.close()

            self.assertEqual(errors, [])
            self.assertEqual(committed, 1)
            summary_progress = [p for p in emitter.of("progress") if p.get("stage") == "summary"]
            self.assertTrue(summary_progress)
            self.assertGreater(summary_progress[-1]["elapsed_seconds"], 0)
            minutes_complete = emitter.of("minutes_complete")
            self.assertTrue(minutes_complete)
            self.assertGreater(minutes_complete[-1]["elapsed_seconds"], 0)

    def test_startup_reconciliation_removes_partial_triplets(self) -> None:
        # A crashed commit leaves a non-committed entry plus artifact finals;
        # the next run must delete them before processing.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            manifest = Manifest.open(output)
            try:
                # Simulate a crashed run: entry stuck at 'summarizing' with
                # two of three finals on disk for a key not in this run.
                manifest.begin("ghost", "fp", "summarizing")
                ghost_a = output / "ghost_화자스크립트.md"
                ghost_b = output / "ghost_본문요약.md"
                output.mkdir(parents=True, exist_ok=True)
                ghost_a.write_text("partial", encoding="utf-8")
                ghost_b.write_text("partial", encoding="utf-8")
                manifest.set_state(
                    "ghost",
                    "summarizing",
                    outputs={"speaker_script": str(ghost_a), "body_summary": str(ghost_b)},
                )
            finally:
                manifest.close()

            emitter = _RecordingEmitter()
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(), manifest=manifest, emitter=emitter)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    committed, errors = run_pipeline([media], ctx)
            finally:
                manifest.close()

            self.assertEqual(errors, [])
            self.assertEqual(committed, 1)
            self.assertFalse(ghost_a.exists())
            self.assertFalse(ghost_b.exists())
            removed = emitter.of("artifacts_removed")
            self.assertTrue(removed)
            self.assertEqual(removed[0]["key"], "ghost")
            self.assertGreaterEqual(removed[0]["count"], 2)
            names = _artifact_names(output)
            self.assertEqual(len(names), 3)
            self.assertTrue(all(name.startswith("meeting_") for name in names))

    def test_failed_rerun_preserves_prior_committed_artifacts(self) -> None:
        # A rerun that fails must NOT destroy the previous run's committed
        # triplet: abort removes only this run's partials, never a complete
        # recorded set (architect finding A2).
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(), manifest=manifest)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    run_pipeline([media], ctx)
            finally:
                manifest.close()
            self.assertEqual(len(_artifact_names(output)), 3)

            # Force a re-run that fails at STT: change the media so the
            # fingerprint no longer matches, then fail transcription.
            media.write_bytes(b"changed-media")
            emitter = _RecordingEmitter()
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(fail_names={"meeting.m4a"}), manifest=manifest, emitter=emitter)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    committed, errors = run_pipeline([media], ctx)
            finally:
                manifest.close()

            self.assertEqual(committed, 0)
            self.assertEqual(len(errors), 1)
            # Prior committed triplet survives the failed rerun.
            self.assertEqual(
                _artifact_names(output),
                ["meeting_본문요약.md", "meeting_본문전체.md", "meeting_화자스크립트.md"],
            )
            self.assertFalse(emitter.of("artifacts_removed"))

    def test_same_stem_different_path_gets_distinct_key(self) -> None:
        # Architect A1: a same-stem file from a different directory must not
        # overwrite the committed triplet of the earlier input — the key is
        # disambiguated and both artifact sets survive.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dir_a = root / "2024-01"
            dir_b = root / "2024-02"
            dir_a.mkdir()
            dir_b.mkdir()
            media_a = _media(dir_a, "meeting.m4a")
            media_b = _media(dir_b, "meeting.m4a")
            output = root / "output"

            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(), manifest=manifest)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    run_pipeline([media_a], ctx)
            finally:
                manifest.close()
            self.assertEqual(len(_artifact_names(output)), 3)

            emitter = _RecordingEmitter()
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(), manifest=manifest, emitter=emitter)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    committed, errors = run_pipeline([media_b], ctx)
            finally:
                manifest.close()

            self.assertEqual(errors, [])
            self.assertEqual(committed, 1)
            # First input's triplet untouched; second input got a hashed key.
            names = _artifact_names(output)
            self.assertEqual(len(names), 6)
            self.assertTrue(any(name.startswith("meeting_") for name in names))
            self.assertTrue(any(name.startswith("meeting.m4a.") for name in names))
            self.assertTrue(emitter.of("config_warning"))

    def test_title_change_re_summarizes_committed_input(self) -> None:
        # Architect A3: summary-shaping inputs (title/provider/models/chunk
        # size) are part of the skip fingerprint — a different title must
        # re-summarize (reusing the transcript checkpoint) instead of
        # silently serving the stale artifact.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            model = _FakeModel()
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=model, manifest=manifest)
                ctx = replace(ctx, meeting_title="1분기 회의")
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    run_pipeline([media], ctx)
            finally:
                manifest.close()
            self.assertEqual(len(model.calls), 1)

            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=model, manifest=manifest)
                ctx = replace(ctx, meeting_title="2분기 회의")
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok) as summary:
                    committed, errors = run_pipeline([media], ctx)
            finally:
                manifest.close()

            self.assertEqual(errors, [])
            self.assertEqual(committed, 1)
            summary.assert_called_once()  # re-summarized, not skipped
            self.assertEqual(len(model.calls), 1)  # transcript checkpoint reused — no STT

    def test_legacy_entry_without_media_path_is_rekeyed(self) -> None:
        # A manifest entry committed before media_path binding existed has
        # no recorded path — a different same-stem input must NOT reuse that
        # key and overwrite the old committed artifacts.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dir_a = root / "a"
            dir_b = root / "b"
            dir_a.mkdir()
            dir_b.mkdir()
            media_a = _media(dir_a, "meeting.m4a")
            media_b = _media(dir_b, "meeting.m4a")
            output = root / "output"

            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(), manifest=manifest)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    run_pipeline([media_a], ctx)
            finally:
                manifest.close()
            # Simulate a legacy entry: strip media_path from the record.
            data = json.loads((output / ".mediascribe" / "manifest.json").read_text(encoding="utf-8"))
            data["files"]["meeting"].pop("media_path", None)
            (output / ".mediascribe" / "manifest.json").write_text(json.dumps(data), encoding="utf-8")

            emitter = _RecordingEmitter()
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(), manifest=manifest, emitter=emitter)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    committed, errors = run_pipeline([media_b], ctx)
            finally:
                manifest.close()

            self.assertEqual(errors, [])
            self.assertEqual(committed, 1)
            names = _artifact_names(output)
            self.assertEqual(len(names), 6)  # old triplet preserved + new hashed-key triplet
            self.assertTrue(emitter.of("config_warning"))

    def test_failed_commit_does_not_preserve_mixed_set(self) -> None:
        # A rerun whose commit partially replaced the prior triplet must not
        # keep the mixed set: recorded digests detect the replacement and the
        # abort removes this run's partial finals.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(), manifest=manifest)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    run_pipeline([media], ctx)
            finally:
                manifest.close()
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(), manifest=manifest)
                ctx = replace(ctx, meeting_title="다른 제목")
                # Fail the second *artifact* rename (marker publish and
                # manifest persists also go through os.replace — only count
                # destinations that look like a final artifact).
                real_replace = os.replace
                calls = {"n": 0}

                def flaky_replace(src, dst):
                    if str(dst).endswith((".md", ".html", ".txt")) and "_본문" in str(dst) or "_화자스크립트" in str(dst):
                        calls["n"] += 1
                        if calls["n"] == 2:
                            raise OSError("simulated mid-commit failure")
                    return real_replace(src, dst)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok), \
                     patch("mediascribe_cli.pipeline.render_speaker_script", return_value="DIFFERENT CONTENT"), \
                     patch("mediascribe_cli.artifacts.os.replace", side_effect=flaky_replace):
                    committed, errors = run_pipeline([media], ctx)
            finally:
                manifest.close()

            self.assertEqual(committed, 0)
            self.assertEqual(len(errors), 1)
            # Mixed set removed — only the commit journal marker remains
            # (reconcile_commits reclaims it on the next run).
            self.assertEqual([name for name in _artifact_names(output) if name.startswith("meeting_")], [])

    def test_no_resume_failure_preserves_on_disk_triplet(self) -> None:
        # --no-resume: no manifest to vouch for the prior commit, but a
        # complete on-disk triplet is still preserved on failure.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            manifest = Manifest.open(output)
            try:
                ctx = _ctx(root, model=_FakeModel(), manifest=manifest)
                with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=_summary_ok):
                    run_pipeline([media], ctx)
            finally:
                manifest.close()
            self.assertEqual(len(_artifact_names(output)), 3)

            emitter = _RecordingEmitter()
            ctx = _ctx(root, model=_FakeModel(), manifest=None, emitter=emitter)
            ctx = replace(ctx, meeting_title="다른 제목")
            with patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=ProviderError("요약 실패")):
                committed, errors = run_pipeline([media], ctx)

            self.assertEqual(committed, 0)
            self.assertEqual(len(errors), 1)
            self.assertEqual(len(_artifact_names(output)), 3)  # prior triplet kept



def manifest_entry(output: Path, key: str) -> dict:
    data = json.loads((output / ".mediascribe" / "manifest.json").read_text(encoding="utf-8"))
    return data["files"][key]


# --------------------------------------------------------------------------
# CLI-level tests (main() with fake faster_whisper module)
# --------------------------------------------------------------------------


def _fake_whisper_module(model):
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = lambda *a, **k: model
    return {"faster_whisper": module}


def _clean_env(**overrides):
    return patch.dict(os.environ, overrides, clear=True)


class CLIPipelineTests(unittest.TestCase):
    def _run_cli(self, argv, model=None, summary=None, env=None):
        modules = _fake_whisper_module(model or _FakeModel())
        stdout = io.StringIO()
        stderr = io.StringIO()
        patches = [patch.dict(sys.modules, modules), _clean_env(**(env or {}))]
        if summary is not None:
            patches.append(patch("mediascribe_cli.pipeline.generate_body_summary", side_effect=summary))
        with contextlib.ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_summary_failure_exits_4_with_zero_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            home = root / "home"
            home.mkdir()
            code, _stdout, _stderr = self._run_cli(
                ["transcribe", str(media), "--output-dir", str(output), "--json-lines"],
                summary=ProviderError("모든 공급자 요청에 실패했습니다."),
                env={"HOME": str(home)},
            )
            self.assertEqual(code, 4)
            self.assertEqual(_artifact_names(output), [])
            self.assertTrue((output / ".mediascribe" / "checkpoints" / "meeting.transcript.json").is_file())

    def test_cli_mixed_failure_exits_5(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _media(root, "bad.m4a")
            _media(root, "good.m4a")
            output = root / "output"
            home = root / "home"
            home.mkdir()
            code, _stdout, _stderr = self._run_cli(
                ["transcribe", str(root), "--output-dir", str(output)],
                model=_FakeModel(fail_names={"bad.m4a"}),
                summary=_summary_ok,
                env={"HOME": str(home)},
            )
            self.assertEqual(code, 5)
            names = _artifact_names(output)
            self.assertEqual(len(names), 3)
            self.assertTrue(all(name.startswith("good_") for name in names))

    def test_cli_dry_run_creates_no_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            home = root / "home"
            home.mkdir()
            code, stdout, _stderr = self._run_cli(
                ["transcribe", str(media), "--output-dir", str(output), "--dry-run", "--json-lines"],
                env={"HOME": str(home)},
            )
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())
            events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
            dry = [event for event in events if event["type"] == "dry_run"]
            self.assertEqual(len(dry), 1)
            self.assertEqual(dry[0]["files"][0]["key"], "meeting")
            self.assertFalse(dry[0]["files"][0]["will_skip"])

    def test_cli_lock_conflict_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            home = root / "home"
            home.mkdir()
            held = Manifest.open(output)
            try:
                code, _stdout, _stderr = self._run_cli(
                    ["transcribe", str(media), "--output-dir", str(output)],
                    env={"HOME": str(home)},
                )
            finally:
                held.close()
            self.assertEqual(code, 2)

    def test_cli_no_resume_leaves_no_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            home = root / "home"
            home.mkdir()
            code, _stdout, _stderr = self._run_cli(
                ["transcribe", str(media), "--output-dir", str(output), "--no-resume"],
                summary=_summary_ok,
                env={"HOME": str(home)},
            )
            self.assertEqual(code, 0)
            self.assertEqual(len(_artifact_names(output)), 3)
            self.assertFalse((output / ".mediascribe").exists())

    def test_cli_jsonl_events_parseable_and_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root, "meeting.m4a")
            output = root / "output"
            home = root / "home"
            home.mkdir()
            code, stdout, _stderr = self._run_cli(
                ["transcribe", str(media), "--output-dir", str(output), "--json-lines"],
                summary=_summary_ok,
                env={"HOME": str(home)},
            )
            self.assertEqual(code, 0)
            events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
            types_seen = [event["type"] for event in events]
            self.assertEqual(types_seen[0], "run_started")
            self.assertEqual(types_seen[-1], "run_complete")
            for expected in ("file_started", "checkpoint_saved", "summary_started", "file_complete"):
                self.assertIn(expected, types_seen)


if __name__ == "__main__":
    unittest.main()
