from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mediascribe_cli.cli import main  # noqa: E402
from mediascribe_cli.meeting_minutes import GeneratedMinutes  # noqa: E402
from mediascribe_cli.events import EventEmitter  # noqa: E402


class FakeWord:
    def __init__(self, start, end, word, probability):
        self.start = start
        self.end = end
        self.word = word
        self.probability = probability


class FakeSegment:
    def __init__(self):
        self.start = 0.0
        self.end = 1.5
        self.text = "미디어 스크라이브 회의"
        self.words = [
            FakeWord(0.0, 0.6, "미디어 스크라이브 ", 0.95),
            FakeWord(0.8, 1.4, "회의", 0.85),
        ]


class FakeInfo:
    language = "ko"
    language_probability = 0.99
    duration = 2.0


class FakeWhisperModel:
    calls = 0

    def __init__(self, *_args, **_kwargs):
        pass

    def transcribe(self, *_args, **_kwargs):
        type(self).calls += 1
        return iter([FakeSegment()]), FakeInfo()


class FakeTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakePipeline:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def __call__(self, *_args, **_kwargs):
        return self

    def itertracks(self, yield_label=True):
        return iter([(FakeTurn(0, 0.7), None, "SPEAKER_00"), (FakeTurn(0.7, 2), None, "SPEAKER_01")])


def _fake_ml_modules():
    faster_whisper = types.ModuleType("faster_whisper")
    faster_whisper.WhisperModel = FakeWhisperModel
    pyannote = types.ModuleType("pyannote")
    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = FakePipeline
    return {"faster_whisper": faster_whisper, "pyannote": pyannote, "pyannote.audio": pyannote_audio}


def _clean_env(**overrides):
    """Isolated environment: no real keys, no user-level env file."""
    return patch.dict(os.environ, overrides, clear=True)


class CLISmokeTests(unittest.TestCase):
    def test_fake_transcription_shows_progress_and_writes_three_artifacts(self) -> None:
        FakeWhisperModel.calls = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "meeting.m4a"
            media.write_bytes(b"fake")
            glossary = root / "glossary.json"
            glossary.write_text('{"terms":{"MediaScribe":["미디어 스크라이브"]}}', encoding="utf-8")
            output = root / "output"
            home = root / "home"
            home.mkdir()
            stdout = io.StringIO()
            with _clean_env(HOME=str(home), HF_TOKEN="env-token"), patch.dict(sys.modules, _fake_ml_modules()):
                with contextlib.redirect_stdout(stdout):
                    with patch(
                        "mediascribe_cli.pipeline.generate_body_summary",
                        return_value=GeneratedMinutes("# 요약\n\n- 일정 논의\n", "openrouter", "test-model"),
                    ):
                        exit_code = main([
                            "transcribe",
                            str(media),
                            "--output-dir",
                            str(output),
                            "--format",
                            "md",
                            "--diarize",
                            "--speaker-labels",
                            "A,B",
                            "--glossary",
                            str(glossary),
                            "--json-lines",
                        ])

            self.assertEqual(exit_code, 0)
            events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
            progress_events = [event for event in events if event["type"] == "progress"]
            self.assertTrue(any(event.get("stage") == "stt" and event.get("progress") is not None for event in progress_events))
            self.assertTrue(any(event.get("stage") == "summary" for event in progress_events))
            self.assertEqual(events[-1]["type"], "run_complete")
            self.assertEqual(events[-1]["failed"], 0)
            self.assertEqual(
                sorted(path.name for path in output.iterdir() if path.is_file()),
                ["meeting_본문요약.md", "meeting_본문전체.md", "meeting_화자스크립트.md"],
            )
            self.assertIn("A: MediaScribe", (output / "meeting_화자스크립트.md").read_text(encoding="utf-8"))
            self.assertIn("일정 논의", (output / "meeting_본문요약.md").read_text(encoding="utf-8"))
            # Manifest metadata lives under .mediascribe/, never among artifacts.
            self.assertTrue((output / ".mediascribe" / "manifest.json").is_file())

    def test_dry_run_reports_real_skip_decision(self) -> None:
        # Dry-run must report the same skip decision the real run makes:
        # a format change (or title/provider change) means will_skip=false
        # even though a committed entry exists.
        FakeWhisperModel.calls = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "meeting.m4a"
            media.write_bytes(b"fake")
            output = root / "output"
            home = root / "home"
            home.mkdir()
            env = {"HOME": str(home), "OPENROUTER_API_KEY": "k"}
            with _clean_env(**env), patch.dict(sys.modules, _fake_ml_modules()):
                with patch(
                    "mediascribe_cli.pipeline.generate_body_summary",
                    return_value=GeneratedMinutes("# 요약\n", "openrouter", "test-model"),
                ):
                    self.assertEqual(main(["transcribe", str(media), "--output-dir", str(output), "--format", "md"]), 0)
            # Dry-run with a different format: committed md entry exists but
            # the html run would reprocess → will_skip must be false.
            stdout = io.StringIO()
            with _clean_env(**env), patch.dict(sys.modules, _fake_ml_modules()):
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(
                        main(["transcribe", str(media), "--output-dir", str(output), "--format", "html", "--dry-run", "--json-lines"]),
                        0,
                    )
            events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
            dry = next(event for event in events if event["type"] == "dry_run")
            self.assertEqual(dry["files"][0]["manifest_state"], "committed")
            self.assertFalse(dry["files"][0]["will_skip"])

    def test_jsonl_redacts_transcript_text_by_default(self) -> None:
        # Contract: JSONL transcript text is a sha256 digest by default;
        # MEDIASCRIBE_EVENTS_FULL_TEXT=1 opts out.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "meeting.m4a"
            media.write_bytes(b"fake")
            output = root / "output"
            home = root / "home"
            home.mkdir()
            stdout = io.StringIO()
            with _clean_env(HOME=str(home)), patch.dict(sys.modules, _fake_ml_modules()):
                with contextlib.redirect_stdout(stdout):
                    with patch(
                        "mediascribe_cli.pipeline.generate_body_summary",
                        return_value=GeneratedMinutes("# 요약\n", "openrouter", "test-model"),
                    ):
                        exit_code = main(["transcribe", str(media), "--output-dir", str(output), "--json-lines"])

            self.assertEqual(exit_code, 0)
            events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
            segment_events = [event for event in events if event["type"] == "transcript_segment"]
            self.assertTrue(segment_events)
            for event in segment_events:
                self.assertIsInstance(event["text"], dict)
                self.assertIn("sha256", event["text"])
            self.assertNotIn("미디어 스크라이브 회의", stdout.getvalue())

    def test_events_full_text_env_keeps_transcript_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "meeting.m4a"
            media.write_bytes(b"fake")
            output = root / "output"
            home = root / "home"
            home.mkdir()
            stdout = io.StringIO()
            with _clean_env(HOME=str(home), MEDIASCRIBE_EVENTS_FULL_TEXT="1"), patch.dict(sys.modules, _fake_ml_modules()):
                with contextlib.redirect_stdout(stdout):
                    with patch(
                        "mediascribe_cli.pipeline.generate_body_summary",
                        return_value=GeneratedMinutes("# 요약\n", "openrouter", "test-model"),
                    ):
                        exit_code = main(["transcribe", str(media), "--output-dir", str(output), "--json-lines"])

            self.assertEqual(exit_code, 0)
            events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
            segment_events = [event for event in events if event["type"] == "transcript_segment"]
            self.assertTrue(segment_events)
            for event in segment_events:
                self.assertEqual(event["text"], "미디어 스크라이브 회의")

    def test_events_full_text_flag_keeps_transcript_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "meeting.m4a"
            media.write_bytes(b"fake")
            output = root / "output"
            home = root / "home"
            home.mkdir()
            stdout = io.StringIO()
            with _clean_env(HOME=str(home)), patch.dict(sys.modules, _fake_ml_modules()):
                with contextlib.redirect_stdout(stdout):
                    with patch(
                        "mediascribe_cli.pipeline.generate_body_summary",
                        return_value=GeneratedMinutes("# 요약\n", "openrouter", "test-model"),
                    ):
                        exit_code = main([
                            "transcribe", str(media), "--output-dir", str(output),
                            "--json-lines", "--events-full-text",
                        ])

            self.assertEqual(exit_code, 0)
            events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
            segment_events = [event for event in events if event["type"] == "transcript_segment"]
            self.assertTrue(segment_events)
            self.assertEqual(segment_events[0]["text"], "미디어 스크라이브 회의")


    def test_hf_token_flag_is_rejected(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(["transcribe", "x.m4a", "--hf-token", "secret"])
        self.assertIn("--hf-token", stderr.getvalue())

    def test_help_lists_safety_flags_without_hf_token(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as caught:
            main(["transcribe", "--help"])
        self.assertEqual(caught.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertNotIn("--hf-token", help_text)
        for flag in ("--env-file", "--allow-custom-endpoint", "--no-resume", "--follow-symlinks", "--summary-chunk-chars", "--events-full-text"):
            self.assertIn(flag, help_text)

    def test_human_progress_line_contains_eta(self) -> None:
        stderr = io.StringIO()
        emitter = EventEmitter(stderr=stderr)
        emitter.emit(
            "progress",
            stage="stt",
            file_name="meeting.m4a",
            progress=42.0,
            elapsed_seconds=18.0,
            eta_seconds=24.0,
            text="현재 세그먼트",
        )
        self.assertIn("STT  42.0%", stderr.getvalue())
        self.assertIn("경과 00:18", stderr.getvalue())
        self.assertIn("예상 잔여 00:24", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
