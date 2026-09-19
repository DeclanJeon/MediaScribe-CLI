from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mediascribe_cli.config import RuntimeConfig  # noqa: E402
from mediascribe_cli.errors import DependencyError, UsageError  # noqa: E402
from mediascribe_cli.transcriber import (  # noqa: E402
    _diarize_file,
    _transcribe_once,
    validate_runtime_dependencies,
    collect_media_files,
    transcribe_file,
)


class _RecordingEmitter:
    def __init__(self):
        self.events = []

    def emit(self, event_type, **payload):
        self.events.append((event_type, payload))

    def types(self):
        return [event_type for event_type, _ in self.events]


class _FakeSegment:
    def __init__(self, text, start=0.0, end=1.0):
        self.text = text
        self.start = start
        self.end = end


class _FakeInfo:
    def __init__(self, language="ko", language_probability=0.9, duration=10.0):
        self.language = language
        self.language_probability = language_probability
        self.duration = duration


class _FakeModel:
    """Records transcribe() calls; serves queued segment lists per call."""

    def __init__(self, segment_batches, info=None):
        self.calls = []
        self._batches = list(segment_batches)
        self._info = info or _FakeInfo()

    def transcribe(self, audio, **options):
        # ``audio`` is a binary file object (O_NOFOLLOW open); ``.name``
        # carries the source path.
        self.calls.append({"path": getattr(audio, "name", audio), "options": options})
        batch = self._batches.pop(0) if self._batches else []
        return iter(batch), self._info


def _ctx(config, emitter):
    return types.SimpleNamespace(config=config, emitter=emitter)


def _media(directory: Path, name: str = "meeting.m4a") -> Path:
    media = directory / name
    media.write_bytes(b"fake-media")
    return media


class LanguageOptionTests(unittest.TestCase):
    def test_configured_language_reaches_model_for_both_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            media = _media(Path(directory))
            for task in ("transcribe", "translate"):
                with self.subTest(task=task):
                    model = _FakeModel([[_FakeSegment("안녕")]])
                    config = RuntimeConfig(task=task, language="ko")
                    _transcribe_once(model, media, config)
                    self.assertEqual(model.calls[0]["options"]["language"], "ko")
                    self.assertEqual(model.calls[0]["options"]["task"], task)

class OpenMediaTests(unittest.TestCase):
    def test_open_media_rejects_symlink_swap(self) -> None:
        # O_NOFOLLOW closes the check/use gap: a path swapped to a symlink
        # after collection fails the open instead of being read.
        from mediascribe_cli.transcriber import _open_media

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.m4a"
            outside.write_bytes(b"secret")
            link = root / "media.m4a"
            link.symlink_to(outside)
            with self.assertRaises(UsageError):
                _open_media(link)

    def test_open_media_reads_regular_file(self) -> None:
        from mediascribe_cli.transcriber import _open_media

        with tempfile.TemporaryDirectory() as directory:
            media = _media(Path(directory))
            with _open_media(media) as audio:
                self.assertEqual(audio.read(), b"fake-media")
                self.assertEqual(audio.name, str(media))

    def test_open_media_rejects_file_swap(self) -> None:
        # A different real file swapped in after collection fails the inode
        # identity check — the decoder never reads the wrong media.
        from mediascribe_cli.transcriber import _open_media

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media.m4a"
            media.write_bytes(b"legit")
            other = root / "other.m4a"
            other.write_bytes(b"different")
            # Simulate the check/use gap: lstat on the media path returns
            # the checked entry, then the file is replaced before the open
            # completes — fstat must catch the inode change.
            real_lstat = os.lstat

            def swap_on_media_lstat(path):
                result = real_lstat(path)
                if Path(path) == media:
                    other.rename(media)  # swap AFTER the check, before open
                return result

            with patch("mediascribe_cli.safeio.os.lstat", side_effect=swap_on_media_lstat):
                with self.assertRaises(UsageError):
                    _open_media(media)

    def test_open_media_allows_symlinked_ancestor(self) -> None:
        # Ancestor symlinks are legitimate (macOS /var → /private/var) —
        # the resolved canonical path is what gets opened.
        from mediascribe_cli.transcriber import _open_media

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_dir = root / "real"
            real_dir.mkdir()
            media = real_dir / "media.m4a"
            media.write_bytes(b"legit")
            link_dir = root / "linked"
            link_dir.symlink_to(real_dir, target_is_directory=True)
            with _open_media(link_dir / "media.m4a") as audio:
                self.assertEqual(audio.read(), b"legit")


class CollectMediaFilesTests(unittest.TestCase):
    def test_single_file_input(self):
        with tempfile.TemporaryDirectory() as directory:
            media = _media(Path(directory))
            self.assertEqual(collect_media_files(media), [media.resolve()])

    def test_directory_collects_supported_files_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _media(root, "b.mp3")
            _media(root, "A.m4a")
            (root / "notes.txt").write_text("x", encoding="utf-8")
            files = collect_media_files(root)
            self.assertEqual([path.name for path in files], ["A.m4a", "b.mp3"])

    def test_symlinked_child_skipped_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _media(root, "real.m4a")
            target = _media(root, "target.mp3")
            link = root / "link.m4a"
            link.symlink_to(target.name)
            files = collect_media_files(root)
            self.assertEqual([path.name for path in files], ["real.m4a", "target.mp3"])

    def test_symlinked_input_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = _media(root, "target.m4a")
            link = root / "input.m4a"
            link.symlink_to(target.name)
            with self.assertRaises(UsageError):
                collect_media_files(link)

    def test_follow_symlinks_accepts_in_tree_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = _media(root, "target.m4a")
            link = root / "link.m4a"
            link.symlink_to(target.name)
            files = collect_media_files(root, follow_symlinks=True)
            self.assertEqual(files, [target.resolve()])

    def test_follow_symlinks_rejects_child_escaping_input_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inside = root / "in"
            inside.mkdir()
            outside = _media(root, "outside.m4a")
            link = inside / "escape.m4a"
            link.symlink_to(outside)
            with self.assertRaises(UsageError):
                collect_media_files(inside, follow_symlinks=True)

    def test_follow_symlinks_accepts_in_tree_input_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            media = _media(real)
            alias = root / "alias"
            alias.symlink_to(real.name, target_is_directory=True)
            files = collect_media_files(alias, follow_symlinks=True)
            self.assertEqual(files, [media.resolve()])

    def test_follow_symlinks_rejects_input_escaping_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sub = root / "sub"
            sub.mkdir()
            outside = _media(root, "outside.m4a")
            link = sub / "input.m4a"
            link.symlink_to(outside)
            with self.assertRaises(UsageError):
                collect_media_files(link, follow_symlinks=True)

    def test_missing_input_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(UsageError):
                collect_media_files(Path(directory) / "nope.m4a")

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(UsageError):
                collect_media_files(directory)


def _fake_pyannote(turns=None, recorder=None):
    """Build fake pyannote/pyannote.audio modules for sys.modules injection."""
    turns = turns or []

    class _Turn:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class _Diarization:
        def itertracks(self, yield_label=False):
            for start, end, speaker in turns:
                yield _Turn(start, end), None, speaker

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            if recorder is not None:
                recorder.append({"model": model, "kwargs": kwargs})
            return lambda path, **options: _Diarization()

    audio = types.ModuleType("pyannote.audio")
    audio.Pipeline = _Pipeline
    package = types.ModuleType("pyannote")
    package.audio = audio
    return {"pyannote": package, "pyannote.audio": audio}


class DiarizationTokenTests(unittest.TestCase):
    def test_diarize_uses_hf_token_env_only(self):
        recorder = []
        config = RuntimeConfig(diarize=True)
        modules = _fake_pyannote(turns=[(0.0, 1.0, "SPEAKER_00")], recorder=recorder)
        with tempfile.TemporaryDirectory() as directory:
            media = _media(Path(directory))
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {"HF_TOKEN": "env-token"}, clear=False):
                turns = _diarize_file(media, config)
        self.assertEqual(recorder[0]["kwargs"].get("token"), "env-token")
        self.assertEqual(turns, [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}])

    def test_diarize_uses_token_from_loaded_environment(self):
        # HF_TOKEN supplied via --env-file/user env (ctx.environment) must
        # reach diarization even when absent from os.environ.
        recorder = []
        config = RuntimeConfig(diarize=True)
        modules = _fake_pyannote(turns=[(0.0, 1.0, "SPEAKER_00")], recorder=recorder)
        env = {key: value for key, value in os.environ.items() if key != "HF_TOKEN"}
        with tempfile.TemporaryDirectory() as directory:
            media = _media(Path(directory))
            with patch.dict(sys.modules, modules), patch.dict(os.environ, env, clear=True):
                turns = _diarize_file(media, config, environment={"HF_TOKEN": "file-token"})
        self.assertEqual(recorder[0]["kwargs"].get("token"), "file-token")
        self.assertEqual(turns, [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}])

    def test_diarize_falls_back_to_use_auth_token_kwarg(self):
        # Older pyannote releases only accept use_auth_token=; the TypeError
        # fallback must still deliver the env token.
        recorder = []

        class _LegacyPipeline:
            @classmethod
            def from_pretrained(cls, model, **kwargs):
                if "token" in kwargs:
                    raise TypeError("unexpected keyword argument 'token'")
                recorder.append(kwargs)
                return lambda source, **options: _fake_diarization()

        def _fake_diarization():
            class _Result:
                def itertracks(self, yield_label=False):
                    yield types.SimpleNamespace(start=0.0, end=1.0), None, "SPEAKER_00"

            return _Result()

        audio = types.ModuleType("pyannote.audio")
        audio.Pipeline = _LegacyPipeline
        package = types.ModuleType("pyannote")
        package.audio = audio
        config = RuntimeConfig(diarize=True)
        with tempfile.TemporaryDirectory() as directory:
            media = _media(Path(directory))
            with patch.dict(sys.modules, {"pyannote": package, "pyannote.audio": audio}), patch.dict(
                os.environ, {"HF_TOKEN": "env-token"}, clear=False
            ):
                turns = _diarize_file(media, config)
        self.assertEqual(recorder[0].get("use_auth_token"), "env-token")
        self.assertEqual(turns, [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}])

    def test_diarize_without_env_token_raises(self):
        config = RuntimeConfig(diarize=True)
        env = {key: value for key, value in os.environ.items() if key != "HF_TOKEN"}
        with patch.dict(sys.modules, _fake_pyannote()), patch.dict(os.environ, env, clear=True):
            with self.assertRaises(DependencyError) as caught:
                _diarize_file(Path("meeting.m4a"), config)
        self.assertIn("HF_TOKEN", str(caught.exception))

    def test_validate_dependencies_requires_env_token(self):
        fake_fw = types.ModuleType("faster_whisper")
        config = RuntimeConfig(diarize=True)
        env = {key: value for key, value in os.environ.items() if key != "HF_TOKEN"}
        modules = {"faster_whisper": fake_fw, **_fake_pyannote()}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, env, clear=True):
            with self.assertRaises(DependencyError) as caught:
                validate_runtime_dependencies(config)
        self.assertIn("HF_TOKEN", str(caught.exception))

    def test_validate_dependencies_accepts_env_file_token(self):
        fake_fw = types.ModuleType("faster_whisper")
        config = RuntimeConfig(diarize=True)
        env = {key: value for key, value in os.environ.items() if key != "HF_TOKEN"}
        modules = {"faster_whisper": fake_fw, **_fake_pyannote()}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, env, clear=True):
            validate_runtime_dependencies(config, environment={"HF_TOKEN": "file-token"})

    def test_validate_dependencies_with_env_token_checks_pyannote(self):
        fake_fw = types.ModuleType("faster_whisper")
        config = RuntimeConfig(diarize=True)
        with patch.dict(sys.modules, {"faster_whisper": fake_fw}), patch.dict(os.environ, {"HF_TOKEN": "env-token"}):
            with self.assertRaises(DependencyError) as caught:
                validate_runtime_dependencies(config)
        self.assertIn("화자 분리 의존성", str(caught.exception))


class TranscribeFileTests(unittest.TestCase):
    def test_returns_result_without_writing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _media(root)
            model = _FakeModel([[_FakeSegment("안녕하세요", 0.0, 2.0), _FakeSegment("반갑습니다", 2.0, 4.0)]])
            emitter = _RecordingEmitter()
            config = RuntimeConfig(language="ko")
            result = transcribe_file(media, config, emitter, model)

            self.assertEqual(result["media_path"], str(media))
            self.assertEqual(result["detected_language"], "ko")
            self.assertEqual([seg["text"] for seg in result["segments"]], ["안녕하세요", "반갑습니다"])
            self.assertIn("metadata", result)
            self.assertNotIn("outputs", result)
            # No artifact files written anywhere near the media file.
            self.assertEqual([path.name for path in root.iterdir()], [media.name])
            # Segment/progress events still flow through ctx.emitter.
            self.assertEqual(emitter.types().count("transcript_segment"), 2)
            self.assertEqual(emitter.types().count("progress"), 2)
            self.assertEqual(model.calls[0]["options"]["language"], "ko")

    def test_vad_retry_when_first_pass_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            media = _media(Path(directory), "meeting.wav")
            model = _FakeModel([[], [_FakeSegment("재시도 성공", 0.0, 1.0)]])
            emitter = _RecordingEmitter()
            config = RuntimeConfig(vad_filter=True)
            result = transcribe_file(media, config, emitter, model)

            self.assertEqual(len(model.calls), 2)
            self.assertTrue(model.calls[0]["options"]["vad_filter"])
            self.assertFalse(model.calls[1]["options"]["vad_filter"])
            self.assertTrue(result["metadata"]["vad_retry_used"])
            self.assertFalse(result["metadata"]["vad_filter_used"])
            self.assertIn("vad_retry", emitter.types())
            self.assertEqual(result["segments"][0]["text"], "재시도 성공")

    def test_no_vad_retry_for_non_audio_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            media = _media(Path(directory), "meeting.mp4")
            model = _FakeModel([[]])
            emitter = _RecordingEmitter()
            config = RuntimeConfig(vad_filter=True)
            result = transcribe_file(media, config, emitter, model)
            self.assertEqual(len(model.calls), 1)
            self.assertFalse(result["metadata"]["vad_retry_used"])

    def test_diarization_events_and_speaker_enrichment(self):
        with tempfile.TemporaryDirectory() as directory:
            media = _media(Path(directory))
            model = _FakeModel([[_FakeSegment("안녕", 0.0, 1.0)]])
            emitter = _RecordingEmitter()
            config = RuntimeConfig(diarize=True)
            modules = _fake_pyannote(turns=[(0.0, 1.0, "SPEAKER_00")])
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {"HF_TOKEN": "env-token"}):
                result = transcribe_file(media, config, emitter, model)
            self.assertIn("diarization_started", emitter.types())
            self.assertIn("diarization_complete", emitter.types())
            self.assertTrue(result["metadata"]["diarization"])
            self.assertEqual(result["segments"][0].get("speaker_id"), "SPEAKER_00")
            self.assertTrue(result["speakers"])


if __name__ == "__main__":
    unittest.main()
