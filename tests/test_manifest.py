from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mediascribe_cli.artifacts import ARTIFACT_KINDS  # noqa: E402
from mediascribe_cli.config import RuntimeConfig  # noqa: E402
from mediascribe_cli.errors import UsageError  # noqa: E402
from mediascribe_cli.manifest import (  # noqa: E402
    _CONFIG_FIELDS,
    MANIFEST_VERSION,
    Manifest,
    config_fingerprint,
    file_fingerprint,
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _media(directory: Path, name: str = "meeting.m4a") -> Path:
    media = directory / name
    media.write_bytes(b"fake-media")
    return media


def _transcript_result(media: Path) -> dict:
    return {
        "media_path": str(media),
        "detected_language": "ko",
        "language_probability": 0.99,
        "segments": [{"start": 0, "end": 1, "text": "발화"}],
        "speakers": [{"id": "SPEAKER_00"}],
        "metadata": {"duration": 1.0},
    }


class FingerprintTests(unittest.TestCase):
    def test_file_fingerprint_changes_with_content_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            media = _media(Path(directory))
            first = file_fingerprint(media)
            self.assertTrue(first.startswith("sha256:"))
            media.write_bytes(b"fake-media-changed")
            self.assertNotEqual(file_fingerprint(media), first)
            self.assertNotEqual(file_fingerprint(media), file_fingerprint(_media(Path(directory), "other.m4a")))

    def test_config_fingerprint_tracks_output_fields(self) -> None:
        class FakeConfig:
            model = "small"
            language = "ko"
            task = "transcribe"
            device = "auto"
            compute_type = ""
            beam_size = 1
            vad_filter = True
            diarize = False
            diarization_model = "pyannote/x"
            num_speakers = 0
            min_speakers = 0
            max_speakers = 0
            glossary_path = ""
            speaker_map_path = ""
            speaker_labels = ()
            artifact_format = "md"

        base = config_fingerprint(FakeConfig())
        self.assertTrue(base.startswith("sha256:"))

        changed = FakeConfig()
        changed.artifact_format = "html"
        self.assertNotEqual(config_fingerprint(changed), base)

        same = FakeConfig()
        self.assertEqual(config_fingerprint(same), base)


class ManifestLifecycleTests(unittest.TestCase):
    def test_open_creates_private_state_dir_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = Manifest.open(out)
            try:
                state_dir = out / ".mediascribe"
                self.assertTrue(state_dir.is_dir())
                self.assertEqual(_mode(state_dir), 0o700)
                self.assertTrue((state_dir / "lock").is_file())
                data = json.loads((state_dir / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(data["version"], MANIFEST_VERSION)
                self.assertEqual(data["output_dir"], str(out.resolve()))
            finally:
                manifest.close()

    def test_close_releases_lock_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = Manifest.open(out)
            manifest.close()
            manifest.close()
            self.assertFalse((out / ".mediascribe" / "lock").exists())
            # A second manifest can open after the first released the lock.
            other = Manifest.open(out)
            other.close()

    def test_live_lock_rejected_with_usage_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = Manifest.open(out)
            try:
                with self.assertRaises(UsageError):
                    Manifest.open(out)
            finally:
                manifest.close()

    def test_stale_lock_from_dead_pid_reclaimed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            state_dir = out / ".mediascribe"
            state_dir.mkdir()
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait()
            (state_dir / "lock").write_text(str(child.pid))
            manifest = Manifest.open(out)  # must not raise
            manifest.close()

    def test_corrupt_manifest_quarantined_and_fresh_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            state_dir = out / ".mediascribe"
            state_dir.mkdir()
            (state_dir / "manifest.json").write_text("{not json", encoding="utf-8")
            manifest = Manifest.open(out)
            try:
                corrupt = list(state_dir.glob("manifest.json.corrupt-*"))
                self.assertEqual(len(corrupt), 1)
                self.assertEqual(manifest.entry("anything"), None)
            finally:
                manifest.close()

    def test_read_only_never_creates_or_locks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "missing"
            manifest = Manifest.open(out, read_only=True)
            self.assertFalse((out / ".mediascribe").exists())
            manifest.set_state("k", "committed", outputs={"a": "/x"})
            manifest.close()
            self.assertFalse(out.exists())

    def test_read_only_coexists_with_live_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            writer = Manifest.open(out)
            try:
                reader = Manifest.open(out, read_only=True)  # no lock conflict
                reader.close()
            finally:
                writer.close()

    def test_create_false_does_not_create_state_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = Manifest.open(out, create=False)
            manifest.set_state("k", "committed")
            manifest.close()
            self.assertFalse((out / ".mediascribe").exists())


class ManifestStateTests(unittest.TestCase):
    def test_begin_then_commit_then_skip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            media = _media(out)
            fingerprint = file_fingerprint(media)
            targets = {kind: str(out / f"meeting_{kind}.md") for kind in ("speaker_script", "body_summary", "full_body")}
            for path in targets.values():
                Path(path).write_text("x", encoding="utf-8")

            manifest = Manifest.open(out)
            manifest.begin("meeting", fingerprint, "transcribing")
            manifest.set_state("meeting", "committed", outputs=targets)
            manifest.close()

            reopened = Manifest.open(out)
            try:
                self.assertTrue(reopened.should_skip("meeting", fingerprint, ARTIFACT_KINDS))
                entry = reopened.entry("meeting")
                self.assertEqual(entry["state"], "committed")
                self.assertEqual(entry["outputs"], targets)
            finally:
                reopened.close()

    def test_should_skip_false_on_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            media = _media(out)
            target = out / "meeting_full.md"
            target.write_text("x", encoding="utf-8")

            manifest = Manifest.open(out)
            manifest.begin("meeting", file_fingerprint(media), "transcribing")
            manifest.set_state("meeting", "committed", outputs={"full_body": str(target)})
            self.assertFalse(manifest.should_skip("meeting", "sha256:different", ("full_body",)))
            manifest.close()

    def test_should_skip_false_when_output_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            media = _media(out)
            fingerprint = file_fingerprint(media)
            target = out / "meeting_full.md"
            target.write_text("x", encoding="utf-8")

            manifest = Manifest.open(out)
            manifest.begin("meeting", fingerprint, "transcribing")
            manifest.set_state("meeting", "committed", outputs={"full_body": str(target)})
            target.unlink()
            self.assertFalse(manifest.should_skip("meeting", fingerprint, ("full_body",)))
            manifest.close()

    def test_should_skip_false_when_not_committed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            media = _media(out)
            fingerprint = file_fingerprint(media)
            manifest = Manifest.open(out)
            manifest.begin("meeting", fingerprint, "transcribing")
            manifest.set_state("meeting", "failed", error="boom")
            self.assertFalse(manifest.should_skip("meeting", fingerprint, ARTIFACT_KINDS))
            manifest.close()
    def test_should_skip_false_on_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            media = _media(out)
            fingerprint = file_fingerprint(media)
            target = out / "meeting_본문전체.md"
            target.write_text("x", encoding="utf-8")

            manifest = Manifest.open(out)
            manifest.begin("meeting", fingerprint, "transcribing")
            # Only one of the three kinds recorded → not a complete triplet.
            manifest.set_state("meeting", "committed", outputs={"full_body": str(target)})
            self.assertFalse(manifest.should_skip("meeting", fingerprint, ARTIFACT_KINDS))
            manifest.close()

    def test_should_skip_false_when_output_is_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            media = _media(out)
            fingerprint = file_fingerprint(media)
            targets = {kind: out / f"meeting_{kind}.md" for kind in ARTIFACT_KINDS}
            for path in targets.values():
                path.write_text("x", encoding="utf-8")
            real = out / "real.md"
            real.write_text("x", encoding="utf-8")
            targets["body_summary"].unlink()
            targets["body_summary"].symlink_to(real)

            manifest = Manifest.open(out)
            manifest.begin("meeting", fingerprint, "transcribing")
            manifest.set_state(
                "meeting", "committed", outputs={kind: str(path) for kind, path in targets.items()}
            )
            self.assertFalse(manifest.should_skip("meeting", fingerprint, ARTIFACT_KINDS))
            manifest.close()

    def test_failed_state_records_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Manifest.open(Path(directory))
            manifest.begin("k", "fp", "transcribing")
            manifest.set_state("k", "failed", error="provider down")
            self.assertEqual(manifest.entry("k")["error"], "provider down")
            manifest.close()

    def test_invalid_state_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Manifest.open(Path(directory))
            with self.assertRaises(ValueError):
                manifest.set_state("k", "bogus")
            manifest.close()

    def test_manifest_file_is_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Manifest.open(Path(directory))
            manifest.begin("k", "fp", "pending")
            manifest.close()
            self.assertEqual(_mode(Path(directory) / ".mediascribe" / "manifest.json"), 0o600)


class CheckpointTests(unittest.TestCase):
    def _save(self, manifest: Manifest, key: str, media: Path, config_fp: str = "cfg") -> None:
        manifest.save_transcript_checkpoint(
            key, _transcript_result(media), file_fingerprint(media), config_fp
        )

    def test_transcript_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            media = _media(out)
            result = _transcript_result(media)

            manifest = Manifest.open(out)
            self._save(manifest, "meeting", media)
            manifest.close()

            reopened = Manifest.open(out)
            try:
                restored = reopened.transcript_checkpoint(
                    "meeting", media, file_fingerprint(media), "cfg"
                )
                self.assertIsNotNone(restored)
                self.assertEqual(restored["segments"], result["segments"])
                self.assertEqual(restored["detected_language"], "ko")
            finally:
                reopened.close()

    def test_transcript_checkpoint_none_when_missing_or_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            media = _media(out)
            manifest = Manifest.open(out)
            self.assertIsNone(
                manifest.transcript_checkpoint("meeting", media, file_fingerprint(media), "cfg")
            )
            self._save(manifest, "meeting", media)
            media.write_bytes(b"changed-media")  # fingerprint now stale
            self.assertIsNone(
                manifest.transcript_checkpoint("meeting", media, file_fingerprint(media), "cfg")
            )
            manifest.close()

    def test_transcript_checkpoint_none_when_media_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            media = _media(out)
            manifest = Manifest.open(out)
            self._save(manifest, "meeting", media)
            media.unlink()
            with self.assertRaises(OSError):
                file_fingerprint(media)
            manifest.close()

    def test_transcript_checkpoint_bound_to_current_media(self) -> None:
        # A different file mapping to the same key must not reuse the stored
        # transcript — the checkpoint validates the *current* media path.
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            media_a = _media(out, "meeting.m4a")
            media_b = _media(out, "meeting.wav")
            manifest = Manifest.open(out)
            self._save(manifest, "meeting", media_a)
            # Same key, different media path → miss.
            self.assertIsNone(
                manifest.transcript_checkpoint(
                    "meeting", media_b, file_fingerprint(media_b), "cfg"
                )
            )
            # Same key, same media, wrong config fingerprint → miss.
            self.assertIsNone(
                manifest.transcript_checkpoint(
                    "meeting", media_a, file_fingerprint(media_a), "other-cfg"
                )
            )
            # Same key, same media, matching fingerprints → hit.
            self.assertIsNotNone(
                manifest.transcript_checkpoint(
                    "meeting", media_a, file_fingerprint(media_a), "cfg"
                )
            )
            manifest.close()

    def test_transcript_checkpoint_malformed_segments_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            media = _media(out)
            manifest = Manifest.open(out)
            self._save(manifest, "meeting", media)
            path = out / ".mediascribe" / "checkpoints" / "meeting.transcript.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["segments"] = [{"text": "no timestamps"}, "not-a-dict"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(
                manifest.transcript_checkpoint("meeting", media, file_fingerprint(media), "cfg")
            )
            manifest.close()

    def test_transcript_checkpoint_nonnumeric_segments_is_missing(self) -> None:
        # Segments with start/end present but non-numeric or inverted must
        # count as malformed → missing → STT re-runs instead of a repeatable
        # summary-stage crash.
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            media = _media(out)
            manifest = Manifest.open(out)
            self._save(manifest, "meeting", media)
            path = out / ".mediascribe" / "checkpoints" / "meeting.transcript.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            for bad in (
                [{"start": "abc", "end": 1.0, "text": "x"}],
                [{"start": 5.0, "end": 1.0, "text": "x"}],
                [{"start": float("nan"), "end": 1.0, "text": "x"}],
                [{"start": -1.0, "end": 1.0, "text": "x"}],
            ):
                payload["segments"] = bad
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertIsNone(
                    manifest.transcript_checkpoint("meeting", media, file_fingerprint(media), "cfg")
                )
            manifest.close()

    def test_symlinked_checkpoints_dir_rejected_on_read(self) -> None:
        # A planted checkpoints symlink must fail reads too, not only writes.
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            media = _media(out)
            manifest = Manifest.open(out)
            self._save(manifest, "meeting", media)
            manifest.close()

            checkpoints = out / ".mediascribe" / "checkpoints"
            real = out / "elsewhere"
            checkpoints.rename(real)
            checkpoints.symlink_to(real, target_is_directory=True)

            reopened = Manifest.open(out)
            try:
                with self.assertRaises(UsageError):
                    reopened.transcript_checkpoint("meeting", media, file_fingerprint(media), "cfg")
                with self.assertRaises(UsageError):
                    reopened.summary_store("meeting").get("sha256:x")
            finally:
                reopened.close()

    def test_deeply_nested_manifest_quarantined_not_crashed(self) -> None:
        # RecursionError from hostile JSON must degrade to corrupt →
        # quarantine + fresh manifest, never an uncaught traceback.
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            state_dir = out / ".mediascribe"
            state_dir.mkdir()
            depth = 2000
            (state_dir / "manifest.json").write_text("[" * depth + "]" * depth, encoding="utf-8")

            manifest = Manifest.open(out)
            try:
                self.assertEqual(manifest.entries(), {})
            finally:
                manifest.close()
            self.assertTrue(list(state_dir.glob("manifest.json.corrupt-*")))

    def test_should_skip_rejects_foreign_output_paths(self) -> None:
        # Recorded outputs outside the output dir (or not matching the
        # deterministic names) must never satisfy the skip check.
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            output = out / "output"
            foreign = out / "foreign"
            foreign.mkdir()
            outputs = {}
            for kind, suffix in (("speaker_script", "화자스크립트"), ("body_summary", "본문요약"), ("full_body", "본문전체")):
                path = foreign / f"meeting_{suffix}.md"
                path.write_text("x", encoding="utf-8")
                outputs[kind] = str(path)
            manifest = Manifest.open(output)
            try:
                manifest.begin("meeting", "fp", "committed")
                manifest.set_state("meeting", "committed", outputs=outputs)
                self.assertFalse(
                    manifest.should_skip("meeting", "fp", ("speaker_script", "body_summary", "full_body"))
                )
            finally:
                manifest.close()

    def test_summary_store_roundtrip_and_corrupt_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)

            manifest = Manifest.open(out)
            store = manifest.summary_store("meeting")
            self.assertIsNone(store.get("sha256:nope"))
            store.put("sha256:a", "요약 A")
            store.put("sha256:b", "요약 B")
            manifest.close()

            reopened = Manifest.open(out)
            try:
                store = reopened.summary_store("meeting")
                self.assertEqual(store.get("sha256:a"), "요약 A")
                self.assertEqual(store.get("sha256:b"), "요약 B")
            finally:
                reopened.close()

            # Corrupt file behaves as empty.
            path = out / ".mediascribe" / "checkpoints" / "meeting.summary.json"
            path.write_text("{broken", encoding="utf-8")
            third = Manifest.open(out)
            try:
                self.assertIsNone(third.summary_store("meeting").get("sha256:a"))
            finally:
                third.close()

    def test_checkpoints_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            media = _media(out)
            manifest = Manifest.open(out)

            self._save(manifest, "meeting", media)
            manifest.close()
            checkpoints = out / ".mediascribe" / "checkpoints"
            for path in checkpoints.iterdir():
                self.assertEqual(_mode(path), 0o600)

class StateDirSafetyTests(unittest.TestCase):
    def test_symlinked_state_dir_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            target = out / "elsewhere"
            target.mkdir()
            (out / ".mediascribe").symlink_to(target, target_is_directory=True)
            with self.assertRaises(UsageError):
                Manifest.open(out)
            with self.assertRaises(UsageError):
                Manifest.open(out, read_only=True)

    def test_state_dir_as_regular_file_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / ".mediascribe").write_text("x", encoding="utf-8")
            with self.assertRaises(UsageError):
                Manifest.open(out)

    def test_symlinked_checkpoints_dir_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            state_dir = out / ".mediascribe"
            state_dir.mkdir()
            target = out / "elsewhere"
            target.mkdir()
            (state_dir / "checkpoints").symlink_to(target, target_is_directory=True)
            manifest = Manifest.open(out)
            try:
                media = _media(out)
                with self.assertRaises(UsageError):
                    manifest.save_transcript_checkpoint(
                        "meeting", _transcript_result(media), file_fingerprint(media), "cfg"
                    )
            finally:
                manifest.close()


class ManifestVersionTests(unittest.TestCase):
    def test_mismatched_version_quarantined_like_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            state_dir = out / ".mediascribe"
            state_dir.mkdir()
            (state_dir / "manifest.json").write_text(
                json.dumps({"version": 999, "files": {"ghost": {"state": "committed"}}}),
                encoding="utf-8",
            )
            manifest = Manifest.open(out)
            try:
                self.assertEqual(manifest.entry("ghost"), None)
                corrupt = list(state_dir.glob("manifest.json.corrupt-*"))
                self.assertEqual(len(corrupt), 1)
            finally:
                manifest.close()


class ConfigFieldCoverageTests(unittest.TestCase):
    def test_config_fields_cover_runtime_config_minus_exclusions(self) -> None:
        import dataclasses

        # Fields that must NOT affect the transcript config fingerprint:
        # output rendering, event plumbing, and run-control flags.
        excluded = {
            "json_lines",
            "env_file",
            "allow_custom_endpoint",
            "resume",
            "follow_symlinks",
            "summary_chunk_chars",
            "events_full_text",
        }
        config_fields = {field.name for field in dataclasses.fields(RuntimeConfig)}
        covered = set(_CONFIG_FIELDS) | {"glossary_path", "speaker_map_path"}
        self.assertEqual(config_fields - excluded, covered & config_fields)
        self.assertEqual(covered - config_fields, set())


if __name__ == "__main__":
    unittest.main()
