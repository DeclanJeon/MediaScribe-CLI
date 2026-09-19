from __future__ import annotations

import hashlib
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mediascribe_cli.artifacts import (  # noqa: E402
    ARTIFACT_KINDS,
    ArtifactStager,
    ArtifactTarget,
    cleanup_staging,
    output_key,
    plan_artifacts,
    plan_output_keys,
    reconcile_commits,
)
from mediascribe_cli.errors import UsageError  # noqa: E402


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class OutputKeyTests(unittest.TestCase):
    def test_unique_stem_uses_stem(self) -> None:
        files = [Path("meeting.m4a"), Path("other.wav")]
        self.assertEqual(output_key(files[0], files), "meeting")

    def test_same_stem_disambiguates_with_extension(self) -> None:
        files = [Path("meeting.m4a"), Path("meeting.wav")]
        self.assertEqual(output_key(files[0], files), "meeting.m4a")
        self.assertEqual(output_key(files[1], files), "meeting.wav")

    def test_same_name_in_different_dirs_still_distinct(self) -> None:
        files = [Path("/a/meeting.m4a"), Path("/b/meeting.m4a")]
        key_a = output_key(files[0], files)
        key_b = output_key(files[1], files)
        self.assertNotEqual(key_a, key_b)
        self.assertTrue(key_a.startswith("meeting.m4a"))

    def test_deterministic(self) -> None:
        files = [Path("meeting.m4a"), Path("meeting.wav")]
        self.assertEqual(output_key(files[0], files), output_key(files[0], files))


class PlanArtifactsTests(unittest.TestCase):
    def test_three_kinds_with_korean_suffixes(self) -> None:
        targets = plan_artifacts(Path("/out"), "meeting", "md")
        self.assertEqual(set(targets), set(ARTIFACT_KINDS))
        self.assertEqual(targets["speaker_script"].path, Path("/out/meeting_화자스크립트.md"))
        self.assertEqual(targets["body_summary"].path, Path("/out/meeting_본문요약.md"))
        self.assertEqual(targets["full_body"].path, Path("/out/meeting_본문전체.md"))

    def test_invalid_format_rejected(self) -> None:
        with self.assertRaises(UsageError):
            plan_artifacts(Path("/out"), "meeting", "pdf")


class ArtifactStagerTests(unittest.TestCase):
    def test_commit_produces_exactly_three_finals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            stager = ArtifactStager(plan_artifacts(out, "meeting", "md"))
            for kind in ARTIFACT_KINDS:
                stager.stage(kind, f"content-{kind}")
            committed = stager.commit()

            self.assertEqual(set(committed), set(ARTIFACT_KINDS))
            finals = sorted(out.iterdir())
            self.assertEqual(len(finals), 3)
            for kind in ARTIFACT_KINDS:
                final = Path(committed[kind])
                self.assertTrue(final.is_file())
                self.assertEqual(final.read_text(encoding="utf-8"), f"content-{kind}")
                self.assertEqual(_mode(final), 0o600)
            self.assertEqual(list(out.glob("*.tmp-*")), [])

    def test_stage_writes_private_temp_in_final_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            stager = ArtifactStager(plan_artifacts(out, "meeting", "md"))
            temp = stager.stage("speaker_script", "x")
            self.assertEqual(temp.parent, out)
            self.assertIn(".tmp-", temp.name)
            self.assertEqual(_mode(temp), 0o600)

    def test_abort_leaves_zero_finals_and_zero_temps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            stager = ArtifactStager(plan_artifacts(out, "meeting", "md"))
            stager.stage("speaker_script", "x")
            stager.stage("body_summary", "y")
            stager.abort()
            stager.abort()  # idempotent
            self.assertEqual(list(out.iterdir()), [])

    def test_commit_without_all_staged_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            stager = ArtifactStager(plan_artifacts(out, "meeting", "md"))
            stager.stage("speaker_script", "x")
            with self.assertRaises(UsageError):
                stager.commit()
            stager.abort()
            self.assertEqual(list(out.iterdir()), [])

    def test_unknown_kind_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stager = ArtifactStager(plan_artifacts(Path(directory), "meeting", "md"))
            with self.assertRaises(UsageError):
                stager.stage("bogus", "x")

    def test_single_target_commit_preserves_sibling_files(self) -> None:
        # minutes --output report.md stages one target; unrelated same-stem
        # files (report.html, report.txt) are NOT stale artifacts and must
        # survive the commit.
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for name in ("report.html", "report.txt"):
                (out / name).write_text("keep", encoding="utf-8")
            stager = ArtifactStager({"body_summary": ArtifactTarget(kind="body_summary", path=out / "report.md")})
            stager.stage("body_summary", "summary")
            stager.commit()

            self.assertEqual((out / "report.md").read_text(encoding="utf-8"), "summary")
            self.assertEqual((out / "report.html").read_text(encoding="utf-8"), "keep")
            self.assertEqual((out / "report.txt").read_text(encoding="utf-8"), "keep")

    def test_artifact_key_with_control_chars_rejected(self) -> None:
        # Newline/tab in a key would inject records into the line-oriented
        # commit journal — rejected at planning time.
        with self.assertRaises(UsageError):
            plan_artifacts(Path("/out"), "evil\nmeeting", "md")
        with self.assertRaises(UsageError):
            plan_artifacts(Path("/out"), "evil\tmeeting", "md")

    def test_rerun_md_to_html_leaves_exactly_three_html(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            first = ArtifactStager(plan_artifacts(out, "meeting", "md"))
            for kind in ARTIFACT_KINDS:
                first.stage(kind, "old")
            first.commit()

            second = ArtifactStager(plan_artifacts(out, "meeting", "html"))
            for kind in ARTIFACT_KINDS:
                second.stage(kind, "new")
            second.commit()

            finals = sorted(out.iterdir())
            self.assertEqual(len(finals), 3)
            self.assertTrue(all(path.suffix == ".html" for path in finals))
            self.assertEqual(list(out.glob("*.md")), [])

    def test_reconcile_rejects_duplicate_marker_targets(self) -> None:
        # A marker repeating one valid entry three times is not a complete
        # 3-target commit — the partial set must be removed, not kept.
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            final = out / "meeting_본문요약.md"
            final.write_text("partial", encoding="utf-8")
            digest = hashlib.sha256(b"partial").hexdigest()
            marker = out / f".mediascribe-commit-{os.getpid() + 999999}"
            marker.write_text(
                "targets\t3\n"
                + f"{final.name}\t{digest}\n" * 3,
                encoding="utf-8",
            )
            removed = reconcile_commits(out)
            self.assertGreaterEqual(removed, 1)
            self.assertFalse(final.exists())
            self.assertEqual(list(out.glob("*.tmp-*")), [])

    def test_symlink_target_replaced_not_followed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            outside = out / "outside.txt"
            outside.write_text("victim", encoding="utf-8")
            targets = plan_artifacts(out, "meeting", "md")
            targets["speaker_script"].path.symlink_to(outside)

            stager = ArtifactStager(targets)
            stager.stage("speaker_script", "new")
            stager.stage("body_summary", "s")
            stager.stage("full_body", "f")
            stager.commit()

            final = targets["speaker_script"].path
            self.assertFalse(final.is_symlink())
            self.assertEqual(final.read_text(encoding="utf-8"), "new")
            self.assertEqual(outside.read_text(encoding="utf-8"), "victim")

    def test_same_stem_inputs_produce_six_distinct_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            files = [Path("meeting.m4a"), Path("meeting.wav")]
            for media in files:
                key = output_key(media, files)
                stager = ArtifactStager(plan_artifacts(out, key, "md"))
                for kind in ARTIFACT_KINDS:
                    stager.stage(kind, f"{key}-{kind}")
                stager.commit()
            names = sorted(path.name for path in out.iterdir())
            self.assertEqual(len(names), 6)
            self.assertTrue(any(name.startswith("meeting.m4a_") for name in names))
            self.assertTrue(any(name.startswith("meeting.wav_") for name in names))


class CleanupStagingTests(unittest.TestCase):
    def test_removes_leftover_temps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "meeting_본문요약.md.tmp-99999-body_summary").write_text("x")
            (out / "meeting_본문전체.md.tmp-99999-full_body").write_text("x")
            (out / "meeting_화자스크립트.md").write_text("keep")
            (out / "notes.tmp-1").write_text("user file, not a stager temp")
            (out / "x.tmp-abc-body_summary").write_text("non-numeric pid")
            self.assertEqual(cleanup_staging(out), 2)
            remaining = sorted(path.name for path in out.iterdir())
            self.assertEqual(
                remaining,
                ["meeting_화자스크립트.md", "notes.tmp-1", "x.tmp-abc-body_summary"],
            )

    def test_missing_directory_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(cleanup_staging(Path(directory) / "nope"), 0)

class StagingTempSafetyTests(unittest.TestCase):
    def test_stage_never_follows_planted_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            victim = out / "victim.txt"
            victim.write_text("precious", encoding="utf-8")
            targets = plan_artifacts(out, "meeting", "md")
            final = targets["speaker_script"].path
            planted = final.with_name(f"{final.name}.tmp-{os.getpid()}-speaker_script")
            planted.symlink_to(victim)

            stager = ArtifactStager(targets)
            temp = stager.stage("speaker_script", "new content")

            self.assertNotEqual(temp, planted)
            self.assertTrue(temp.name.endswith("-1"))
            self.assertEqual(victim.read_text(encoding="utf-8"), "precious")
            self.assertTrue(planted.is_symlink())  # never unlinked/followed
            stager.abort()

    def test_stage_retries_past_existing_temp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            targets = plan_artifacts(out, "meeting", "md")
            final = targets["body_summary"].path
            leftover = final.with_name(f"{final.name}.tmp-{os.getpid()}-body_summary")
            leftover.write_text("stale", encoding="utf-8")

            stager = ArtifactStager(targets)
            temp = stager.stage("body_summary", "fresh")

            self.assertTrue(temp.name.endswith("-1"))
            self.assertEqual(temp.read_text(encoding="utf-8"), "fresh")
            self.assertEqual(leftover.read_text(encoding="utf-8"), "stale")
            stager.abort()

    def test_created_output_dir_is_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new-output"
            stager = ArtifactStager(plan_artifacts(out, "meeting", "md"))
            stager.stage("speaker_script", "x")
            self.assertEqual(_mode(out), 0o700)
            stager.abort()

    def test_preexisting_output_dir_mode_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "existing"
            out.mkdir(mode=0o755)
            stager = ArtifactStager(plan_artifacts(out, "meeting", "md"))
            stager.stage("speaker_script", "x")
            self.assertEqual(_mode(out), 0o755)
            stager.abort()


class PlanOutputKeysTests(unittest.TestCase):
    def test_matches_output_key_for_every_input(self) -> None:
        files = [Path("meeting.m4a"), Path("meeting.wav"), Path("other.mp3")]
        keys = plan_output_keys(files)
        self.assertEqual(set(keys), set(files))
        for media in files:
            self.assertEqual(keys[media], output_key(media, files))
        self.assertEqual(keys[Path("meeting.m4a")], "meeting.m4a")
        self.assertEqual(keys[Path("other.mp3")], "other")


class CommitReconcileTests(unittest.TestCase):
    def test_dead_pid_marker_removes_partial_finals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            partial = out / "meeting_화자스크립트.md"
            partial.write_text("partial", encoding="utf-8")
            marker = out / ".mediascribe-commit-99999999"
            # Journal format: "targets\tN" header + "name\t<sha256>" records.
            # Digests that don't match the file → incomplete commit → delete.
            bogus = hashlib.sha256(b"committed-content").hexdigest()
            marker.write_text(
                f"targets\t2\nmeeting_화자스크립트.md\t{bogus}\nmeeting_본문요약.md\t{bogus}\n",
                encoding="utf-8",
            )

            removed = reconcile_commits(out)

            self.assertFalse(partial.exists())
            self.assertFalse(marker.exists())
            self.assertGreaterEqual(removed, 2)

    def test_live_pid_marker_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            final = out / "meeting_화자스크립트.md"
            final.write_text("live", encoding="utf-8")
            marker = out / ".mediascribe-commit-42424242"
            marker.write_text("meeting_화자스크립트.md\n", encoding="utf-8")

            with unittest.mock.patch("mediascribe_cli.artifacts._pid_alive", return_value=True):
                reconcile_commits(out)

            self.assertTrue(final.exists())  # live foreign pid → untouched
            self.assertTrue(marker.exists())

    def test_stale_marker_with_complete_triplet_only_removes_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            finals = [
                out / "meeting_화자스크립트.md",
                out / "meeting_본문요약.md",
                out / "meeting_본문전체.md",
            ]
            lines = ["targets\t3"]
            for final in finals:
                final.write_text("done", encoding="utf-8")
                digest = hashlib.sha256(final.read_bytes()).hexdigest()
                lines.append(f"{final.name}\t{digest}")
            marker = out / ".mediascribe-commit-99999996"
            marker.write_text("\n".join(lines) + "\n", encoding="utf-8")

            removed = reconcile_commits(out)

            for final in finals:
                self.assertTrue(final.exists())  # completed commit is never deleted
            self.assertFalse(marker.exists())
            self.assertEqual(removed, 1)

    def test_dead_marker_removes_mixed_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            finals = [
                out / "meeting_화자스크립트.md",
                out / "meeting_본문요약.md",
                out / "meeting_본문전체.md",
            ]
            # Crash after replacing only the first target over an old triplet:
            # all three names exist but only the first matches the journal.
            finals[0].write_text("new", encoding="utf-8")
            finals[1].write_text("old", encoding="utf-8")
            finals[2].write_text("old", encoding="utf-8")
            lines = [
                "targets\t3",
                f"{finals[0].name}\t{hashlib.sha256(b'new').hexdigest()}",
                f"{finals[1].name}\t{hashlib.sha256(b'new').hexdigest()}",
                f"{finals[2].name}\t{hashlib.sha256(b'new').hexdigest()}",
            ]
            marker = out / ".mediascribe-commit-99999995"
            marker.write_text("\n".join(lines) + "\n", encoding="utf-8")

            removed = reconcile_commits(out)

            for final in finals:
                self.assertFalse(final.exists())  # mixed set fully removed
            self.assertFalse(marker.exists())
            self.assertEqual(removed, 4)

    def test_marker_never_deletes_outside_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            out.mkdir()
            victim = Path(directory) / "victim.txt"
            victim.write_text("precious", encoding="utf-8")
            marker = out / ".mediascribe-commit-99999998"
            marker.write_text("../victim.txt\n", encoding="utf-8")

            reconcile_commits(out)

            self.assertTrue(victim.exists())
            self.assertFalse(marker.exists())

    def test_cleanup_staging_skips_live_pid_temps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            live = out / "meeting_본문요약.md.tmp-42424241-body_summary"
            live.write_text("live", encoding="utf-8")
            dead = out / "meeting_본문요약.md.tmp-99999997-body_summary"
            dead.write_text("dead", encoding="utf-8")

            with unittest.mock.patch("mediascribe_cli.artifacts._pid_alive", side_effect=lambda pid: pid == 42424241):
                removed = cleanup_staging(out)

            self.assertTrue(live.exists())
            self.assertFalse(dead.exists())
            self.assertEqual(removed, 1)


if __name__ == "__main__":
    unittest.main()
