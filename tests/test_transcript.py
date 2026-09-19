from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mediascribe_cli.artifacts import ArtifactStager, plan_artifacts  # noqa: E402
from mediascribe_cli.transcript import (  # noqa: E402
    apply_glossary,
    normalize_artifact_format,
    render_full_body,
    render_speaker_script,
    render_summary,
    split_segments_by_speaker,
    transcript_lines,
)


class TranscriptTests(unittest.TestCase):
    def test_glossary_preserves_raw_text_and_cleans_text(self) -> None:
        clean, terms = apply_glossary("미디어 스크라이브 회의", {"MediaScribe": ("MediaScribe", "미디어 스크라이브")})
        self.assertEqual(clean, "MediaScribe 회의")
        self.assertEqual(terms, ["미디어 스크라이브 → MediaScribe"])

    def test_word_timestamps_are_split_by_speaker_and_labeled(self) -> None:
        segments, speakers = split_segments_by_speaker(
            [
                {
                    "start": 0,
                    "end": 2,
                    "text": "안녕하세요 반갑습니다",
                    "words": [
                        {"start": 0, "end": 0.5, "word": "안녕하세요 ", "probability": 0.9},
                        {"start": 1, "end": 1.5, "word": "반갑습니다", "probability": 0.8},
                    ],
                }
            ],
            diarization_turns=[
                {"start": 0, "end": 0.7, "speaker": "SPEAKER_00"},
                {"start": 0.8, "end": 2, "speaker": "SPEAKER_01"},
            ],
            requested_labels=("진행자", "참석자"),
        )
        self.assertEqual([item["speaker_label"] for item in segments], ["진행자", "참석자"])
        self.assertEqual([item["id"] for item in speakers], ["SPEAKER_00", "SPEAKER_01"])
        self.assertAlmostEqual(segments[0]["confidence"], 0.9)
        self.assertAlmostEqual(segments[1]["confidence"], 0.8)

    def test_three_artifacts_share_the_selected_format(self) -> None:
        segments = [
            {"start": 0, "end": 1, "text": "발화", "clean_text": "발화", "speaker_label": "A"},
            {"start": 1, "end": 2, "text": "다음 발화", "clean_text": "다음 발화", "speaker_label": "B"},
        ]
        self.assertEqual(normalize_artifact_format("HTML"), "html")
        self.assertIn("A: 발화", render_speaker_script("meeting.m4a", segments, "txt"))
        self.assertIn("본문 전체본", render_full_body("meeting.m4a", segments, "md"))
        self.assertIn("<html", render_summary("# 요약\n\n내용", "회의", "html"))

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            stager = ArtifactStager(plan_artifacts(out, "meeting", "md"))
            stager.stage("speaker_script", render_speaker_script("meeting.m4a", segments, "md"))
            stager.stage("body_summary", render_summary("# 요약\n\n내용", "회의", "md"))
            stager.stage("full_body", render_full_body("meeting.m4a", segments, "md"))
            outputs = stager.commit()
            self.assertEqual(set(outputs), {"speaker_script", "body_summary", "full_body"})
            self.assertTrue(outputs["speaker_script"].endswith("_화자스크립트.md"))
            self.assertTrue(outputs["body_summary"].endswith("_본문요약.md"))
            self.assertTrue(outputs["full_body"].endswith("_본문전체.md"))
            self.assertEqual(len(list(out.iterdir())), 3)

    def test_transcript_lines_format_is_evidence_contract(self) -> None:
        segments = [
            {"start": 1, "end": 4, "text": "안녕하세요.", "speaker_label": "A"},
            {"start": 5, "end": 8, "text": "반갑습니다.", "speaker_id": "SPEAKER_01"},
            {"start": 9, "end": 10, "text": ""},
        ]
        self.assertEqual(
            transcript_lines(segments),
            ["[00:00:01–00:00:04] A: 안녕하세요.", "[00:00:05–00:00:08] SPEAKER_01: 반갑습니다."],
        )

    def test_render_summary_strips_unsafe_markup(self) -> None:
        dirty = "# 요약\n\n<script>alert(1)</script>내용 [x](javascript:alert(1))"
        rendered = render_summary(dirty, "회의", "md")
        self.assertNotIn("<script", rendered)
        self.assertNotIn("javascript:", rendered)
        self.assertIn("내용", rendered)

    def test_md_escapes_transcript_markup(self) -> None:
        segments = [{"start": 0, "end": 1, "text": "# 제목 & <태그>", "speaker_label": "A"}]
        body = render_full_body("meeting.m4a", segments, "md")
        self.assertIn("\\# 제목 &amp; &lt;태그&gt;", body)

    def test_md_escape_applies_to_every_line(self) -> None:
        # A segment containing a newline must escape leading markup on each
        # line, not just the first.
        segments = [{"start": 0, "end": 1, "text": "첫 줄\n# heading\n- item", "speaker_label": "A"}]
        body = render_full_body("meeting.m4a", segments, "md")
        self.assertIn("\\# heading", body)
        self.assertIn("\\- item", body)

    def test_md_escapes_inline_markup_and_links(self) -> None:
        # Inline Markdown (links, emphasis, code spans) in transcript text
        # must render literally — a [x](javascript:…) line stays inert.
        segments = [{"start": 0, "end": 1, "text": "[open](javascript:alert(1)) and *bold* `code`", "speaker_label": "A"}]
        body = render_full_body("meeting.m4a", segments, "md")
        self.assertIn("\\[open\\](javascript:alert(1))", body)
        self.assertIn("\\*bold\\*", body)
        self.assertIn("\\`code\\`", body)

    def test_md_escapes_media_derived_title(self) -> None:
        # A hostile filename stem must not inject markup into the heading.
        segments = [{"start": 0, "end": 1, "text": "x"}]
        body = render_full_body("evil [x](javascript:alert(1)).m4a", segments, "md")
        self.assertIn("\\[x\\]", body.splitlines()[0])

    def test_read_transcript_source_rejects_symlink(self) -> None:
        from mediascribe_cli.errors import UsageError
        from mediascribe_cli.transcript import read_transcript_source

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real.txt"
            real.write_text("transcript", encoding="utf-8")
            link = root / "link.txt"
            link.symlink_to(real)
            with self.assertRaises(UsageError):
                read_transcript_source(link)

    def test_read_transcript_source_deep_json_is_usage_error(self) -> None:
        from mediascribe_cli.errors import UsageError
        from mediascribe_cli.transcript import read_transcript_source

        with tempfile.TemporaryDirectory() as directory:
            deep = Path(directory) / "deep.json"
            depth = 200000
            deep.write_text("[" * depth + "]" * depth, encoding="utf-8")
            with self.assertRaises(UsageError):
                read_transcript_source(deep)


if __name__ == "__main__":
    unittest.main()
