from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mediascribe_cli.chunking import (  # noqa: E402
    DEFAULT_CHUNK_CHARS,
    MAX_CHUNK_CHARS,
    chunk_digest,
    chunk_transcript,
)


class ChunkTranscriptTests(unittest.TestCase):
    def test_empty_input_returns_no_chunks(self) -> None:
        self.assertEqual(chunk_transcript([], 100), [])

    def test_single_line_fits_one_chunk(self) -> None:
        self.assertEqual(chunk_transcript(["hello"], 100), [["hello"]])

    def test_packing_respects_budget_including_separators(self) -> None:
        # "aaa" + "\n" + "aaa" = 7 chars > 6, so the second line starts a new chunk.
        chunks = chunk_transcript(["aaa", "aaa"], 6)
        self.assertEqual(chunks, [["aaa"], ["aaa"]])

    def test_lines_pack_greedily_on_boundaries(self) -> None:
        lines = ["aa", "bb", "cc", "dd"]
        # "aa\nbb" = 5; adding "\ncc" would reach 8 > 6.
        chunks = chunk_transcript(lines, 6)
        self.assertEqual(chunks, [["aa", "bb"], ["cc", "dd"]])

    def test_every_chunk_within_budget(self) -> None:
        lines = [f"line-{i} " + "x" * (i % 50) for i in range(1000)]
        chunks = chunk_transcript(lines, DEFAULT_CHUNK_CHARS)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len("\n".join(chunk)), DEFAULT_CHUNK_CHARS)
            self.assertTrue(chunk)  # never an empty chunk

    def test_roundtrip_preserves_all_lines_in_order(self) -> None:
        lines = ["", "alpha", "", "beta gamma", "zz"]
        chunks = chunk_transcript(lines, 10)
        flattened = [line for chunk in chunks for line in chunk]
        self.assertEqual(flattened, lines)

    def test_hard_split_pieces_concatenate_to_original(self) -> None:
        line = "x" * 30
        chunks = chunk_transcript([line], 10)
        self.assertEqual("".join(piece for chunk in chunks for piece in chunk), line)

    def test_oversized_line_is_hard_split_deterministically(self) -> None:
        line = "abcdefghij"  # 10 chars, budget 4 → 4/4/2
        chunks = chunk_transcript([line], 4)
        self.assertEqual(chunks, [["abcd"], ["efgh"], ["ij"]])
        self.assertEqual(chunk_transcript([line], 4), chunks)  # deterministic

    def test_oversized_line_between_normal_lines(self) -> None:
        chunks = chunk_transcript(["hi", "abcdef", "yo"], 3)
        self.assertEqual(chunks, [["hi"], ["abc"], ["def"], ["yo"]])

    def test_deterministic_for_same_input(self) -> None:
        lines = [f"[00:00:{i:02d}–00:01:{i:02d}] SPEAKER_{i % 3}: 발화 {i}" for i in range(500)]
        first = chunk_transcript(lines, DEFAULT_CHUNK_CHARS)
        second = chunk_transcript(lines, DEFAULT_CHUNK_CHARS)
        self.assertEqual(first, second)
        self.assertEqual([chunk_digest(c) for c in first], [chunk_digest(c) for c in second])


class ChunkDigestTests(unittest.TestCase):
    def test_digest_is_stable_sha256(self) -> None:
        import hashlib

        lines = ["a", "b"]
        expected = "sha256:" + hashlib.sha256("a\nb".encode("utf-8")).hexdigest()
        self.assertEqual(chunk_digest(lines), expected)
        self.assertEqual(chunk_digest(lines), chunk_digest(["a", "b"]))

    def test_digest_distinguishes_content(self) -> None:
        self.assertNotEqual(chunk_digest(["a", "b"]), chunk_digest(["a", "c"]))
        self.assertNotEqual(chunk_digest(["ab"]), chunk_digest(["a", "b"]))

    def test_max_chunk_chars_is_hard_ceiling(self) -> None:
        self.assertGreaterEqual(MAX_CHUNK_CHARS, DEFAULT_CHUNK_CHARS)


if __name__ == "__main__":
    unittest.main()
