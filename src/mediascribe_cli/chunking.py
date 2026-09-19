"""Deterministic transcript chunking for bounded provider requests.

Stdlib-only. Chunk budgets are measured in characters, not tokens: boring,
deterministic, and tokenizer-free. The summary pipeline joins each chunk with
``"\\n"`` before sending, so the budget accounts for the separators too.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

DEFAULT_CHUNK_CHARS = 24_000
MAX_CHUNK_CHARS = 256_000  # hard ceiling; config above this is clamped
MAX_TRANSCRIPT_BYTES = 32 * 1024 * 1024  # minutes-command input cap
MAX_RESPONSE_BYTES = 1 * 1024 * 1024
MAX_ERROR_BODY_BYTES = 64 * 1024


def chunk_transcript(lines: Sequence[str], max_chars: int) -> list[list[str]]:
    """Greedy line-boundary packing of ``lines`` into chunks.

    A line longer than ``max_chars`` is hard-split at ``max_chars`` boundaries
    (deterministic, no mid-word requirement); the pieces become separate chunk
    lines that concatenate back to the original. Lines within budget are
    preserved verbatim and in order. Never returns an empty chunk; empty
    input returns ``[]``.
    """
    budget = max(1, int(max_chars))
    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for line in lines:
        text = str(line)
        pieces = [text[i : i + budget] for i in range(0, len(text), budget)] or [""]
        for piece in pieces:
            needed = len(piece) + (1 if current else 0)
            if current and current_chars + needed > budget:
                chunks.append(current)
                current = []
                current_chars = 0
                needed = len(piece)
            current.append(piece)
            current_chars += needed
    if current:
        chunks.append(current)
    return chunks


def chunk_digest(chunk_lines: Sequence[str]) -> str:
    """``sha256:`` digest of ``"\\n".join(lines)`` — stable checkpoint key."""
    payload = "\n".join(str(line) for line in chunk_lines)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
