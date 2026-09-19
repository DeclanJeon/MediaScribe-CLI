from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import ARTIFACT_EXTENSIONS, DEFAULT_ARTIFACT_FORMAT
from .chunking import MAX_TRANSCRIPT_BYTES
from .errors import UsageError
from .safeio import open_verified


def normalize_artifact_format(value: str | None) -> str:
    normalized = str(value or DEFAULT_ARTIFACT_FORMAT).strip().lower()
    if normalized not in ARTIFACT_EXTENSIONS:
        raise ValueError(f"출력 형식은 html, md, txt 중 하나여야 합니다: {value}")
    return normalized


_MAX_AUX_JSON_BYTES = 16 * 1024 * 1024


def _read_aux_json(path: str | Path, label: str) -> Any:
    """Read a small user-supplied JSON file (glossary, speaker map) with a
    hard size cap — these are config inputs, not bulk data."""
    source = Path(path).expanduser().resolve()
    try:
        if source.stat().st_size > _MAX_AUX_JSON_BYTES:
            raise UsageError(f"{label} 파일이 허용 한도({_MAX_AUX_JSON_BYTES // (1024 * 1024)} MiB)를 초과합니다: {source}")
        return json.loads(source.read_text(encoding="utf-8"))
    except (ValueError, RecursionError, MemoryError) as exc:
        if isinstance(exc, UsageError):
            raise
        raise UsageError(f"{label} JSON을 해석할 수 없습니다: {source}") from exc


def load_glossary(path: str | Path | None) -> dict[str, tuple[str, ...]]:
    if not path:
        return {}
    payload = _read_aux_json(path, "용어 사전")
    if isinstance(payload, dict) and isinstance(payload.get("terms"), dict):
        payload = payload["terms"]
    if not isinstance(payload, dict):
        raise ValueError("용어 사전은 JSON object 또는 terms object여야 합니다.")

    glossary: dict[str, tuple[str, ...]] = {}
    for key, value in payload.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if isinstance(value, list):
            canonical = key_text
            aliases = [str(item).strip() for item in value]
        elif isinstance(value, str):
            canonical = value.strip()
            aliases = [key_text]
        else:
            raise ValueError(f"용어 사전 항목 {key_text!r}은 문자열 또는 배열이어야 합니다.")
        if canonical:
            glossary[canonical] = tuple(dict.fromkeys([canonical, *aliases]))
    return glossary


def apply_glossary(text: str, glossary: Mapping[str, Sequence[str]] | None) -> tuple[str, list[str]]:
    cleaned = str(text or "").strip()
    if not cleaned or not glossary:
        return cleaned, []
    replacements: list[tuple[str, str]] = []
    for canonical, aliases in glossary.items():
        canonical_text = str(canonical).strip()
        for alias in aliases:
            alias_text = str(alias).strip()
            if alias_text and alias_text != canonical_text:
                replacements.append((alias_text, canonical_text))
    normalized_terms: list[str] = []
    for alias, canonical in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if alias in cleaned:
            cleaned = cleaned.replace(alias, canonical)
            normalized_terms.append(f"{alias} → {canonical}")
    return cleaned, list(dict.fromkeys(normalized_terms))


def format_clock(seconds: float) -> str:
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_speaker_map(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    payload = _read_aux_json(path, "화자 매핑")
    if not isinstance(payload, dict):
        raise ValueError("화자 매핑은 JSON object여야 합니다.")
    return {str(key): str(value).strip() for key, value in payload.items() if str(value).strip()}


def _word_dict(word: Any) -> dict[str, Any]:
    if isinstance(word, Mapping):
        return {
            "start": word.get("start"),
            "end": word.get("end"),
            "word": str(word.get("word", "")),
            "probability": word.get("probability"),
        }
    return {
        "start": getattr(word, "start", None),
        "end": getattr(word, "end", None),
        "word": str(getattr(word, "word", "")),
        "probability": getattr(word, "probability", None),
    }


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _best_speaker(start: float, end: float, turns: Sequence[Mapping[str, Any]]) -> str:
    if not turns:
        return ""
    midpoint = (float(start) + float(end)) / 2
    ranked: list[tuple[float, float, str]] = []
    for turn in turns:
        turn_start = float(turn["start"])
        turn_end = float(turn["end"])
        overlap = _overlap(float(start), float(end), turn_start, turn_end)
        distance = 0.0 if turn_start <= midpoint <= turn_end else min(abs(midpoint - turn_start), abs(midpoint - turn_end))
        ranked.append((overlap, -distance, str(turn["speaker"])))
    ranked.sort(reverse=True)
    best_overlap, negative_distance, speaker = ranked[0]
    return speaker if best_overlap > 0 or negative_distance >= -1.0 else ""


def _letter_label(index: int) -> str:
    value = int(index)
    label = ""
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("A") + remainder) + label
        if value == 0:
            return label
        value -= 1


def _speaker_labels(
    speaker_ids: Sequence[str],
    requested: Sequence[str] | None,
    explicit_map: Mapping[str, str] | None,
) -> dict[str, str]:
    mapped = {str(key): str(value).strip() for key, value in (explicit_map or {}).items() if str(value).strip()}
    requested_values = [str(item).strip() for item in (requested or ()) if str(item).strip()]
    return {
        speaker_id: mapped.get(speaker_id) or (requested_values[index] if index < len(requested_values) else _letter_label(index))
        for index, speaker_id in enumerate(speaker_ids)
    }


def split_segments_by_speaker(
    segments: Sequence[Mapping[str, Any]],
    diarization_turns: Sequence[Mapping[str, Any]] | None = None,
    glossary: Mapping[str, Sequence[str]] | None = None,
    requested_labels: Sequence[str] | None = None,
    speaker_map: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    turns = list(diarization_turns or [])
    materialized: list[dict[str, Any]] = []
    speaker_order: list[str] = []

    def remember(speaker_id: str) -> None:
        if speaker_id and speaker_id not in speaker_order:
            speaker_order.append(speaker_id)

    def confidence_for(words: Sequence[Mapping[str, Any]]) -> float | None:
        probabilities = [float(word["probability"]) for word in words if word.get("probability") is not None]
        return sum(probabilities) / len(probabilities) if probabilities else None

    def add_segment(start: float, end: float, raw_text: str, speaker_id: str, words: Sequence[Mapping[str, Any]], confidence: float | None = None) -> None:
        raw_value = str(raw_text or "").strip()
        if not raw_value:
            return
        clean_value, normalized_terms = apply_glossary(raw_value, glossary)
        remember(speaker_id)
        resolved_confidence = confidence if confidence is not None else confidence_for(words)
        segment: dict[str, Any] = {
            "start": float(start),
            "end": float(end),
            "text": clean_value,
            "raw_text": raw_value,
            "clean_text": clean_value,
            "speaker_id": speaker_id,
            "words": [dict(word) for word in words],
            "normalized_terms": normalized_terms,
        }
        if resolved_confidence is not None:
            segment["confidence"] = resolved_confidence
        materialized.append(segment)

    for source in segments:
        start = float(source.get("start", 0.0))
        end = float(source.get("end", start))
        raw_text = str(source.get("raw_text") or source.get("text") or "").strip()
        words = [_word_dict(word) for word in (source.get("words") or [])]
        words = [word for word in words if word["word"].strip()]
        if not turns or not words or not all(word.get("start") is not None and word.get("end") is not None for word in words):
            add_segment(start, end, raw_text, _best_speaker(start, end, turns), words, source.get("confidence"))
            continue

        groups: list[dict[str, Any]] = []
        for word in words:
            word_start = float(word["start"])
            word_end = float(word["end"])
            speaker_id = _best_speaker(word_start, word_end, turns)
            if groups and groups[-1]["speaker_id"] == speaker_id and word_start - groups[-1]["end"] <= 1.2:
                groups[-1]["end"] = word_end
                groups[-1]["words"].append(word)
                groups[-1]["raw_text"] += word["word"]
            else:
                groups.append({"start": word_start, "end": word_end, "speaker_id": speaker_id, "words": [word], "raw_text": word["word"]})
        for group in groups:
            add_segment(group["start"], group["end"], group["raw_text"], group["speaker_id"], group["words"])

    labels = _speaker_labels(speaker_order, requested_labels, speaker_map)
    for segment in materialized:
        segment["speaker_label"] = labels.get(segment["speaker_id"], "")
        if not segment["words"]:
            segment.pop("words", None)
    speakers = [{"id": speaker_id, "label": labels.get(speaker_id, ""), "first_seen": index} for index, speaker_id in enumerate(speaker_order)]
    return materialized, speakers


def text_for_segment(segment: Mapping[str, Any]) -> str:
    return str(segment.get("clean_text") or segment.get("text") or segment.get("raw_text") or "").strip()


def _speaker_rows(segments: Sequence[Mapping[str, Any]]) -> list[str]:
    rows = []
    for segment in segments:
        text = text_for_segment(segment)
        if text:
            label = str(segment.get("speaker_label") or segment.get("speaker_id") or "-")
            rows.append(f"[{format_clock(segment['start'])}–{format_clock(segment['end'])}] {label}: {text}")
    return rows


def _speaker_rows_md(segments: Sequence[Mapping[str, Any]]) -> list[str]:
    return [_md_escape(row) for row in _speaker_rows(segments)]


def _full_body_lines(segments: Sequence[Mapping[str, Any]]) -> list[str]:
    return [text_for_segment(segment) for segment in segments if text_for_segment(segment)]


def _full_body_lines_md(segments: Sequence[Mapping[str, Any]]) -> list[str]:
    return [_md_escape(line) for line in _full_body_lines(segments)]

def transcript_lines(segments: Sequence[Mapping[str, Any]]) -> list[str]:
    """``'[HH:MM:SS–HH:MM:SS] label: text'`` lines for summary prompts.

    This format is the contract for evidence timestamps; both the transcribe
    pipeline and the ``minutes`` command feed it to the chunked summarizer.
    """
    lines: list[str] = []
    for segment in segments:
        text = text_for_segment(segment)
        if not text:
            continue
        label = str(segment.get("speaker_label") or segment.get("speaker_id") or "").strip()
        speaker = f" {label}:" if label else ""
        lines.append(f"[{format_clock(segment['start'])}–{format_clock(segment['end'])}]{speaker} {text}".strip())
    return lines


_MD_LEADING_MARKUP_RE = re.compile(r"^(\s{0,3})([#\-|>+~=`]|\d+[.)])", re.MULTILINE)
_MD_INLINE_META_RE = re.compile(r"([\\`*_\[\]])")


def _md_escape(text: str) -> str:
    """Escape transcript-derived text for Markdown output.

    Neutralizes inline markup (links, emphasis, code spans) and leading
    block markers (headings, lists, quotes, fences, tables, ordered lists)
    so transcript text can never become active markup — including
    ``[x](javascript:…)`` links. HTML metacharacters are entity-escaped.
    """
    escaped = str(text)
    escaped = _MD_INLINE_META_RE.sub(r"\\\1", escaped)
    escaped = escaped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _MD_LEADING_MARKUP_RE.sub(r"\1\\\2", escaped)


def _html_document(title: str, body: str) -> str:
    return f"<!doctype html>\n<html lang=\"ko\">\n<head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{html.escape(title)}</title><style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 20px;line-height:1.7;color:#172033}}h1{{border-bottom:1px solid #d8deea;padding-bottom:12px}}.speaker{{margin:12px 0;padding:10px 14px;border-left:3px solid #64748b;background:#f8fafc}}.time{{color:#64748b;font-variant-numeric:tabular-nums}}pre{{white-space:pre-wrap;font:inherit}}</style></head>\n<body><h1>{html.escape(title)}</h1>\n{body}\n</body></html>\n"


def render_speaker_script(media_path: str | Path, segments: Sequence[Mapping[str, Any]], artifact_format: str) -> str:
    format_name = normalize_artifact_format(artifact_format)
    title = f"{Path(media_path).stem} 화자 스크립트"
    rows = _speaker_rows(segments)
    if format_name == "txt":
        return "\n".join(rows) + ("\n" if rows else "")
    if format_name == "md":
        body = "\n".join(f"- {row}" for row in _speaker_rows_md(segments))
        return f"# {_md_escape(title)}\n\n{body}\n" if body else f"# {_md_escape(title)}\n"
    body = "\n".join(f"<p class=\"speaker\"><span class=\"time\">{html.escape(row.split(' ', 1)[0])}</span>{html.escape(row.split(' ', 1)[1] if ' ' in row else '')}</p>" for row in rows)
    return _html_document(title, body)


def render_full_body(media_path: str | Path, segments: Sequence[Mapping[str, Any]], artifact_format: str) -> str:
    format_name = normalize_artifact_format(artifact_format)
    title = f"{Path(media_path).stem} 본문 전체본"
    lines = _full_body_lines(segments)
    if format_name == "txt":
        return "\n".join(lines) + ("\n" if lines else "")
    if format_name == "md":
        lines_md = _full_body_lines_md(segments)
        return f"# {_md_escape(title)}\n\n" + "\n\n".join(lines_md) + ("\n" if lines_md else "")
    body = "\n".join(f"<p>{html.escape(line)}</p>" for line in lines)
    return _html_document(title, body)


def _summary_plain_text(summary: str) -> str:
    lines = []
    for line in str(summary or "").splitlines():
        stripped = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        stripped = re.sub(r"^\s*[-*+]\s+", "", stripped)
        if stripped.strip() and not re.fullmatch(r"\s*[|: -]+\s*", stripped):
            lines.append(stripped.rstrip())
    return "\n".join(lines).strip() + ("\n" if lines else "")


_UNSAFE_TAG_RE = re.compile(r"</?\s*(script|iframe|object|embed)\b[^>]*>", re.IGNORECASE)
_UNSAFE_URL_RE = re.compile(r"(?i)(javascript|data:text/html)\s*:")


def _strip_unsafe_summary_markup(summary: str) -> str:
    """Strip ``<script``/``<iframe``/``<object``/``<embed`` tags and
    ``javascript:``/``data:text/html`` URLs from provider-generated markdown.

    Defense-in-depth, not sanitization: provider output is markdown by
    contract and is not otherwise escaped.
    """
    cleaned = _UNSAFE_TAG_RE.sub("", str(summary))
    return _UNSAFE_URL_RE.sub("removed:", cleaned)


def render_summary(summary: str, title: str, artifact_format: str) -> str:
    format_name = normalize_artifact_format(artifact_format)
    normalized = _strip_unsafe_summary_markup(summary).strip()
    if format_name == "txt":
        return _summary_plain_text(normalized)
    if format_name == "md":
        return normalized + ("\n" if normalized else "")
    return _html_document(title, f"<pre>{html.escape(normalized)}</pre>")


def read_transcript_source(path: str | Path) -> str:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        # Same default-reject policy as media inputs: a transcript link must
        # never silently redirect provider-bound content outside the path
        # the user supplied. There is no --follow-symlinks on minutes.
        raise UsageError(f"심볼릭 링크 입력은 지원하지 않습니다: {raw}")
    # Open by descriptor, not pathname: a symlink swapped into any path
    # component after the check above fails the open instead of being read.
    try:
        fd = open_verified(raw)
    except OSError as exc:
        raise UsageError(f"전사 파일을 읽을 수 없습니다: {raw}") from exc
    try:
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            if handle.seek(0, os.SEEK_END) > MAX_TRANSCRIPT_BYTES:
                raise UsageError(
                    f"전사 파일이 허용 한도({MAX_TRANSCRIPT_BYTES // (1024 * 1024)} MiB)를 초과합니다: {raw}"
                )
            handle.seek(0)
            content = handle.read().decode("utf-8")
    finally:
        if fd >= 0:
            os.close(fd)
    source = raw.resolve()
    if source.suffix.lower() != ".json":
        return content.strip()
    try:
        payload = json.loads(content)
    except (ValueError, RecursionError, MemoryError) as exc:
        raise UsageError(f"전사 JSON을 해석할 수 없습니다: {source}") from exc
    segments = payload.get("segments") if isinstance(payload, dict) else payload
    if not isinstance(segments, list):
        return content.strip()
    lines: list[str] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        text = text_for_segment(segment)
        if not text:
            continue
        label = str(segment.get("speaker_label") or segment.get("speaker_id") or "").strip()
        prefix = f"[{format_clock(segment['start'])}–{format_clock(segment['end'])}]" if "start" in segment and "end" in segment else ""
        speaker_prefix = f" {label}:" if label else ""
        lines.append(f"{prefix}{speaker_prefix} {text}".strip())
    return "\n".join(lines).strip()
