"""Artifact naming, staged writes, atomic commit, and stale cleanup.

A successful input produces exactly three artifacts in the selected format:
``{key}_화자스크립트``, ``{key}_본문요약``, ``{key}_본문전체``. Writes are staged
as ``<final>.tmp-<pid>-<kind>`` files (mode 0600, fsync'd) and committed with
``os.replace`` so a failed or interrupted run leaves zero final artifacts.

Depends only on stdlib + ``errors``.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from .errors import UsageError

ARTIFACT_KINDS = ("speaker_script", "body_summary", "full_body")
ARTIFACT_SUFFIX = {"speaker_script": "화자스크립트", "body_summary": "본문요약", "full_body": "본문전체"}
ARTIFACT_EXTENSIONS = ("html", "md", "txt")
DEFAULT_ARTIFACT_FORMAT = "md"


@dataclass(frozen=True)
class ArtifactTarget:
    kind: str  # one of ARTIFACT_KINDS
    path: Path  # final destination


def plan_output_keys(all_media: Sequence[Path]) -> dict[Path, str]:
    """``{media_path: output_key}`` for every input, computed in one pass.

    ``'meeting'`` when the stem is unique across ``all_media``;
    ``'meeting.m4a'`` when ≥2 inputs share the stem (extension disambiguates).
    If the full file name is still ambiguous (same name in different
    directories), a short digest of the resolved path disambiguates.
    """
    media = [Path(item) for item in all_media]
    stem_counts = Counter(item.stem for item in media)
    name_counts = Counter(item.name for item in media)
    keys: dict[Path, str] = {}
    for item in media:
        if stem_counts[item.stem] <= 1:
            keys[item] = item.stem
        elif name_counts[item.name] <= 1:
            keys[item] = item.name
        else:
            keys[item] = disambiguated_key(item)
    return keys


def disambiguated_key(media_path: Path) -> str:
    """``'<name>.<hash8>'`` key for a same-name input from a different path —
    the single source for path-digest disambiguation (``plan_output_keys``
    and the manifest media_path binding both use it)."""
    media = Path(media_path)
    digest = hashlib.sha256(str(media.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{media.name}.{digest}"
    return keys


def output_key(media_path: Path, all_media: Sequence[Path]) -> str:
    """Deterministic artifact stem — see ``plan_output_keys``."""
    keys = plan_output_keys(all_media)
    media = Path(media_path)
    if media in keys:
        return keys[media]
    # media_path not in all_media: compute against the set plus itself.
    return plan_output_keys([*all_media, media])[media]

def _safe_artifact_name(name: str) -> str:
    """Reject names that would corrupt the line-oriented commit journal or
    escape a single-line record: tab and any line-break character."""
    if "\t" in name or len(str(name).splitlines()) != 1:
        raise UsageError(f"산출물 이름에 사용할 수 없는 제어 문자가 포함되어 있습니다: {name!r}")
    return name


def plan_artifacts(output_dir: Path, key: str, artifact_format: str) -> dict[str, ArtifactTarget]:
    """``{kind: ArtifactTarget}`` for the three artifacts. Pure function, no I/O."""
    format_name = str(artifact_format or DEFAULT_ARTIFACT_FORMAT).strip().lower()
    if format_name not in ARTIFACT_EXTENSIONS:
        raise UsageError(f"지원하지 않는 산출물 포맷입니다: {artifact_format}")
    safe_key = _safe_artifact_name(str(key))
    return {
        kind: ArtifactTarget(kind=kind, path=Path(output_dir) / f"{safe_key}_{ARTIFACT_SUFFIX[kind]}.{format_name}")
        for kind in ARTIFACT_KINDS
    }



_STAGING_TEMP_RE = re.compile(r"\.tmp-(\d+)-(speaker_script|body_summary|full_body)(-\d+)?$")
_COMMIT_STAGE_RE = re.compile(r"^\.mediascribe-commit-(\d+)(-\d+)?\.stage$")
_COMMIT_MARKER_RE = re.compile(r"^\.mediascribe-commit-(\d+)(-\d+)?$")

_MARKER_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) TERMINATES the process on Windows — never call it.
        # OpenProcess alone is insufficient: a dead process object survives
        # while any handle stays open (e.g. Popen._handle), so probe the
        # signaled state — WAIT_OBJECT_0 means terminated.
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        try:
            result = kernel32.WaitForSingleObject(handle, 0)
        finally:
            kernel32.CloseHandle(handle)
        return result != 0  # WAIT_OBJECT_0=0 → dead; TIMEOUT/FAILED → alive
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return  # directories can't be opened/fsync'd on Windows
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fchmod_private(fd: int) -> None:
    if os.name != "nt":
        os.fchmod(fd, 0o600)


class ArtifactStager:
    """Stage artifact contents as temp files, then commit them atomically."""

    def __init__(self, targets: Mapping[str, ArtifactTarget]):
        self._targets = dict(targets)
        self._temps: dict[str, Path] = {}
        self._digests: dict[str, str] = {}

    @property
    def digests(self) -> dict[str, str]:
        """``{kind: sha256}`` of staged contents — recorded in the manifest
        at commit so later runs can verify a preserved set is untampered."""
        return dict(self._digests)

    def stage(self, kind: str, content: str) -> Path:
        """Write ``content`` to ``<final>.tmp-<pid>-<kind>[-<n>]`` in the final
        dir, mode 0600, fsync'd. Returns the temp path.

        The temp file is created with ``O_EXCL`` (+``O_NOFOLLOW`` where
        available) so a planted symlink or leftover file is never followed
        or truncated; a collision retries with a ``-<n>`` suffix.
        """
        target = self._targets.get(kind)
        if target is None:
            raise UsageError(f"알 수 없는 아티팩트 종류입니다: {kind}")
        final = target.path
        if not final.parent.is_dir():
            final.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(final.parent, 0o700)
            except OSError:
                pass
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        temp: Path | None = None
        fd = -1
        for attempt in range(10):
            candidate = final.with_name(
                f"{final.name}.tmp-{os.getpid()}-{kind}" + (f"-{attempt}" if attempt else "")
            )
            try:
                fd = os.open(candidate, flags, 0o600)
                temp = candidate
                break
            except FileExistsError:
                continue
        if temp is None or fd < 0:
            raise UsageError(f"임시 파일을 만들 수 없습니다: {final}")
        try:
            _fchmod_private(fd)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = -1
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fd >= 0:
                os.close(fd)
        self._temps[kind] = temp
        self._digests[kind] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return temp

    def _write_commit_marker(self) -> Path:
        """Journal the pending commit so a crash mid-rename is recoverable.

        ``.mediascribe-commit-<pid>`` lists the final basenames about to be
        replaced. If the process dies between renames, ``reconcile_commits``
        finds the dead-pid marker and removes the partial set. The marker is
        deleted after the directory fsync that makes the commit durable.
        """
        directory = next(iter(self._targets.values())).path.parent
        payload = f"targets\t{len(self._targets)}\n" + "".join(
            f"{_safe_artifact_name(target.path.name)}\t{self._digests.get(kind, '')}\n"
            for kind, target in sorted(self._targets.items())
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        for attempt in range(10):
            marker = directory / (f".mediascribe-commit-{os.getpid()}" + (f"-{attempt}" if attempt else ""))
            staging = marker.with_name(f"{marker.name}.stage")
            try:
                fd = os.open(staging, flags, 0o600)
            except FileExistsError:
                continue
            try:
                _fchmod_private(fd)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    fd = -1
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if fd >= 0:
                    os.close(fd)
            # Atomic publish: a marker at its final name is always complete —
            # torn writes only ever leave the .stage temp behind.
            os.replace(staging, marker)
            _fsync_directory(directory)
            return marker
        raise UsageError(f"커밋 마커를 만들 수 없습니다: {directory}")

    def commit(self) -> dict[str, str]:
        """Atomically move every staged temp file to its final path.

        1) Write a ``.mediascribe-commit-<pid>`` journal marker listing the
        final basenames. 2) For each target: ``os.replace(tmp, final)`` —
        atomic, replaces files/symlinks. 3) fsync the directory, then delete
        the marker. 4) Delete stale siblings: same stem, extension in
        ``ARTIFACT_EXTENSIONS`` minus the committed one. Returns
        ``{kind: str(final_path)}``.

        A crash between renames leaves the marker behind; the next run's
        ``reconcile_commits`` removes the partial set so no incomplete
        triplet survives. Never delete before rename.
        """
        missing = [kind for kind in self._targets if kind not in self._temps]
        if missing:
            raise UsageError(f"스테이징되지 않은 아티팩트가 있습니다: {', '.join(missing)}")
        marker = self._write_commit_marker()
        committed: dict[str, str] = {}
        directories: set[Path] = set()
        try:
            for kind, target in self._targets.items():
                temp = self._temps[kind]
                os.replace(temp, target.path)
                committed[kind] = str(target.path)
                directories.add(target.path.parent)
            self._temps.clear()
            for directory in directories:
                _fsync_directory(directory)
        except Exception:
            # Keep the journal marker: reconcile_commits removes the partial
            # set on the next run. Temps already moved are gone; remaining
            # temps are cleaned by abort()/cleanup_staging.
            raise
        # Marker removal must succeed or fail loudly: a silently retained
        # marker would be reconciled later (harmless for a complete triplet,
        # but a pending journal must never be hidden from the caller).
        marker_removed = False
        for _attempt in range(3):
            try:
                marker.unlink()
                marker_removed = True
                break
            except OSError:
                time.sleep(0.01)
        if not marker_removed:
            raise UsageError(f"커밋 마커를 제거할 수 없습니다: {marker}")
        for directory in directories:
            _fsync_directory(directory)  # make marker removal durable too
        # Stale-format cleanup only applies to the managed three-artifact
        # contract: a single-target commit (minutes --output) must never
        # delete unrelated same-stem files the user chose as a destination.
        if set(self._targets) == set(ARTIFACT_KINDS):
            for target in self._targets.values():
                self._remove_stale_siblings(target.path)
        return committed

    def abort(self) -> None:
        """Delete all temp files. Idempotent."""
        for temp in self._temps.values():
            try:
                temp.unlink()
            except OSError:
                pass
        self._temps.clear()

    @staticmethod
    def _remove_stale_siblings(final: Path) -> None:
        for extension in ARTIFACT_EXTENSIONS:
            if extension == final.suffix.lstrip("."):
                continue
            sibling = final.with_suffix(f".{extension}")
            try:
                if os.path.lexists(sibling):
                    sibling.unlink()
            except OSError:
                pass


def cleanup_staging(output_dir: Path) -> int:
    """Delete leftover stager temp files from crashed runs.

    Matches only the stager's own ``<final>.tmp-<pid>-<kind>[-<n>]`` pattern
    and torn ``.mediascribe-commit-<pid>[-<n>].stage`` marker temps, so user
    files that happen to contain ``.tmp-`` are never touched. Temps
    belonging to a live pid are skipped, so this is safe to call without a
    lock even alongside a concurrent run. Returns the number removed.
    """
    removed = 0
    try:
        candidates = list(Path(output_dir).glob("*.tmp-*"))
        candidates += list(Path(output_dir).glob(".mediascribe-commit-*.stage"))
    except OSError:
        return 0
    for candidate in candidates:
        match = _STAGING_TEMP_RE.search(candidate.name)
        if match:
            pid = int(match.group(1))
        else:
            # ``.mediascribe-commit-<pid>[-<n>].stage`` — a torn marker write
            # left behind by a crash; same dead-pid reclaim rule.
            stage_match = _COMMIT_STAGE_RE.match(candidate.name)
            if not stage_match:
                continue
            pid = int(stage_match.group(1))
        if pid != os.getpid() and _pid_alive(pid):
            continue  # belongs to a live concurrent run
        try:
            candidate.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def reconcile_commits(output_dir: Path) -> int:
    """Remove partial artifact sets left by crashed commits.

    ``ArtifactStager.commit`` journals ``name<TAB>sha256`` lines in
    ``.mediascribe-commit-<pid>``. A dead-pid marker is stale: the commit is
    complete only when every listed final exists AND its content hash
    matches the journaled digest — this detects partial replacement over an
    older triplet, which existence checks cannot. Otherwise the commit died
    mid-rename and every listed final is deleted so no mixed/partial set
    survives. Names are basename-validated and confined to ``output_dir``.
    Live-pid markers belong to a concurrent run and are left alone.
    Returns the number of files removed.
    """
    removed = 0
    try:
        markers = list(Path(output_dir).glob(".mediascribe-commit-*"))
    except OSError:
        return 0
    for marker in markers:
        match = _COMMIT_MARKER_RE.match(marker.name)
        if not match:
            continue
        pid = int(match.group(1))
        if pid != os.getpid() and _pid_alive(pid):
            continue
        try:
            lines = marker.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        expected = -1
        entries: list[tuple[Path, str]] = []
        for line in lines:
            name, _, digest = line.partition("\t")
            if name == "targets":
                try:
                    expected = int(digest)
                except ValueError:
                    expected = -1
                continue
            if not name or Path(name).name != name:
                continue  # never follow a tampered marker outside output_dir
            if not _MARKER_DIGEST_RE.fullmatch(digest):
                continue  # malformed/injected record — never unlink it
            entries.append((Path(output_dir) / name, digest))

        def _final_digest(path: Path) -> str | None:
            try:
                if path.is_symlink():
                    return None  # a symlink final is never a real artifact
                return hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                return None

        # Complete only when the journal declares a target count, every
        # listed final exists as a real file, each content hash matches, and
        # the targets are distinct — a marker repeating one valid entry can
        # never pass as a finished multi-file commit.
        complete = (
            expected > 0
            and len(entries) == expected
            and len({candidate for candidate, _digest in entries}) == expected
            and all(digest and _final_digest(candidate) == digest for candidate, digest in entries)
        )

        if not complete:
            for candidate, _digest in entries:
                try:
                    if os.path.lexists(candidate):
                        candidate.unlink()
                        removed += 1
                except OSError:
                    pass
        try:
            marker.unlink()
            removed += 1
        except OSError:
            pass
    return removed
