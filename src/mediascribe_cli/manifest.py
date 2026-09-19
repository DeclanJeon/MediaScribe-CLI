"""Run manifest, per-file states, transcript/summary checkpoints, lockfile.

State machine per input file::

    pending → transcribing → transcribed → summarizing → committed
                  ↘              (checkpoint on disk)        ↘ failed (retryable)

All run state lives under ``<output_dir>/.mediascribe/`` (mode 0700, files
0600) — hidden metadata that never counts toward the three-artifact contract.
Depends only on stdlib + ``errors``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .errors import UsageError

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime dependency
    from .config import RuntimeConfig

MANIFEST_VERSION = 1
FILE_STATES = ("pending", "transcribing", "transcribed", "summarizing", "committed", "failed")

_STATE_DIR_NAME = ".mediascribe"
_MANIFEST_NAME = "manifest.json"
_LOCK_NAME = "lock"
_CHECKPOINT_DIR_NAME = "checkpoints"

_STATE_TEMP_RE = re.compile(r"\.tmp-(\d+)(-\d+)?$")

_CONFIG_FIELDS = (
    "model",
    "language",
    "task",
    "device",
    "compute_type",
    "beam_size",
    "vad_filter",
    "diarize",
    "diarization_model",
    "num_speakers",
    "min_speakers",
    "max_speakers",
    "speaker_labels",
    "artifact_format",
)

_TRANSCRIPT_CHECKPOINT_FIELDS = (
    "media_path",
    "detected_language",
    "language_probability",
    "segments",
    "speakers",
    "metadata",
)

_MAX_CHECKPOINT_SEGMENTS = 1_000_000


def _valid_checkpoint_segment(segment: Any) -> bool:
    """Shape check for a checkpointed segment: numeric finite bounds with
    ``start <= end``. Anything else is treated as a missing checkpoint so a
    malformed payload re-runs STT instead of crashing the summary stage."""
    if not isinstance(segment, dict):
        return False
    start, end = segment.get("start"), segment.get("end")
    if isinstance(start, bool) or isinstance(end, bool):
        return False
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return False
    try:
        if not (math.isfinite(start) and math.isfinite(end)):
            return False
    except (OverflowError, ValueError):
        return False  # huge JSON ints overflow float() — malformed, not fatal
    return 0 <= start <= end


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _exclusive_create(path: Path) -> tuple[int, Path]:
    """Open ``path`` with ``O_WRONLY|O_CREAT|O_EXCL`` (+``O_NOFOLLOW`` when
    available), mode 0600. Retries with ``-<n>`` suffixes on collision and
    never unlinks or follows an existing path. Returns ``(fd, temp_path)``."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for attempt in range(10):
        temp = path if attempt == 0 else path.with_name(f"{path.name}-{attempt}")
        try:
            return os.open(temp, flags, 0o600), temp
        except FileExistsError:
            continue
    raise UsageError(f"임시 파일을 만들 수 없습니다: {path}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as mode-0600 JSON via tmp + ``os.replace``, fsync'd."""
    _ensure_private_dir(path.parent)
    fd, temp = _exclusive_create(path.with_name(f"{path.name}.tmp-{os.getpid()}"))
    try:
        _fchmod_private(fd)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    os.replace(temp, path)
    _fsync_directory(path.parent)


def _ensure_private_dir(path: Path) -> None:
    """Create ``path`` as mode-0700, refusing a pre-existing symlink or
    non-directory so state can never be redirected outside the output dir.
    Only used for managed state dirs (``.mediascribe``, ``checkpoints``), so
    the 0700 contract is re-applied on every call."""
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_dir():
            raise UsageError(f"상태 디렉터리가 안전하지 않습니다: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_stamp() -> str:
    """Filename-safe UTC stamp — ``:`` is illegal in Windows filenames."""
    return _utc_now().replace(":", "-")


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


_MAX_STATE_JSON_BYTES = 64 * 1024 * 1024


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        # Bound the read itself, not just the stat: a file that grows past
        # the cap between stat() and read() must still be rejected.
        with open(path, "rb") as handle:
            raw = handle.read(_MAX_STATE_JSON_BYTES + 1)
        if len(raw) > _MAX_STATE_JSON_BYTES:
            return None  # oversized state file → treat as corrupt
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, RecursionError, MemoryError):
        # RecursionError/MemoryError: deeply nested JSON must degrade to
        # "corrupt" (quarantine/missing), never an uncaught traceback.
        return None
    return payload if isinstance(payload, dict) else None


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


def _safe_key(key: str) -> str:
    """Strip any directory components so keys cannot escape the state dir."""
    name = Path(str(key)).name
    return name if name not in ("", ".", "..") else "_"


def file_fingerprint(path: Path) -> str:
    """``sha256:`` of ``f'{resolved_path}|{size}|{mtime_ns}'``. Cheap, no content read."""
    source = Path(path)
    stat = source.stat()
    return _sha256_text(f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}")


def config_fingerprint(config: "RuntimeConfig") -> str:
    """``sha256:`` of JSON of the fields that change transcript output:
    model, language, task, device, compute_type, beam_size, vad_filter,
    diarize, diarization_model, num/min/max_speakers, glossary_path+hash,
    speaker_map_path+hash, speaker_labels, artifact_format."""
    fields: dict[str, Any] = {name: getattr(config, name, None) for name in _CONFIG_FIELDS}
    for path_field in ("glossary_path", "speaker_map_path"):
        value = str(getattr(config, path_field, "") or "")
        entry: dict[str, Any] = {"path": value, "hash": ""}
        if value:
            try:
                entry["hash"] = _file_sha256(Path(value).expanduser())
            except OSError:
                entry["hash"] = "unreadable"
        fields[path_field] = entry
    canonical = json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str)
    return _sha256_text(canonical)


class SummaryCheckpoint:
    """Per-file summary checkpoint: ``get(digest)`` / ``put(digest, text)``.

    Backed by ``checkpoints/<key>.summary.json``. Each ``put`` rewrites the
    file atomically; a corrupt or missing file behaves as empty.
    """

    def __init__(self, manifest: "Manifest", key: str):
        self._manifest = manifest
        self._key = key

    @property
    def _path(self) -> Path:
        return self._manifest._checkpoint_path(self._key, "summary")

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self._path)
        if payload is None:
            return {"version": MANIFEST_VERSION, "key": self._key, "entries": {}}
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            payload["entries"] = {}
        return payload

    def get(self, digest: str) -> str | None:
        entries = self._load().get("entries") or {}
        value = entries.get(digest)
        return value if isinstance(value, str) else None

    def put(self, digest: str, text: str) -> None:
        if not self._manifest._writable():
            return
        payload = self._load()
        payload["entries"][digest] = text
        _write_json_atomic(self._path, payload)


class Manifest:
    """Version-1 run manifest bound to one output directory."""

    def __init__(self, output_dir: Path, data: dict[str, Any], *, read_only: bool, persistent: bool):
        self._output_dir = output_dir
        self._data = data
        self._read_only = read_only
        self._persistent = persistent
        self._lock_held = False

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(cls, output_dir: Path, *, create: bool = True, read_only: bool = False) -> "Manifest":
        """Load ``<output_dir>/.mediascribe/manifest.json``.

        Corrupt JSON or a mismatched ``version`` → renamed to
        ``manifest.json.corrupt-<ts>`` (nothing is emitted; the caller warns)
        and a fresh manifest starts. Acquires
        ``<output_dir>/.mediascribe/lock`` (``O_CREAT|O_EXCL``); a stale lock
        (dead pid) is reclaimed, a live lock raises ``UsageError``.
        ``read_only=True`` (``--dry-run``): no lock, no writes, no
        ``.mediascribe`` creation. A pre-existing ``.mediascribe`` symlink or
        non-directory is rejected with ``UsageError`` in every mode.
        """
        resolved = Path(output_dir).expanduser().resolve()
        state_dir = resolved / _STATE_DIR_NAME
        manifest_path = state_dir / _MANIFEST_NAME
        if os.path.lexists(state_dir) and (state_dir.is_symlink() or not state_dir.is_dir()):
            raise UsageError(f"상태 디렉터리가 안전하지 않습니다: {state_dir}")

        persistent = not read_only and (create or state_dir.is_dir())
        manifest = cls(resolved, cls._fresh_data(resolved), read_only=read_only, persistent=persistent)

        if persistent:
            manifest._ensure_state_dir()
            manifest._acquire_lock()

        payload = _read_json(manifest_path)
        if payload is not None and payload.get("version") != MANIFEST_VERSION:
            payload = None  # unknown schema → quarantine like corrupt JSON
        if payload is None:
            if manifest_path.exists() and not read_only and persistent:
                corrupt = manifest_path.with_name(f"{_MANIFEST_NAME}.corrupt-{_utc_stamp()}")
                try:
                    manifest_path.rename(corrupt)
                except OSError:
                    pass
        else:
            files = payload.get("files")
            if not isinstance(files, dict):
                payload["files"] = {}
            manifest._data = payload

        if persistent:
            manifest._reclaim_temp_litter()
            manifest._persist()
        return manifest

    def _reclaim_temp_litter(self) -> None:
        """Delete ``*.tmp-<pid>[-<n>]`` leftovers from crashed atomic writes
        under ``.mediascribe`` (manifest and checkpoint temps). Dead-pid
        temps only — a live concurrent run's in-flight write is untouched."""
        try:
            candidates = list(self._state_dir().glob("*.tmp-*")) + list((self._state_dir() / _CHECKPOINT_DIR_NAME).glob("*.tmp-*"))
        except OSError:
            return
        for candidate in candidates:
            match = _STATE_TEMP_RE.search(candidate.name)
            if not match:
                continue
            pid = int(match.group(1))
            if pid != os.getpid() and _pid_alive(pid):
                continue
            try:
                candidate.unlink()
            except OSError:
                pass

    def close(self) -> None:
        """Release the lockfile. Called in ``finally``. Idempotent."""
        if not self._lock_held:
            return
        self._lock_held = False
        try:
            self._lock_path().unlink()
        except OSError:
            pass

    def entry(self, key: str) -> dict | None:
        entry = self._files().get(key)
        return entry if isinstance(entry, dict) else None

    def entries(self) -> dict[str, dict]:
        """``{key: entry}`` for every recorded file entry (dicts only)."""
        return {key: entry for key, entry in self._files().items() if isinstance(entry, dict)}

    def begin(self, key: str, fingerprint: str, state: str, media_path: str = "") -> None:
        """Start (or restart) processing ``key`` at ``state``.

        ``outputs`` is preserved across restarts: it always records the last
        *committed* artifact set, so abort/reconcile can tell a prior run's
        valid triplet from this run's partials. ``media_path`` binds the
        entry to the input identity — a same-stem file from a different path
        must not silently reuse or overwrite this key's artifacts.
        """
        self._validate_state(state)
        prior = self._files().get(key)
        outputs = prior.get("outputs") if isinstance(prior, dict) and isinstance(prior.get("outputs"), dict) else {}
        output_digests = (
            prior.get("output_digests")
            if isinstance(prior, dict) and isinstance(prior.get("output_digests"), dict)
            else {}
        )
        self._files()[key] = {
            "state": state,
            "fingerprint": fingerprint,
            "config_fingerprint": None,
            "summary_fingerprint": None,
            "media_path": media_path,
            "outputs": outputs,
            "output_digests": output_digests,
            "error": None,
            "updated_at": _utc_now(),
        }
        self._persist()

    def set_state(self, key: str, state: str, **fields: Any) -> None:
        """Update an entry, writing the manifest via tmp+``os.replace``, fsync'd."""
        self._validate_state(state)
        entry = self._files().get(key)
        if not isinstance(entry, dict):
            entry = {"fingerprint": None, "config_fingerprint": None, "outputs": {}, "error": None}
            self._files()[key] = entry
        entry["state"] = state
        for name, value in fields.items():
            entry[name] = value
        entry["updated_at"] = _utc_now()
        self._persist()

    def should_skip(
        self,
        key: str,
        fingerprint: str,
        expected_kinds,
        expected_outputs=None,
        summary_fingerprint: str | None = None,
    ) -> bool:
        """True iff state is ``committed`` AND the fingerprint matches AND the
        recorded outputs cover exactly ``expected_kinds``, each a real file
        that is not a symlink.

        When ``expected_outputs`` (``{kind: path}``) is given, every recorded
        path must additionally resolve to exactly that path inside this
        manifest's output directory — a tampered manifest cannot point the
        skip check at foreign files. ``summary_fingerprint`` binds the skip
        to summary-shaping inputs (title/provider/models/chunk size); an
        entry committed before fingerprints existed never matches, so it is
        conservatively reprocessed once.
        """
        entry = self.entry(key)
        if entry is None:
            return False
        if entry.get("state") != "committed" or entry.get("fingerprint") != fingerprint:
            return False
        if summary_fingerprint is not None and entry.get("summary_fingerprint") != summary_fingerprint:
            return False
        outputs = entry.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != set(expected_kinds):
            return False
        if not self.recorded_outputs_intact(entry, expected_kinds):
            return False
        if expected_outputs is not None:
            for kind, recorded in outputs.items():
                expected = expected_outputs.get(kind)
                if expected is None:
                    return False
                try:
                    if Path(str(recorded)).resolve() != Path(expected).resolve():
                        return False
                except OSError:
                    return False
        return True

    def recorded_outputs_intact(self, entry: dict | None, expected_kinds) -> bool:
        """True iff the entry's recorded outputs still form a complete,
        untampered artifact set inside this manifest's output directory:
        exactly ``expected_kinds``, each a real non-symlink file resolving
        under ``output_dir``, and — when ``output_digests`` was recorded at
        commit time — each file's sha256 still matching. A partially
        replaced or edited set fails, so a mixed triplet is never mistaken
        for a prior run's valid commit."""
        outputs = entry.get("outputs") if isinstance(entry, dict) else None
        if not isinstance(outputs, dict) or set(outputs) != set(expected_kinds):
            return False
        digests = entry.get("output_digests") if isinstance(entry, dict) else None
        if not isinstance(digests, dict):
            digests = {}
        for kind, recorded in outputs.items():
            path = Path(str(recorded))
            if not path.is_file() or path.is_symlink():
                return False
            try:
                resolved = path.resolve()
            except OSError:
                return False
            if not resolved.is_relative_to(self._output_dir):
                return False
            expected_digest = str(digests.get(kind) or "")
            if expected_digest:
                try:
                    actual = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError:
                    return False
                if actual != expected_digest:
                    return False
        return True

    # -- checkpoints --------------------------------------------------------

    def transcript_checkpoint(
        self,
        key: str,
        media_path: Path,
        fingerprint: str,
        config_fingerprint: str,
    ) -> dict | None:
        """Return the ``transcribe_file`` result dict for ``key``, or ``None``.

        The checkpoint is bound to the *current* media: it is returned only
        when the stored ``media_path`` resolves to the same file, the stored
        ``fingerprint`` equals ``fingerprint`` (computed by the caller before
        STT), and the stored ``config_fingerprint`` equals the current
        transcript config fingerprint. Malformed payloads — including
        segments that are not dicts with ``start``/``end`` — count as
        missing, so a corrupt checkpoint can never wedge a run.
        """
        payload = _read_json(self._checkpoint_path(key, "transcript"))
        if payload is None:
            return None
        stored_media = payload.get("media_path")
        if not stored_media or not isinstance(stored_media, str):
            return None
        try:
            if Path(stored_media).resolve() != Path(media_path).resolve():
                return None
        except OSError:
            return None
        if payload.get("fingerprint") != fingerprint:
            return None
        if payload.get("config_fingerprint") != config_fingerprint:
            return None
        if payload.get("version") != MANIFEST_VERSION:
            return None
        segments = payload.get("segments")
        if (
            not isinstance(segments, list)
            or len(segments) > _MAX_CHECKPOINT_SEGMENTS
            or not all(_valid_checkpoint_segment(segment) for segment in segments)
        ):
            return None
        return payload

    def save_transcript_checkpoint(
        self,
        key: str,
        result: dict,
        fingerprint: str,
        config_fingerprint: str,
    ) -> None:
        """Atomic write of ``{version, fingerprint, config_fingerprint,
        media_path, detected_language, language_probability, segments,
        speakers, metadata}``.

        ``fingerprint`` must be the media fingerprint computed *before*
        ``transcribe_file`` — never recomputed after STT.
        """
        if not self._writable():
            return
        payload = {
            "version": MANIFEST_VERSION,
            "fingerprint": fingerprint,
            "config_fingerprint": config_fingerprint,
        }
        for field in _TRANSCRIPT_CHECKPOINT_FIELDS:
            payload[field] = result.get(field)
        _write_json_atomic(self._checkpoint_path(key, "transcript"), payload)

    def summary_store(self, key: str) -> SummaryCheckpoint:
        """Return the ``SummaryCheckpoint`` for ``key``."""
        return SummaryCheckpoint(self, key)

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _fresh_data(output_dir: Path) -> dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "run_id": f"{_utc_now()}-{os.getpid()}",
            "output_dir": str(output_dir),
            "files": {},
        }

    @staticmethod
    def _validate_state(state: str) -> None:
        if state not in FILE_STATES:
            raise ValueError(f"unknown manifest file state: {state!r}")

    def _files(self) -> dict[str, Any]:
        files = self._data.get("files")
        if not isinstance(files, dict):
            files = {}
            self._data["files"] = files
        return files

    def _state_dir(self) -> Path:
        return self._output_dir / _STATE_DIR_NAME

    def _lock_path(self) -> Path:
        return self._state_dir() / _LOCK_NAME

    def _checkpoint_path(self, key: str, kind: str) -> Path:
        """Path under ``checkpoints/`` — the directory itself is validated on
        every access so a planted symlink is rejected on reads too, not only
        when a write happens to call ``_ensure_private_dir``."""
        checkpoints = self._state_dir() / _CHECKPOINT_DIR_NAME
        if os.path.lexists(checkpoints) and (checkpoints.is_symlink() or not checkpoints.is_dir()):
            raise UsageError(f"체크포인트 디렉터리가 안전하지 않습니다: {checkpoints}")
        return checkpoints / f"{_safe_key(key)}.{kind}.json"

    def _writable(self) -> bool:
        return self._persistent and not self._read_only

    def _ensure_state_dir(self) -> None:
        _ensure_private_dir(self._state_dir())
        self._persistent = True


    def _persist(self) -> None:
        if not self._writable():
            return
        self._ensure_state_dir()
        _write_json_atomic(self._state_dir() / _MANIFEST_NAME, self._data)

    def _read_lock_pid(self) -> int | None:
        """``None`` = no lock file; ``0`` = present but unparsable."""
        try:
            text = self._lock_path().read_text(encoding="utf-8").strip()
        except OSError:
            return None
        try:
            return int(text)
        except ValueError:
            return 0

    def _acquire_lock(self) -> None:
        for _attempt in range(3):
            try:
                fd = os.open(self._lock_path(), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                if self._reclaim_stale_lock():
                    continue
                raise UsageError(f"다른 MediaScribe 실행이 출력 디렉터리를 사용 중입니다: {self._lock_path()}")
            try:
                _fchmod_private(fd)
                os.write(fd, str(os.getpid()).encode("ascii"))
                os.fsync(fd)
            finally:
                os.close(fd)
            self._lock_held = True
            return
        raise UsageError(f"출력 디렉터리 잠금을 획득할 수 없습니다: {self._lock_path()}")

    def _reclaim_stale_lock(self) -> bool:
        """True when the existing lock was removed and may be recreated."""
        pid = self._read_lock_pid()
        if pid is None:
            return True  # vanished between O_EXCL and read — retry
        if pid == 0:
            # Unparsable: possibly a lock being created right now. Give the
            # creator a moment to write its pid, then re-check.
            time.sleep(0.05)
            pid = self._read_lock_pid()
            if pid is None:
                return True
            if pid == 0:
                pid = -1  # still garbage → treat as dead
        if _pid_alive(pid):
            return False
        try:
            self._lock_path().unlink()
        except OSError:
            return False
        return True
