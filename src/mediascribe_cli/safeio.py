"""Symlink-safe file opening.

Two layers, both stdlib-only:

1. ``open_no_symlinks`` resolves the path (canonicalizing ancestor
   symlinks — ``/var`` → ``/private/var`` on macOS is legitimate), then
   re-opens every component of the *resolved* path with ``O_NOFOLLOW``.
   A symlink swapped into the canonical path after resolution fails the
   open instead of redirecting the read.

2. ``open_verified`` adds an ``lstat`` → ``fstat`` device+inode identity
   check around it: the file opened is the same file that was checked,
   closing the check/use gap on the final component for symlink *and*
   rename swaps.

Residual gap (documented, accepted): an ancestor directory swapped
*before* ``resolve()`` runs is followed like any pathname lookup — the
opened file is still what the path resolves to, and the manifest
``media_path`` binding treats a changed resolution as a different input.
Closing that fully needs ``openat2(RESOLVE_BENEATH)`` or fd-pinning at
collection time, neither of which the stdlib exposes.

Platforms without ``dir_fd`` (Windows) fall back to a single
``O_NOFOLLOW`` open of the resolved path — final-component protection
only.
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd


def open_no_symlinks(path: Path, flags: int = os.O_RDONLY) -> int:
    """Open ``path`` for reading without following post-resolution symlinks.

    Returns a file descriptor for the resolved regular file. Raises
    ``OSError`` (``ELOOP``/``ENOENT``/``ENOTDIR``) when a component of the
    resolved path is a symlink or the path cannot be opened.
    """
    resolved = Path(path).resolve()
    if not _SUPPORTS_DIR_FD:
        # Windows: no dir_fd — single open of the canonical path.
        return os.open(resolved, flags | _NOFOLLOW)
    parts = resolved.parts
    if not parts:
        raise FileNotFoundError(str(resolved))
    fd = os.open(parts[0], os.O_RDONLY | _NOFOLLOW | _DIRECTORY)
    try:
        for index, part in enumerate(parts[1:], start=1):
            last = index == len(parts) - 1
            component_flags = flags if last else (os.O_RDONLY | _DIRECTORY)
            next_fd = os.open(part, component_flags | _NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_verified(path: Path, flags: int = os.O_RDONLY) -> int:
    """Open ``path`` proving the opened file is the checked file.

    ``lstat`` rejects a final-component symlink, then after the
    ``O_NOFOLLOW`` component walk the opened file's device+inode must
    equal the checked entry's — a swap to a symlink *or* a different real
    file between check and open fails with ``ELOOP``.
    """
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise OSError(errno.ELOOP, os.strerror(errno.ELOOP), str(path))
    fd = open_no_symlinks(path, flags)
    try:
        fst = os.fstat(fd)
    except BaseException:
        os.close(fd)
        raise
    if (fst.st_dev, fst.st_ino) != (st.st_dev, st.st_ino):
        os.close(fd)
        raise OSError(errno.ELOOP, "path changed between check and open", str(path))
    return fd
