"""Symlink-safe file opening.

``os.open(path, O_NOFOLLOW)`` only refuses a symlink in the *final* path
component — an ancestor directory swapped for a symlink after path
collection is still followed. ``open_no_symlinks`` normalizes the path
*lexically* (``abspath``, which unlike ``resolve()`` never follows links)
and then re-opens every component with ``O_NOFOLLOW``, so a symlink
anywhere in the supplied path fails the open instead of redirecting the
read.

Residual gap (documented, accepted): a *rename* swap — replacing a real
directory or file with a different real one between normalization and
open — cannot be closed without ``openat2(RESOLVE_BENEATH)``, which
Python's stdlib does not expose. An attacker who can rename inside the
input tree can already replace file contents directly, so this adds no
new capability.
"""

from __future__ import annotations

import os
from pathlib import Path


def open_no_symlinks(path: Path, flags: int = os.O_RDONLY) -> int:
    """Open ``path`` for reading without following any symlink component.

    Returns a file descriptor for the regular file. Raises ``OSError``
    (``ELOOP``/``ENOENT``/``ENOTDIR``) when any component is a symlink or
    the path cannot be opened.
    """
    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    if not parts:
        raise FileNotFoundError(str(absolute))
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    fd = os.open(parts[0], os.O_RDONLY | nofollow | directory_flag)
    try:
        for index, part in enumerate(parts[1:], start=1):
            last = index == len(parts) - 1
            component_flags = flags if last else (os.O_RDONLY | directory_flag)
            next_fd = os.open(part, component_flags | nofollow, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise
