"""Read untrusted local files once, without following links.

Every path component is opened with O_NOFOLLOW from a directory descriptor, so a link anywhere
below the root is refused. The bytes returned are the bytes that were read, so a caller
hashes and parses the same data rather than checking a path and opening it again. This guards
against mistakes and accidental substitution on a local disk; it is not protection against a
process that can write to the same directory while a check runs.
"""

import json
import os
import stat
from pathlib import Path
from typing import Any

MAX_BYTES = 256 * 1024 * 1024


class FileProblem(ValueError):
    """A file or locator is unsafe or not what it must be."""


class FileMissing(FileProblem):
    """The file or one of its directories does not exist."""


def _parts(relative: str) -> list[str]:
    parts = relative.split("/") if isinstance(relative, str) else []
    if (
        not parts
        or "\\" in relative
        or "\0" in relative
        or relative.startswith("~")
        or any(p in ("", ".", "..") for p in parts)
    ):
        raise FileProblem(f"unsafe locator {relative!r}")
    return parts


def read_regular(root: Path, relative: str, *, max_bytes: int = MAX_BYTES) -> bytes:
    """Bytes of the regular file `relative` below `root`; links are refused."""
    parts = _parts(relative)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as error:
        raise FileProblem(f"root cannot be opened: {error}") from error
    try:
        for part in parts[:-1]:
            try:
                inner = os.open(part, flags | os.O_DIRECTORY, dir_fd=fd)
            except FileNotFoundError as error:
                raise FileMissing(f"missing directory {part} in {relative}") from error
            except OSError as error:
                raise FileProblem(f"{part} in {relative} is a link or not a directory") from error
            os.close(fd)
            fd = inner
        name = parts[-1]
        try:
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError as error:
            raise FileMissing(f"missing file {relative}") from error
        if stat.S_ISLNK(info.st_mode):
            raise FileProblem(f"{relative} is a link")
        if not stat.S_ISREG(info.st_mode):
            raise FileProblem(f"{relative} is not a regular file")
        try:
            descriptor = os.open(name, flags | os.O_NONBLOCK, dir_fd=fd)
        except OSError as error:
            raise FileProblem(f"{relative} cannot be opened: {error}") from error
    finally:
        os.close(fd)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise FileProblem(f"{relative} is not a regular file")
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise FileProblem(f"{relative} is larger than {max_bytes} bytes")
    return data


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite JSON value {value}")


def loads_strict(data: bytes) -> Any:
    """Parse one JSON document; duplicate keys, NaN/Infinity, bad UTF-8 and trailing data fail."""
    try:
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=_no_duplicates, parse_constant=_nonfinite
        )
    except ValueError as error:
        raise FileProblem(str(error)) from error
