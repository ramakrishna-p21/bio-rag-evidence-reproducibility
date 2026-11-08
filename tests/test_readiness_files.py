import json
import os

import pytest

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.readiness.files import FileProblem, loads_strict, read_regular


def test_reads_bytes_and_returns_their_hash(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f.json").write_bytes(b'{"x":1}')
    data = read_regular(tmp_path, "a/f.json")
    assert data == b'{"x":1}'
    assert sha256_bytes(data) == sha256_bytes(b'{"x":1}')


@pytest.mark.parametrize(
    "bad", ["", "/etc/passwd", "../x", "a/../x", "a//b", "a\\b", "./a", "a/\0b", "~/x"]
)
def test_refuses_untrusted_locators(tmp_path, bad):
    with pytest.raises(FileProblem):
        read_regular(tmp_path, bad)


def test_missing_file_and_directory(tmp_path):
    (tmp_path / "d").mkdir()
    with pytest.raises(FileProblem, match="missing"):
        read_regular(tmp_path, "nope")
    with pytest.raises(FileProblem, match="regular"):
        read_regular(tmp_path, "d")


def test_symlinked_file_and_symlinked_parent_are_refused(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "f").write_bytes(b"x")
    os.symlink(tmp_path / "real" / "f", tmp_path / "link")
    os.symlink(tmp_path / "real", tmp_path / "dirlink")
    with pytest.raises(FileProblem, match="link"):
        read_regular(tmp_path, "link")
    with pytest.raises(FileProblem, match="link"):
        read_regular(tmp_path, "dirlink/f")


def test_size_limit(tmp_path):
    (tmp_path / "big").write_bytes(b"x" * 11)
    with pytest.raises(FileProblem, match="larger"):
        read_regular(tmp_path, "big", max_bytes=10)


def test_fifo_is_not_opened(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(FileProblem, match="regular"):
        read_regular(tmp_path, "pipe")


def test_strict_json_refuses_duplicate_keys_and_nonfinite():
    assert loads_strict(b'{"a":1}') == {"a": 1}
    with pytest.raises(FileProblem, match="duplicate"):
        loads_strict(b'{"a":1,"a":2}')
    with pytest.raises(FileProblem, match="nonfinite"):
        loads_strict(b'{"a":NaN}')
    with pytest.raises(FileProblem):
        loads_strict(b"\xff")
    with pytest.raises(FileProblem):
        loads_strict(b'{"a":1}{"b":2}')
    assert json.loads(b"[1]") == loads_strict(b"[1]")
