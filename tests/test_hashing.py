import struct

import pytest

from biorag_repro.core.hashing import (
    HashMismatchError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    verify_file_sha256,
)


def test_rfc_8785_sample_canonical_bytes():
    # RFC 8785, Sections 3.2.2–3.2.4: https://www.rfc-editor.org/rfc/rfc8785.html
    value = {
        "numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 1e-27],
        "string": '€$\x0f\nA\'B"\\\\"/',
        "literals": [None, True, False],
    }
    expected = (
        b'{"literals":[null,true,false],"numbers":'
        b"[333333333.3333333,1e+30,4.5,0.002,1e-27],"
        b'"string":"\xe2\x82\xac$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
    )
    assert canonical_json_bytes(value) == expected


def test_rfc_8785_utf16_key_order():
    # RFC 8785, Section 3.2.3: supplementary-plane keys sort as UTF-16.
    value = {"\ufb33": 7, "\u20ac": 5, "\U0001f600": 6, "\r": 1, "1": 2, "\u0080": 3, "\u00f6": 4}
    expected = '{"\\r":1,"1":2,"\u0080":3,"ö":4,"€":5,"😀":6,"דּ":7}'.encode()
    assert canonical_json_bytes(value) == expected


def test_nested_objects_are_sorted_without_reordering_arrays():
    value = {"z": [{"b": 1, "a": 2}, {"d": 3, "c": 4}], "a": 0}
    assert canonical_json_bytes(value) == b'{"a":0,"z":[{"a":2,"b":1},{"c":4,"d":3}]}'


@pytest.mark.parametrize(
    ("ieee754_hex", "expected"),
    [
        ("8000000000000000", b"0"),
        ("0000000000000001", b"5e-324"),
        ("3eb0c6f7a0b5ed8d", b"0.000001"),
        ("444b1ae4d6e2ef50", b"1e+21"),
    ],
)
def test_rfc_8785_appendix_b_numbers(ieee754_hex, expected):
    # RFC 8785, Appendix B: IEEE 754 bit patterns and expected JSON tokens.
    number = struct.unpack(">d", bytes.fromhex(ieee754_hex))[0]
    assert canonical_json_bytes(number) == expected


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), "\ud800", {1: "one"}, (1, 2)]
)
def test_invalid_canonical_values_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        canonical_json_bytes(value)


def test_sha256_bytes_uses_known_digest():
    assert sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_file_hashes_exact_bytes(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == sha256_bytes(b"abc")
    path.write_bytes(b"abc\n")
    assert sha256_file(path) == sha256_bytes(b"abc\n")


@pytest.mark.parametrize("content", [b"abc", b""])
def test_verify_file_sha256_accepts_exact_bytes(tmp_path, content):
    path = tmp_path / "artifact.bin"
    path.write_bytes(content)
    assert verify_file_sha256(path, sha256_bytes(content)) is None


def test_verify_file_sha256_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_file_sha256(tmp_path / "missing.bin", sha256_bytes(b"abc"))


@pytest.mark.parametrize("digest", ["abc", "A" * 64, "aA" * 32, "g" * 64])
def test_verify_file_sha256_rejects_malformed_expected_digest(tmp_path, digest):
    with pytest.raises(ValueError, match="SHA-256"):
        verify_file_sha256(tmp_path / "missing.bin", digest)


def test_verify_file_sha256_reports_mismatch_and_changed_bytes(tmp_path):
    path = tmp_path / "artifact.bin"
    expected = sha256_bytes(b"abc")
    path.write_bytes(b"other")
    with pytest.raises(HashMismatchError):
        verify_file_sha256(path, expected)
    assert path.read_bytes() == b"other"

    path.write_bytes(b"abc")
    verify_file_sha256(path, expected)
    path.write_bytes(b"abc\n")
    with pytest.raises(HashMismatchError):
        verify_file_sha256(path, expected)
