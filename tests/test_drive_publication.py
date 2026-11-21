"""Synthetic publication checks for a mounted Drive folder."""

import pytest

from biorag_repro.core.hashing import sha256_file
from biorag_repro.finaltest.drive_publication import (
    DrivePublicationError,
    publish_drive_batch,
    verify_drive_batch,
)


def test_copy_resume_readback_and_duplicate_prevention(tmp_path):
    source = tmp_path / "local" / ("run_" + "a" * 32)
    source.mkdir(parents=True)
    (source / "batch.json").write_bytes(b"batch")
    (source / "results").mkdir()
    (source / "results" / "one.json").write_bytes(b"one")
    drive = tmp_path / "drive"
    partial = drive / source.name
    partial.mkdir(parents=True)
    (partial / "batch.json").write_bytes(b"batch")
    destination = publish_drive_batch(source, drive)
    assert verify_drive_batch(destination) == {
        "batch.json": sha256_file(source / "batch.json"),
        "results/one.json": sha256_file(source / "results" / "one.json"),
    }
    assert publish_drive_batch(source, drive) == destination


def test_missing_and_altered_records_are_refused(tmp_path):
    source = tmp_path / ("run_" + "b" * 32)
    source.mkdir()
    (source / "batch.json").write_bytes(b"batch")
    (source / "result.json").write_bytes(b"result")
    destination = publish_drive_batch(source, tmp_path / "drive")
    (destination / "result.json").write_bytes(b"altered")
    with pytest.raises(DrivePublicationError):
        verify_drive_batch(destination)
    with pytest.raises(DrivePublicationError):
        publish_drive_batch(source, tmp_path / "drive")
    (destination / "result.json").unlink()
    with pytest.raises(DrivePublicationError):
        verify_drive_batch(destination)
