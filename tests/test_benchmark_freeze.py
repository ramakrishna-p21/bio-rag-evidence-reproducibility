from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from biorag_repro.benchmark.freeze import _path, verify_implementation
from biorag_repro.benchmark.release import BenchmarkRelease
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import serialize_project_json


def ref(name: str) -> ArtifactRef:
    return ArtifactRef(artifact_id=name, logical_ref=f"benchmark/{name}.json", sha256="a" * 64)


def release() -> BenchmarkRelease:
    return BenchmarkRelease(
        benchmark_version="benchmark_v1",
        corpus_version="corpus_v1",
        corpus_content_sha256="b" * 64,
        corpus_release=ref("corpus"),
        implementation_git_sha="c" * 40,
        seed_registry=ref("seeds"),
        relatedness=ref("relatedness"),
        split=ref("split"),
        items=ref("items"),
        decisions=ref("decisions"),
        audit=ref("audit"),
        validation_report=ref("validation"),
        item_count=5,
        development_paper_count=10,
        final_test_paper_count=40,
    )


def test_release_round_trip_and_counts() -> None:
    record = release()
    assert (
        BenchmarkRelease.model_validate_json(serialize_project_json(record, BenchmarkRelease))
        == record
    )
    with pytest.raises(ValidationError):
        BenchmarkRelease.model_validate({**record.model_dump(), "development_paper_count": 9})


def test_release_reference_cannot_leave_root(tmp_path: Path) -> None:
    assert _path(tmp_path, ref("items")) == tmp_path / "benchmark/items.json"
    with pytest.raises(ValueError):
        _path(tmp_path, ArtifactRef(artifact_id="x", logical_ref="../outside", sha256="a" * 64))


def test_freeze_requires_reviewed_head_and_clean_tree(tmp_path: Path) -> None:
    with (
        patch("biorag_repro.benchmark.freeze.subprocess.check_output", side_effect=["d" * 40]),
        pytest.raises(ValueError, match="HEAD"),
    ):
        verify_implementation(tmp_path, "c" * 40)
    with (
        patch(
            "biorag_repro.benchmark.freeze.subprocess.check_output",
            side_effect=["c" * 40, " M file.py"],
        ),
        pytest.raises(ValueError, match="uncommitted"),
    ):
        verify_implementation(tmp_path, "c" * 40)
