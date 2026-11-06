"""Local model snapshot inventory and the execution part of the environment record.

Snapshots here are a few small synthetic files. Nothing is downloaded and no real weights
are involved.
"""

import json
import os

import pytest
from pydantic import ValidationError
from test_generation_environment import MODEL, NOW, cpu_probes

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.generation.environment import (
    BackendIdentity,
    EnvironmentRecord,
    ExecutionRecord,
    ModelFile,
    ModelSnapshot,
    capture_environment,
    environment_sha256,
)
from biorag_repro.generation.model_snapshot import SnapshotError, inventory_snapshot

REVISION = "1" * 40


def make_snapshot(
    tmp_path,
    *,
    shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
    revision=REVISION,
):
    directory = tmp_path / "models" / revision
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": [1]}))
    for index, name in enumerate(shards):
        (directory / name).write_bytes(f"weights {index}".encode())
    if len(shards) == 1 and shards[0] == "model.safetensors":
        return directory
    weight_map = {f"layer.{i}": name for i, name in enumerate(shards)}
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return directory


def test_inventory_hashes_every_file_and_names_the_manifest(tmp_path):
    directory = make_snapshot(tmp_path)
    snapshot = inventory_snapshot(directory, REVISION)
    names = [f.path for f in snapshot.files]
    assert names == sorted(names)
    assert "config.json" in names and "model.safetensors.index.json" in names
    by_name = {f.path: f for f in snapshot.files}
    shard = by_name["model-00001-of-00002.safetensors"]
    assert shard.sha256 == sha256_bytes(b"weights 0") and shard.size == len(b"weights 0")
    assert snapshot.directory == str(directory.resolve())
    assert inventory_snapshot(directory, REVISION) == snapshot


def test_manifest_hash_changes_with_any_file(tmp_path):
    directory = make_snapshot(tmp_path)
    before = inventory_snapshot(directory, REVISION).manifest_sha256
    (directory / "model-00002-of-00002.safetensors").write_bytes(b"weights X")
    assert inventory_snapshot(directory, REVISION).manifest_sha256 != before


def test_manifest_hash_is_independent_of_where_the_snapshot_lives(tmp_path):
    first = inventory_snapshot(make_snapshot(tmp_path / "a"), REVISION)
    second = inventory_snapshot(make_snapshot(tmp_path / "b"), REVISION)
    assert first.directory != second.directory
    assert first.manifest_sha256 == second.manifest_sha256


def test_directory_must_be_named_by_the_configured_revision(tmp_path):
    directory = make_snapshot(tmp_path)
    with pytest.raises(SnapshotError, match="revision"):
        inventory_snapshot(directory, "2" * 40)


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(SnapshotError, match="directory"):
        inventory_snapshot(tmp_path / REVISION, REVISION)


def test_a_missing_config_is_refused(tmp_path):
    directory = make_snapshot(tmp_path)
    (directory / "config.json").unlink()
    with pytest.raises(SnapshotError, match="config.json"):
        inventory_snapshot(directory, REVISION)


def test_a_missing_shard_named_by_the_index_is_refused(tmp_path):
    directory = make_snapshot(tmp_path)
    (directory / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(SnapshotError, match="index"):
        inventory_snapshot(directory, REVISION)


def test_a_shard_the_index_does_not_name_is_refused(tmp_path):
    directory = make_snapshot(tmp_path)
    (directory / "model-00009-of-00009.safetensors").write_bytes(b"extra")
    with pytest.raises(SnapshotError, match="index"):
        inventory_snapshot(directory, REVISION)


def test_single_file_weights_need_no_index(tmp_path):
    directory = make_snapshot(tmp_path, shards=("model.safetensors",))
    assert any(f.path == "model.safetensors" for f in inventory_snapshot(directory, REVISION).files)


def test_no_weights_are_refused(tmp_path):
    directory = make_snapshot(tmp_path, shards=("model.safetensors",))
    (directory / "model.safetensors").unlink()
    with pytest.raises(SnapshotError, match="weight"):
        inventory_snapshot(directory, REVISION)


def test_two_unindexed_weight_files_are_refused(tmp_path):
    directory = make_snapshot(tmp_path, shards=("model.safetensors",))
    (directory / "other.safetensors").write_bytes(b"x")
    with pytest.raises(SnapshotError, match="weight"):
        inventory_snapshot(directory, REVISION)


@pytest.mark.parametrize(
    "name", ["pytorch_model.bin", "model.pt", "weights.pth", "x.ckpt", "x.pkl"]
)
def test_pickle_weight_files_are_refused(tmp_path, name):
    directory = make_snapshot(tmp_path)
    (directory / name).write_bytes(b"pickle")
    with pytest.raises(SnapshotError, match="pickle"):
        inventory_snapshot(directory, REVISION)


def test_a_quantization_setting_in_the_model_config_is_refused(tmp_path):
    directory = make_snapshot(tmp_path)
    (directory / "config.json").write_text(json.dumps({"quantization_config": {"bits": 4}}))
    with pytest.raises(SnapshotError, match="quantiz"):
        inventory_snapshot(directory, REVISION)


def test_an_unreadable_model_config_is_refused(tmp_path):
    directory = make_snapshot(tmp_path)
    (directory / "config.json").write_text("not json")
    with pytest.raises(SnapshotError, match="config.json"):
        inventory_snapshot(directory, REVISION)


def test_download_bookkeeping_is_not_part_of_the_snapshot(tmp_path):
    directory = make_snapshot(tmp_path)
    before = inventory_snapshot(directory, REVISION)
    cache = directory / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "stamp.metadata").write_text("2026-11-05T00:00:00")
    assert inventory_snapshot(directory, REVISION) == before


def test_symlinked_files_are_hashed_by_content(tmp_path):
    directory = make_snapshot(tmp_path, shards=("model.safetensors",))
    blob = tmp_path / "blob"
    blob.write_bytes(b"blob weights")
    (directory / "model.safetensors").unlink()
    os.symlink(blob, directory / "model.safetensors")
    by_name = {f.path: f for f in inventory_snapshot(directory, REVISION).files}
    assert by_name["model.safetensors"].sha256 == sha256_bytes(b"blob weights")


def test_a_broken_symlink_is_refused(tmp_path):
    directory = make_snapshot(tmp_path)
    os.symlink(tmp_path / "missing", directory / "dangling.json")
    with pytest.raises(SnapshotError, match="dangling"):
        inventory_snapshot(directory, REVISION)


# ---- the model records ----------------------------------------------------------------------------


def file(path, content=b"x"):
    return ModelFile(path=path, size=len(content), sha256=sha256_bytes(content))


def test_snapshot_model_validates_its_own_manifest_and_order(tmp_path):
    snapshot = inventory_snapshot(make_snapshot(tmp_path), REVISION)
    data = snapshot.model_dump(mode="json")
    ModelSnapshot.model_validate(data)
    for change in (
        {"manifest_sha256": "0" * 64},
        {"files": list(reversed(data["files"]))},
        {"files": data["files"] + data["files"][:1]},
        {"files": [f for f in data["files"] if f["path"] != "config.json"]},
    ):
        with pytest.raises(ValidationError):
            ModelSnapshot.model_validate({**data, **change})


def execution_for(tmp_path, **changes):
    fields = {
        "snapshot": changes.pop("snapshot", None)
        or inventory_snapshot(make_snapshot(tmp_path), REVISION),
        "tokenizer_file_sha256": {"tokenizer.json": "a" * 64},
        "device": "cpu",
        "device_name": None,
        "dtype": "bfloat16",
        "max_position_embeddings": 8192,
    }
    return ExecutionRecord(**{**fields, **changes})


def capture(execution=None):
    return capture_environment(
        backend=BackendIdentity(kind="smoke", name="synthetic", version=None),
        model=MODEL,
        tokenizer=MODEL,
        packages=("pydantic",),
        probes=cpu_probes(),
        execution=execution,
        captured_utc=NOW,
    )


def test_environment_without_execution_is_unchanged_in_meaning(tmp_path):
    record = capture()
    assert record.execution is None
    legacy = record.model_dump(mode="json")
    del legacy["execution"]
    assert EnvironmentRecord.model_validate(legacy) == record  # earlier records still load


def test_execution_is_part_of_the_environment_identity(tmp_path):
    plain = capture()
    with_execution = capture(execution_for(tmp_path))
    assert environment_sha256(plain) != environment_sha256(with_execution)
    other = execution_for(tmp_path / "other", device="cuda:0")
    assert environment_sha256(capture(other)) != environment_sha256(with_execution)


def test_execution_rejects_a_blank_device_or_dtype(tmp_path):
    snapshot = inventory_snapshot(make_snapshot(tmp_path), REVISION)
    for change in ({"device": ""}, {"dtype": ""}, {"max_position_embeddings": 0}):
        with pytest.raises(ValidationError):
            execution_for(tmp_path, snapshot=snapshot, **change)
