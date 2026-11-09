"""Checkpoint verification on small synthetic safetensors snapshots.

No weights are downloaded and nothing is executed. The tokenizer pin, revision and tensors
are synthetic; the tool only inventories and checks files, and never approves anything.
"""

import json
import os
import struct

import pytest
from pydantic import ValidationError

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.generation import checkpoint
from biorag_repro.generation import config as generator_config
from biorag_repro.generation.checkpoint import (
    CheckpointError,
    CheckpointInventory,
    CheckpointManifest,
    build_inventory,
    read_artifacts,
    verify_against,
    write_artifacts,
)
from biorag_repro.generation.model_snapshot import inventory_snapshot

REVISION = "3" * 40
REPOSITORY = "Qwen/Qwen3-8B"
TOKENIZER = {
    "merges.txt": b"merges",
    "tokenizer.json": b"tokenizer",
    "tokenizer_config.json": b"tokenizer config",
    "vocab.json": b"vocab",
}
PINS = {name: sha256_bytes(data) for name, data in TOKENIZER.items()}


def safetensors(tensors: dict[str, str]) -> bytes:
    """A real safetensors container: 8-byte header length, JSON header, then tensor bytes."""
    header, body = {}, b""
    for name, dtype in tensors.items():
        data = bytes(2)  # one element of a 2-byte dtype
        header[name] = {"dtype": dtype, "shape": [1], "data_offsets": [len(body), len(body) + 2]}
        body += data
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + body


def make(tmp_path, *, dtype="BF16", name=REVISION):
    directory = tmp_path / name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"], "torch_dtype": "bfloat16"})
    )
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": [1]}))
    for filename, data in TOKENIZER.items():
        (directory / filename).write_bytes(data)
    shards = {
        "model-00001-of-00002.safetensors": {"layer.0": dtype},
        "model-00002-of-00002.safetensors": {"layer.1": dtype},
    }
    weight_map = {}
    for shard, tensors in shards.items():
        (directory / shard).write_bytes(safetensors(tensors))
        weight_map.update({tensor: shard for tensor in tensors})
    index = {"metadata": {"total_size": 4}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    return directory


def build(directory, **changes):
    arguments = {"repository": REPOSITORY, "revision": REVISION, "tokenizer_assets": PINS}
    return build_inventory(directory, **{**arguments, **changes})


def test_a_valid_snapshot_is_inventoried_with_the_existing_manifest_identity(tmp_path):
    directory = make(tmp_path)
    inventory = build(directory)
    assert inventory.manifest_sha256 == inventory_snapshot(directory, REVISION).manifest_sha256
    assert inventory.weight_shards == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert (inventory.repository, inventory.revision) == (REPOSITORY, REVISION)
    assert str(directory) not in inventory.model_dump_json()  # no machine-specific path


def test_a_missing_weight_shard_is_refused(tmp_path):
    directory = make(tmp_path)
    (directory / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(CheckpointError, match="index"):
        build(directory)


def test_altered_weight_bytes_are_detected_against_the_stored_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "PINNED_TOKENIZER_ASSETS", PINS)
    directory = make(tmp_path)
    inventory_path, manifest_path = write_artifacts(build(directory), tmp_path / "out")
    shard = directory / "model-00001-of-00002.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1] + b"\x01")  # same size, same header
    with pytest.raises(CheckpointError, match="differs"):
        verify_against(directory, inventory_path, manifest_path)


def test_a_wrong_revision_or_repository_is_refused(tmp_path):
    directory = make(tmp_path)
    with pytest.raises(CheckpointError, match="revision"):
        build(directory, revision="4" * 40)
    with pytest.raises(CheckpointError, match="repository"):
        build(directory, repository="Qwen/Qwen3-8B-AWQ")
    with pytest.raises(CheckpointError, match="revision"):
        build(directory, revision="main")


@pytest.mark.parametrize(
    "index",
    [
        "not json",
        json.dumps({"metadata": {}}),
        json.dumps({"weight_map": ["a"]}),
        json.dumps({"weight_map": {"layer.0": "model-00001-of-00002.safetensors"}}),
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00001-of-00002.safetensors",  # lives in shard 2
                }
            }
        ),
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                    "layer.9": "model-00002-of-00002.safetensors",  # not in any header
                }
            }
        ),
    ],
)
def test_a_malformed_or_inconsistent_weight_index_is_refused(tmp_path, index):
    directory = make(tmp_path)
    (directory / "model.safetensors.index.json").write_text(index)
    with pytest.raises(CheckpointError):
        build(directory)


def test_a_truncated_shard_is_refused(tmp_path):
    directory = make(tmp_path)
    shard = directory / "model-00001-of-00002.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(CheckpointError, match="shard"):
        build(directory)


@pytest.mark.parametrize("name", ["notes.txt", "extra.json", "sub/readme.md"])
def test_unexpected_files_are_refused(tmp_path, name):
    directory = make(tmp_path)
    (directory / name).parent.mkdir(exist_ok=True)
    (directory / name).write_text("x")
    with pytest.raises(CheckpointError, match="unexpected"):
        build(directory)


def test_documentation_files_of_a_published_repository_are_allowed(tmp_path):
    directory = make(tmp_path)
    for name in ("README.md", "LICENSE", ".gitattributes"):
        (directory / name).write_text("x")
    assert {"README.md", "LICENSE", ".gitattributes"} <= {f.path for f in build(directory).files}


def test_a_duplicated_weight_file_is_refused(tmp_path):
    directory = make(tmp_path)
    first = directory / "model-00001-of-00002.safetensors"
    (directory / "model-00003-of-00002.safetensors").write_bytes(first.read_bytes())
    with pytest.raises(CheckpointError):
        build(directory)


def test_paths_that_collide_when_case_is_ignored_are_refused(tmp_path):
    directory = make(tmp_path)
    (directory / "README.md").write_text("x")
    (directory / "readme.md").write_text("y")
    with pytest.raises(CheckpointError, match="collide"):
        build(directory)


@pytest.mark.parametrize("change", ["altered", "missing"])
def test_an_inconsistent_tokenizer_identity_is_refused(tmp_path, change):
    directory = make(tmp_path)
    if change == "altered":
        (directory / "tokenizer.json").write_bytes(b"other tokenizer")
    else:
        (directory / "vocab.json").unlink()
    with pytest.raises(CheckpointError, match="tokenizer"):
        build(directory)


def test_the_project_tokenizer_pin_is_the_default(tmp_path):
    directory = make(tmp_path)
    with pytest.raises(CheckpointError, match="tokenizer"):
        build_inventory(directory, repository=REPOSITORY, revision=REVISION)


@pytest.mark.parametrize(
    "config",
    [
        {"quantization_config": {"bits": 4}},
        {"torch_dtype": "float16"},
        {"dtype": "float32"},
    ],
)
def test_configurations_incompatible_with_the_official_condition_are_refused(tmp_path, config):
    directory = make(tmp_path)
    (directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(CheckpointError):
        build(directory)


@pytest.mark.parametrize("dtype", ["F16", "U8", "I32", "F8_E4M3"])
def test_non_bfloat16_weights_are_refused(tmp_path, dtype):
    with pytest.raises(CheckpointError, match="dtype"):
        build(make(tmp_path, dtype=dtype))


def test_pickle_weights_and_symlinked_directories_are_refused(tmp_path):
    directory = make(tmp_path)
    (directory / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(CheckpointError, match="pickle"):
        build(directory)
    (directory / "pytorch_model.bin").unlink()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "hidden.json").write_text("{}")
    os.symlink(elsewhere, directory / "linked")
    with pytest.raises(CheckpointError, match="symbolic link"):
        build(directory)


def test_file_symlinks_to_blobs_are_hashed_by_content(tmp_path):
    directory = make(tmp_path)
    blob = tmp_path / "blob"
    blob.write_bytes(b"vocab")
    (directory / "vocab.json").unlink()
    os.symlink(blob, directory / "vocab.json")
    assert build(directory).tokenizer_files["vocab.json"] == sha256_bytes(b"vocab")


def test_repeated_inventories_are_byte_identical(tmp_path):
    directory = make(tmp_path)
    first = write_artifacts(build(directory), tmp_path / "one")
    second = write_artifacts(build(directory), tmp_path / "two")
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes()
    other = write_artifacts(build(make(tmp_path / "copy")), tmp_path / "three")
    assert other[1].read_bytes() == first[1].read_bytes()  # independent of storage location


def test_written_artifacts_round_trip_and_are_never_overwritten(tmp_path):
    inventory = build(make(tmp_path))
    inventory_path, manifest_path = write_artifacts(inventory, tmp_path / "out")
    read_inventory, manifest = read_artifacts(inventory_path, manifest_path)
    assert read_inventory == inventory
    assert manifest.inventory_sha256 == sha256_bytes(inventory_path.read_bytes())
    assert manifest.manifest_sha256 == inventory.manifest_sha256
    assert manifest.approval == "not_approved"
    with pytest.raises(CheckpointError, match="already exist"):
        write_artifacts(inventory, tmp_path / "out")


def edit(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize(
    "tamper",
    [
        lambda d: d["files"][0].update(sha256="0" * 64),
        lambda d: d["files"].pop(),
        lambda d: d.update(revision="4" * 40),
        lambda d: d.update(manifest_sha256="0" * 64),
    ],
)
def test_a_tampered_inventory_is_rejected(tmp_path, tamper):
    paths = write_artifacts(build(make(tmp_path)), tmp_path / "out")
    edit(paths[0], tamper)
    with pytest.raises((CheckpointError, ValidationError)):
        read_artifacts(*paths)


def test_a_tampered_manifest_or_a_mismatched_pair_is_rejected(tmp_path):
    paths = write_artifacts(build(make(tmp_path)), tmp_path / "out")
    edit(paths[1], lambda d: d.update(manifest_sha256="0" * 64))
    with pytest.raises(CheckpointError, match="manifest"):
        read_artifacts(*paths)
    other = write_artifacts(build(make(tmp_path / "b", dtype="BF16")), tmp_path / "other")
    edit(other[0], lambda d: d.update(weight_shards=d["weight_shards"][:1]))
    with pytest.raises((CheckpointError, ValidationError)):
        read_artifacts(*other)


def test_the_inventory_models_validate_their_own_identity():
    with pytest.raises(ValidationError):
        CheckpointManifest.model_validate({})
    assert "approval" in CheckpointManifest.model_fields
    assert "manifest_sha256" in CheckpointInventory.model_fields


def run_cli(capsys, *args):
    code = checkpoint.main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def cli_args(directory, out):
    return [
        "inventory",
        "--directory",
        str(directory),
        "--revision",
        REVISION,
        "--output-dir",
        str(out),
    ]


def test_the_cli_writes_artifacts_and_shows_but_never_applies_a_candidate_entry(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(checkpoint, "PINNED_TOKENIZER_ASSETS", PINS)
    directory = make(tmp_path)
    before = dict(generator_config.APPROVED_CHECKPOINTS)
    code, out, _ = run_cli(capsys, *cli_args(directory, tmp_path / "out"))
    assert code == 0
    manifest = inventory_snapshot(directory, REVISION).manifest_sha256
    assert f'("{REPOSITORY}", "{REVISION}"): "{manifest}"' in out
    assert "NOT APPROVED" in out
    assert generator_config.APPROVED_CHECKPOINTS == before == {}
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "checkpoint_inventory.json",
        "checkpoint_manifest.json",
    ]


def test_an_unapproved_manifest_does_not_make_a_checkpoint_official(tmp_path, monkeypatch):
    monkeypatch.setattr(generator_config, "APPROVED_CHECKPOINTS", {})
    inventory = build(make(tmp_path))
    key = (inventory.repository, inventory.revision)
    assert generator_config.APPROVED_CHECKPOINTS.get(key) != inventory.manifest_sha256


def test_the_cli_refuses_a_bad_snapshot_with_a_nonzero_exit(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(checkpoint, "PINNED_TOKENIZER_ASSETS", PINS)
    directory = make(tmp_path)
    (directory / "model-00002-of-00002.safetensors").unlink()
    code, out, err = run_cli(capsys, *cli_args(directory, tmp_path / "out"))
    assert code == 1 and "index" in err and "candidate" not in out.lower()
    assert not (tmp_path / "out").exists()


def test_the_cli_verify_command_detects_a_changed_directory(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(checkpoint, "PINNED_TOKENIZER_ASSETS", PINS)
    directory = make(tmp_path)
    assert run_cli(capsys, *cli_args(directory, tmp_path / "out"))[0] == 0
    verify = [
        "verify",
        "--directory",
        str(directory),
        "--inventory",
        str(tmp_path / "out" / "checkpoint_inventory.json"),
        "--manifest",
        str(tmp_path / "out" / "checkpoint_manifest.json"),
    ]
    assert run_cli(capsys, *verify)[0] == 0
    (directory / "notes.txt").write_text("added later")
    code, _, err = run_cli(capsys, *verify)
    assert code == 1 and "unexpected" in err


def test_verify_rejects_stored_tokenizer_pins_that_are_not_the_project_pins(tmp_path, monkeypatch):
    directory = make(tmp_path)
    paths = write_artifacts(build(directory, tokenizer_assets={}), tmp_path / "weak")
    monkeypatch.setattr(checkpoint, "PINNED_TOKENIZER_ASSETS", PINS)
    with pytest.raises(CheckpointError, match="tokenizer"):
        verify_against(directory, *paths)


@pytest.mark.parametrize(
    "tensors",
    [
        {},  # no tensors
        {"layer.0": {"dtype": "BF16", "shape": [1000, 1000], "data_offsets": [0, 2]}},
        {"layer.0": {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]}},  # gap at the start
    ],
)
def test_shard_tensor_layout_must_match_shapes_and_be_contiguous(tmp_path, tensors):
    directory = make(tmp_path)
    shard = directory / "model-00001-of-00002.safetensors"
    raw = json.dumps(tensors).encode()
    data = bytes(max((t["data_offsets"][1] for t in tensors.values()), default=0))
    shard.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    index = {
        "weight_map": {
            **{t: shard.name for t in tensors},
            "layer.1": "model-00002-of-00002.safetensors",
        }
    }
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(CheckpointError):
        build(directory)


def test_a_file_changed_after_hashing_is_not_inventoried(tmp_path, monkeypatch):
    directory = make(tmp_path)
    real = checkpoint._check_weights

    def change_then_check(path, shards):
        (path / "LICENSE").write_text("added during verification")
        real(path, shards)

    monkeypatch.setattr(checkpoint, "_check_weights", change_then_check)
    with pytest.raises(CheckpointError, match="changed"):
        build(directory)


def test_a_failed_write_leaves_no_artifacts(tmp_path, monkeypatch):
    inventory = build(make(tmp_path))
    out = tmp_path / "out"
    calls = []
    real = os.link

    def failing_link(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("disk full")
        real(source, target)

    monkeypatch.setattr(os, "link", failing_link)
    with pytest.raises(OSError, match="disk full"):
        write_artifacts(inventory, out)
    assert list(out.iterdir()) == []
    monkeypatch.setattr(os, "link", real)
    write_artifacts(inventory, out)


def test_a_malformed_index_value_is_a_clean_failure_not_a_traceback(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(checkpoint, "PINNED_TOKENIZER_ASSETS", PINS)
    directory = make(tmp_path)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": ["a"]}})
    )
    code, _, err = run_cli(capsys, *cli_args(directory, tmp_path / "out"))
    assert code == 1 and "failed" in err


def test_a_config_that_does_not_state_bfloat16_is_refused(tmp_path):
    directory = make(tmp_path)
    (directory / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    with pytest.raises(CheckpointError, match="dtype"):
        build(directory)


@pytest.mark.parametrize("name", [".cache/payload.dat", ".cache/other/x.json", ".cache"])
def test_only_the_hugging_face_download_state_is_exempt(tmp_path, name):
    directory = make(tmp_path)
    target = directory / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x")
    with pytest.raises(CheckpointError, match="unexpected"):
        build(directory)
    target.unlink()
    state = directory / ".cache" / "huggingface" / "stamp.metadata"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("x")
    build(directory)


def test_inventory_model_rejects_inconsistent_content(tmp_path):
    data = build(make(tmp_path)).model_dump(mode="json")
    CheckpointInventory.model_validate(data)
    other = dict(data["files"][0], sha256="0" * 64)
    for change in (
        {"files": [other, *data["files"][1:]]},
        {"weight_shards": data["weight_shards"][:1]},
        {"tokenizer_files": {**data["tokenizer_files"], "vocab.json": "0" * 64}},
        {"files": list(reversed(data["files"]))},
    ):
        with pytest.raises(ValidationError):
            CheckpointInventory.model_validate({**data, **change})
