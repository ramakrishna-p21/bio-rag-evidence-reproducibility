"""Verify a local model snapshot and write its inventory and manifest.

This checks files; it never approves a checkpoint. A passing inventory produces a
candidate `APPROVED_CHECKPOINTS` entry that is only printed. Adding it to
`generation.config` is a separate, manual and reviewed step.
"""

import argparse
import json
import math
import os
import re
import struct
import sys
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.records import Record
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.generation.config import OFFICIAL_REPOSITORY, GitRevision, Sha256
from biorag_repro.generation.environment import ModelFile, manifest_sha256
from biorag_repro.generation.model_snapshot import (
    BOOKKEEPING,
    INDEX,
    PICKLE_SUFFIXES,
    SnapshotError,
    inventory_snapshot,
)
from biorag_repro.retrieval.context import ASSETS as PINNED_TOKENIZER_ASSETS

INVENTORY_FILE = "checkpoint_inventory.json"
MANIFEST_FILE = "checkpoint_manifest.json"
MAX_HEADER_BYTES = 100_000_000  # the safetensors format limit

# What a published Qwen3 repository contains. Anything else is reviewed by hand, not loaded.
ALLOWED_FILES = {
    "config.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    INDEX,
    "model.safetensors",
    "README.md",
    "LICENSE",
    ".gitattributes",
}
SHARD_NAME = re.compile(r"model-\d{5}-of-\d{5}\.safetensors")


class CheckpointError(SnapshotError):
    """The snapshot, or a stored inventory of it, fails verification."""


class CheckpointInventory(Record):
    schema_version: Literal["1"] = "1"
    repository: str = Field(min_length=1)
    revision: GitRevision
    files: list[ModelFile]
    manifest_sha256: Sha256
    weight_shards: list[str]
    tokenizer_files: dict[str, Sha256]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        paths = [f.path for f in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("inventory files must be unique and sorted by path")
        if manifest_sha256(self.files) != self.manifest_sha256:
            raise ValueError("inventory manifest hash does not match its files")
        if self.weight_shards != [p for p in paths if p.endswith(".safetensors")]:
            raise ValueError("weight shards differ from the safetensors files")
        by_path = {f.path: f.sha256 for f in self.files}
        if any(by_path.get(name) != digest for name, digest in self.tokenizer_files.items()):
            raise ValueError("tokenizer hashes differ from the inventoried files")
        return self


class CheckpointManifest(Record):
    """What a maintainer reviews. `approval` records that inventorying approves nothing."""

    schema_version: Literal["1"] = "1"
    repository: str = Field(min_length=1)
    revision: GitRevision
    manifest_sha256: Sha256
    inventory_sha256: Sha256  # hash of the stored inventory file's bytes
    file_count: int = Field(gt=0)
    total_size_bytes: int = Field(ge=0)
    approval: Literal["not_approved"] = "not_approved"


def _scan(directory: Path) -> None:
    """Refuse unsafe or unexpected entries before any file is hashed."""
    seen: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        name = relative.as_posix()
        if path.is_symlink() and path.is_dir():
            raise CheckpointError(f"{name} is a symbolic link to a directory")
        if path.is_dir():
            continue
        if relative.parts[:2] == (BOOKKEEPING, "huggingface"):
            continue  # download state, left out of the inventory by inventory_snapshot
        if path.suffix.lower() in PICKLE_SUFFIXES:
            raise CheckpointError(f"{name} is a pickle-based file and is refused")
        if seen.setdefault(name.lower(), name) != name:
            raise CheckpointError(f"{name} and {seen[name.lower()]} collide when case is ignored")
        if name not in ALLOWED_FILES and not SHARD_NAME.fullmatch(name):
            raise CheckpointError(f"unexpected file {name}")


def _start(path: Path, tensor: str, entry: object) -> int:
    try:
        start = entry["data_offsets"][0]  # type: ignore[index]
    except (TypeError, KeyError, IndexError) as error:
        raise CheckpointError(f"shard {path.name} has a malformed entry {tensor}") from error
    if type(start) is not int:
        raise CheckpointError(f"shard {path.name} has a malformed entry {tensor}")
    return start


def _header(path: Path) -> dict[str, dict[str, object]]:
    """Tensor table of a safetensors file; the data region must match the file size."""
    size = path.stat().st_size
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise CheckpointError(f"shard {path.name} is too short")
        (length,) = struct.unpack("<Q", prefix)
        if length > MAX_HEADER_BYTES or 8 + length > size:
            raise CheckpointError(f"shard {path.name} has an invalid header length")
        try:
            table = json.loads(source.read(length))
        except ValueError as error:
            raise CheckpointError(f"shard {path.name} has an unreadable header") from error
    if not isinstance(table, dict):
        raise CheckpointError(f"shard {path.name} has an invalid header")
    table.pop("__metadata__", None)
    if not table:
        raise CheckpointError(f"shard {path.name} holds no tensors")
    end = 0
    for tensor, entry in sorted(table.items(), key=lambda item: _start(path, *item)):
        try:
            dtype, (start, stop), shape = entry["dtype"], entry["data_offsets"], entry["shape"]
            width = math.prod(shape) * 2  # BF16 is two bytes
        except (TypeError, KeyError, ValueError) as error:
            raise CheckpointError(f"shard {path.name} has a malformed entry {tensor}") from error
        if dtype != "BF16":
            raise CheckpointError(f"tensor {tensor} has dtype {dtype}, not BF16")
        if type(start) is not int or type(stop) is not int or start != end or stop - start != width:
            raise CheckpointError(f"tensor {tensor} does not match its shape or leaves a gap")
        end = stop
    if 8 + length + end != size:
        raise CheckpointError(f"shard {path.name} is truncated or has trailing bytes")
    return table


def _weight_map(directory: Path) -> dict[str, str] | None:
    path = directory / INDEX
    if not path.is_file():
        return None
    try:
        weight_map = json.loads(path.read_text(encoding="utf-8"))["weight_map"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CheckpointError(f"{INDEX} is unreadable") from error
    if not isinstance(weight_map, dict) or not all(
        isinstance(shard, str) for shard in weight_map.values()
    ):
        raise CheckpointError(f"{INDEX} has a malformed weight map")
    return weight_map


def _check_weights(directory: Path, shards: list[str]) -> None:
    owners: dict[str, str] = {}
    for name in shards:
        for tensor in _header(directory / name):
            if owners.setdefault(tensor, name) != name:
                raise CheckpointError(f"tensor {tensor} is stored in two shards")
    weight_map = _weight_map(directory)
    if weight_map is not None and weight_map != owners:
        raise CheckpointError(f"the tensors in the shards differ from the {INDEX} weight map")


def build_inventory(
    directory: Path,
    *,
    repository: str,
    revision: str,
    tokenizer_assets: dict[str, str] | None = None,
) -> CheckpointInventory:
    if repository != OFFICIAL_REPOSITORY:
        raise CheckpointError(f"repository {repository!r} is not {OFFICIAL_REPOSITORY}")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise CheckpointError("revision must be a full 40-character commit hash")
    pins = PINNED_TOKENIZER_ASSETS if tokenizer_assets is None else tokenizer_assets
    if not directory.is_dir():
        raise CheckpointError(f"model directory {directory} does not exist")
    try:
        _scan(directory)
        _weight_map(directory)  # a malformed index is reported here, not deeper down
        snapshot = inventory_snapshot(directory, revision)
    except CheckpointError:
        raise
    except SnapshotError as error:
        raise CheckpointError(str(error)) from error

    by_path = {f.path: f.sha256 for f in snapshot.files}
    for name, digest in pins.items():
        if by_path.get(name) != digest:
            raise CheckpointError(f"tokenizer file {name} is missing or differs from the pin")
    settings = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    stated = [settings[k] for k in ("torch_dtype", "dtype") if k in settings]
    if not stated or set(stated) != {"bfloat16"}:
        raise CheckpointError(f"config.json must state a bfloat16 dtype, not {stated or 'none'}")
    shards = [f.path for f in snapshot.files if f.path.endswith(".safetensors")]
    _check_weights(directory, shards)
    try:
        unchanged = inventory_snapshot(directory, revision) == snapshot
    except SnapshotError:
        unchanged = False
    if not unchanged:
        raise CheckpointError("model files changed while they were being verified")
    return CheckpointInventory(
        repository=repository,
        revision=revision,
        files=snapshot.files,
        manifest_sha256=snapshot.manifest_sha256,
        weight_shards=shards,
        tokenizer_files={name: by_path[name] for name in sorted(pins)},
    )


def write_artifacts(inventory: CheckpointInventory, out_dir: Path) -> tuple[Path, Path]:
    inventory_bytes = serialize_project_json(inventory, CheckpointInventory)
    manifest = CheckpointManifest(
        repository=inventory.repository,
        revision=inventory.revision,
        manifest_sha256=inventory.manifest_sha256,
        inventory_sha256=sha256_bytes(inventory_bytes),
        file_count=len(inventory.files),
        total_size_bytes=sum(f.size for f in inventory.files),
    )
    paths = (out_dir / INVENTORY_FILE, out_dir / MANIFEST_FILE)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(p.exists() for p in paths):
        raise CheckpointError(f"checkpoint artifacts already exist in {out_dir}")
    contents = (inventory_bytes, serialize_project_json(manifest, CheckpointManifest))
    linked: list[Path] = []
    try:
        for path, data in zip(paths, contents, strict=True):
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_bytes(data)
            try:
                os.link(temporary, path)  # fails rather than replacing an existing file
                linked.append(path)
            finally:
                temporary.unlink()
    except BaseException:
        for path in linked:
            path.unlink()
        raise
    return paths


def read_artifacts(
    inventory_path: Path, manifest_path: Path
) -> tuple[CheckpointInventory, CheckpointManifest]:
    inventory_bytes = inventory_path.read_bytes()
    manifest = CheckpointManifest.model_validate_json(manifest_path.read_bytes())
    if sha256_bytes(inventory_bytes) != manifest.inventory_sha256:
        raise CheckpointError("the inventory file differs from the one the manifest cites")
    inventory = CheckpointInventory.model_validate_json(inventory_bytes)
    if (
        (manifest.repository, manifest.revision, manifest.manifest_sha256)
        != (inventory.repository, inventory.revision, inventory.manifest_sha256)
        or manifest.file_count != len(inventory.files)
        or manifest.total_size_bytes != sum(f.size for f in inventory.files)
    ):
        raise CheckpointError("the manifest does not describe the inventory")
    return inventory, manifest


def verify_against(
    directory: Path, inventory_path: Path, manifest_path: Path
) -> CheckpointInventory:
    """Re-inventory a directory and require it to equal the stored, validated inventory."""
    stored, _ = read_artifacts(inventory_path, manifest_path)
    if stored.tokenizer_files != PINNED_TOKENIZER_ASSETS:
        raise CheckpointError("the stored tokenizer hashes are not the project's tokenizer pin")
    current = build_inventory(
        directory,
        repository=stored.repository,
        revision=stored.revision,
        tokenizer_assets=stored.tokenizer_files,
    )
    if current != stored:
        raise CheckpointError("the directory differs from the stored inventory")
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("inventory", help="inventory a snapshot and write the artifacts")
    make.add_argument("--directory", type=Path, required=True)
    make.add_argument("--revision", required=True)
    make.add_argument("--repository", default=OFFICIAL_REPOSITORY)
    make.add_argument("--output-dir", type=Path, required=True)
    check = commands.add_parser("verify", help="compare a directory with stored artifacts")
    check.add_argument("--directory", type=Path, required=True)
    check.add_argument("--inventory", type=Path, required=True)
    check.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            inventory = build_inventory(
                args.directory, repository=args.repository, revision=args.revision
            )
            write_artifacts(inventory, args.output_dir)
            print(f"inventoried {len(inventory.files)} files, written to {args.output_dir}")
            print("Candidate APPROVED_CHECKPOINTS entry (NOT APPROVED; nothing was modified):")
            print(
                f'    ("{inventory.repository}", "{inventory.revision}"): '
                f'"{inventory.manifest_sha256}",'
            )
        else:
            inventory = verify_against(args.directory, args.inventory, args.manifest)
            print(f"directory matches the stored inventory (manifest {inventory.manifest_sha256})")
    except (ValueError, OSError) as error:  # pydantic's ValidationError is a ValueError
        print(f"checkpoint verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
