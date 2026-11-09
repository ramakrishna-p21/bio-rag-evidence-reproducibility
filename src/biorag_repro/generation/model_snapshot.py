"""Inventory of a local model directory.

The inventory is a verified list of the files that were present, not proof of which
weights executed: it hashes the directory at one moment, and a later load reads the files
again. The configured revision is checked against the directory name (the layout the
tokenizer assets already use) and, for official use, against an approved manifest hash.
"""

import json
from pathlib import Path

from biorag_repro.core.hashing import sha256_file
from biorag_repro.generation.environment import ModelFile, ModelSnapshot, manifest_sha256

PICKLE_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
INDEX = "model.safetensors.index.json"
BOOKKEEPING = ".cache"  # Hugging Face download state, which changes between downloads


class SnapshotError(ValueError):
    """The model directory is missing, inconsistent or not loadable under the project rules."""


def _files(directory: Path) -> list[ModelFile]:
    found = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if BOOKKEEPING in relative.parts or path.is_dir():
            continue
        if not path.is_file():  # broken symlinks and special files
            raise SnapshotError(f"{relative.as_posix()} is not a readable regular file")
        if path.suffix.lower() in PICKLE_SUFFIXES:
            raise SnapshotError(f"{relative.as_posix()} is a pickle-based file and is refused")
        found.append(
            ModelFile(path=relative.as_posix(), size=path.stat().st_size, sha256=sha256_file(path))
        )
    return found


def _check_weights(directory: Path, names: list[str]) -> None:
    shards = sorted(n for n in names if n.endswith(".safetensors"))
    if INDEX in names:
        try:
            mapped = set(
                json.loads((directory / INDEX).read_text(encoding="utf-8"))["weight_map"].values()
            )
        except (OSError, ValueError, KeyError, AttributeError) as error:
            raise SnapshotError(f"{INDEX} is unreadable") from error
        if mapped != set(shards):
            raise SnapshotError("the weight files differ from those named by the index")
    elif shards != ["model.safetensors"]:
        raise SnapshotError(
            "without an index there must be exactly one model.safetensors weight file"
        )


def inventory_snapshot(directory: Path, revision: str) -> ModelSnapshot:
    if not directory.is_dir():
        raise SnapshotError(f"model directory {directory} does not exist")
    if directory.name != revision:
        raise SnapshotError(
            f"model directory is named {directory.name}, not the configured revision {revision}"
        )
    files = _files(directory)
    names = [f.path for f in files]
    if "config.json" not in names:
        raise SnapshotError("model directory has no config.json")
    try:
        settings = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SnapshotError("config.json is unreadable") from error
    if not isinstance(settings, dict) or "quantization_config" in settings:
        raise SnapshotError(
            "config.json is not a plain model configuration or requests quantization"
        )
    _check_weights(directory, names)
    return ModelSnapshot(
        directory=str(directory.resolve()), files=files, manifest_sha256=manifest_sha256(files)
    )
