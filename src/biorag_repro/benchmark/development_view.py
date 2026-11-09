"""Development-only view of a frozen benchmark release.

The release JSONL mixes development and protected final-test records. This module
reads it programmatically, keeps only records of papers the validated split assigns
to development, and publishes them as a derived artifact. Records of other papers are
validated for integrity and then discarded; neither their content nor their identities
are returned, logged or placed in an exception. The view is not a new benchmark
version: it is a read-only subset of the existing release.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from biorag_repro.benchmark.models import BenchmarkItem, Record
from biorag_repro.benchmark.split import PaperSplit
from biorag_repro.benchmark.storage import decode_jsonl, encode_jsonl
from biorag_repro.core.hashing import sha256_bytes, verify_file_sha256
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import (
    ArtifactRef,
    BuildInput,
    BuildManifest,
    DerivedOutput,
)
from biorag_repro.core.storage import (
    publish_bytes,
    read_manifest,
    read_project_json,
    serialize_manifest,
    serialize_project_json,
)

PROCEDURE = "development_paper_filter_v1"
ITEMS_FILE = "items.jsonl"
MANIFEST_FILE = "build_manifest.json"
RECORD_FILE = "development_view.json"


class DerivationError(ValueError):
    """The derivation was refused. Messages never carry record content."""


class DevelopmentView(Record):
    schema_version: Literal["1"] = "1"
    access_classification: Literal["development_only"] = "development_only"
    procedure: str = PROCEDURE
    benchmark_version: str
    corpus_version: str
    source_release: ArtifactRef
    source_items: ArtifactRef
    split: ArtifactRef
    output_items: ArtifactRef
    build_manifest: ArtifactRef
    source_item_count: int
    development_item_count: int
    development_paper_ids: list[str]
    contributing_paper_ids: list[str]


class PublishedView(Record):
    directory: str
    record_sha256: str
    items_sha256: str


def derive_development_items(
    source: bytes,
    split: PaperSplit,
    *,
    benchmark_version: str,
    corpus_version: str,
    expected_source_items: int,
    expected_development_items: int,
) -> tuple[bytes, list[BenchmarkItem]]:
    """Return the canonical bytes and records of development items, in source order."""
    development = set(split.development_ids)
    final_test = set(split.final_test_ids)
    lines = source.split(b"\n")
    if not source or lines.pop() != b"":
        raise DerivationError("source must be non-empty newline-terminated JSONL")
    seen: set[str] = set()
    selected: list[BenchmarkItem] = []
    for number, line in enumerate(lines, start=1):
        try:
            item = BenchmarkItem.model_validate(json.loads(line), strict=True)
        except (ValueError, ValidationError):
            raise DerivationError(f"record {number} is not a valid benchmark item") from None
        if encode_jsonl([item]) != line + b"\n":
            raise DerivationError(f"record {number} is not in canonical form")
        if item.benchmark_version != benchmark_version or item.corpus_version != corpus_version:
            raise DerivationError(f"record {number} has a different benchmark or corpus version")
        if item.candidate_status != "accepted":
            raise DerivationError(f"record {number} is not an accepted item")
        if item.paper_id in development:
            expected_split = "development"
        elif item.paper_id in final_test:
            expected_split = "final_test"
        else:
            raise DerivationError(f"record {number} belongs to a paper outside the split")
        if item.split != expected_split:
            raise DerivationError(f"record {number} split label conflicts with the paper split")
        if item.item_id in seen:
            raise DerivationError(f"record {number} repeats an earlier item identity")
        seen.add(item.item_id)
        if expected_split == "development":
            selected.append(item)
    if len(lines) != expected_source_items:
        raise DerivationError("source item count differs from the release authority")
    if len(selected) != expected_development_items:
        raise DerivationError("development item count differs from the release authority")
    return encode_jsonl(selected), selected


def check_development_items(
    items: list[BenchmarkItem], development_ids: list[str], benchmark_version: str
) -> None:
    """Independent check applied to an output before publication and on every read."""
    ids = [item.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise DerivationError("development view repeats an item identity")
    for item in items:
        if (
            item.split != "development"
            or item.paper_id not in development_ids
            or item.candidate_status != "accepted"
            or item.benchmark_version != benchmark_version
        ):
            raise DerivationError("development view contains a non-development record")


def publish_development_view(
    *,
    data_root: Path,
    items_ref: ArtifactRef,
    split_ref: ArtifactRef,
    release_ref: ArtifactRef,
    benchmark_version: str,
    corpus_version: str,
    expected_source_items: int,
    expected_development_items: int,
) -> PublishedView:
    """Verify sources, derive, check, and publish immutable development-only files."""
    items_path = data_root / items_ref.logical_ref
    split_path = data_root / split_ref.logical_ref
    verify_file_sha256(items_path, items_ref.sha256)
    split = read_project_json(split_path, split_ref.sha256, PaperSplit)
    content, selected = derive_development_items(
        items_path.read_bytes(),
        split,
        benchmark_version=benchmark_version,
        corpus_version=corpus_version,
        expected_source_items=expected_source_items,
        expected_development_items=expected_development_items,
    )
    check_development_items(selected, split.development_ids, benchmark_version)
    if decode_jsonl(content, BenchmarkItem) != selected:
        raise DerivationError("derived bytes do not reproduce the selected records")

    directory_ref = f"benchmark/derived/{benchmark_version}_development"
    items_sha = sha256_bytes(content)
    lineage = [
        ArtifactInput("benchmark_items", items_ref.artifact_id, items_ref.sha256),
        ArtifactInput("paper_split", split_ref.artifact_id, split_ref.sha256),
    ]
    output = DerivedOutput(
        artifact=ArtifactRef(
            artifact_id=derived_artifact_id(
                artifact_kind="benchmark_development_items",
                inputs=lineage,
                transformation_id=f"{PROCEDURE}:{benchmark_version}",
                transformation_settings_sha256=None,
                output_sha256=items_sha,
                output_format="jsonl",
                serialization_version="1",
            ),
            logical_ref=f"{directory_ref}/{ITEMS_FILE}",
            sha256=items_sha,
        ),
        artifact_kind="benchmark_development_items",
        output_format="jsonl",
        serialization_version="1",
        identity_version="1",
    )
    manifest = BuildManifest(
        schema_version="1",
        inputs=[
            BuildInput(role="benchmark_items", artifact=items_ref),
            BuildInput(role="paper_split", artifact=split_ref),
        ],
        transformation_id=f"{PROCEDURE}:{benchmark_version}",
        outputs=[output],
        status="complete",
    )
    manifest_bytes = serialize_manifest(manifest)
    manifest_sha = sha256_bytes(manifest_bytes)
    view = DevelopmentView(
        benchmark_version=benchmark_version,
        corpus_version=corpus_version,
        source_release=release_ref,
        source_items=items_ref,
        split=split_ref,
        output_items=output.artifact,
        build_manifest=ArtifactRef(
            artifact_id=f"{benchmark_version}_development_build_manifest",
            logical_ref=f"{directory_ref}/{MANIFEST_FILE}",
            sha256=manifest_sha,
        ),
        source_item_count=expected_source_items,
        development_item_count=len(selected),
        development_paper_ids=list(split.development_ids),
        contributing_paper_ids=sorted({item.paper_id for item in selected}),
    )
    record_bytes = serialize_project_json(view, DevelopmentView)
    record_sha = sha256_bytes(record_bytes)

    directory = data_root / directory_ref
    directory.mkdir(parents=True, exist_ok=True)
    publish_bytes(directory / ITEMS_FILE, content, items_sha)
    publish_bytes(directory / MANIFEST_FILE, manifest_bytes, manifest_sha)
    publish_bytes(directory / RECORD_FILE, record_bytes, record_sha)
    read_development_view(directory, record_sha)
    return PublishedView(directory=directory_ref, record_sha256=record_sha, items_sha256=items_sha)


def read_development_view(
    directory: Path, record_sha256: str
) -> tuple[DevelopmentView, list[BenchmarkItem]]:
    """Read a published view using only its own files; the mixed release is not opened."""
    view = read_project_json(directory / RECORD_FILE, record_sha256, DevelopmentView)
    verify_file_sha256(directory / ITEMS_FILE, view.output_items.sha256)
    items = decode_jsonl((directory / ITEMS_FILE).read_bytes(), BenchmarkItem)
    manifest = read_manifest(directory / MANIFEST_FILE, view.build_manifest.sha256)
    inputs = {entry.role: entry.artifact for entry in manifest.inputs}
    if (
        manifest.status != "complete"
        or [o.artifact for o in manifest.outputs] != [view.output_items]
        or inputs != {"benchmark_items": view.source_items, "paper_split": view.split}
    ):
        raise DerivationError("build manifest does not match the view record")
    if (
        len(view.development_paper_ids) != 10
        or view.development_paper_ids != sorted(view.development_paper_ids)
        or view.development_item_count != len(items)
        or view.contributing_paper_ids != sorted({item.paper_id for item in items})
        or not set(view.contributing_paper_ids) <= set(view.development_paper_ids)
    ):
        raise DerivationError("view record does not match its items")
    check_development_items(items, view.development_paper_ids, view.benchmark_version)
    return view, items
