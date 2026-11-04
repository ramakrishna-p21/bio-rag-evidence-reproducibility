"""Frozen paired-source lookup, Phase 4 builds, and lineage validation."""

import json
import re
from pathlib import Path
from typing import Any, Literal, Self
from xml.etree import ElementTree as ET

from pydantic import Field, field_validator, model_validator

from biorag_repro.core.hashing import (
    sha256_bytes,
    validate_sha256,
    verify_file_sha256,
)
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id, raw_source_artifact_id
from biorag_repro.core.manifest import ArtifactRef, BuildManifest
from biorag_repro.core.storage import (
    publish_bytes,
    read_manifest,
    read_project_json,
    serialize_project_json,
)
from biorag_repro.corpus.manifest import (
    PaperRecord,
    ReleaseRecord,
    corpus_content_hash,
    manifest_semantic_hash,
    parse_csv,
    verify_source_bytes,
)
from biorag_repro.documents.jats import ingest_jats
from biorag_repro.documents.models import CanonicalDocument, Record
from biorag_repro.documents.pdf_ingest import (
    ConversionAttempt,
    PdfCanonicalBuild,
    canonical_build_ref,
    canonicalize_saved_native,
    convert_pdf,
)
from biorag_repro.documents.storage import publish_canonical_jats

Arm = Literal["JATS", "PDF_DOCLING"]


class JatsBuild(Record):
    schema_version: Literal["1"] = "1"
    paper_id: str
    source: ArtifactRef
    configuration_sha256: str
    status: Literal["success"] = "success"
    build_status: Literal["complete"] = "complete"
    canonical_artifact: ArtifactRef
    build_manifest: ArtifactRef


class Phase4Validation(Record):
    schema_version: Literal["2"] = "2"
    corpus_version: str
    paper_id: str
    representation: Arm
    source: ArtifactRef
    configuration_sha256: str
    native_artifact: ArtifactRef | None = None
    native_build_manifest: ArtifactRef | None = None
    attempt_artifact: ArtifactRef | None = None
    canonical_build_artifact: ArtifactRef | None = None
    canonical_artifact: ArtifactRef
    build_manifest: ArtifactRef
    environment_sha256: str | None = None
    asset_inventory_sha256: str | None = None
    warnings: list[str] = Field(default_factory=list)
    jats_source_elements: int | None = None
    jats_source_slots: int | None = None
    jats_overlapping_slots: int | None = None
    native_objects: int | None = None
    native_text_objects: int | None = None
    native_table_cells: int | None = None
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def valid_arm(self) -> Self:
        validate_sha256(self.configuration_sha256)
        if self.representation == "JATS" and None in (
            self.jats_source_elements,
            self.jats_source_slots,
            self.jats_overlapping_slots,
        ):
            raise ValueError("JATS validation requires complete source coverage counts")
        if self.representation == "PDF_DOCLING" and None in (
            self.native_objects,
            self.native_text_objects,
            self.native_table_cells,
        ):
            raise ValueError("PDF validation requires complete native coverage counts")
        if self.representation == "PDF_DOCLING" and not all(
            (
                self.native_artifact,
                self.native_build_manifest,
                self.attempt_artifact,
                self.environment_sha256,
                self.asset_inventory_sha256,
            )
        ):
            raise ValueError(
                "PDF validation requires native, attempt, environment, and asset identities"
            )
        if self.representation == "JATS" and any(
            (
                self.native_artifact,
                self.native_build_manifest,
                self.attempt_artifact,
                self.canonical_build_artifact,
                self.environment_sha256,
                self.asset_inventory_sha256,
            )
        ):
            raise ValueError("JATS validation cannot claim PDF identities")
        return self


class Phase4GateIndex(Record):
    """Local evidence index tied to one reviewed implementation and frozen pair."""

    schema_version: Literal["2"] = "2"
    corpus_version: str
    paper_id: str
    corpus_release_sha256: str
    implementation_git_sha: str
    jats_validation: ArtifactRef
    pdf_validation: ArtifactRef

    @field_validator("corpus_release_sha256")
    @classmethod
    def release_digest(cls, value: str) -> str:
        validate_sha256(value)
        return value

    @field_validator("implementation_git_sha")
    @classmethod
    def commit_digest(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("implementation Git SHA must be a full lowercase SHA-1")
        return value


def _publish_evidence(
    value: Phase4Validation | Phase4GateIndex,
    model: type[Phase4Validation] | type[Phase4GateIndex],
    data_root: Path,
    kind: str,
    inputs: list[ArtifactInput],
) -> ArtifactRef:
    content = serialize_project_json(value, model)
    digest = sha256_bytes(content)
    artifact_id = derived_artifact_id(
        artifact_kind=kind,
        inputs=inputs,
        transformation_id="phase4_evidence_v2",
        transformation_settings_sha256=None,
        output_sha256=digest,
        output_format="json",
        serialization_version="1",
    )
    ref = ArtifactRef(
        artifact_id=artifact_id,
        logical_ref=f"corpus/derived/phase4_evidence/{artifact_id}.json",
        sha256=digest,
    )
    target = data_root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        content,
        digest,
        validate_temp=lambda path: read_project_json(path, digest, model),
    )
    return ref


def publish_validation(value: Phase4Validation, data_root: Path) -> ArtifactRef:
    return _publish_evidence(
        value,
        Phase4Validation,
        data_root,
        "phase4_validation",
        [
            ArtifactInput(
                "canonical", value.canonical_artifact.artifact_id, value.canonical_artifact.sha256
            )
        ],
    )


def publish_gate_index(value: Phase4GateIndex, data_root: Path, corpus_root: Path) -> ArtifactRef:
    for ref in (value.jats_validation, value.pdf_validation):
        checked = read_project_json(data_root / ref.logical_ref, ref.sha256, Phase4Validation)
        if checked.paper_id != value.paper_id or checked.corpus_version != value.corpus_version:
            raise ValueError("gate validation paper or corpus mismatch")
    jats = read_project_json(
        data_root / value.jats_validation.logical_ref,
        value.jats_validation.sha256,
        Phase4Validation,
    )
    pdf = read_project_json(
        data_root / value.pdf_validation.logical_ref, value.pdf_validation.sha256, Phase4Validation
    )
    if jats.representation != "JATS" or pdf.representation != "PDF_DOCLING":
        raise ValueError("gate index requires both Phase 4 arms")
    for report in (jats, pdf):
        raw_kind = "JATS" if report.representation == "JATS" else "PDF"
        if report.source.artifact_id != raw_source_artifact_id(
            paper_id=report.paper_id, representation=raw_kind, sha256=report.source.sha256
        ):
            raise ValueError("gate source artifact ID mismatch")
        source_path = corpus_root / "corpus/raw" / report.source.logical_ref
        verify_file_sha256(source_path, report.source.sha256)
        inputs = [report.source]
        if report.native_artifact is not None:
            inputs.append(report.native_artifact)
            assert report.native_build_manifest is not None
            _check_manifest(
                report.native_build_manifest, data_root, report.native_artifact, [report.source]
            )
            assert report.attempt_artifact is not None
            attempt = read_project_json(
                data_root / report.attempt_artifact.logical_ref,
                report.attempt_artifact.sha256,
                ConversionAttempt,
            )
            if (
                attempt.status != "success"
                or attempt.build_status != "complete"
                or attempt.source != report.source
                or attempt.native_artifact != report.native_artifact
                or attempt.environment_sha256 != report.environment_sha256
                or attempt.asset_inventory_sha256 != report.asset_inventory_sha256
            ):
                raise ValueError("gate PDF attempt lineage mismatch")
            if report.canonical_build_artifact is not None:
                canonical_build = read_project_json(
                    data_root / report.canonical_build_artifact.logical_ref,
                    report.canonical_build_artifact.sha256,
                    PdfCanonicalBuild,
                )
                if (
                    canonical_build_ref(canonical_build) != report.canonical_build_artifact
                    or canonical_build.original_attempt_artifact != report.attempt_artifact
                    or canonical_build.native_artifact != report.native_artifact
                    or canonical_build.canonical_artifact != report.canonical_artifact
                    or canonical_build.build_manifest != report.build_manifest
                ):
                    raise ValueError("gate saved-native canonicalization lineage mismatch")
            elif (
                attempt.canonical_artifact != report.canonical_artifact
                or attempt.build_manifest != report.build_manifest
            ):
                raise ValueError("gate PDF attempt canonical lineage mismatch")
        manifest = _check_manifest(
            report.build_manifest, data_root, report.canonical_artifact, inputs
        )
        doc = read_project_json(
            data_root / report.canonical_artifact.logical_ref,
            report.canonical_artifact.sha256,
            CanonicalDocument,
        )
        if (
            doc.paper_id != report.paper_id
            or doc.representation != report.representation
            or doc.source != report.source
            or doc.native_artifact != report.native_artifact
            or doc.configuration_sha256 != report.configuration_sha256
            or doc.status != "complete"
        ):
            raise ValueError("gate canonical lineage mismatch")
        if manifest.transformation_id != doc.transformation_id:
            raise ValueError("gate canonical transformation differs")
        if report.native_artifact is not None:
            native_data = json.loads((data_root / report.native_artifact.logical_ref).read_bytes())
            if validate_pdf_coverage(doc, native_data) != (
                report.native_objects,
                report.native_text_objects,
                report.native_table_cells,
            ):
                raise ValueError("gate PDF coverage summary differs")
        else:
            if validate_jats_coverage(doc, source_path.read_bytes()) != (
                report.jats_source_elements,
                report.jats_source_slots,
                report.jats_overlapping_slots,
            ):
                raise ValueError("gate JATS coverage summary differs")
    return _publish_evidence(
        value,
        Phase4GateIndex,
        data_root,
        "phase4_gate_index",
        [
            ArtifactInput(
                "jats_validation", value.jats_validation.artifact_id, value.jats_validation.sha256
            ),
            ArtifactInput(
                "pdf_validation", value.pdf_validation.artifact_id, value.pdf_validation.sha256
            ),
        ],
    )


def load_frozen_paper(
    corpus_root: Path,
    release_path: Path,
    release_sha256: str,
    corpus_version: str,
    paper_id: str,
) -> tuple[ReleaseRecord, PaperRecord]:
    """Resolve a paper only through the validated frozen release and exact CSV."""
    release = read_project_json(release_path, release_sha256, ReleaseRecord)
    if release.corpus_version != corpus_version:
        raise ValueError("corpus version mismatch")
    csv_path = corpus_root / release.manifest_artifact.logical_ref
    if csv_path.is_symlink() or not csv_path.is_file():
        raise ValueError("frozen corpus manifest must be a regular file")
    content = csv_path.read_bytes()
    if sha256_bytes(content) != release.csv_sha256:
        raise ValueError("frozen corpus CSV SHA-256 mismatch")
    manifest = parse_csv(content)
    if corpus_content_hash(manifest) != release.corpus_content_sha256:
        raise ValueError("frozen corpus content hash mismatch")
    if manifest_semantic_hash(manifest) != release.manifest_semantic_sha256:
        raise ValueError("frozen corpus semantic hash mismatch")
    selected = [paper for paper in manifest.papers if paper.paper_id == paper_id]
    if len(selected) != 1:
        raise ValueError("paper ID missing or duplicated in frozen corpus")
    paper = selected[0]
    if paper.corpus_version != corpus_version or paper.pair.pair_status != "verified_exact_pair":
        raise ValueError("frozen JATS/PDF pair is not verified")
    for kind in ("JATS", "PDF"):
        source = paper.jats if kind == "JATS" else paper.pdf
        if source.artifact_id != raw_source_artifact_id(
            paper_id=paper_id, representation=kind, sha256=source.sha256
        ):
            raise ValueError(f"{kind} source artifact ID mismatch")
    return release, paper


def _source(paper: PaperRecord, arm: Arm, corpus_root: Path) -> tuple[Path, ArtifactRef]:
    kind: Literal["jats", "pdf"] = "jats" if arm == "JATS" else "pdf"
    record = paper.jats if kind == "jats" else paper.pdf
    representation = "JATS" if kind == "jats" else "PDF"
    if record.artifact_id != raw_source_artifact_id(
        paper_id=paper.paper_id, representation=representation, sha256=record.sha256
    ):
        raise ValueError("source artifact ID mismatch")
    path = corpus_root / "corpus/raw" / record.logical_ref
    verify_source_bytes(paper, kind, path)
    return path, ArtifactRef(
        artifact_id=record.artifact_id, logical_ref=record.logical_ref, sha256=record.sha256
    )


def build_arm(
    paper: PaperRecord,
    paper_id: str,
    arm: Arm,
    corpus_root: Path,
    data_root: Path,
    configuration_sha256: str,
    *,
    assets_root: Path | None = None,
    lock_path: Path | None = None,
    reuse_attempt_id: str | None = None,
) -> JatsBuild | ConversionAttempt | PdfCanonicalBuild:
    """Verify exact frozen bytes before running the selected existing adapter."""
    if paper.paper_id != paper_id:
        raise ValueError("paper ID mismatch")
    if paper.pair.pair_status != "verified_exact_pair":
        raise ValueError("frozen pair is not verified")
    validate_sha256(configuration_sha256)
    source_path, source = _source(paper, arm, corpus_root)
    if arm == "PDF_DOCLING":
        if reuse_attempt_id is not None:
            if re.fullmatch(r"[0-9a-f]{32}", reuse_attempt_id) is None:
                raise ValueError("invalid reuse attempt ID")
            attempt_path = data_root / f"corpus/derived/attempts/{reuse_attempt_id}.json"
            prior = ConversionAttempt.model_validate_json(attempt_path.read_bytes())
            return canonicalize_saved_native(
                source_path, source, paper_id, prior, configuration_sha256, data_root
            )
        if assets_root is None or lock_path is None:
            raise ValueError("qualified PDF assets and lock are required")
        return convert_pdf(
            source_path, source, paper_id, configuration_sha256, assets_root, lock_path, data_root
        )
    if arm != "JATS":
        raise ValueError("unknown Phase 4 representation")
    source_content = source_path.read_bytes()
    doc = ingest_jats(
        source_content,
        paper_id=paper_id,
        source_artifact_id=source.artifact_id,
        source_sha256=source.sha256,
        source_logical_ref=source.logical_ref,
        configuration_sha256=configuration_sha256,
    )
    canonical, manifest = publish_canonical_jats(doc, data_root, source_content)
    return JatsBuild(
        paper_id=paper_id,
        source=source,
        configuration_sha256=configuration_sha256,
        canonical_artifact=canonical,
        build_manifest=manifest,
    )


def _check_manifest(
    ref: ArtifactRef,
    data_root: Path,
    expected_output: ArtifactRef,
    expected_inputs: list[ArtifactRef],
) -> BuildManifest:
    if ref.artifact_id != "build_" + ref.sha256:
        raise ValueError("build manifest artifact identity mismatch")
    manifest = read_manifest(data_root / ref.logical_ref, ref.sha256)
    if manifest.status != "complete" or len(manifest.outputs) != 1:
        raise ValueError("complete build manifest is required")
    if manifest.outputs[0].artifact != expected_output:
        raise ValueError("build output lineage mismatch")
    if [item.artifact for item in manifest.inputs] != expected_inputs:
        raise ValueError("build input lineage mismatch")
    verify_file_sha256(data_root / expected_output.logical_ref, expected_output.sha256)
    return manifest


def _jats_paths(content: bytes) -> dict[str, ET.Element]:
    root = ET.fromstring(content)
    paths: dict[str, ET.Element] = {}

    def visit(element: ET.Element, path: str) -> None:
        paths[path] = element
        counts: dict[str, int] = {}
        for child in element:
            counts[child.tag] = counts.get(child.tag, 0) + 1
            visit(child, f"{path}/{child.tag}[{counts[child.tag]}]")

    visit(root, f"/{root.tag}[1]")
    return paths


def _native_ref(data: object, reference: str) -> object:
    if not reference.startswith("#/"):
        raise ValueError("invalid native object reference")
    current = data
    for part in reference[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(key)]
        elif isinstance(current, dict):
            current = current[key]
        else:
            raise TypeError("unresolved native object reference")
    return current


def validate_jats_coverage(doc: CanonicalDocument, content: bytes) -> tuple[int, int, int]:
    """Compare canonical XML paths and slots with the original XML inventory."""
    paths = _jats_paths(content)
    nodes = {node.mapping.path: node for node in doc.nodes if node.mapping.path is not None}
    if set(nodes) != set(paths) or len(nodes) != len(doc.nodes):
        raise ValueError("JATS source element coverage differs")
    source_slots = {}
    for path, element in paths.items():
        for name, value in (("text()", element.text), ("tail()", element.tail)):
            if value is not None:
                source_slots[f"{path}/{name}"] = value
    represented: dict[str, list[str]] = {}
    for path, node in nodes.items():
        element = paths[path]
        if (
            node.source_tag != element.tag
            or node.source_id != element.attrib.get("id")
            or node.attributes != element.attrib
            or node.mapping.state != "exact_native"
        ):
            raise ValueError("JATS source element differs")
        for ordinal, part in enumerate(node.segments):
            slot = part.mapping.path
            if slot is None:
                raise ValueError("JATS source slot missing path")
            if (
                slot not in source_slots
                or part.text != source_slots[slot]
                or part.mapping.segment_ordinal != ordinal
            ):
                raise ValueError("JATS source slot differs")
            represented.setdefault(slot, []).append(node.node_id)
    if set(represented) != set(source_slots):
        raise ValueError("JATS source slot coverage differs")
    ordered = list(paths)
    if [node.mapping.path for node in doc.nodes] != ordered:
        raise ValueError("JATS source element order differs")
    path_by_element = {element: path for path, element in paths.items()}
    parent_by_element = {child: parent for parent in paths.values() for child in parent}

    def subtree_slots(element: ET.Element) -> list[str]:
        result = []
        own = f"{path_by_element[element]}/text()"
        if own in source_slots:
            result.append(own)
        for child in element:
            result.extend(subtree_slots(child))
            tail = f"{path_by_element[child]}/tail()"
            if tail in source_slots:
                result.append(tail)
        return result

    xml_id_nodes = {
        element.attrib["id"]: nodes[path].node_id
        for path, element in paths.items()
        if "id" in element.attrib
    }
    for path, element in paths.items():
        parent = parent_by_element.get(element)
        expected_parent = nodes[path_by_element[parent]].node_id if parent is not None else None
        if nodes[path].parent_node_id != expected_parent:
            raise ValueError("JATS source parent differs")
        direct = [f"{path}/text()"] + [f"{path_by_element[child]}/tail()" for child in element]
        direct = [slot for slot in direct if slot in source_slots]
        actual = [part.mapping.path for part in nodes[path].segments if part.mapping.path in direct]
        if actual != direct:
            # Semantic nodes may also contain descendant slots, but their direct
            # slots still appear in their original mixed-content order.
            raise ValueError("JATS mixed content source order differs")
        if nodes[path].kind in {
            "title",
            "paragraph",
            "table_cell",
            "caption",
            "formula",
            "xref",
            "reference",
        } and [part.mapping.path for part in nodes[path].segments] != subtree_slots(element):
            raise ValueError("JATS subtree mixed content order differs")
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in {"disp-formula", "inline-formula"} and nodes[path].kind != "formula":
            raise ValueError("JATS formula structure differs")
        if tag == "xref" and (
            nodes[path].rid != element.attrib.get("rid")
            or nodes[path].ref_type != element.attrib.get("ref-type")
        ):
            raise ValueError("JATS xref fields differ")
        if tag == "xref":
            rid = element.attrib.get("rid", "")
            targets = [xml_id_nodes[target] for target in rid.split() if target in xml_id_nodes]
            if nodes[path].target_node_ids != targets:
                raise ValueError("JATS xref targets differ")
        if tag == "table":
            table = nodes[path].table
            if table is None:
                raise ValueError("JATS table omitted")
            rows = [child for child in element.iter() if child.tag.rsplit("}", 1)[-1] == "tr"]
            source_cells = [
                cell for row in rows for cell in row if cell.tag.rsplit("}", 1)[-1] in {"td", "th"}
            ]
            by_id = {node.node_id: node for node in doc.nodes}
            if table.row_count != len(rows) or len(table.cells) != len(source_cells):
                raise ValueError("JATS table cell coverage differs")
            wrapper = parent_by_element.get(element)
            wrapper = (
                wrapper
                if wrapper is not None and wrapper.tag.rsplit("}", 1)[-1] == "table-wrap"
                else element
            )
            captions = [
                nodes[path_by_element[child]].node_id
                for child in wrapper
                if child.tag.rsplit("}", 1)[-1] == "caption"
            ]
            footnotes = [
                nodes[path_by_element[child]].node_id
                for child in wrapper
                if child.tag.rsplit("}", 1)[-1] == "table-wrap-foot"
            ]
            if table.caption_node_ids != captions or table.footnote_node_ids != footnotes:
                raise ValueError("JATS table relations differ")
            for source_cell, cell in zip(source_cells, table.cells, strict=True):
                if (
                    by_id[cell.node_id].mapping.path != path_by_element[source_cell]
                    or cell.row_span != int(source_cell.attrib.get("rowspan", "1"))
                    or cell.column_span != int(source_cell.attrib.get("colspan", "1"))
                    or cell.text != "".join(source_cell.itertext())
                ):
                    raise ValueError("JATS table cell differs")
    return len(paths), len(source_slots), sum(len(refs) > 1 for refs in represented.values())


def validate_pdf_coverage(doc: CanonicalDocument, data: dict[str, Any]) -> tuple[int, int, int]:
    """Compare saved native objects and required fields against canonical nodes."""
    if data.get("form_items") or data.get("key_value_items"):
        raise ValueError("unsupported populated native structured item")
    source = {}
    for root in ("body", "furniture"):
        if isinstance(data.get(root), dict):
            source[data[root]["self_ref"]] = data[root]
    for array in ("groups", "texts", "pictures", "tables", "form_items", "key_value_items"):
        for item in data.get(array, []):
            ref = item["self_ref"]
            if ref in source:
                raise ValueError("duplicate native object")
            source[ref] = item
    by_ref = {node.mapping.native_ref: node for node in doc.nodes if node.kind != "table_cell"}
    if len(by_ref) != sum(node.kind != "table_cell" for node in doc.nodes) or set(by_ref) != set(
        source
    ):
        raise ValueError("native object coverage differs")
    positioned = []
    seen = set()

    def visit(ref: str) -> None:
        if ref in seen or ref not in source:
            raise ValueError("native child relation invalid")
        seen.add(ref)
        positioned.append(ref)
        for child in source[ref].get("children", []):
            visit(child["$ref"])

    for root in ("#/body", "#/furniture"):
        if root in source:
            visit(root)
    expected_order = positioned + [ref for ref in source if ref not in seen]
    actual_order = [node.mapping.native_ref for node in doc.nodes if node.kind != "table_cell"]
    if actual_order != expected_order:
        raise ValueError("native reading order differs")
    table_cell_count = 0
    for ref, item in source.items():
        node = by_ref[ref]
        parent_ref = (item.get("parent") or {}).get("$ref")
        if parent_ref is not None and parent_ref not in by_ref:
            raise ValueError("native parent unresolved")
        if node.parent_node_id != (by_ref[parent_ref].node_id if parent_ref else None):
            raise ValueError("native parent differs")
        children = [link["$ref"] for link in item.get("children", [])]
        if node.native_child_node_ids != [by_ref[child].node_id for child in children]:
            raise ValueError("native children differ")
        if node.native_order_state != ("linked" if ref in seen else "unpositioned"):
            raise ValueError("native order state differs")
        if node.native_label != item.get("label") or node.native_content_layer != item.get(
            "content_layer"
        ):
            raise ValueError("native label or layer differs")
        if node.kind == "page_furniture" and item.get("content_layer") != "furniture":
            raise ValueError("invented page furniture")
        if (
            item.get("content_layer") == "furniture"
            and ref != "#/furniture"
            and node.kind != "page_furniture"
        ):
            raise ValueError("native page furniture type differs")
        if node.native_orig != item.get("orig"):
            raise ValueError("native orig differs")
        original = item.get("orig")
        original_text = item.get("text")
        expected_qualification = (
            (
                "unverified_formula"
                if item.get("label") == "formula" and original != original_text
                else "marker_inclusive"
                if item.get("label") == "list_item" and original != original_text
                else "same_as_text"
                if original == original_text
                else "unclassified_variant"
            )
            if original is not None
            else None
        )
        if node.orig_qualification != expected_qualification:
            raise ValueError("native orig qualification differs")
        if item.get("label") == "formula" and not original_text:
            state = "unverified_native_rendition" if original else "marker_only"
            if node.content_state != state:
                raise ValueError("native formula state differs")
        if node.native_marker != item.get("marker") or node.native_enumerated != item.get(
            "enumerated"
        ):
            raise ValueError("native list fields differ")
        if node.native_heading_level != item.get("level") or node.native_group_name != item.get(
            "name"
        ):
            raise ValueError("native heading or group field differs")
        if "text" in item and (len(node.segments) != 1 or node.segments[0].text != item["text"]):
            raise ValueError("native text differs")
        if "text" not in item and node.segments:
            raise ValueError("invented native text")
        if len(node.mapping.provenance) != len(item.get("prov", [])):
            raise ValueError("native provenance coverage differs")
        for index, native_prov in enumerate(item.get("prov", [])):
            record = node.mapping.provenance[index]
            span = native_prov.get("charspan")
            expected_state = (
                "unavailable" if span is None else "placeholder" if span == [0, 0] else "meaningful"
            )
            if (
                record.page_no != native_prov.get("page_no")
                or record.charspan != native_prov.get("charspan")
                or record.charspan_state != expected_state
                or record.ordinal != index
                or record.state != "approximate"
                or (record.bbox.model_dump() if record.bbox else None) != native_prov.get("bbox")
            ):
                raise ValueError("native provenance differs")
        if item.get("annotations") or item.get("references") or item.get("footnotes"):
            raise ValueError("unsupported populated native relation")
        captions = [link["$ref"] for link in item.get("captions", [])]
        if node.kind == "figure" and node.target_node_ids != [
            by_ref[target].node_id for target in captions
        ]:
            raise ValueError("native figure captions differ")
        if node.kind == "table":
            if node.table is None:
                raise ValueError("native table structure omitted")
            table = item["data"]
            if (node.table.row_count, node.table.column_count) != (
                table["num_rows"],
                table["num_cols"],
            ):
                raise ValueError("native table dimensions differ")
            if node.table.caption_node_ids != [by_ref[target].node_id for target in captions]:
                raise ValueError("native table captions differ")
            if node.table.caption_relation_state != ("present" if captions else "unavailable"):
                raise ValueError("native table caption state differs")
            if node.table.footnote_node_ids or node.table.footnote_relation_state != "unavailable":
                raise ValueError("native table footnote relation was inferred")
            if node.table.native_orientation != table.get("orientation"):
                raise ValueError("native table orientation differs")
            expected_cells = {
                f"{ref}/data/table_cells/{i}": cell
                for i, cell in enumerate(table.get("table_cells", []))
            }
            for row_index, row in enumerate(table.get("grid", [])):
                for column_index, cell in enumerate(row):
                    if cell is not None and cell not in expected_cells.values():
                        expected_cells[f"{ref}/data/grid/{row_index}/{column_index}"] = cell
            actual_cells = {cell.mapping.native_ref: cell for cell in node.table.cells}
            table_cell_count += len(expected_cells)
            if set(actual_cells) != set(expected_cells):
                raise ValueError("native table cell coverage differs")
            for cell_ref, native_cell in expected_cells.items():
                cell = actual_cells[cell_ref]
                row_header = native_cell.get("row_header")
                column_header = native_cell.get("column_header")
                expected_header = (
                    bool(row_header or column_header)
                    if row_header is not None and column_header is not None
                    else None
                )
                if (
                    cell.text != native_cell.get("text")
                    or cell.row != native_cell.get("start_row_offset_idx")
                    or cell.column != native_cell.get("start_col_offset_idx")
                    or cell.row_span != native_cell.get("row_span")
                    or cell.column_span != native_cell.get("col_span")
                    or cell.native_row_header != row_header
                    or cell.native_column_header != column_header
                    or cell.native_row_section != native_cell.get("row_section")
                    or cell.native_fillable != native_cell.get("fillable")
                    or cell.is_header != expected_header
                ):
                    raise ValueError("native table cell differs")
                if (
                    native_cell.get("end_row_offset_idx") != cell.row + cell.row_span
                    or native_cell.get("end_col_offset_idx") != cell.column + cell.column_span
                ):
                    raise ValueError("native table end offsets differ")
                expected_bbox = native_cell.get("bbox")
                provenance = cell.mapping.provenance
                if expected_bbox is None:
                    if provenance:
                        raise ValueError("invented native table cell provenance")
                elif (
                    len(provenance) != 1
                    or provenance[0].bbox is None
                    or provenance[0].bbox.model_dump() != expected_bbox
                    or provenance[0].page_no is not None
                    or provenance[0].charspan is not None
                    or provenance[0].state != "ambiguous"
                ):
                    raise ValueError("native table cell provenance differs")
            grid = table.get("grid", [])
            if len(grid) != len(node.table.grid_cell_node_ids):
                raise ValueError("native table grid differs")
            for row, canonical_row in zip(grid, node.table.grid_cell_node_ids, strict=True):
                if len(row) != len(canonical_row):
                    raise ValueError("native table grid differs")
                for native_cell, cell_id in zip(row, canonical_row, strict=True):
                    if native_cell is None:
                        if cell_id is not None:
                            raise ValueError("invented native grid cell")
                        continue
                    matches = [
                        cell_ref
                        for cell_ref, value in expected_cells.items()
                        if value == native_cell
                    ]
                    if len(matches) != 1 or cell_id != actual_cells[matches[0]].node_id:
                        raise ValueError("native table grid differs or is ambiguous")
    for node in doc.nodes:
        ref = node.mapping.native_ref
        if ref is None or _native_ref(data, ref) is None:
            raise ValueError("invented native reference")
    return len(source), len(data.get("texts", [])), table_cell_count


def validate_build(
    build: JatsBuild | ConversionAttempt | PdfCanonicalBuild,
    data_root: Path,
    *,
    corpus_root: Path | None = None,
    corpus_version: str = "corpus_v1",
) -> Phase4Validation:
    """Verify saved artifacts and source mappings without benchmark outcomes."""
    if build.build_status != "complete" or build.status != "success":
        raise ValueError("complete successful build is required")
    if build.canonical_artifact is None or build.build_manifest is None:
        raise ValueError("complete build has missing artifacts")
    native = (
        build.native_artifact if isinstance(build, (ConversionAttempt, PdfCanonicalBuild)) else None
    )
    if corpus_root is not None:
        source_path = corpus_root / "corpus/raw" / build.source.logical_ref
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError("frozen source must be a regular file")
        verify_file_sha256(source_path, build.source.sha256)
    _check_manifest(
        build.build_manifest,
        data_root,
        build.canonical_artifact,
        [build.source, native] if native else [build.source],
    )
    doc = read_project_json(
        data_root / build.canonical_artifact.logical_ref,
        build.canonical_artifact.sha256,
        CanonicalDocument,
    )
    if (
        doc.source != build.source
        or doc.paper_id != build.paper_id
        or doc.configuration_sha256 != build.configuration_sha256
        or doc.status != "complete"
    ):
        raise ValueError("canonical identity or completion mismatch")
    warnings: list[str] = []
    jats_coverage: tuple[int, int, int] | None = None
    pdf_coverage: tuple[int, int, int] | None = None
    if native is None:
        if doc.representation != "JATS":
            raise ValueError("JATS build has wrong representation")
        if corpus_root is None:
            raise ValueError("JATS validation requires frozen source root")
        paths = None
        source_content = source_path.read_bytes()
        paths = _jats_paths(source_content)
        jats_coverage = validate_jats_coverage(doc, source_content)
        for node in doc.nodes:
            if paths is not None and node.mapping.path not in paths:
                raise ValueError("unresolved JATS XML node path")
            for segment in node.segments:
                if paths is not None:
                    path, slot = segment.mapping.path.rsplit("/", 1)  # type: ignore[union-attr]
                    if path not in paths or slot not in {"text()", "tail()"}:
                        raise ValueError("unresolved JATS XML segment path")
                    original = paths[path].text if slot == "text()" else paths[path].tail
                    if original != segment.text:
                        raise ValueError("JATS mixed content differs from source")
    else:
        if doc.representation != "PDF_DOCLING" or doc.native_artifact != native:
            raise ValueError("PDF native lineage mismatch")
        assert isinstance(build, (ConversionAttempt, PdfCanonicalBuild))
        if build.native_build_manifest is None:
            raise ValueError("native build manifest missing")
        _check_manifest(build.native_build_manifest, data_root, native, [build.source])
        native_data = json.loads((data_root / native.logical_ref).read_bytes())
        pdf_coverage = validate_pdf_coverage(doc, native_data)
        id_by_ref = {node.mapping.native_ref: node.node_id for node in doc.nodes}
        for node in doc.nodes:
            source_item = _native_ref(native_data, node.mapping.native_ref or "")
            if not isinstance(source_item, dict):
                raise TypeError("native reference is not an object")
            if node.native_label is None:
                raise ValueError("PDF parser-native label missing")
            source_label = source_item.get("label", "table_cell")
            if node.native_label != source_label:
                raise ValueError("PDF parser-native label differs")
            if (
                "text" in source_item
                and node.segments
                and node.segments[0].text != source_item["text"]
            ):
                raise ValueError("native text differs")
            for record in node.mapping.provenance:
                original = source_item.get("prov", [])
                if original:
                    if record.ordinal >= len(original):
                        raise ValueError("unresolved native provenance ordinal")
                    entry = original[record.ordinal]
                elif node.kind == "table_cell" and record.ordinal == 0:
                    entry = {"bbox": source_item.get("bbox")}
                else:
                    raise ValueError("unresolved native provenance ordinal")
                if record.page_no != entry.get("page_no"):
                    raise ValueError("native page differs")
                if record.charspan != entry.get("charspan"):
                    raise ValueError("native charspan differs")
                if record.charspan == [0, 0] and record.charspan_state != "placeholder":
                    raise ValueError("placeholder native charspan was misqualified")
                if record.bbox is not None:
                    bbox = entry.get("bbox")
                    if bbox is None or any(
                        getattr(record.bbox, key) != bbox[key] for key in ("l", "t", "r", "b")
                    ):
                        raise ValueError("native bbox differs")
                    if record.bbox.coord_origin != bbox.get("coord_origin"):
                        raise ValueError("native coordinate origin differs")
            if node.table:
                native_table = source_item.get("data", {})
                if node.table.row_count != native_table.get(
                    "num_rows"
                ) or node.table.column_count != native_table.get("num_cols"):
                    raise ValueError("native table dimensions differ")
                native_grid = native_table.get("grid", [])
                if len(node.table.grid_cell_node_ids) != len(native_grid):
                    raise ValueError("native table grid row count differs")
                cell_by_id = {cell.node_id: cell for cell in node.table.cells}
                for row_ids, row_source in zip(
                    node.table.grid_cell_node_ids, native_grid, strict=True
                ):
                    if len(row_ids) != len(row_source):
                        raise ValueError("native table grid column count differs")
                    for cell_id, grid_source in zip(row_ids, row_source, strict=True):
                        if cell_id is None:
                            if grid_source is not None:
                                raise ValueError("native table grid cell was dropped")
                        elif (
                            _native_ref(native_data, cell_by_id[cell_id].mapping.native_ref or "")
                            != grid_source
                        ):
                            raise ValueError("native table grid cell differs")
                grid_complete = len(native_grid) == node.table.row_count and all(
                    len(row) == node.table.column_count and all(cell is not None for cell in row)
                    for row in native_grid
                )
                if node.table.structure_state != (
                    "exact_native" if grid_complete else "parser_damaged"
                ):
                    raise ValueError("native table damage state differs")
                native_captions = [link["$ref"] for link in source_item.get("captions", [])]
                expected_captions = [id_by_ref[ref] for ref in native_captions]
                if node.table.caption_node_ids != expected_captions:
                    raise ValueError("native table caption relation differs")
                if node.table.caption_relation_state != (
                    "present" if native_captions else "unavailable"
                ):
                    raise ValueError("native caption relation state differs")
                if (
                    node.table.footnote_node_ids
                    or node.table.footnote_relation_state != "unavailable"
                ):
                    raise ValueError("native table footnote relation was inferred")
                for cell in node.table.cells:
                    cell_source = _native_ref(native_data, cell.mapping.native_ref or "")
                    if not isinstance(cell_source, dict) or cell.text != cell_source.get("text"):
                        raise ValueError("native table cell differs")
                    if (
                        cell.row != cell_source.get("start_row_offset_idx")
                        or cell.column != cell_source.get("start_col_offset_idx")
                        or cell.row_span != cell_source.get("row_span")
                        or cell.column_span != cell_source.get("col_span")
                    ):
                        raise ValueError("native table cell geometry differs")
                if node.table.caption_relation_state == "unavailable":
                    warnings.append("table_caption_link_unavailable")
                if node.table.footnote_relation_state == "unavailable":
                    warnings.append("table_footnote_link_unavailable")
    for node in doc.nodes:
        if node.kind == "source_construct":
            warnings.append("source_construct_preserved")
        if node.content_state == "marker_only":
            warnings.append("formula_marker_only")
        if node.relation_state == "unavailable":
            warnings.append("unresolved_relation")
        if any(p.charspan_state == "placeholder" for p in node.mapping.provenance):
            warnings.append("native_charspan_placeholder")
        if node.interpretation_state in {"ambiguous", "parser_damaged"}:
            warnings.append("native_classification_uncertain")
        if node.table and node.table.structure_state == "parser_damaged":
            warnings.append("table_structure_damaged")
    attempt_ref = None
    if isinstance(build, PdfCanonicalBuild):
        attempt_ref = build.original_attempt_artifact
        record_ref = canonical_build_ref(build)
        if (
            read_project_json(
                data_root / record_ref.logical_ref, record_ref.sha256, PdfCanonicalBuild
            )
            != build
        ):
            raise ValueError("PDF canonical build read-back mismatch")
    elif isinstance(build, ConversionAttempt):
        attempt_path = data_root / build.attempt_logical_ref
        attempt_digest = sha256_bytes(attempt_path.read_bytes())
        if read_project_json(attempt_path, attempt_digest, ConversionAttempt) != build:
            raise ValueError("PDF attempt read-back mismatch")
        attempt_ref = ArtifactRef(
            artifact_id="attempt_" + build.attempt_id,
            logical_ref=build.attempt_logical_ref,
            sha256=attempt_digest,
        )
    return Phase4Validation(
        corpus_version=corpus_version,
        paper_id=doc.paper_id,
        representation=doc.representation,
        source=doc.source,
        configuration_sha256=doc.configuration_sha256,
        native_artifact=native,
        native_build_manifest=build.native_build_manifest
        if isinstance(build, (ConversionAttempt, PdfCanonicalBuild))
        else None,
        attempt_artifact=attempt_ref,
        canonical_build_artifact=canonical_build_ref(build)
        if isinstance(build, PdfCanonicalBuild)
        else None,
        jats_source_elements=jats_coverage[0] if jats_coverage else None,
        jats_source_slots=jats_coverage[1] if jats_coverage else None,
        jats_overlapping_slots=jats_coverage[2] if jats_coverage else None,
        native_objects=pdf_coverage[0] if pdf_coverage else None,
        native_text_objects=pdf_coverage[1] if pdf_coverage else None,
        native_table_cells=pdf_coverage[2] if pdf_coverage else None,
        canonical_artifact=build.canonical_artifact,
        build_manifest=build.build_manifest,
        environment_sha256=build.environment_sha256
        if isinstance(build, (ConversionAttempt, PdfCanonicalBuild))
        else None,
        asset_inventory_sha256=(
            build.asset_inventory_sha256
            if isinstance(build, (ConversionAttempt, PdfCanonicalBuild))
            else None
        ),
        warnings=sorted(set(warnings)),
    )
