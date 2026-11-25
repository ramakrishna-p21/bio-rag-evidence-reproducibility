"""Paired-source lookup, JATS and PDF evidence builds, and validation of a build's lineage."""

import json
import re
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from biorag_repro.core.hashing import (
    sha256_bytes,
    validate_sha256,
    verify_file_sha256,
)
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id, raw_source_artifact_id
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import (
    publish_bytes,
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
from biorag_repro.documents.coverage import (
    jats_paths,
    pdf_objects,
    resolve_native_ref,
    validate_jats_coverage,
    validate_pdf_coverage,
)
from biorag_repro.documents.jats import TRANSFORMATION_ID as JATS_TRANSFORMATION
from biorag_repro.documents.jats import ingest_jats
from biorag_repro.documents.lineage import check_manifest, check_manifest_contract, runtime_settings
from biorag_repro.documents.models import CanonicalDocument, Record
from biorag_repro.documents.pdf import (
    FIGURE_FOOTNOTE_TRANSFORMATION_ID,
    FOOTNOTE_TRANSFORMATION_ID,
    native_classification,
)
from biorag_repro.documents.pdf import TRANSFORMATION_ID as PDF_TRANSFORMATION
from biorag_repro.documents.pdf_ingest import (
    NATIVE_TRANSFORMATION,
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
    if (
        not isinstance(value, Phase4Validation)
        or model is not Phase4Validation
        or kind != "phase4_validation"
    ):
        raise ValueError("independent human authorization is unavailable for gate publication")
    content = serialize_project_json(value, model)
    ref = evidence_ref(value, model, kind, inputs)
    target = data_root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        content,
        ref.sha256,
        validate_temp=lambda path: read_project_json(path, ref.sha256, model),
    )
    return ref


def evidence_ref(
    value: Phase4Validation | Phase4GateIndex,
    model: type[Phase4Validation] | type[Phase4GateIndex],
    kind: str,
    inputs: list[ArtifactInput],
) -> ArtifactRef:
    digest = sha256_bytes(serialize_project_json(value, model))
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
    """Historical gate publication is disabled until independent approval exists."""
    raise ValueError("independent human authorization is unavailable; publish a candidate only")


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


def frozen_source(paper: PaperRecord, arm: Arm, corpus_root: Path) -> tuple[Path, ArtifactRef]:
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
    source_path, source = frozen_source(paper, arm, corpus_root)
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
    check_manifest(
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

    settings = build.configuration_sha256
    if isinstance(build, (ConversionAttempt, PdfCanonicalBuild)):
        settings = runtime_settings(
            settings, build.environment_sha256, build.asset_inventory_sha256
        )
    pdf_transform = doc.transformation_id
    if native and pdf_transform not in {
        PDF_TRANSFORMATION,
        FOOTNOTE_TRANSFORMATION_ID,
        FIGURE_FOOTNOTE_TRANSFORMATION_ID,
    }:
        raise ValueError("unsupported canonical PDF adapter")
    check_manifest_contract(
        build.build_manifest,
        data_root,
        build.canonical_artifact,
        [build.source, native] if native else [build.source],
        ["raw_pdf", "docling_native"] if native else ["raw_jats"],
        "canonical_pdf_docling" if native else "canonical_jats",
        pdf_transform if native else JATS_TRANSFORMATION,
        settings,
        doc.schema_version,
    )
    if doc.transformation_id != (pdf_transform if native else JATS_TRANSFORMATION):
        raise ValueError("canonical adapter identity mismatch")
    if native is not None:
        assert isinstance(build, (ConversionAttempt, PdfCanonicalBuild))
        assert build.native_build_manifest is not None
        native_data = json.loads((data_root / native.logical_ref).read_bytes())
        check_manifest_contract(
            build.native_build_manifest,
            data_root,
            native,
            [build.source],
            ["raw_pdf"],
            "docling_native",
            NATIVE_TRANSFORMATION,
            settings,
            native_data["version"],
        )
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
        paths = jats_paths(source_content)
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
        check_manifest(build.native_build_manifest, data_root, native, [build.source])
        native_data = json.loads((data_root / native.logical_ref).read_bytes())
        pdf_coverage = validate_pdf_coverage(doc, native_data)
        if any(
            native_classification(ref, item["label"], item.get("content_layer"))[1]
            in {"ambiguous", "parser_damaged"}
            for ref, item in pdf_objects(native_data).items()
        ):
            warnings.append("native_classification_uncertain")
        id_by_ref = {node.mapping.native_ref: node.node_id for node in doc.nodes}
        for node in doc.nodes:
            source_item = resolve_native_ref(native_data, node.mapping.native_ref or "")
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
                            resolve_native_ref(
                                native_data, cell_by_id[cell_id].mapping.native_ref or ""
                            )
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
                footnotes = (
                    [link["$ref"] for link in source_item.get("footnotes", [])]
                    if doc.transformation_id
                    in {FOOTNOTE_TRANSFORMATION_ID, FIGURE_FOOTNOTE_TRANSFORMATION_ID}
                    else []
                )
                if any(ref not in id_by_ref for ref in footnotes):
                    raise ValueError("unresolved native table footnote")
                if node.table.footnote_node_ids != [
                    id_by_ref[ref] for ref in footnotes
                ] or node.table.footnote_relation_state != (
                    "present" if footnotes else "unavailable"
                ):
                    raise ValueError("native table footnote relation differs or was inferred")
                for cell in node.table.cells:
                    cell_source = resolve_native_ref(native_data, cell.mapping.native_ref or "")
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
        if native is None and node.interpretation_state in {"ambiguous", "parser_damaged"}:
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
