"""Immutable qualified PDF conversion artifacts and attempt evidence."""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, verify_file_sha256
from biorag_repro.core.identity import (
    ArtifactInput,
    derived_artifact_id,
    new_attempt_id,
    raw_source_artifact_id,
)
from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_project_json,
    serialize_project_json,
)
from biorag_repro.documents.models import CanonicalDocument, Record
from biorag_repro.documents.pdf import (
    TRANSFORMATION_ID,
    PdfConfiguration,
    canonicalize_native,
    verify_assets,
    verify_effective_options,
    verify_environment,
    verify_qualified_lock,
)

NATIVE_TRANSFORMATION = "qualified_docling_native_v1"


class ConversionAttempt(Record):
    schema_version: Literal["1"] = "1"
    attempt_id: str = Field(min_length=1)
    attempt_logical_ref: str = Field(min_length=1)
    paper_id: str
    source: ArtifactRef
    configuration_sha256: str
    environment_sha256: str
    asset_inventory_sha256: str
    recorded_utc: str
    status: Literal["success", "partial_success", "failure"]
    build_status: Literal["complete", "incomplete", "failed"]
    errors: list[str] = Field(default_factory=list)
    page_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    native_artifact: ArtifactRef | None = None
    native_build_manifest: ArtifactRef | None = None
    canonical_artifact: ArtifactRef | None = None
    build_manifest: ArtifactRef | None = None

    @model_validator(mode="after")
    def valid_outcome(self) -> Self:
        if self.native_build_manifest and not self.native_artifact:
            raise ValueError("native manifest requires native artifact")
        if self.canonical_artifact and not self.native_artifact:
            raise ValueError("canonical artifact requires native artifact")
        if self.build_manifest and not self.canonical_artifact:
            raise ValueError("complete manifest requires canonical artifact")
        if self.build_status == "complete":
            if self.status != "success" or not all(
                (
                    self.native_artifact,
                    self.native_build_manifest,
                    self.canonical_artifact,
                    self.build_manifest,
                )
            ):
                raise ValueError("complete build requires successful full lineage")
        elif self.build_manifest is not None:
            raise ValueError("incomplete or failed build cannot have complete manifest")
        if self.status == "failure" and self.build_status != "failed":
            raise ValueError("failed Docling conversion cannot complete")
        if self.status == "partial_success" and self.build_status == "complete":
            raise ValueError("partial conversion cannot complete")
        if self.build_status == "incomplete" and self.status != "partial_success":
            raise ValueError("incomplete inspection requires partial conversion")
        if self.build_status == "failed" and not self.errors:
            raise ValueError("failed build requires diagnostics")
        return self


def _ref(
    kind: str,
    inputs: list[ArtifactInput],
    transform: str,
    config_hash: str,
    content: bytes,
    serialization_version: str,
    logical_dir: str,
) -> ArtifactRef:
    digest = sha256_bytes(content)
    artifact_id = derived_artifact_id(
        artifact_kind=kind,
        inputs=inputs,
        transformation_id=transform,
        transformation_settings_sha256=config_hash,
        output_sha256=digest,
        output_format="json",
        serialization_version=serialization_version,
    )
    return ArtifactRef(
        artifact_id=artifact_id,
        logical_ref=f"corpus/derived/{logical_dir}/{artifact_id}.json",
        sha256=digest,
    )


def _manifest(
    inputs: list[BuildInput],
    transform: str,
    config_hash: str,
    output: ArtifactRef,
    kind: str,
    version: str,
    data_root: Path,
) -> ArtifactRef:
    manifest = BuildManifest(
        schema_version="1",
        inputs=inputs,
        transformation_id=transform,
        transformation_settings_sha256=config_hash,
        outputs=[
            DerivedOutput(
                artifact=output,
                artifact_kind=kind,
                output_format="json",
                serialization_version=version,
                identity_version="1",
            )
        ],
        status="complete",
    )
    logical_ref = f"corpus/derived/manifests/{output.artifact_id}.build.json"
    target = data_root / logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = publish_manifest(
        manifest, target, {output.logical_ref: data_root / output.logical_ref}
    )
    return ArtifactRef(artifact_id="build_" + digest, logical_ref=logical_ref, sha256=digest)


def _publish_attempt(attempt: ConversionAttempt, data_root: Path) -> ConversionAttempt:
    content = serialize_project_json(attempt, ConversionAttempt)
    target = data_root / attempt.attempt_logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(content)
    publish_bytes(
        target,
        content,
        digest,
        validate_temp=lambda path: read_project_json(path, digest, ConversionAttempt),
    )
    return attempt


def publish_pdf_result(
    result: Any,
    source_path: Path,
    source: ArtifactRef,
    paper_id: str,
    configuration_sha256: str,
    environment_sha256: str,
    asset_inventory_sha256: str,
    data_root: Path,
    *,
    load_native: Any = None,
) -> ConversionAttempt:
    """Persist one converter result; only success may produce a complete build."""
    verify_file_sha256(source_path, source.sha256)
    if source.artifact_id != raw_source_artifact_id(
        paper_id=paper_id, representation="PDF", sha256=source.sha256
    ):
        raise ValueError("raw PDF artifact ID mismatch")
    status = result.status.value
    if status not in {"success", "partial_success", "failure"}:
        raise ValueError(f"unknown Docling status: {status}")
    attempt_id = new_attempt_id()
    runtime_settings_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "configuration_sha256": configuration_sha256,
                "environment_sha256": environment_sha256,
                "asset_inventory_sha256": asset_inventory_sha256,
            }
        )
    )
    errors = [str(getattr(error, "error_message", error)) for error in result.errors]
    build_status: Literal["complete", "incomplete", "failed"] = (
        "complete"
        if status == "success"
        else "incomplete"
        if status == "partial_success"
        else "failed"
    )
    if status == "success" and result.document is None:
        build_status = "failed"
        errors.append("Docling reported success without a native document")
    if status == "failure" and not errors:
        errors.append("Docling conversion failed without diagnostics")
    pages = [
        {
            "page_no": getattr(page, "page_no", None),
            "has_parsed_page": getattr(page, "parsed_page", None) is not None,
        }
        for page in result.pages
    ]
    native_ref = native_manifest = canonical_ref = build_ref = None
    loader = load_native
    if loader is None:
        from docling_core.types.doc import DoclingDocument

        loader = lambda path: DoclingDocument.load_from_json(path, allow_local_images=False)
    try:
        if result.document is not None and status in {"success", "partial_success"}:
            data_root.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=".docling-native-", suffix=".json", dir=data_root
            )
            os.close(descriptor)
            temporary = Path(name)
            try:
                result.document.save_as_json(
                    temporary, sort_keys=True, coord_precision=None, confid_precision=None
                )
                native_bytes = temporary.read_bytes()
                header = json.loads(native_bytes)
                if header.get("schema_name") != "DoclingDocument" or not header.get("version"):
                    raise ValueError("invalid native Docling JSON header")
                loader(temporary)
            finally:
                temporary.unlink(missing_ok=True)
            native_version = str(header["version"])
            raw_input = ArtifactInput("raw_pdf", source.artifact_id, source.sha256)
            native_ref = _ref(
                "docling_native",
                [raw_input],
                NATIVE_TRANSFORMATION,
                runtime_settings_sha256,
                native_bytes,
                native_version,
                "native/PDF_DOCLING",
            )
            native_target = data_root / native_ref.logical_ref
            native_target.parent.mkdir(parents=True, exist_ok=True)
            publish_bytes(
                native_target,
                native_bytes,
                native_ref.sha256,
                validate_temp=lambda path: loader(path),
            )
            verify_file_sha256(native_target, native_ref.sha256)
            reloaded = loader(native_target)
            native_manifest = _manifest(
                [BuildInput(role="raw_pdf", artifact=source)],
                NATIVE_TRANSFORMATION,
                runtime_settings_sha256,
                native_ref,
                "docling_native",
                native_version,
                data_root,
            )
            if status == "success":
                document = canonicalize_native(
                    reloaded, paper_id, source, native_ref, configuration_sha256
                )
                canonical_bytes = serialize_project_json(document, CanonicalDocument)
                inputs = [
                    raw_input,
                    ArtifactInput("docling_native", native_ref.artifact_id, native_ref.sha256),
                ]
                canonical_ref = _ref(
                    "canonical_pdf_docling",
                    inputs,
                    TRANSFORMATION_ID,
                    runtime_settings_sha256,
                    canonical_bytes,
                    "2",
                    "canonical/PDF_DOCLING",
                )
                canonical_target = data_root / canonical_ref.logical_ref
                canonical_target.parent.mkdir(parents=True, exist_ok=True)
                publish_bytes(
                    canonical_target,
                    canonical_bytes,
                    canonical_ref.sha256,
                    validate_temp=lambda path: read_project_json(
                        path, canonical_ref.sha256, CanonicalDocument
                    ),
                )
                verify_file_sha256(canonical_target, canonical_ref.sha256)
                build_ref = _manifest(
                    [
                        BuildInput(role="raw_pdf", artifact=source),
                        BuildInput(role="docling_native", artifact=native_ref),
                    ],
                    TRANSFORMATION_ID,
                    runtime_settings_sha256,
                    canonical_ref,
                    "canonical_pdf_docling",
                    "2",
                    data_root,
                )
    except FileExistsError:
        raise
    except Exception as exc:  # noqa: BLE001 - retain scientific failure evidence
        build_status = "failed"
        errors.append(f"{type(exc).__name__}: {exc}")
    return _publish_attempt(
        ConversionAttempt(
            attempt_id=attempt_id,
            attempt_logical_ref=f"corpus/derived/attempts/{attempt_id}.json",
            paper_id=paper_id,
            source=source,
            configuration_sha256=configuration_sha256,
            environment_sha256=environment_sha256,
            asset_inventory_sha256=asset_inventory_sha256,
            recorded_utc=datetime.now(UTC).isoformat(),
            status=status,
            build_status=build_status,
            errors=errors,
            page_diagnostics=pages,
            native_artifact=native_ref,
            native_build_manifest=native_manifest,
            canonical_artifact=canonical_ref,
            build_manifest=build_ref,
        ),
        data_root,
    )


def convert_pdf(
    source_path: Path,
    source: ArtifactRef,
    paper_id: str,
    configuration_sha256: str,
    assets_root: Path,
    lock_path: Path,
    data_root: Path,
) -> ConversionAttempt:
    """Run the fixed offline single-thread CPU Docling conversion."""
    import os

    import torch
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        LayoutObjectDetectionOptions,
        PdfPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes

    config = PdfConfiguration()
    actual_config = sha256_bytes(canonical_json_bytes(config.model_dump(mode="python")))
    if actual_config != configuration_sha256:
        raise ValueError("qualified configuration identity mismatch")
    environment = verify_environment()
    assets_hash = verify_assets(assets_root, config)
    verify_qualified_lock(lock_path, config)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"qualified offline/thread environment mismatch: {name}")
    verify_file_sha256(source_path, source.sha256)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    layout = LayoutObjectDetectionOptions()
    layout.model_spec = layout.model_spec.model_copy(update={"revision": config.heron_revision})
    options = PdfPipelineOptions(
        artifacts_path=assets_root,
        do_ocr=False,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE, do_cell_matching=True
        ),
        do_formula_enrichment=False,
        heading_hierarchy_options={"enabled": False},
        layout_options=layout,
        generate_page_images=False,
        generate_picture_images=False,
        generate_table_images=False,
        enable_remote_services=False,
        allow_external_plugins=False,
        document_timeout=None,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=1),
    )
    backend = ThreadedDoclingParseBackendOptions(parser_threads=1)
    verify_effective_options(
        options.model_dump(mode="json", exclude_none=False),
        backend.model_dump(mode="json", exclude_none=False),
        config,
    )
    try:
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options, backend_options=backend)
            },
        )
        result = converter.convert(source_path, raises_on_error=False)
    except Exception as exc:  # noqa: BLE001 - converter failures are recorded as attempts
        result = SimpleNamespace(
            status=SimpleNamespace(value="failure"),
            errors=[SimpleNamespace(error_message=f"{type(exc).__name__}: {exc}")],
            pages=[],
            document=None,
        )
    env_hash = sha256_bytes(canonical_json_bytes(environment))
    return publish_pdf_result(
        result,
        source_path,
        source,
        paper_id,
        configuration_sha256,
        env_hash,
        assets_hash,
        data_root,
    )
