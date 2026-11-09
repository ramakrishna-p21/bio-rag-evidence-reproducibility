"""Index of the files that make up a synthetic analysis rehearsal, with their hashes.

Records only. The index lists the R primary result directory, the R secondary result directory
and the input files they cite, each with its size and SHA-256, and it lists separately the
resources that are not part of the package and must be acquired or requested elsewhere. No
protected scientific content is copied. Every reference is checked when the index is built and
again when it is read: the files are re-hashed, the primary run record and the secondary record
must agree with the files and with each other, and the cited input hashes must match.

The index is for a synthetic rehearsal. It carries no scientific authority and is never final.
"""

import json
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from biorag_repro.benchmark.models import Record
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.storage import publish_bytes
from biorag_repro.evaluation.analysis_result import (
    SCIENTIFIC_FILES,
    ResultError,
    parse_run_record,
    read_result_files,
    scientific_result_id,
)
from biorag_repro.evaluation.supplement import (
    RECORD_FILE as SUPPLEMENT_RECORD,
)
from biorag_repro.evaluation.supplement import (
    TABLES as SUPPLEMENT_TABLES,
)
from biorag_repro.evaluation.supplement import (
    SupplementRecord,
    serialize_record,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
BANNER: Final = "SYNTHETIC REHEARSAL — NOT SCIENTIFIC FINDINGS"

SECONDARY_TABLES: Final = (
    "secondary_answer_class.csv",
    "secondary_completion.csv",
    "secondary_correct_without_evidence.csv",
    "secondary_discordance.csv",
    "secondary_populations.csv",
    "secondary_repeat_outcomes.csv",
    "secondary_repeat_variability.csv",
    "secondary_robustness.csv",
    "secondary_scope.csv",
    "secondary_stage_a.csv",
    "secondary_stage_b_failure.csv",
    "secondary_unavailable.csv",
)
SECONDARY_PRESENTATION: Final = ("report.md", "coprimary_effects.svg")
SECONDARY_RECORD: Final = "secondary_record.json"
SECONDARY_FILES: Final = (*SECONDARY_TABLES, *SECONDARY_PRESENTATION, SECONDARY_RECORD)

# The R summary of the optional supplement (analysis/R/supplement.R).
SUPPLEMENT_SUMMARY_TABLES: Final = (
    "supplement_evidence_outcomes.csv",
    "supplement_evidence_summary.csv",
    "supplement_pairs_summary.csv",
    "supplement_populations.csv",
    "supplement_stability_summary.csv",
    "supplement_stage_forms.csv",
)
SUPPLEMENT_SUMMARY_RECORD: Final = "supplement_summary_record.json"
SUPPLEMENT_PYTHON_FILES: Final = (*SUPPLEMENT_TABLES, SUPPLEMENT_RECORD)
SUPPLEMENT_SUMMARY_FILES: Final = (*SUPPLEMENT_SUMMARY_TABLES, SUPPLEMENT_SUMMARY_RECORD)

# Resources the package does not contain. Their absence is part of the index.
EXTERNAL_RESOURCES: Final = {
    "frozen_source_publications": "source JATS and PDF files: rights not inferred from availability; authorized reacquisition",
    "parser_and_embedding_models": "Docling and BGE model files: pinned revisions, acquired separately",
    "generator_weights": "Qwen3-8B checkpoint: pinned revision, acquired separately",
    "benchmark_questions_gold_evidence": "protected benchmark content: never copied into an index",
    "real_final_test_observations": "no real final-test observation exists; none is represented here",
}


class IncludedArtifact(Record):
    role: str = Field(min_length=1)
    path: str = Field(min_length=1)
    bytes: int = Field(ge=1)
    sha256: Sha256


class ExternalResource(Record):
    role: str
    status: Literal["externally_acquired_or_restricted"]
    reason: str


class IndexCode(Record):
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: bool
    script_sha256: dict[str, Sha256]


class ReleaseIndex(Record):
    kind: Literal["release_index"] = "release_index"
    schema_version: Literal["1"] = "1"
    purpose: Literal["synthetic_rehearsal"] = "synthetic_rehearsal"
    scientific_authority: Literal["none"] = "none"
    final: Literal[False] = False
    banner: Literal["SYNTHETIC REHEARSAL — NOT SCIENTIFIC FINDINGS"] = BANNER
    handoff_sha256: Sha256
    csv_sha256: Sha256
    primary_scientific_result_sha256: Sha256
    secondary_result_sha256: Sha256
    # Present only when a supplement is indexed; absent otherwise so earlier indexes keep their bytes.
    supplement_sha256: Sha256 | None = Field(default=None, exclude_if=lambda value: value is None)
    supplement_summary_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    r_version_required: str
    renv_lock_sha256: Sha256
    primary_code: IndexCode
    secondary_code: IndexCode
    included: list[IncludedArtifact]
    external: list[ExternalResource]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        roles = [a.role for a in self.included]
        if len(set(roles)) != len(roles) or len({a.path for a in self.included}) != len(roles):
            raise ValueError("included roles and paths must be unique")
        if [e.role for e in self.external] != list(EXTERNAL_RESOURCES):
            raise ValueError("the external resources must be exactly the declared ones, in order")
        if set(roles) & set(EXTERNAL_RESOURCES):
            raise ValueError("an external resource cannot also be included")
        if (self.supplement_sha256 is None) != (self.supplement_summary_sha256 is None):
            raise ValueError("a supplement and its summary are indexed together")
        return self


def _json(data: bytes, what: str) -> dict[str, Any]:
    try:
        parsed = json.loads(data.decode("utf-8"))
    except ValueError as error:
        raise ResultError("record_structure", f"{what} is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ResultError("record_structure", f"{what} is not a JSON object")
    return parsed


def _secondary_id(handoff: str, csv: str, tables: dict[str, str]) -> str:
    payload = {
        "csv_sha256": csv,
        "files": [{"name": n, "sha256": tables[n]} for n in sorted(tables)],
        "handoff_sha256": handoff,
        "kind": "secondary_tables",
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _file(root: Path, relative: str) -> bytes:
    parts = relative.split("/")
    if not parts or any(p in ("", ".", "..") for p in parts) or "\\" in relative:
        raise ResultError("result_location", "an input file must be a plain relative path")
    directory = "/".join(parts[:-1])
    if not directory:
        raise ResultError("result_location", "an input file must be inside a directory")
    return read_result_files(root, directory, (parts[-1],), exact=False)[parts[-1]]


def _check(
    root: Path, primary_dir: str, secondary_dir: str, inputs: dict[str, str]
) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, bytes]]:
    """Read and cross-check everything the index cites; raise on any disagreement."""
    primary = read_result_files(root, primary_dir)
    secondary = read_result_files(root, secondary_dir, SECONDARY_FILES)
    record = parse_run_record(primary["run_record.json"])
    for name, entry in record.outputs.files.items():
        if name not in primary or sha256_bytes(primary[name]) != entry.sha256:
            raise ResultError("hash", f"primary {name} differs from the hash in its run record")
    if scientific_result_id(
        record.inputs.handoff_sha256,
        record.inputs.csv_sha256,
        record.seed.value,
        {n: record.outputs.files[n].sha256 for n in SCIENTIFIC_FILES},
    ) != (record.outputs.scientific_result_sha256):
        raise ResultError("identity", "the primary scientific result hash does not match its files")
    sec = _json(secondary[SECONDARY_RECORD], SECONDARY_RECORD)
    listed = {**sec.get("scientific_files", {}), **sec.get("presentation", {})}
    if set(listed) != set(SECONDARY_FILES) - {SECONDARY_RECORD}:
        raise ResultError("file_set", "the secondary record does not list the expected files")
    for name, entry in listed.items():
        if sha256_bytes(secondary[name]) != entry.get("sha256") or len(
            secondary[name]
        ) != entry.get("bytes"):
            raise ResultError("hash", f"secondary {name} differs from its record")
    ids = {n: sec["scientific_files"][n]["sha256"] for n in SECONDARY_TABLES}
    inputs_cited = sec.get("inputs", {})
    if _secondary_id(
        inputs_cited.get("handoff_sha256", ""), inputs_cited.get("csv_sha256", ""), ids
    ) != sec.get("secondary_result_sha256"):
        raise ResultError("identity", "the secondary result hash does not match its tables")
    cited = sec.get("primary_result", {})
    if (
        cited.get("run_record_sha256") != sha256_bytes(primary["run_record.json"])
        or cited.get("scientific_result_sha256") != record.outputs.scientific_result_sha256
        or inputs_cited.get("handoff_sha256") != record.inputs.handoff_sha256
        or inputs_cited.get("csv_sha256") != record.inputs.csv_sha256
        or sec.get("purpose") != "synthetic_rehearsal"
        or sec.get("scientific_authority") != "none"
        or sec.get("banner") != BANNER
    ):
        raise ResultError("citation", "the secondary record and the primary result disagree")
    for name in SECONDARY_PRESENTATION:
        if BANNER.encode() not in secondary[name]:
            raise ResultError("banner", f"secondary {name} does not state that it is synthetic")
    expected = {"handoff": record.inputs.handoff_sha256, "analysis_table": record.inputs.csv_sha256}
    if set(inputs) != {"handoff", "analysis_table", "preflight"}:
        raise ResultError(
            "inputs", "the inputs are the handoff, the analysis table and the preflight"
        )
    input_bytes = {role: _file(root, path) for role, path in inputs.items()}
    expected["preflight"] = record.inputs.python_preflight_sha256
    for role, data in input_bytes.items():
        if sha256_bytes(data) != expected[role]:
            raise ResultError(
                "hash", f"input {role} differs from the hash the primary record cites"
            )
    return primary, secondary, input_bytes


def _summary_id(handoff: str, csv: str, supplement: str, tables: dict[str, str]) -> str:
    payload = {
        "csv_sha256": csv,
        "files": [{"name": n, "sha256": tables[n]} for n in sorted(tables)],
        "handoff_sha256": handoff,
        "kind": "supplement_summary",
        "supplement_sha256": supplement,
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _check_supplement(
    root: Path, supplement_dir: str, summary_dir: str, handoff: str, csv: str
) -> tuple[dict[str, bytes], dict[str, bytes], str, str]:
    """Cross-check the Python supplement and its R summary with each other and with the primary."""
    py = read_result_files(root, supplement_dir, SUPPLEMENT_PYTHON_FILES)
    summary = read_result_files(root, summary_dir, SUPPLEMENT_SUMMARY_FILES)
    record = SupplementRecord.model_validate_json(py[SUPPLEMENT_RECORD], strict=True)
    if serialize_record(record) != py[SUPPLEMENT_RECORD]:
        raise ResultError("canonical_form", "the supplement record is not in canonical form")
    for name in SUPPLEMENT_TABLES:
        entry = record.files[name]
        if sha256_bytes(py[name]) != entry.sha256 or len(py[name]) != entry.bytes:
            raise ResultError("hash", f"supplement {name} differs from its record")
    if (record.handoff_sha256, record.analysis_table_sha256) != (handoff, csv):
        raise ResultError("citation", "the supplement cites other inputs than the primary result")
    cited = _json(summary[SUPPLEMENT_SUMMARY_RECORD], SUPPLEMENT_SUMMARY_RECORD)
    listed = cited.get("scientific_files", {})
    if set(listed) != set(SUPPLEMENT_SUMMARY_TABLES):
        raise ResultError("file_set", "the supplement summary record lists other tables")
    for name in SUPPLEMENT_SUMMARY_TABLES:
        if sha256_bytes(summary[name]) != listed[name].get("sha256") or len(summary[name]) != (
            listed[name].get("bytes")
        ):
            raise ResultError("hash", f"supplement summary {name} differs from its record")
    inputs_cited = cited.get("inputs", {})
    if (
        (inputs_cited.get("handoff_sha256"), inputs_cited.get("csv_sha256")) != (handoff, csv)
        or inputs_cited.get("supplement_sha256") != record.supplement_sha256
        or inputs_cited.get("supplement_record_sha256") != sha256_bytes(py[SUPPLEMENT_RECORD])
        or cited.get("purpose") != "synthetic_rehearsal"
        or cited.get("scientific_authority") != "none"
        or cited.get("banner") != BANNER
    ):
        raise ResultError("citation", "the summary and the supplement disagree")
    summary_id = _summary_id(
        handoff, csv, record.supplement_sha256, {n: listed[n]["sha256"] for n in listed}
    )
    if summary_id != cited.get("supplement_summary_sha256"):
        raise ResultError("identity", "the supplement summary hash does not match its tables")
    return py, summary, record.supplement_sha256, summary_id


def build_release_index(
    root: Path,
    *,
    primary_dir: str,
    secondary_dir: str,
    inputs: dict[str, str],
    supplement_dir: str | None = None,
    supplement_summary_dir: str | None = None,
) -> ReleaseIndex:
    if (supplement_dir is None) != (supplement_summary_dir is None):
        raise ResultError("inputs", "a supplement and its summary directory are given together")
    primary, secondary, input_bytes = _check(root, primary_dir, secondary_dir, inputs)
    record = parse_run_record(primary["run_record.json"])
    sec = _json(secondary[SECONDARY_RECORD], SECONDARY_RECORD)
    code = sec["code"]
    included = [
        IncludedArtifact(
            role=f"input:{role}", path=inputs[role], bytes=len(data), sha256=sha256_bytes(data)
        )
        for role, data in sorted(input_bytes.items())
    ]
    for label, directory, files in (
        ("r_primary", primary_dir, primary),
        ("r_secondary", secondary_dir, secondary),
    ):
        included += [
            IncludedArtifact(
                role=f"{label}:{name}",
                path=f"{directory}/{name}",
                bytes=len(data),
                sha256=sha256_bytes(data),
            )
            for name, data in files.items()
        ]
    supplement_id: str | None = None
    summary_id: str | None = None
    if supplement_dir is not None and supplement_summary_dir is not None:
        py, summary, supplement_id, summary_id = _check_supplement(
            root,
            supplement_dir,
            supplement_summary_dir,
            record.inputs.handoff_sha256,
            record.inputs.csv_sha256,
        )
        for label, directory, files in (
            ("supplement", supplement_dir, py),
            ("supplement_summary", supplement_summary_dir, summary),
        ):
            included += [
                IncludedArtifact(
                    role=f"{label}:{name}",
                    path=f"{directory}/{name}",
                    bytes=len(data),
                    sha256=sha256_bytes(data),
                )
                for name, data in files.items()
            ]
    return ReleaseIndex(
        supplement_sha256=supplement_id,
        supplement_summary_sha256=summary_id,
        handoff_sha256=record.inputs.handoff_sha256,
        csv_sha256=record.inputs.csv_sha256,
        primary_scientific_result_sha256=record.outputs.scientific_result_sha256,
        secondary_result_sha256=sec["secondary_result_sha256"],
        r_version_required=record.environment_intended.r_version_required,
        renv_lock_sha256=record.environment_intended.renv_lock_sha256,
        primary_code=IndexCode(
            git_commit=record.code.git_commit,
            dirty=record.code.dirty,
            script_sha256=record.code.script_sha256,
        ),
        secondary_code=IndexCode(
            git_commit=code["git_commit"], dirty=code["dirty"], script_sha256=code["script_sha256"]
        ),
        included=included,
        external=[
            ExternalResource(role=r, status="externally_acquired_or_restricted", reason=why)
            for r, why in EXTERNAL_RESOURCES.items()
        ],
    )


def serialize_index(index: ReleaseIndex) -> bytes:
    return canonical_json_bytes(index.model_dump(mode="json")) + b"\n"


def publish_release_index(index: ReleaseIndex, root: Path, relative: str) -> str:
    """Write the index under `root` without replacing anything; return its SHA-256."""
    data = serialize_index(index)
    target = root / relative
    publish_bytes(target, data, sha256_bytes(data))
    return sha256_bytes(data)


def read_release_index(
    root: Path,
    relative: str,
    expected_sha256: str,
    *,
    primary_dir: str,
    secondary_dir: str,
    inputs: dict[str, str],
    supplement_dir: str | None = None,
    supplement_summary_dir: str | None = None,
) -> ReleaseIndex:
    """Read an index by its trusted hash and re-establish every reference it makes."""
    data = _file(root, relative)
    if sha256_bytes(data) != expected_sha256:
        raise ResultError("hash", "the release index differs from the trusted SHA-256")
    index = ReleaseIndex.model_validate(_json(data, "release index"), strict=True)
    if serialize_index(index) != data:
        raise ResultError("canonical_form", "the release index is not in its canonical form")
    if (
        build_release_index(
            root,
            primary_dir=primary_dir,
            secondary_dir=secondary_dir,
            inputs=inputs,
            supplement_dir=supplement_dir,
            supplement_summary_dir=supplement_summary_dir,
        )
        != index
    ):
        raise ResultError("references", "the index no longer matches the files it cites")
    return index
