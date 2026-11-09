"""Python verification of a published R result directory, and the typed result record.

The R result directory is untrusted input. It is read through no-follow file descriptors under
the artifact root, the bytes that are parsed are the bytes that are hashed, and every statement
in R's run record is checked against what Python can establish independently: the file set and
hashes, the scientific result identity, the CSV contract, the binding to the Python preflight
and handoff, and small deterministic arithmetic (counts, rates, the paired risk difference)
recomputed from the Python-approved accounting.

Not done here, and stated in the record: the bootstrap, CR2/Satterthwaite and Holm values are
R's and are not recomputed; the R runtime and code identity are what R reported, not what
Python observed; source-run lineage and the production seed are not established; a hash that
the caller supplies is not an authentication credential. A passing verification describes the
bytes observed during it, not protection against later modification.

The record is deterministic for identical evidence and verifier identity (no nonce, no clock
reading of its own). Publication never replaces an artifact. The record file and its build
manifest are two files, so publication is not atomic as a pair; a reader must go through the
manifest, and file data is fsynced but directory entries are not claimed durable across a crash.
"""

import csv
import io
import json
import math
import os
import re
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, field_validator, model_validator

from biorag_repro.benchmark.models import Record
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_manifest,
    read_project_json,
    serialize_manifest,
    serialize_project_json,
)
from biorag_repro.evaluation.analysis_handoff import (
    EXECUTION_PROVENANCE_REQUIRED,
    AnalysisHandoff,
    OutcomeAccounting,
    OutcomeRule,
)
from biorag_repro.evaluation.analysis_preflight import VerifiedPreflight, read_preflight
from biorag_repro.generation.environment import GitState, check_utc, system_probes

SCHEMA_VERSION = "1"
TRANSFORMATION = "analysis_result_v1"
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

ResultName = Literal[
    "coprimary_results.csv",
    "paper_counts.csv",
    "state_counts.csv",
    "package_inventory.csv",
    "run_record.json",
]
RESULT_FILES: Final[tuple[ResultName, ...]] = (
    "coprimary_results.csv",
    "paper_counts.csv",
    "state_counts.csv",
    "package_inventory.csv",
    "run_record.json",
)
SCIENTIFIC_FILES: Final = RESULT_FILES[:3]
DATA_FILES: Final = RESULT_FILES[:4]
MAX_FILE_BYTES: Final = 16 * 1024 * 1024
REQUIRED_CLUBSANDWICH_VERSION: Final = "0.7.0"

# OpenBLAS picks its kernel from the CPU, and the last bit of the CR2 standard error depends on
# the kernel. The result hash is byte-compared, so R always runs with the kernel the synthetic
# golden was produced with and one thread. analysis/run_analysis.R repeats these values and a test
# compares them; R has to see them in its environment before it starts.
R_BLAS_PIN: Final = {"OPENBLAS_CORETYPE": "HASWELL", "OPENBLAS_NUM_THREADS": "1"}


def r_process_env() -> dict[str, str]:
    """The environment for every R subprocess: the caller's, with the BLAS kernel pinned."""
    return {**os.environ, **R_BLAS_PIN}


# R's column schemas, repeated here on purpose so that a change on either side is noticed: a test
# compares these with the declarations in analysis/R/results.R.
COPRIMARY_SCHEMA: Final = {
    "outcome": "chr",
    "n_eligible": "int",
    "n_complete": "int",
    "n_papers": "int",
    "summary_status": "chr",
    "rate_jats": "dbl",
    "rate_pdf": "dbl",
    "rd": "dbl",
    "rd_pp": "dbl",
    "equal_paper_rd": "dbl",
    "boot_status": "chr",
    "boot_n_resamples": "int",
    "boot_n_empty": "int",
    "boot_ci_lower": "dbl",
    "boot_ci_upper": "dbl",
    "test_status": "chr",
    "test_reason": "chr",
    "test_se": "dbl",
    "test_df": "dbl",
    "test_t": "dbl",
    "test_p": "dbl",
    "holm_status": "chr",
    "holm_p": "dbl",
    "bound_lower": "dbl",
    "bound_upper": "dbl",
    "bound_n_unknown_jats": "int",
    "bound_n_unknown_pdf": "int",
}
PAPER_COUNTS_SCHEMA: Final = {
    "outcome": "chr",
    "paper_id": "chr",
    "eligible": "int",
    "complete": "int",
}
STATE_COUNTS_SCHEMA: Final = {"outcome": "chr", "condition": "chr", "state": "chr", "n": "int"}
INVENTORY_SCHEMA: Final = {
    "package": "chr",
    "version": "chr",
    "priority": "chr",
    "library_role": "chr",
    "in_lock": "chr",
}
SCHEMAS: Final = {
    "coprimary_results.csv": COPRIMARY_SCHEMA,
    "paper_counts.csv": PAPER_COUNTS_SCHEMA,
    "state_counts.csv": STATE_COUNTS_SCHEMA,
    "package_inventory.csv": INVENTORY_SCHEMA,
}

# The only columns that may hold NA. Everywhere else a missing value is a malformed file.
NULLABLE: Final = {
    "coprimary_results.csv": {
        "rate_jats",
        "rate_pdf",
        "rd",
        "rd_pp",
        "equal_paper_rd",
        "boot_ci_lower",
        "boot_ci_upper",
        "test_reason",
        "test_se",
        "test_df",
        "test_t",
        "test_p",
        "holm_p",
    },
    "paper_counts.csv": set(),
    "state_counts.csv": set(),
    "package_inventory.csv": {"priority"},
}

# Rates and differences are quotients of small integers, rounded once. R and Python agree to
# the last bit in practice; 1e-12 leaves room for an alternative summation order while staying
# many orders of magnitude below the 1/n^2 spacing of distinct rationals at any realistic n.
ARITHMETIC_TOLERANCE: Final = 1e-12
TEST_REASONS: Final = (
    "no_complete_pairs",
    "single_cluster",
    "zero_variance",
    "zero_robust_variance",
    "nonfinite_result",
)

Status = Literal["verified_by_python", "reported_by_r", "not_established"]
Rerun = Literal[
    "not_performed", "byte_identical", "numerically_equal_not_byte_identical", "failed_to_reproduce"
]

CHECKS: Final[tuple[tuple[str, Status, str], ...]] = (
    (
        "r_result_files",
        "verified_by_python",
        (
            "exactly the five expected regular files, read without following links; "
            "the bytes parsed are the bytes hashed"
        ),
    ),
    (
        "r_run_record_bytes",
        "verified_by_python",
        "run_record.json bytes equal the SHA-256 supplied by the caller; this is not authentication",
    ),
    (
        "r_run_record_structure",
        "verified_by_python",
        "strict schema; the fixed synthetic-rehearsal claims; duplicate keys refused",
    ),
    (
        "r_output_hashes",
        "verified_by_python",
        "the four listed data files match their recorded size and SHA-256",
    ),
    (
        "scientific_result_identity",
        "verified_by_python",
        "scientific_result_sha256 rebuilt from handoff, CSV, seed and the three scientific files",
    ),
    (
        "r_csv_contract",
        "verified_by_python",
        "schemas, canonical bytes, NA semantics, row order, status vocabularies, value ranges",
    ),
    (
        "preflight_binding",
        "verified_by_python",
        (
            "cited preflight hash equals the stored bytes; the preflight, its handoff and table are "
            "re-verified; handoff and CSV hashes agree along the chain"
        ),
    ),
    (
        "complete_case_arithmetic",
        "verified_by_python",
        (
            "counts, per-paper counts, state counts, rates, item-weighted paired risk difference and "
            "missingness bounds recomputed from the Python accounting"
        ),
    ),
    (
        "inference_structure",
        "verified_by_python",
        "statuses, reasons, 10000 resamples, the two-outcome Holm family, NA only where undefined",
    ),
    (
        "statistical_values",
        "reported_by_r",
        (
            "bootstrap interval, CR2 standard error, Satterthwaite df, t, p and Holm p are R's and "
            "are not recomputed"
        ),
    ),
    (
        "r_environment_consistency",
        "verified_by_python",
        "recorded lock, package inventory and package versions agree with one another",
    ),
    (
        "r_runtime_and_code",
        "reported_by_r",
        (
            "R version, platform, commit, dirty state and script hashes are what R recorded; Python "
            "did not observe the runtime or read the historical source"
        ),
    ),
    (
        "source_run_lineage",
        "not_established",
        "a synthetic handoff has no source runs; none were traversed",
    ),
    (
        "seed_approval",
        "not_established",
        "the production primary_bootstrap seed is awaiting approval",
    ),
    (
        "record_authentication",
        "not_established",
        "no signature or key; a hash supplied by the caller is not a credential",
    ),
)

NOT_ESTABLISHED: Final = (
    "official source-run lineage",
    "approval of the production primary_bootstrap seed",
    "independent replication of the bootstrap, CR2, Satterthwaite or Holm computation",
    "that the R runtime and code were as recorded: R reported them and Python did not observe them",
    "authenticity: no signature is involved and a caller-supplied hash is not a credential",
    "that the files are unchanged after this verification: it describes the bytes observed",
    "any scientific authority: a synthetic rehearsal has none",
)


class ResultError(ValueError):
    """A result directory or result record fails verification; `stage` names the boundary."""

    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


# ---- reading the result directory without following links ---------------------------------


def _relative_parts(relative: str) -> list[str]:
    parts = relative.split("/") if isinstance(relative, str) else []
    if (
        not parts
        or "\\" in relative
        or "\0" in relative
        or any(p in ("", ".", "..") for p in parts)
    ):
        raise ResultError("result_location", "the result directory must be a plain relative path")
    return parts


def read_result_files(
    root: Path, relative: str, names: tuple[str, ...] | None = None, *, exact: bool = True
) -> dict[str, bytes]:
    """Read the files of a result directory located under `root` (the five R files by default).

    Every path component is opened with O_NOFOLLOW from a directory descriptor, so a link
    anywhere below the root is refused rather than followed. Only regular files are read.
    """
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as error:
        raise ResultError(
            "result_location", f"the artifact root cannot be opened: {error}"
        ) from error
    try:
        for part in _relative_parts(relative):
            try:
                inner = os.open(part, flags | os.O_DIRECTORY, dir_fd=fd)
            except OSError as error:
                raise ResultError(
                    "result_location",
                    f"result directory is missing, not a directory or a link: {part}",
                ) from error
            os.close(fd)
            fd = inner
        order = RESULT_FILES if names is None else names
        expected = set(order)
        present = set(os.listdir(fd))
        if (present != expected) if exact else not expected <= present:
            raise ResultError(
                "file_set",
                f"missing {sorted(expected - present)}, unexpected {sorted(present - expected)}",
            )
        return {name: _read_one(fd, name) for name in order}
    finally:
        os.close(fd)


def _read_one(directory_fd: int, name: str) -> bytes:
    try:
        if not stat.S_ISREG(os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode):
            raise ResultError("file_set", f"{name} is not a regular file")
        descriptor = os.open(
            name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
        )
    except OSError as error:
        raise ResultError(
            "file_set", f"{name} cannot be opened as a regular file: {error}"
        ) from error
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            raise ResultError("file_set", f"{name} is not a regular file of acceptable size")
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ResultError("file_set", f"{name} is larger than {MAX_FILE_BYTES} bytes")
    if not data:
        raise ResultError("file_set", f"{name} is empty")
    return data


# ---- R's CSV format, strictly -----------------------------------------------------------


def _format_cell(value: Any, kind: str) -> str:
    if value is None:
        return "NA"
    if kind == "chr":
        if value in ("", "NA") or re.search(r"[\r\n]", value):
            raise ResultError("csv", "a text cell cannot be written unambiguously")
        if re.search(r'[,"]', value):
            return '"' + str(value).replace('"', '""') + '"'
        return str(value)
    if kind == "int":
        return str(value)
    return "0" if value == 0 else format(value, ".17g")


def write_csv(schema: dict[str, str], rows: list[dict[str, Any]]) -> bytes:
    lines = [",".join(schema)]
    lines += [",".join(_format_cell(r[c], k) for c, k in schema.items()) for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse_csv_with(
    schema: dict[str, str], name: str, data: bytes, nullable: set[str] | None = None
) -> list[dict[str, Any]]:
    """Parse a file written by R's csv_bytes; the same bytes must be reproduced on writing back.

    `nullable` names the columns that may be NA; by default every column may.
    """

    def bad(why: str) -> ResultError:
        return ResultError("csv", f"{name}: {why}")

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise bad("not valid UTF-8") from error
    if not text.endswith("\n") or "\r" in text or text.startswith("\ufeff"):
        raise bad("not UTF-8 text with LF line endings and no byte-order mark")
    try:
        cells = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise bad("cannot be parsed as CSV") from error
    if not cells or cells[0] != list(schema):
        raise bad("columns differ from the schema")
    rows: list[dict[str, Any]] = []
    for line in cells[1:]:
        if len(line) != len(schema):
            raise bad("a row has the wrong number of cells")
        row: dict[str, Any] = {}
        for (column, kind), cell in zip(schema.items(), line, strict=True):
            if cell == "NA":
                if nullable is not None and column not in nullable:
                    raise bad(f"column {column} is never missing in a result file")
                row[column] = None
            elif kind == "chr":
                row[column] = cell
            elif kind == "int":
                if not re.fullmatch(r"-?[0-9]+", cell) or abs(int(cell)) > 2**31 - 1:
                    raise bad(f"column {column} is not a 32-bit integer")
                row[column] = int(cell)
            else:
                try:
                    number = float(cell)
                except ValueError as error:
                    raise bad(f"column {column} is not numeric") from error
                if not math.isfinite(number):
                    raise bad(f"column {column} is not finite")
                row[column] = number
        rows.append(row)
    if write_csv(schema, rows) != data:
        raise bad("is not in the canonical form R writes")
    return rows


def parse_csv(name: str, data: bytes) -> list[dict[str, Any]]:
    return parse_csv_with(SCHEMAS[name], name, data, NULLABLE[name])


# ---- what R's run record says ----------------------------------------------------------------


class RVerification(Record):
    input_origin: Literal["synthetic_rehearsal"]
    r_byte_verification: Literal[
        "handoff_and_csv_bytes_matched_supplied_sha256_and_reading_contract"
    ]
    python_preflight: Literal["cited_bytes_hashed_not_interpreted"]
    python_preflight_scope: Literal[
        "bytes_hashed_by_r_only_contents_not_read_python_verification_not_repeated"
    ]
    lineage_verified_by_r: Literal[False]
    official_source_runs: Literal["not_applicable_synthetic_rehearsal"]
    statistical_execution: Literal["completed"]
    output_integrity: Literal["hashes_recorded_in_this_record_only"]
    record_authentication: Literal["none_without_external_record_sha256"]


class RInputs(Record):
    handoff_sha256: Sha256
    csv_sha256: Sha256
    input_origin: Literal["synthetic_rehearsal"]
    handoff_kind: Literal["analysis_handoff"]
    handoff_schema_version: Literal["1"]
    analysis_spec_version: Literal["analysis_spec_v1"]
    python_preflight_sha256: Sha256
    python_preflight_bytes: int = Field(ge=1)


class RSeed(Record):
    value: int = Field(ge=-(2**31) + 1, le=2**31 - 1)
    label: Literal["synthetic_rehearsal"]
    spec_name: Literal["primary_bootstrap"]
    registry: Literal["not_registered"]
    status: Literal["synthetic_rehearsal_not_scientific"]
    production_seed: Literal["awaiting_approval"]
    rng: dict[str, str]
    quantile_type: Literal[7]
    n_resamples: Literal[10000]

    @field_validator("rng")
    @classmethod
    def frozen_generator(cls, value: dict[str, str]) -> dict[str, str]:
        if value != {
            "kind": "Mersenne-Twister",
            "normal.kind": "Inversion",
            "sample.kind": "Rejection",
        }:
            raise ValueError("the random number generator is not the frozen one")
        return value


class REnvIntended(Record):
    r_version_required: Literal["4.6.1"]
    renv_lock_sha256: Sha256
    lock_packages: dict[str, str]


class REnvActual(Record):
    r_version: str
    platform: str
    os: str | None
    blas: str | None
    lapack: str | None
    lapack_version: str | None
    library_roles: list[str]
    renv_synchronized: Literal[True]
    installed_package_inventory_sha256: Sha256
    package_versions_used: dict[str, str]


class RCode(Record):
    status: Literal["ok"]
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: bool
    tracked_modified: list[str]
    tracked_diff_sha256: Sha256 | None
    untracked_source: dict[str, Sha256] | list[str]
    script_sha256: dict[str, Sha256]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if isinstance(self.untracked_source, list) and self.untracked_source:
            raise ValueError("untracked source must be a mapping of file to hash")
        changed = bool(self.tracked_modified) or bool(self.untracked_source)
        if self.dirty != changed:
            raise ValueError("dirty flag disagrees with the changed files")
        if (self.tracked_diff_sha256 is not None) != bool(self.tracked_modified):
            raise ValueError("a diff hash is present exactly when tracked files differ")
        if not self.script_sha256:
            raise ValueError("no script hashes recorded")
        return self


class RExecution(Record):
    run_id: str = Field(min_length=1)
    started_utc: str
    finished_utc: str
    hostname: str

    _utc = field_validator("started_utc", "finished_utc")(check_utc)


class RFile(Record):
    bytes: int = Field(ge=1)
    sha256: Sha256


class ROutputs(Record):
    files: dict[str, RFile]
    scientific_result_sha256: Sha256


class RProvenance(Record):
    required: list[str]


class RRunRecord(Record):
    kind: Literal["r_run_record"]
    schema_version: Literal["1"]
    purpose: Literal["synthetic_rehearsal"]
    scientific_authority: Literal["none"]
    verification: RVerification
    inputs: RInputs
    seed: RSeed
    environment_intended: REnvIntended
    environment_actual: REnvActual
    code: RCode
    execution: RExecution
    outputs: ROutputs
    provenance: RProvenance


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate JSON key")
    return dict(pairs)


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON value {value}")


def parse_run_record(data: bytes) -> RRunRecord:
    try:
        parsed = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
        return RRunRecord.model_validate(parsed, strict=True)
    except ValueError as error:  # JSON errors, duplicate keys, bad UTF-8 and schema errors
        raise ResultError(
            "record_structure", f"run_record.json is not acceptable: {error}"
        ) from error


# ---- identity of the scientific result ---------------------------------------------------------


def scientific_result_id(
    handoff_sha256: str, csv_sha256: str, seed: int, file_sha256: dict[str, str]
) -> str:
    """The payload R hashes: inputs, seed and the hash of each scientific file, as canonical JSON."""
    payload = {
        "csv_sha256": csv_sha256,
        "files": [{"name": n, "sha256": file_sha256[n]} for n in sorted(file_sha256)],
        "handoff_sha256": handoff_sha256,
        "seed": seed,
    }
    return sha256_bytes(canonical_json_bytes(payload))


# ---- the record Python publishes ---------------------------------------------------------------


class ResultFile(Record):
    name: ResultName
    bytes: int = Field(ge=1)
    sha256: Sha256


class ResultMethod(Record):
    bootstrap_method: str
    bootstrap_resamples: Literal[10000]
    interval_percent: int
    variance: str
    test: str
    package: str
    package_version: str
    multiplicity: str
    multiplicity_family: list[str]


class RecordedByR(Record):
    """What R asserted. Python read it from the run record and did not observe it."""

    seed: RSeed
    environment_intended: REnvIntended
    environment_actual: REnvActual
    code: RCode
    run_id: str
    started_utc: str
    finished_utc: str


class ArithmeticCheck(Record):
    outcome: str
    n_eligible: int = Field(ge=0)
    n_complete: int = Field(ge=0)
    n_papers: int = Field(ge=0)
    n_jats_success: int = Field(ge=0)
    n_pdf_success: int = Field(ge=0)
    n_unknown_jats: int = Field(ge=0)
    n_unknown_pdf: int = Field(ge=0)


class InferenceStatus(Record):
    outcome: str
    summary_status: Literal["ok", "no_complete_pairs"]
    boot_status: Literal["ok", "single_cluster_degenerate", "no_complete_pairs"]
    test_status: Literal["ok", "inference_undefined"]
    test_reason: str | None
    holm_status: Literal["ok", "holm_undefined"]


class ResultCheck(Record):
    name: str
    status: Status
    scope: str


class ResultVerifier(Record):
    tool: Literal["analysis_result_v1"]
    settings_sha256: Sha256
    module_sha256: Sha256
    git: GitState


class AnalysisResultRecord(Record):
    schema_version: Literal["1"] = "1"
    kind: Literal["analysis_result_record"] = "analysis_result_record"
    purpose: Literal["synthetic_rehearsal"]
    input_origin: Literal["synthetic_rehearsal"]
    scientific_authority: Literal["none"]
    preflight_build: ArtifactRef
    preflight: ArtifactRef
    handoff_build: ArtifactRef
    handoff: ArtifactRef
    table_build: ArtifactRef
    csv: ArtifactRef
    result_dir: str  # relative to the artifact root; R's files stay where R wrote them
    r_files: list[ResultFile] = Field(min_length=5, max_length=5)
    run_record_sha256: Sha256
    scientific_result_sha256: Sha256
    applied_bootstrap_seed: int
    r_dependency_lock_sha256: Sha256
    installed_package_inventory_sha256: Sha256
    method: ResultMethod
    arithmetic: list[ArithmeticCheck] = Field(min_length=2, max_length=2)
    inference: list[InferenceStatus] = Field(min_length=2, max_length=2)
    recorded_by_r: RecordedByR
    rerun: Rerun
    checks: list[ResultCheck]
    not_established: list[str]
    verifier: ResultVerifier

    @field_validator("result_dir")
    @classmethod
    def plain_relative(cls, value: str) -> str:
        _relative_parts(value)
        return value

    @model_validator(mode="after")
    def claims_are_the_fixed_set(self) -> Self:
        if [(c.name, c.status, c.scope) for c in self.checks] != list(CHECKS):
            raise ValueError("checks differ from the claims this schema version can make")
        if self.not_established != list(NOT_ESTABLISHED):
            raise ValueError("the statement of what is not established was changed")
        if [f.name for f in self.r_files] != list(RESULT_FILES):
            raise ValueError("r_files must list the five result files in order")
        by_name = {f.name: f.sha256 for f in self.r_files}
        r = self.recorded_by_r
        if (
            by_name["run_record.json"] != self.run_record_sha256
            or by_name["package_inventory.csv"] != self.installed_package_inventory_sha256
            or r.seed.value != self.applied_bootstrap_seed
            or r.environment_intended.renv_lock_sha256 != self.r_dependency_lock_sha256
            or r.environment_actual.installed_package_inventory_sha256
            != self.installed_package_inventory_sha256
        ):
            raise ValueError("a bound identity disagrees with the file or record it is taken from")
        return self


def settings_sha256() -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "checks": [list(c) for c in CHECKS],
                "arithmetic_tolerance": ARITHMETIC_TOLERANCE,
            }
        )
    )


# ---- verification ------------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedRResult:
    claims: dict[str, Any]  # every AnalysisResultRecord field that is derived from evidence
    run: RRunRecord
    preflight: VerifiedPreflight


def _rule(handoff: AnalysisHandoff, name: str) -> OutcomeRule:
    return next(r for r in handoff.analysis_spec.outcomes if r.name == name)


def _close(actual: float | None, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return abs(actual - expected) <= ARITHMETIC_TOLERANCE


def _check_orders(tables: dict[str, list[dict[str, Any]]], family: list[str]) -> None:
    def bad(why: str) -> ResultError:
        return ResultError("csv", why)

    if [r["outcome"] for r in tables["coprimary_results.csv"]] != family:
        raise bad("coprimary_results.csv must hold the two co-primary outcomes in the frozen order")
    for name, key in (
        (
            "paper_counts.csv",
            lambda r: (family.index(r["outcome"]) if r["outcome"] in family else -1, r["paper_id"]),
        ),
        (
            "state_counts.csv",
            lambda r: (
                family.index(r["outcome"]) if r["outcome"] in family else -1,
                f"{r['condition']}|{r['state']}",
            ),
        ),
    ):
        keys = [key(r) for r in tables[name]]
        if any(k[0] < 0 for k in keys) or any(a >= b for a, b in pairwise(keys)):
            raise bad(
                f"{name} rows are not in the stable order (outcome, then key) without duplicates"
            )
    inventory = [r["package"] for r in tables["package_inventory.csv"]]
    if any(a >= b for a, b in pairwise(inventory)) or not inventory:
        raise bad("package_inventory.csv must be sorted by package without duplicates")
    for row in tables["package_inventory.csv"]:
        if (
            row["priority"] not in (None, "base", "recommended")
            or row["library_role"] not in ("project", "r_base", "r_other")
            or row["in_lock"] not in ("yes", "no")
        ):
            raise bad("package_inventory.csv holds a value outside its vocabulary")


def _check_ranges(coprimary: list[dict[str, Any]]) -> None:
    def inside(value: float | None, low: float, high: float) -> bool:
        return value is None or low <= value <= high

    for r in coprimary:
        if (
            r["summary_status"] not in ("ok", "no_complete_pairs")
            or r["boot_status"] not in ("ok", "single_cluster_degenerate", "no_complete_pairs")
            or r["test_status"] not in ("ok", "inference_undefined")
            or r["holm_status"] not in ("ok", "holm_undefined")
            or (r["test_reason"] is not None and r["test_reason"] not in TEST_REASONS)
        ):
            raise ResultError("csv", "a status or reason is outside the known vocabulary")
        if not (
            all(inside(r[c], 0, 1) for c in ("rate_jats", "rate_pdf", "test_p", "holm_p"))
            and all(
                inside(r[c], -1, 1)
                for c in ("rd", "boot_ci_lower", "boot_ci_upper", "bound_lower", "bound_upper")
            )
            and inside(r["rd_pp"], -100, 100)
            and inside(r["equal_paper_rd"], -1, 1)
            and all(
                r[c] >= 0
                for c in (
                    "n_eligible",
                    "n_complete",
                    "n_papers",
                    "boot_n_resamples",
                    "boot_n_empty",
                )
            )
            and r["n_complete"] <= r["n_eligible"]
            and r["n_papers"] <= r["n_complete"]
        ):
            raise ResultError("csv", "a value lies outside its range")


def _outcome_arithmetic(
    handoff: AnalysisHandoff, accounting: OutcomeAccounting, row: dict[str, Any]
) -> ArithmeticCheck:
    """Recompute one outcome's counts, rates and bounds from the Python accounting and compare."""
    rule = _rule(handoff, accounting.outcome)
    evaluable = {rule.success, rule.failure}
    pairs: list[tuple[str, int, int]] = []
    unknown_jats = unknown_pdf = 0
    upper = lower = 0
    for item in accounting.eligible_items:
        if item.jats is None or item.pdf_docling is None:
            raise ResultError(
                "arithmetic", f"eligible item {item.item_id} lacks a planned representation"
            )
        j, p = (
            None if c.state not in evaluable else int(c.state == rule.success)
            for c in (item.jats, item.pdf_docling)
        )
        unknown_jats += j is None
        unknown_pdf += p is None
        upper += (1 if j is None else j) - (0 if p is None else p)
        lower += (0 if j is None else j) - (1 if p is None else p)
        if j is not None and p is not None:
            pairs.append((item.paper_id, j, p))
    n_eligible, n = len(accounting.eligible_items), len(pairs)
    jats_yes, pdf_yes = sum(j for _, j, _ in pairs), sum(p for _, _, p in pairs)
    papers = sorted({paper for paper, _, _ in pairs})
    expected: dict[str, Any] = {
        "n_eligible": n_eligible,
        "n_complete": n,
        "n_papers": len(papers),
        "summary_status": "ok" if n else "no_complete_pairs",
        "bound_n_unknown_jats": unknown_jats,
        "bound_n_unknown_pdf": unknown_pdf,
    }
    floats: dict[str, float | None] = {
        "rate_jats": jats_yes / n if n else None,
        "rate_pdf": pdf_yes / n if n else None,
        "rd": (jats_yes - pdf_yes) / n if n else None,
        "rd_pp": 100 * (jats_yes - pdf_yes) / n if n else None,
        "equal_paper_rd": (
            sum(
                sum(j - p for q, j, p in pairs if q == paper) / sum(q == paper for q, _, _ in pairs)
                for paper in papers
            )
            / len(papers)
            if n
            else None
        ),
        "bound_lower": lower / n_eligible,
        "bound_upper": upper / n_eligible,
    }
    for field, value in expected.items():
        if row[field] != value:
            raise ResultError(
                "arithmetic",
                f"{accounting.outcome}: {field} is {row[field]!r}, recomputed {value!r}",
            )
    for field, value in floats.items():
        if not _close(row[field], value):
            raise ResultError(
                "arithmetic",
                f"{accounting.outcome}: {field} is {row[field]!r}, recomputed {value!r}",
            )
    _inference_structure(row, n, len(papers), {j - p for _, j, p in pairs})
    return ArithmeticCheck(
        outcome=accounting.outcome,
        n_eligible=n_eligible,
        n_complete=n,
        n_papers=len(papers),
        n_jats_success=jats_yes,
        n_pdf_success=pdf_yes,
        n_unknown_jats=unknown_jats,
        n_unknown_pdf=unknown_pdf,
    )


def _inference_structure(
    row: dict[str, Any], n: int, n_papers: int, distinct_differences: set[int]
) -> None:
    """Statuses and presence of values only. The statistical values themselves are R's."""
    name = row["outcome"]

    def bad(why: str) -> ResultError:
        return ResultError("inference", f"{name}: {why}")

    boot = "no_complete_pairs" if n == 0 else "single_cluster_degenerate" if n_papers == 1 else "ok"
    if row["boot_status"] != boot:
        raise bad(f"bootstrap status is {row['boot_status']!r}, expected {boot!r}")
    if row["boot_n_resamples"] != (0 if n == 0 else 10000):
        raise bad("the bootstrap did not use exactly 10000 resamples")
    if row["boot_n_empty"] > row["boot_n_resamples"]:
        raise bad("more empty resamples than resamples")
    interval = (row["boot_ci_lower"], row["boot_ci_upper"])
    if (n == 0) != (interval[0] is None and interval[1] is None) or (interval[0] is None) != (
        interval[1] is None
    ):
        raise bad("the bootstrap interval is missing or present contrary to the complete cases")
    if interval[0] is not None and interval[0] > interval[1]:
        raise bad("the bootstrap interval is reversed")
    forced = (
        "no_complete_pairs"
        if n == 0
        else "single_cluster"
        if n_papers < 2
        else "zero_variance"
        if len(distinct_differences) == 1
        else None
    )
    undefined = row["test_status"] == "inference_undefined"
    values = [row[c] for c in ("test_se", "test_df", "test_t", "test_p")]
    if forced is not None and row["test_reason"] != forced:
        raise bad(f"the test must be undefined with reason {forced!r}")
    if undefined != (row["test_reason"] is not None) or undefined != all(v is None for v in values):
        raise bad("test status, reason and values disagree")
    if (
        undefined
        and forced is None
        and row["test_reason"] not in ("zero_robust_variance", "nonfinite_result")
    ):
        raise bad("an undefined test whose cause is not structural needs a numerical reason")
    if not undefined:
        if row["test_se"] <= 0 or row["test_df"] <= 0:
            raise bad("a defined test needs a positive standard error and degrees of freedom")
        if abs(row["test_t"] * row["test_se"] - row["rd"]) > 1e-9:
            raise bad("t times its standard error does not reproduce the risk difference")


def _check_holm(coprimary: list[dict[str, Any]]) -> None:
    undefined = any(r["test_p"] is None for r in coprimary)
    for r in coprimary:
        if (
            r["holm_status"] != ("holm_undefined" if undefined else "ok")
            or (r["holm_p"] is None) != undefined
        ):
            raise ResultError(
                "inference", f"{r['outcome']}: the Holm family status disagrees with the p-values"
            )
        if not undefined and r["holm_p"] < r["test_p"]:
            raise ResultError(
                "inference", f"{r['outcome']}: an adjusted p-value is below its raw p-value"
            )


def _check_accounting_tables(
    handoff: AnalysisHandoff, tables: dict[str, list[dict[str, Any]]]
) -> None:
    papers = [
        {
            "outcome": a.outcome,
            "paper_id": p.paper_id,
            "eligible": p.eligible,
            "complete": p.complete,
        }
        for a in handoff.accounting
        for p in a.papers
    ]
    states = [
        {
            "outcome": a.outcome,
            "condition": key.split("|", 1)[0],
            "state": key.split("|", 1)[1],
            "n": n,
        }
        for a in handoff.accounting
        for key, n in sorted(a.state_counts.items())
    ]
    if tables["paper_counts.csv"] != papers:
        raise ResultError(
            "arithmetic", "paper_counts.csv differs from the per-paper counts of the accounting"
        )
    if tables["state_counts.csv"] != states:
        raise ResultError(
            "arithmetic", "state_counts.csv differs from the state counts of the accounting"
        )


def _cr2_test_ran(coprimary: list[dict[str, Any]]) -> bool:
    """True when at least one outcome had a complete-case test, so R must have called clubSandwich."""
    return any(r["test_status"] != "inference_undefined" for r in coprimary)


def _check_environment(
    run: RRunRecord,
    inventory: list[dict[str, Any]],
    handoff: AnalysisHandoff,
    *,
    test_ran: bool,
) -> str:
    """Internal consistency of the recorded lock, inventory and versions; returns the method package version.

    The method package must be locked and installed at the required version. It must also appear
    among the packages R used, unless no outcome had a complete-case test to run: R loads it
    only when it calls it, so a run whose tests are all undefined legitimately never loads it.
    """

    def bad(why: str) -> ResultError:
        return ResultError("environment", why)

    installed = {r["package"]: r["version"] for r in inventory}
    locked = run.environment_intended.lock_packages
    flagged = {r["package"] for r in inventory if r["in_lock"] == "yes"}
    if set(locked) != flagged:
        raise bad("the packages flagged in_lock differ from the locked packages")
    for package, version in locked.items():
        if installed.get(package) != version:
            raise bad(
                f"{package}: the inventory has {installed.get(package)}, the lock has {version}"
            )
    if any(r["library_role"] == "project" and r["in_lock"] != "yes" for r in inventory):
        raise bad("the project library holds a package that is not locked")
    for package, version in run.environment_actual.package_versions_used.items():
        if installed.get(package) != version:
            raise bad(
                f"{package} was used at {version} but the inventory has {installed.get(package)}"
            )
    if not run.environment_actual.r_version.startswith("R version 4.6.1 "):
        raise bad("the recorded R version is not the required 4.6.1")
    method_version = installed.get(handoff.analysis_spec.package)
    used = run.environment_actual.package_versions_used.get(handoff.analysis_spec.package)
    if method_version != REQUIRED_CLUBSANDWICH_VERSION or (test_ran and used != method_version):
        raise bad(
            f"{handoff.analysis_spec.package} {REQUIRED_CLUBSANDWICH_VERSION} must be locked, installed and used"
        )
    return REQUIRED_CLUBSANDWICH_VERSION


def _bind_preflight(
    root: Path, preflight_build: ArtifactRef, run: RRunRecord
) -> tuple[VerifiedPreflight, ArtifactRef]:
    def bad(why: str) -> ResultError:
        return ResultError("preflight", why)

    try:
        checked = read_preflight(preflight_build, root)
        stored = (
            read_manifest(root / preflight_build.logical_ref, preflight_build.sha256)
            .outputs[0]
            .artifact
        )
        size = len((root / stored.logical_ref).read_bytes())
    except (ValueError, OSError) as error:
        raise bad(f"the preflight does not verify: {error}") from error
    pre = checked.preflight
    if (
        run.inputs.python_preflight_sha256 != stored.sha256
        or run.inputs.python_preflight_bytes != size
    ):
        raise bad("the preflight cited by R is not the preflight stored here")
    if run.inputs.handoff_sha256 != pre.handoff.sha256 or run.inputs.csv_sha256 != pre.csv.sha256:
        raise bad("the handoff or table hash cited by R is not the one the preflight verified")
    return checked, stored


def verify_r_result(
    root: Path, result_dir: str, preflight_build: ArtifactRef, run_record_sha256: str
) -> VerifiedRResult:
    """Verify a published R result directory against the Python preflight it cites.

    `run_record_sha256` is the hash R printed for run_record.json, carried here by the caller.
    It ties the record to a value given from outside this directory; it is not authentication.
    """
    files = read_result_files(root, result_dir)
    hashes = {name: sha256_bytes(data) for name, data in files.items()}
    if hashes["run_record.json"] != run_record_sha256:
        raise ResultError(
            "record_hash",
            f"run_record.json has sha256 {hashes['run_record.json']}, the supplied value is {run_record_sha256}",
        )
    run = parse_run_record(files["run_record.json"])
    if set(run.outputs.files) != set(DATA_FILES):
        raise ResultError("output_hashes", "the record must list exactly the four data files")
    for name in DATA_FILES:
        listed = run.outputs.files[name]
        if listed.sha256 != hashes[name] or listed.bytes != len(files[name]):
            raise ResultError(
                "output_hashes", f"{name} differs from the size and hash recorded for it"
            )
    if run.environment_actual.installed_package_inventory_sha256 != hashes["package_inventory.csv"]:
        raise ResultError(
            "output_hashes", "the recorded inventory hash differs from package_inventory.csv"
        )
    identity = scientific_result_id(
        run.inputs.handoff_sha256,
        run.inputs.csv_sha256,
        run.seed.value,
        {n: hashes[n] for n in SCIENTIFIC_FILES},
    )
    if identity != run.outputs.scientific_result_sha256:
        raise ResultError(
            "result_identity", "scientific_result_sha256 does not match the files, inputs and seed"
        )

    if sorted(run.provenance.required) != sorted(EXECUTION_PROVENANCE_REQUIRED):
        raise ResultError(
            "record_structure", "the provenance checklist is not the one the handoff requires"
        )
    tables: dict[str, list[dict[str, Any]]] = {n: parse_csv(n, files[n]) for n in DATA_FILES}
    checked, preflight_file = _bind_preflight(root, preflight_build, run)
    handoff = checked.handoff.handoff
    family = list(handoff.analysis_spec.multiplicity_family)
    _check_orders(tables, family)
    coprimary = tables["coprimary_results.csv"]
    _check_ranges(coprimary)
    _check_accounting_tables(handoff, tables)
    arithmetic = [
        _outcome_arithmetic(
            handoff, next(a for a in handoff.accounting if a.outcome == r["outcome"]), r
        )
        for r in coprimary
    ]
    _check_holm(coprimary)
    package_version = _check_environment(
        run,
        tables["package_inventory.csv"],
        handoff,
        test_ran=_cr2_test_ran(coprimary),
    )

    spec = handoff.analysis_spec
    pre = checked.preflight
    claims: dict[str, Any] = {
        "purpose": "synthetic_rehearsal",
        "input_origin": pre.input_origin,
        "scientific_authority": "none",
        "preflight_build": preflight_build,
        "preflight": preflight_file,
        "handoff_build": pre.handoff_build,
        "handoff": pre.handoff,
        "table_build": pre.table_build,
        "csv": pre.csv,
        "result_dir": result_dir,
        "r_files": [
            ResultFile(name=n, bytes=len(files[n]), sha256=hashes[n]) for n in RESULT_FILES
        ],
        "run_record_sha256": run_record_sha256,
        "scientific_result_sha256": identity,
        "applied_bootstrap_seed": run.seed.value,
        "r_dependency_lock_sha256": run.environment_intended.renv_lock_sha256,
        "installed_package_inventory_sha256": hashes["package_inventory.csv"],
        "method": ResultMethod(
            bootstrap_method=spec.bootstrap_method,
            bootstrap_resamples=run.seed.n_resamples,
            interval_percent=spec.interval_percent,
            variance=spec.variance,
            test=spec.test,
            package=spec.package,
            package_version=package_version,
            multiplicity=spec.multiplicity,
            multiplicity_family=family,
        ),
        "arithmetic": arithmetic,
        "inference": [
            InferenceStatus(
                outcome=r["outcome"],
                summary_status=r["summary_status"],
                boot_status=r["boot_status"],
                test_status=r["test_status"],
                test_reason=r["test_reason"],
                holm_status=r["holm_status"],
            )
            for r in coprimary
        ],
        "recorded_by_r": RecordedByR(
            seed=run.seed,
            environment_intended=run.environment_intended,
            environment_actual=run.environment_actual,
            code=run.code,
            run_id=run.execution.run_id,
            started_utc=run.execution.started_utc,
            finished_utc=run.execution.finished_utc,
        ),
    }
    return VerifiedRResult(claims, run, checked)


# ---- optional diagnostics ---------------------------------------------------------------------


def compare_current_environment(record: AnalysisResultRecord, repository: Path) -> dict[str, str]:
    """Compare what R recorded with today's checkout. Informational: it never fails a record.

    A historical result stays valid when the repository has moved on. The installed R packages
    are not compared because that needs R.
    """
    code = record.recorded_by_r.code
    out = {}
    git = system_probes(repository).git()
    out["git_commit"] = (
        "unavailable"
        if git.commit is None
        else "same"
        if git.commit == code.git_commit
        else "different"
    )
    lock = repository / "analysis" / "renv.lock"
    out["renv_lock"] = (
        "unavailable"
        if not lock.is_file()
        else "same"
        if sha256_file(lock) == record.r_dependency_lock_sha256
        else "different"
    )
    analysis = repository / "analysis"
    current = {
        p.relative_to(analysis).as_posix(): sha256_file(p)
        for p in [*sorted((analysis / "R").glob("*.R")), analysis / "run_analysis.R"]
        if p.is_file()
    }
    out["r_scripts"] = (
        "unavailable" if not current else "same" if current == code.script_sha256 else "different"
    )
    return out


def rerun_in_r(
    verified: VerifiedRResult,
    root: Path,
    analysis_dir: Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Rerun:
    """Ask R to recompute the result from the original inputs (`verify --rerun`).

    This repeats R's own computation; it is not an independent implementation of the method.
    A failure of R itself (not R's explicit finding of a difference) is an error, not a result.
    """
    claims = verified.claims
    handoff, csv_ref = claims["handoff"], claims["csv"]
    command = [
        "Rscript",
        "run_analysis.R",
        "verify",
        "--result",
        str(root / claims["result_dir"]),
        "--record-sha256",
        claims["run_record_sha256"],
        "--rerun",
        "--handoff",
        str(root / handoff.logical_ref),
        "--handoff-sha256",
        handoff.sha256,
        "--csv",
        str(root / csv_ref.logical_ref),
    ]
    try:
        done = run(
            command,
            cwd=analysis_dir,
            capture_output=True,
            text=True,
            timeout=1800,
            env=r_process_env(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ResultError("rerun", f"R could not be run: {error}") from error
    found = re.search(
        r"^re-execution: (byte_identical|numerically_equal_not_byte_identical)$",
        done.stdout,
        re.MULTILINE,
    )
    if done.returncode == 0 and found:
        return found[1]  # type: ignore[return-value]
    if re.search(r"re-execution differs", done.stderr):
        return "failed_to_reproduce"
    raise ResultError(
        "rerun", f"R failed without a reproduction verdict: {done.stderr.strip()[-300:]}"
    )


# ---- publication and read-back ---------------------------------------------------------------


@dataclass(frozen=True)
class PublishedResult:
    ref: ArtifactRef
    build: ArtifactRef
    created: bool  # False when identical bytes were already published


def _inputs(record: AnalysisResultRecord) -> list[BuildInput]:
    return [
        BuildInput(role="analysis_preflight_build", artifact=record.preflight_build),
        BuildInput(role="analysis_preflight", artifact=record.preflight),
    ]


def _publish_once(
    target: Path, data: bytes, digest: str, validate: Callable[[Path], object]
) -> bool:
    """Publish without replacing. Identical existing bytes are accepted; different bytes are a conflict."""
    try:
        publish_bytes(target, data, digest, validate_temp=validate)
        return True
    except FileExistsError:
        if target.is_symlink() or not target.is_file() or target.read_bytes() != data:
            raise ResultError(
                "publication",
                f"{target.name} already exists with different content; nothing replaced",
            ) from None
        return False


def _store(record: AnalysisResultRecord, root: Path) -> PublishedResult:
    data = serialize_project_json(record, AnalysisResultRecord)
    digest = sha256_bytes(data)
    inputs = _inputs(record)
    artifact_id = derived_artifact_id(
        artifact_kind="analysis_result_record",
        inputs=[ArtifactInput(i.role, i.artifact.artifact_id, i.artifact.sha256) for i in inputs],
        transformation_id=TRANSFORMATION,
        transformation_settings_sha256=settings_sha256(),
        output_sha256=digest,
        output_format="json",
        serialization_version="1",
    )
    ref = ArtifactRef(
        artifact_id=artifact_id, logical_ref=f"analysis/result/{artifact_id}.json", sha256=digest
    )
    manifest = BuildManifest(
        schema_version="1",
        inputs=inputs,
        transformation_id=TRANSFORMATION,
        transformation_settings_sha256=settings_sha256(),
        outputs=[
            DerivedOutput(
                artifact=ref,
                artifact_kind="analysis_result_record",
                output_format="json",
                serialization_version="1",
                identity_version="1",
            )
        ],
        status="complete",
    )
    target = root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    created = _publish_once(
        target, data, digest, lambda p: read_project_json(p, digest, AnalysisResultRecord)
    )
    build_path = f"analysis/result/{sha256_bytes(canonical_json_bytes([artifact_id]))}.build.json"
    manifest_bytes = serialize_manifest(manifest)
    manifest_digest = sha256_bytes(manifest_bytes)
    try:
        publish_manifest(manifest, root / build_path, {ref.logical_ref: target})
        created = True
    except FileExistsError:
        existing = root / build_path
        if (
            existing.is_symlink()
            or not existing.is_file()
            or existing.read_bytes() != manifest_bytes
        ):
            raise ResultError(
                "publication", "a different build manifest already exists; nothing replaced"
            ) from None
    return PublishedResult(
        ref,
        ArtifactRef(
            artifact_id="build_" + manifest_digest, logical_ref=build_path, sha256=manifest_digest
        ),
        created,
    )


def publish_result_record(
    root: Path,
    result_dir: str,
    preflight_build: ArtifactRef,
    run_record_sha256: str,
    *,
    code: GitState,
    rerun: Rerun = "not_performed",
) -> PublishedResult:
    """Verify the R result, then publish the result record and read it back.

    Nothing is written unless every mandatory verification passes. A record that fails its
    read-back after being stored stays on disk (artifacts are never replaced) and the error is raised.
    """
    verified = verify_r_result(root, result_dir, preflight_build, run_record_sha256)
    record = AnalysisResultRecord(
        **verified.claims,
        rerun=rerun,
        checks=[ResultCheck(name=n, status=s, scope=c) for n, s, c in CHECKS],
        not_established=list(NOT_ESTABLISHED),
        verifier=ResultVerifier(
            tool="analysis_result_v1",
            settings_sha256=settings_sha256(),
            module_sha256=sha256_file(Path(__file__)),
            git=code,
        ),
    )
    published = _store(record, root)
    read_result_record(published.build, root)
    return published


@dataclass(frozen=True)
class VerifiedResultRecord:
    record: AnalysisResultRecord
    verified: VerifiedRResult


def read_result_record(build: ArtifactRef, root: Path) -> VerifiedResultRecord:
    """Check bytes and lineage, then re-verify every derived claim against the R files and the preflight."""
    manifest = read_manifest(root / build.logical_ref, build.sha256)
    if len(manifest.outputs) != 1 or manifest.outputs[0].artifact_kind != "analysis_result_record":
        raise ResultError("read_back", "build manifest does not describe an analysis result record")
    if manifest.transformation_id != TRANSFORMATION or (
        manifest.transformation_settings_sha256 != settings_sha256()
    ):
        raise ResultError("read_back", "result record was built under other settings")
    ref = manifest.outputs[0].artifact
    record = read_project_json(root / ref.logical_ref, ref.sha256, AnalysisResultRecord)
    if manifest.inputs != _inputs(record):
        raise ResultError("read_back", "result record lineage differs from its build manifest")
    verified = verify_r_result(
        root, record.result_dir, record.preflight_build, record.run_record_sha256
    )
    differing = sorted(k for k, v in verified.claims.items() if getattr(record, k) != v)
    if differing:
        raise ResultError(
            "read_back", f"result record differs from the evidence it describes: {differing}"
        )
    return VerifiedResultRecord(record, verified)
