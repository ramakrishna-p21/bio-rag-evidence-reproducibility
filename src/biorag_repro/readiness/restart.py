"""Canonical verification of the bounded G10 development recovery exercise.

The hashes authenticate retained bytes and their relationships. They do not prove that a
physical key press or signal occurred. Acceptance therefore also requires runner-generated
attempt, GPU and environment records, a SIGINT process record, and matching Drive-to-WSL byte
inventories. The process record and custody records remain evidence supplied by their named
operators; the reader makes no stronger hardware-attestation claim.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, ValidationError, model_validator

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.records import Record
from biorag_repro.evaluation.stage_c_models import GenerationResult
from biorag_repro.generation.environment import EnvironmentRecord, ModelSnapshot, check_utc
from biorag_repro.generation.environment_lock import (
    EnvironmentInventory,
    EnvironmentLockError,
    verify_environment,
)
from biorag_repro.generation.receipt import ExecutionReceipt
from biorag_repro.readiness.files import FileMissing, FileProblem, loads_strict, read_regular
from biorag_repro.readiness.gather import Gathered, file_ref

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
RunId = Annotated[str, StringConstraints(pattern=r"^run_[0-9a-f]{32}$")]
AttemptId = Annotated[str, StringConstraints(pattern=r"^att_[0-9a-f]{32}$")]
ObservationKey = Annotated[str, StringConstraints(pattern=r"^obs_[0-9a-f]{64}$")]
GitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]

EVIDENCE_RECORD = "readiness/evidence/g10_restart_recovery_v1.json"
FROZEN_ARMS = frozenset(
    {
        "confirmatory",
        "dense_robustness",
        "chunk_robustness",
        "scope_robustness",
        "closed_book",
    }
)


@dataclass(frozen=True)
class RestartContract:
    v4_archive_sha256: str
    v5_update_sha256: str
    v5_manifest_sha256: str
    historical_code_revision: str
    historical_run_id: str
    historical_anchor_sha256: str
    planned_observations: int
    contexts_sha256: str
    planned_keys_sha256: str
    generator_configurations: dict[str, str]
    prompt_sha256: str
    seed_registry_sha256: str
    environment_lock_sha256: str
    model_repository: str
    model_revision: str
    model_manifest_sha256: str
    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_file_sha256: dict[str, str]
    approved_gpu_name: str
    minimum_model_positions: int


PRODUCTION_CONTRACT = RestartContract(
    v4_archive_sha256="c8ddc4500329776b9ead10fdf5e3851cd9b4c8624e5e4cde396f0aa4633c0875",
    v5_update_sha256="b067777b4612fafb89e16f6fd732fa4f032ba6736273e1372e4bc8e78ec5348c",
    v5_manifest_sha256="0853ec27e1d7a6b5747c5d4762f1c7c64fc2909b02951aa24e1c3bea61472359",
    historical_code_revision="639955d5e86e87cc16eab4142f797aa4fc56372a",
    historical_run_id="run_78f7738f308d48daab614f6e2f02616f",
    historical_anchor_sha256="e6f7e92c6be2a227f7fad43074eb0fb7cf98c71cbeb2ad53206ccb0dec714c1c",
    planned_observations=1290,
    contexts_sha256="9f728b6bf60a840eaec3805759cb8f457a32a49105ad0cac79562e9e5a59b46c",
    planned_keys_sha256="b618f884c682d501575617a601a444a39c62108544e2f64d8bd00881d076962d",
    generator_configurations={
        "confirmatory": "11290e58973f3eec13ae2ce2712b9848f19fb57900bc42e7a73bd6f687949927",
        "dense_robustness": "38bf4944a798e2b42baf855d3dccda8adb2a51804fa0f8b8c8d868deabee5c9c",
        "chunk_robustness": "2406e990544267d41d86ab0a487ccd14b1e14d26dc80d824352fd03d728c51c0",
        "scope_robustness": "ac88bb4faeaa22907c7b6663c96299983ddf2f946e35c107721339b8c987e3cb",
        "closed_book": "c247b14982c8bdba5d8110816465e1e2dc07789c8f1d7d8debe0948cfe397888",
    },
    prompt_sha256="7d293ebbc6e8fc53d8718dd2d07d18e6ab50a90975b9d25cf778beb30b877b78",
    seed_registry_sha256="4c1ddc3f1323198e5efc84044b73074b61dc3bd42e36c5a162658c18ac23c6c8",
    environment_lock_sha256="96af1876324098a0fbf6c28d9896b790d10a210ba7943538d41056257f15847a",
    model_repository="Qwen/Qwen3-8B",
    model_revision="b968826d9c46dd6066d109eabc6255188de91218",
    model_manifest_sha256="4a133fa0d16ea3fdd42e987d7aa5135f3f8562ce45c55bf478b2f777f53e6fad",
    tokenizer_repository="Qwen/Qwen3-8B",
    tokenizer_revision="b968826d9c46dd6066d109eabc6255188de91218",
    tokenizer_file_sha256={
        "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
        "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
    approved_gpu_name="NVIDIA L4",
    minimum_model_positions=32768,
)


class FileReference(Record):
    logical_ref: str = Field(min_length=1)
    sha256: Sha256


class PackageEvidence(Record):
    package_root: str = Field(min_length=1)
    v4_archive: FileReference
    v5_update: FileReference
    manifest: FileReference


class ExecutionEvidence(Record):
    batch_root: str = Field(min_length=1)
    execution_source_revision: GitSha
    interruption_anchor: FileReference
    recovery_anchor: FileReference
    process_record: FileReference
    command_record: FileReference


class HistoricalEvidence(Record):
    package_code_revision: GitSha
    published_run_id: RunId
    published_anchor_sha256: Sha256


class CustodyEvidence(Record):
    drive_manifest: FileReference
    wsl_manifest: FileReference


class RestartEvidence(Record):
    schema_version: Literal["g10_restart_recovery_evidence_v1"]
    exercise_id: str = Field(min_length=1)
    evidence_class: Literal["live_development_gpu_interruption"]
    researcher: Literal["Rama Krishna Pudota"]
    package: PackageEvidence
    execution: ExecutionEvidence
    historical: HistoricalEvidence
    custody: CustodyEvidence


class InterruptionProcess(Record):
    schema_version: Literal["g10_interruption_process_v1"]
    runner: Literal["scripts/run_generation.py"]
    command_sha256: Sha256
    pid: int = Field(gt=0)
    signal: Literal["SIGINT"]
    exit_code: Literal[130]
    run_id: RunId
    observation_key: ObservationKey
    attempt_id: AttemptId
    session_boot_id: str = Field(min_length=1)
    started_at_utc: str
    signal_at_utc: str
    exited_at_utc: str
    synthetic: Literal[False]

    @model_validator(mode="after")
    def ordered(self) -> Self:
        stamps = [
            check_utc(v) for v in (self.started_at_utc, self.signal_at_utc, self.exited_at_utc)
        ]
        if [datetime.fromisoformat(v) for v in stamps] != sorted(
            datetime.fromisoformat(v) for v in stamps
        ):
            raise ValueError("process timestamps are not ordered")
        return self


class PublicationManifest(Record):
    schema_version: Literal["g10_drive_publication_v1"]
    publication_id: str = Field(min_length=1)
    published_at_utc: str
    files: dict[str, Sha256] = Field(min_length=1)


class CustodyManifest(Record):
    schema_version: Literal["g10_wsl_custody_v1"]
    publication_id: str = Field(min_length=1)
    drive_manifest_sha256: Sha256
    received_at_utc: str
    custody_root_kind: Literal["wsl_outside_drive_sync"]
    files: dict[str, Sha256] = Field(min_length=1)


def _safe(relative: str) -> str:
    path = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise FileProblem(f"unsafe locator {relative!r}")
    return relative


def _join(root: str, relative: str) -> str:
    return _safe(f"{_safe(root)}/{_safe(relative)}")


def _read_ref(root: Path, reference: FileReference) -> bytes:
    data = read_regular(root, _safe(reference.logical_ref))
    if sha256_bytes(data) != reference.sha256:
        raise FileProblem(f"{reference.logical_ref} differs from its recorded SHA-256")
    return data


def _model(data: bytes, model):
    return model.model_validate(loads_strict(data), strict=True)


def _tree_files(root: Path, relative: str) -> set[str]:
    base = root / _safe(relative)
    if not base.is_dir() or base.is_symlink():
        raise FileProblem(f"{relative} is not a regular directory")
    files = set()
    for path in base.rglob("*"):
        if path.is_symlink():
            raise FileProblem(f"{path.relative_to(root)} is a link")
        if path.is_file():
            files.add(path.relative_to(base).as_posix())
        elif not path.is_dir():
            raise FileProblem(f"{path.relative_to(root)} is not a regular file or directory")
    return files


def _package(root: Path, evidence: RestartEvidence, contract: RestartContract) -> dict:
    package = evidence.package
    if (
        package.v4_archive.sha256 != contract.v4_archive_sha256
        or package.v5_update.sha256 != contract.v5_update_sha256
        or package.manifest.sha256 != contract.v5_manifest_sha256
    ):
        raise ValueError("development package identity differs from the frozen v5 package")
    _read_ref(root, package.v4_archive)
    _read_ref(root, package.v5_update)
    manifest_bytes = _read_ref(root, package.manifest)
    manifest = loads_strict(manifest_bytes)
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files"}:
        raise ValueError("package manifest has an unsupported shape")
    if manifest["schema_version"] != "1" or not isinstance(manifest["files"], dict):
        raise ValueError("package manifest is invalid")
    listed = manifest["files"]
    actual = _tree_files(root, package.package_root)
    manifest_name = Path(package.manifest.logical_ref).relative_to(package.package_root).as_posix()
    if actual != set(listed) | {manifest_name}:
        raise ValueError("package files differ from the authenticated manifest")
    for relative, identity in listed.items():
        if (
            not isinstance(identity, dict)
            or set(identity) != {"sha256", "bytes"}
            or not isinstance(identity["sha256"], str)
            or type(identity["bytes"]) is not int
            or identity["bytes"] < 0
        ):
            raise ValueError("package manifest contains an invalid file identity")
        data = read_regular(root, _join(package.package_root, relative))
        if len(data) != identity["bytes"] or sha256_bytes(data) != identity["sha256"]:
            raise ValueError(f"package file {relative} differs from its manifest")
    required = {
        "preflight.json",
        "planned_inputs.json",
        "code_revision.txt",
        "environment/approved_environment_lock_v1.toml",
        "inputs/contexts.json",
        "inputs/prompt.json",
        "inputs/seeds.json",
        *{f"inputs/configs/{arm}.toml" for arm in FROZEN_ARMS},
    }
    if not required <= set(listed):
        raise ValueError("package manifest omits a required development input")
    preflight = loads_strict(read_regular(root, _join(package.package_root, "preflight.json")))
    planned = loads_strict(read_regular(root, _join(package.package_root, "planned_inputs.json")))
    revision = read_regular(root, _join(package.package_root, "code_revision.txt")).decode().strip()
    expected = {
        "status": "development_inputs_checked",
        "scientific_execution": False,
        "contexts_sha256": contract.contexts_sha256,
        "planned_observations": contract.planned_observations,
        "planned_keys_sha256": contract.planned_keys_sha256,
        "generator_configurations": contract.generator_configurations,
        "prompt_sha256": contract.prompt_sha256,
        "seed_registry_sha256": contract.seed_registry_sha256,
    }
    if not isinstance(preflight, dict) or any(preflight.get(k) != v for k, v in expected.items()):
        raise ValueError("package preflight differs from the frozen complete development plan")
    inputs = planned.get("inputs") if isinstance(planned, dict) else None
    if (
        not isinstance(planned, dict)
        or planned.get("status") != "planned_generator_inputs_not_execution_evidence"
        or not isinstance(inputs, dict)
        or len(inputs) != contract.planned_observations
        or any(not isinstance(k, str) or not k.startswith("obs_") or len(k) != 68 for k in inputs)
        or sha256_bytes(canonical_json_bytes(sorted(inputs))) != contract.planned_keys_sha256
    ):
        raise ValueError("planned input universe differs from the frozen plan")
    for value in inputs.values():
        input_ref = ArtifactRef.model_validate(value, strict=True)
        if not input_ref.logical_ref.startswith("inputs/"):
            raise ValueError("planned input has a noncanonical path")
        identity = listed.get("planned/" + input_ref.logical_ref)
        if not isinstance(identity, dict) or identity.get("sha256") != input_ref.sha256:
            raise ValueError("planned input is absent from the authenticated package")
    lock = read_regular(
        root, _join(package.package_root, "environment/approved_environment_lock_v1.toml")
    )
    if sha256_bytes(lock) != contract.environment_lock_sha256:
        raise ValueError("package environment lock differs")
    if revision != contract.historical_code_revision:
        raise ValueError("package code revision differs")
    return planned


def _anchor(root: Path, reference: FileReference, batch_root: str) -> dict:
    if not reference.logical_ref.startswith(_safe(batch_root) + "/anchors/"):
        raise ValueError("anchor is outside the exercise batch")
    data = _read_ref(root, reference)
    value = loads_strict(data)
    if not isinstance(value, dict) or set(value) != {
        "sealed_utc",
        "files",
        "previous_anchor_sha256",
    }:
        raise ValueError("anchor has an unsupported shape")
    check_utc(value["sealed_utc"])
    if not isinstance(value["files"], dict) or not value["files"]:
        raise ValueError("anchor has no file inventory")
    for relative, digest in value["files"].items():
        data = read_regular(root, _join(batch_root, relative))
        if not isinstance(digest, str) or sha256_bytes(data) != digest:
            raise ValueError(f"sealed file {relative} differs")
    return value


def _events(root: Path, batch_root: str, sealed: dict[str, str]) -> list[dict]:
    names = sorted(
        name for name in sealed if name.startswith("attempts/") and name.endswith(".json")
    )
    if names != [f"attempts/{n:06d}.json" for n in range(1, len(names) + 1)]:
        raise ValueError("attempt history is not one canonical sequence")
    events = [loads_strict(read_regular(root, _join(batch_root, name))) for name in names]
    for n, event in enumerate(events, 1):
        if not isinstance(event, dict) or event.get("seq") != n:
            raise ValueError("attempt event sequence differs")
        check_utc(event.get("at_utc"))
    return events


def _result(
    root: Path,
    batch_root: str,
    sealed: dict[str, str],
    key: str,
    run_id: str,
    attempt_id: str,
    contract: RestartContract,
    execution_revision: str,
    hostname: str,
    snapshot: ModelSnapshot,
    planned_input: dict,
) -> tuple[GenerationResult, str]:
    relative = f"results/{key}.json"
    value = loads_strict(read_regular(root, _join(batch_root, relative)))
    result = GenerationResult.model_validate(value["generation_result"], strict=True)
    if (
        result.phase != "development"
        or result.status != "completed"
        or result.run_id != run_id
        or result.observation_key != key
        or result.attempt_id != attempt_id
        or result.resolved_config_sha256 not in set(contract.generator_configurations.values())
    ):
        raise ValueError("completed result differs from the bounded recovery sequence")
    output = result.output
    if output is None or sealed.get(output.logical_ref) != output.sha256:
        raise ValueError("result output is not sealed")
    if sha256_bytes(read_regular(root, _join(batch_root, output.logical_ref))) != output.sha256:
        raise ValueError("result output bytes differ")
    input_ref = value.get("generator_input")
    receipt_ref = value.get("execution_receipt")
    if not isinstance(input_ref, dict) or not isinstance(receipt_ref, dict):
        raise FileProblem("official completion lacks input or execution receipt")
    if input_ref != planned_input:
        raise ValueError("executed generator input differs from the authenticated plan")
    if sealed.get(input_ref.get("logical_ref")) != input_ref.get("sha256"):
        raise ValueError("stored input is not sealed")
    input_bytes = read_regular(root, _join(batch_root, input_ref["logical_ref"]))
    if sha256_bytes(input_bytes) != input_ref["sha256"]:
        raise ValueError("stored input bytes differ")
    if sealed.get(receipt_ref.get("logical_ref")) != receipt_ref.get("sha256"):
        raise ValueError("execution receipt is not sealed")
    receipt_data = read_regular(root, _join(batch_root, receipt_ref["logical_ref"]))
    if sha256_bytes(receipt_data) != receipt_ref["sha256"]:
        raise ValueError("execution receipt bytes differ")
    receipt = _model(receipt_data, ExecutionReceipt)
    if (
        receipt.run_id != run_id.removeprefix("run_")
        or receipt.observation_key != key
        or receipt.attempt_id != attempt_id.removeprefix("att_")
        or receipt.input_record_sha256 != input_ref["sha256"]
        or receipt.raw_output_sha256 != output.sha256
        or receipt.facts.measurements is None
        or receipt.facts.measurements.cuda_memory.status != "measured"
    ):
        raise ValueError("execution receipt differs from the result provenance")
    environment_path = f"environments/{receipt.environment_sha256}.json"
    if sealed.get(environment_path) != receipt.environment_sha256:
        raise ValueError("execution environment is not sealed")
    environment = _model(read_regular(root, _join(batch_root, environment_path)), EnvironmentRecord)
    execution = environment.execution
    if (
        environment.backend.kind != "official"
        or execution is None
        or execution.snapshot != snapshot
        or execution.device_name != contract.approved_gpu_name
        or not execution.device.startswith("cuda:")
        or execution.dtype != "bfloat16"
        or environment.model_repository != contract.model_repository
        or environment.model_revision != contract.model_revision
        or environment.tokenizer_repository != contract.tokenizer_repository
        or environment.tokenizer_revision != contract.tokenizer_revision
        or execution.tokenizer_file_sha256 != contract.tokenizer_file_sha256
        or execution.max_position_embeddings < contract.minimum_model_positions
        or environment.git.status != "ok"
        or environment.git.dirty
        or environment.git.commit != execution_revision
        or environment.hostname != hostname
    ):
        raise ValueError("official model, tokenizer or execution environment differs")
    return result, receipt.environment_sha256


def _custody(
    root: Path,
    evidence: RestartEvidence,
    required: dict[str, str],
) -> tuple[bytes, bytes]:
    drive_bytes = _read_ref(root, evidence.custody.drive_manifest)
    wsl_bytes = _read_ref(root, evidence.custody.wsl_manifest)
    drive = _model(drive_bytes, PublicationManifest)
    wsl = _model(wsl_bytes, CustodyManifest)
    if (
        wsl.publication_id != drive.publication_id
        or wsl.drive_manifest_sha256 != sha256_bytes(drive_bytes)
        or wsl.files != drive.files
        or drive.files != required
        or datetime.fromisoformat(check_utc(wsl.received_at_utc))
        <= datetime.fromisoformat(check_utc(drive.published_at_utc))
    ):
        raise ValueError("Drive publication and independent WSL custody do not agree")
    for relative, digest in drive.files.items():
        if sha256_bytes(read_regular(root, _safe(relative))) != digest:
            raise ValueError(f"custodied bytes differ for {relative}")
    return drive_bytes, wsl_bytes


def restart_recovery(data_root: Path, contract: RestartContract = PRODUCTION_CONTRACT) -> Gathered:
    """Verify the canonical G10 record, returning absence until it is separately published."""
    scope = "data root: authenticated bounded development interruption and recovery exercise"
    try:
        record_bytes = read_regular(data_root, EVIDENCE_RECORD)
    except FileMissing:
        return Gathered("absent", "official_restart_evidence_absent", scope)
    out = Gathered("failed", "restart_evidence_invalid", scope)
    out.refs.append(file_ref("g10_recovery_record", "data/" + EVIDENCE_RECORD, record_bytes))
    try:
        evidence = _model(record_bytes, RestartEvidence)
        planned = _package(data_root, evidence, contract)
        if (
            evidence.historical.package_code_revision != contract.historical_code_revision
            or evidence.historical.published_run_id != contract.historical_run_id
            or evidence.historical.published_anchor_sha256 != contract.historical_anchor_sha256
        ):
            raise ValueError("historical publication identity differs")
        if evidence.execution.execution_source_revision == contract.historical_code_revision:
            raise ValueError("recovery exercise reused the historical package source revision")
        interruption = _anchor(
            data_root, evidence.execution.interruption_anchor, evidence.execution.batch_root
        )
        recovery = _anchor(
            data_root, evidence.execution.recovery_anchor, evidence.execution.batch_root
        )
        if (
            interruption["previous_anchor_sha256"] is not None
            or recovery["previous_anchor_sha256"] != evidence.execution.interruption_anchor.sha256
        ):
            raise ValueError("recovery anchor is not the direct successor of interruption")
        if not set(interruption["files"]) < set(recovery["files"]):
            raise ValueError("recovery anchor does not extend the interruption seal")
        anchors = {
            Path(evidence.execution.interruption_anchor.logical_ref)
            .relative_to(evidence.execution.batch_root)
            .as_posix(),
            Path(evidence.execution.recovery_anchor.logical_ref)
            .relative_to(evidence.execution.batch_root)
            .as_posix(),
        }
        if (
            _tree_files(data_root, evidence.execution.batch_root)
            != set(recovery["files"]) | anchors
        ):
            raise ValueError("batch contains an unsealed or missing file")
        batch_bytes = read_regular(data_root, _join(evidence.execution.batch_root, "batch.json"))
        batch = loads_strict(batch_bytes)
        identity = batch.get("identity") if isinstance(batch, dict) else None
        run_id = batch.get("run_id") if isinstance(batch, dict) else None
        if (
            not isinstance(identity, dict)
            or run_id == contract.historical_run_id
            or identity.get("phase") != "development"
            or identity.get("official_execution") is not True
            or identity.get("contexts_sha256") != contract.contexts_sha256
            or identity.get("planned_keys_sha256") != contract.planned_keys_sha256
            or identity.get("generator_configurations") != contract.generator_configurations
            or identity.get("prompt_sha256") != contract.prompt_sha256
            or identity.get("seed_registry_sha256") != contract.seed_registry_sha256
            or identity.get("environment_lock_sha256") != contract.environment_lock_sha256
            or identity.get("model_manifest_sha256") != contract.model_manifest_sha256
            or identity.get("git", {}).get("head") != evidence.execution.execution_source_revision
            or identity.get("git", {}).get("tracked_changes_sha256") != sha256_bytes(b"")
            or set(batch.get("contexts_header", {}).get("arms", [])) != FROZEN_ARMS
        ):
            raise ValueError("batch identity differs from the complete official development plan")
        snapshot = ModelSnapshot.model_validate(batch.get("model_snapshot"), strict=True)
        if snapshot.manifest_sha256 != contract.model_manifest_sha256:
            raise ValueError("batch model snapshot differs")
        events = _events(data_root, evidence.execution.batch_root, recovery["files"])
        if len(events) != 6:
            raise ValueError("bounded exercise needs exactly six attempt events")
        first_attempt, first_key = events[0]["attempt_id"], events[0]["observation_key"]
        interrupted_attempt, recovered_key = events[2]["attempt_id"], events[2]["observation_key"]
        recovery_attempt = events[4]["attempt_id"]
        expected_sequence = [
            (first_attempt, first_key, "started"),
            (first_attempt, first_key, "completed"),
            (interrupted_attempt, recovered_key, "started"),
            (interrupted_attempt, recovered_key, "interrupted"),
            (recovery_attempt, recovered_key, "started"),
            (recovery_attempt, recovered_key, "completed"),
        ]
        observed_sequence = [
            (event.get("attempt_id"), event.get("observation_key"), event.get("event"))
            for event in events
        ]
        event_times = [datetime.fromisoformat(event["at_utc"]) for event in events]
        planned_keys = set(planned["inputs"])
        if (
            observed_sequence != expected_sequence
            or len({first_attempt, interrupted_attempt, recovery_attempt}) != 3
            or first_key == recovered_key
            or {first_key, recovered_key} - planned_keys
            or event_times != sorted(event_times)
            or any(event.get("run_id") != run_id for event in events)
            or set(interruption["files"])
            & {"attempts/000004.json", "attempts/000005.json", "attempts/000006.json"}
            or not {"attempts/000001.json", "attempts/000002.json", "attempts/000003.json"}
            <= set(interruption["files"])
        ):
            raise ValueError("attempt interruption and recovery sequence differs")
        result_names = sorted(name for name in recovery["files"] if name.startswith("results/"))
        if result_names != sorted([f"results/{first_key}.json", f"results/{recovered_key}.json"]):
            raise ValueError("bounded exercise must contain exactly two completed results")
        if (
            f"results/{first_key}.json" not in interruption["files"]
            or f"results/{recovered_key}.json" in interruption["files"]
        ):
            raise ValueError("interruption seal has the wrong completed-result boundary")
        hostname = identity.get("session_hostname")
        first, first_env = _result(
            data_root,
            evidence.execution.batch_root,
            recovery["files"],
            first_key,
            run_id,
            first_attempt,
            contract,
            evidence.execution.execution_source_revision,
            hostname,
            snapshot,
            planned["inputs"][first_key],
        )
        recovered, recovered_env = _result(
            data_root,
            evidence.execution.batch_root,
            recovery["files"],
            recovered_key,
            run_id,
            recovery_attempt,
            contract,
            evidence.execution.execution_source_revision,
            hostname,
            snapshot,
            planned["inputs"][recovered_key],
        )
        if first_env != recovered_env:
            raise ValueError("recovery used a different execution environment")
        lock = read_regular(
            data_root,
            _join(
                evidence.package.package_root,
                "environment/approved_environment_lock_v1.toml",
            ),
        )
        inventories = []
        for name in ("environment_start.json", "environment_end.json"):
            if name not in recovery["files"]:
                raise ValueError("sealed environment inventory is missing")
            inventory = _model(
                read_regular(data_root, _join(evidence.execution.batch_root, name)),
                EnvironmentInventory,
            )
            report = verify_environment(
                lock,
                inventory,
                expected_spec_sha256=contract.environment_lock_sha256,
                inventory_source="stored",
            )
            if report.outcome != "matches_approved_lock":
                raise ValueError("stored environment differs from the approved lock")
            inventories.append(inventory)
        if inventories[0] != inventories[1]:
            raise ValueError("start and end environments differ")
        session_gpu = identity.get("session_gpu_inventory")
        if (
            not identity.get("session_boot_id")
            or not isinstance(session_gpu, list)
            or not session_gpu
        ):
            raise ValueError("session boot or GPU identity is missing")
        if not any(contract.approved_gpu_name in gpu.get("name", "") for gpu in session_gpu):
            raise ValueError("exercise did not use the approved L4 identity")
        if not any(contract.approved_gpu_name in gpu.name for gpu in inventories[0].gpus):
            raise ValueError("approved environment inventory names another GPU")
        for invocation in (1, 2):
            for suffix in ("start", "end"):
                name = f"gpu_inventories/{invocation:06d}_{suffix}.json"
                if name not in recovery["files"] or (
                    invocation == 1 and name not in interruption["files"]
                ):
                    raise ValueError("GPU invocation inventory is missing from its seal")
                gpu = loads_strict(
                    read_regular(data_root, _join(evidence.execution.batch_root, name))
                )
                if gpu.get("status") != "measured" or gpu.get("gpus") != session_gpu:
                    raise ValueError("GPU identity changed between invocations")
        process_bytes = _read_ref(data_root, evidence.execution.process_record)
        process = _model(process_bytes, InterruptionProcess)
        command_bytes = _read_ref(data_root, evidence.execution.command_record)
        process_times = [
            datetime.fromisoformat(process.started_at_utc),
            datetime.fromisoformat(process.signal_at_utc),
            datetime.fromisoformat(process.exited_at_utc),
        ]
        if (
            process.run_id != run_id
            or process.command_sha256 != sha256_bytes(command_bytes)
            or process.observation_key != recovered_key
            or process.attempt_id != interrupted_attempt
            or process.session_boot_id != identity["session_boot_id"]
            or not (event_times[2] <= process_times[1] <= process_times[2] <= event_times[3])
            or process_times[0] > event_times[0]
            or process_times[2] > datetime.fromisoformat(interruption["sealed_utc"])
            or datetime.fromisoformat(interruption["sealed_utc"]) > event_times[3]
            or datetime.fromisoformat(recovery["sealed_utc"]) < event_times[5]
        ):
            raise ValueError("SIGINT process evidence differs from the active attempt")
        required = {
            **{
                _join(evidence.execution.batch_root, name): digest
                for name, digest in recovery["files"].items()
            },
            evidence.execution.interruption_anchor.logical_ref: evidence.execution.interruption_anchor.sha256,
            evidence.execution.recovery_anchor.logical_ref: evidence.execution.recovery_anchor.sha256,
            evidence.execution.process_record.logical_ref: evidence.execution.process_record.sha256,
            evidence.execution.command_record.logical_ref: evidence.execution.command_record.sha256,
            evidence.package.v4_archive.logical_ref: evidence.package.v4_archive.sha256,
            evidence.package.v5_update.logical_ref: evidence.package.v5_update.sha256,
            evidence.package.manifest.logical_ref: evidence.package.manifest.sha256,
        }
        drive_bytes, wsl_bytes = _custody(data_root, evidence, dict(sorted(required.items())))
        out.state, out.reason = "ok", "checks_passed"
        out.subject = {
            "kind": "development_restart_recovery_v1",
            "evidence_record_sha256": sha256_bytes(record_bytes),
            "exercise_id": evidence.exercise_id,
            "v5_manifest_sha256": contract.v5_manifest_sha256,
            "package_code_revision": contract.historical_code_revision,
            "execution_source_revision": evidence.execution.execution_source_revision,
            "run_id": run_id,
            "planned_observations": contract.planned_observations,
            "completed_observations": 2,
            "campaign_completion_claim": False,
            "preserved_observation_key": first.observation_key,
            "preserved_result_sha256": recovery["files"][f"results/{first_key}.json"],
            "recovered_observation_key": recovered.observation_key,
            "recovered_result_sha256": recovery["files"][f"results/{recovered_key}.json"],
            "interrupted_attempt_id": interrupted_attempt,
            "recovery_attempt_id": recovery_attempt,
            "runner_command_sha256": process.command_sha256,
            "interruption_anchor_sha256": evidence.execution.interruption_anchor.sha256,
            "recovery_anchor_sha256": evidence.execution.recovery_anchor.sha256,
            "environment_sha256": first_env,
            "model_manifest_sha256": contract.model_manifest_sha256,
            "drive_publication_manifest_sha256": sha256_bytes(drive_bytes),
            "wsl_custody_manifest_sha256": sha256_bytes(wsl_bytes),
            "historical_run_id": contract.historical_run_id,
            "historical_anchor_sha256": contract.historical_anchor_sha256,
        }
        out.refs.extend(
            [
                file_ref(
                    "g10_v4_archive",
                    "data/" + evidence.package.v4_archive.logical_ref,
                    _read_ref(data_root, evidence.package.v4_archive),
                ),
                file_ref(
                    "g10_v5_update",
                    "data/" + evidence.package.v5_update.logical_ref,
                    _read_ref(data_root, evidence.package.v5_update),
                ),
                file_ref(
                    "g10_v5_manifest",
                    "data/" + evidence.package.manifest.logical_ref,
                    _read_ref(data_root, evidence.package.manifest),
                ),
                file_ref(
                    "g10_interruption_anchor",
                    "data/" + evidence.execution.interruption_anchor.logical_ref,
                    _read_ref(data_root, evidence.execution.interruption_anchor),
                ),
                file_ref(
                    "g10_recovery_anchor",
                    "data/" + evidence.execution.recovery_anchor.logical_ref,
                    _read_ref(data_root, evidence.execution.recovery_anchor),
                ),
                file_ref(
                    "g10_process_record",
                    "data/" + evidence.execution.process_record.logical_ref,
                    process_bytes,
                ),
                file_ref(
                    "g10_runner_command",
                    "data/" + evidence.execution.command_record.logical_ref,
                    command_bytes,
                ),
                file_ref(
                    "g10_drive_manifest",
                    "data/" + evidence.custody.drive_manifest.logical_ref,
                    drive_bytes,
                ),
                file_ref(
                    "g10_wsl_custody",
                    "data/" + evidence.custody.wsl_manifest.logical_ref,
                    wsl_bytes,
                ),
            ]
        )
        out.detail = (
            f"2 of {contract.planned_observations:,} planned observations authenticated "
            "as bounded G10 gate evidence"
        )
        out.facts = {
            "g10_run_id": run_id,
            "g10_recovery_anchor_sha256": evidence.execution.recovery_anchor.sha256,
        }
    except (
        FileProblem,
        ValueError,
        ValidationError,
        EnvironmentLockError,
        OSError,
        KeyError,
        TypeError,
        AttributeError,
    ) as error:
        out.detail = str(error)[:300]
    return out
