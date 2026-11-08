"""Immutable publication of a readiness report and independent read-back.

Layout below `<data root>/readiness/`: `evidence/<sha>.json` holds canonical-JSON
observations and a copy of the policy used; `reports/<sha>.json` the report; and
`reports/<sha>.build.json` its build manifest. File names are content hashes, nothing is ever
replaced, and publishing the same bytes twice is a success. Publication writes evidence first,
then the report, then the manifest, so an interrupted publication leaves a report without
manifest, which `read_report` refuses until the same publication is repeated. The directory's
existence says nothing about whether the data root is approved storage.

Four things are verified separately on read-back and must not be conflated:

* report and manifest integrity: bytes, strict parse, lineage, policy and recomputed decision;
* retained evidence: `data/` and `snapshot/` evidence is re-read and re-hashed. Evidence that
  is no longer there is reported as not retained; evidence that differs is an error;
* what the evaluator reported: gate statuses are the evaluator's original claims, not a
  re-run. `repo/` evidence (the checkout, including the Git-ignored frozen specifications)
  was never retained and is therefore never described as historically verified;
* the present checkout (`compare_current`), which is informational and does not alter
  validity.

A report is a historical assessment. It is not an authorization credential and carries no
claim about who produced it.
"""

from dataclasses import dataclass, field
from pathlib import Path

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    serialize_manifest,
    serialize_project_json,
)
from biorag_repro.readiness import evidence, git_state
from biorag_repro.readiness.decide import Outcome, ReportInvalid, verify_report
from biorag_repro.readiness.evaluate import Evaluation
from biorag_repro.readiness.files import FileMissing, FileProblem, loads_strict, read_regular
from biorag_repro.readiness.models import ReadinessReport
from biorag_repro.readiness.policy import ApprovalPolicy

TRANSFORMATION = "readiness_report_v1"
ROOT = "readiness"


class PublicationError(ValueError):
    """Publication was refused; nothing existing was replaced."""


class PartialPublication(PublicationError):
    """A report exists without its manifest, or the reverse."""


class ReadbackError(ValueError):
    """The stored report, manifest or retained evidence fails verification."""


def _evidence_ref(digest: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id="snapshot_" + digest,
        logical_ref=f"{ROOT}/evidence/{digest}.json",
        sha256=digest,
    )


def _inputs(report: ReadinessReport) -> list[BuildInput]:
    digests = {report.policy_sha256}
    for gate in report.gates:
        digests |= {r.sha256 for r in gate.evidence if r.locator.startswith("snapshot/")}
    return [BuildInput(role="evidence", artifact=_evidence_ref(d)) for d in sorted(digests)]


def _manifest_for(report: ReadinessReport, report_bytes: bytes) -> tuple[BuildManifest, str]:
    digest = sha256_bytes(report_bytes)
    inputs = _inputs(report)
    settings = report.evaluator.code_sha256
    artifact_id = derived_artifact_id(
        artifact_kind="readiness_report",
        inputs=[ArtifactInput(i.role, i.artifact.artifact_id, i.artifact.sha256) for i in inputs],
        transformation_id=TRANSFORMATION,
        transformation_settings_sha256=settings,
        output_sha256=digest,
        output_format="json",
        serialization_version="1",
    )
    output = ArtifactRef(
        artifact_id=artifact_id, logical_ref=f"{ROOT}/reports/{digest}.json", sha256=digest
    )
    manifest = BuildManifest(
        schema_version="1",
        inputs=inputs,
        transformation_id=TRANSFORMATION,
        transformation_settings_sha256=settings,
        outputs=[
            DerivedOutput(
                artifact=output,
                artifact_kind="readiness_report",
                output_format="json",
                serialization_version="1",
                identity_version="1",
            )
        ],
        status="complete",
    )
    return manifest, digest


def _directories(data_root: Path) -> Path:
    if data_root.is_symlink() or not data_root.is_dir():
        raise PublicationError("the data root must be an existing directory and not a link")
    base = data_root
    for part in (ROOT, f"{ROOT}/evidence", f"{ROOT}/reports"):
        path = data_root / part
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise PublicationError(f"{part} must be a plain directory")
        path.mkdir(exist_ok=True)
    return base


def _store(data_root: Path, relative: str, content: bytes) -> bool:
    """Write without replacing. Returns True if created, False if identical bytes existed."""
    digest = sha256_bytes(content)
    target = data_root / relative
    try:
        publish_bytes(target, content, digest)
        return True
    except FileExistsError:
        pass
    try:
        existing = read_regular(data_root, relative)
    except FileProblem as error:
        raise PublicationError(f"{relative} exists and cannot be read: {error}") from error
    if existing != content:
        raise PublicationError(f"{relative} exists with different bytes; it was not replaced")
    return False


@dataclass(frozen=True)
class Published:
    report_sha256: str
    manifest_sha256: str
    created: list[str]
    reused: list[str]


def publish_report(evaluation: Evaluation, data_root: Path) -> Published:
    report = evaluation.report
    report_bytes = serialize_project_json(report, ReadinessReport)
    manifest, digest = _manifest_for(report, report_bytes)
    needed = {i.artifact.sha256 for i in manifest.inputs}
    if missing := needed - set(evaluation.snapshots):
        raise PublicationError(f"{len(missing)} cited observations were not captured")
    for sha, data in evaluation.snapshots.items():
        if sha256_bytes(data) != sha:
            raise PublicationError("a captured observation does not match its hash")
    _directories(data_root)
    created: list[str] = []
    reused: list[str] = []

    def note(relative: str, was_created: bool) -> None:
        (created if was_created else reused).append(relative)

    for sha in sorted(needed):
        relative = f"{ROOT}/evidence/{sha}.json"
        note(relative, _store(data_root, relative, evaluation.snapshots[sha]))
    report_path = f"{ROOT}/reports/{digest}.json"
    note(report_path, _store(data_root, report_path, report_bytes))
    manifest_path = f"{ROOT}/reports/{digest}.build.json"
    manifest_bytes = serialize_manifest(manifest)
    try:
        publish_manifest(
            manifest,
            data_root / manifest_path,
            {manifest.outputs[0].artifact.logical_ref: data_root / report_path},
        )
        created.append(manifest_path)
    except FileExistsError:
        if read_regular(data_root, manifest_path) != manifest_bytes:
            raise PublicationError(f"{manifest_path} exists with different bytes") from None
        reused.append(manifest_path)
    read_report(data_root, digest, _policy_of(evaluation))
    return Published(digest, sha256_bytes(manifest_bytes), created, reused)


def _policy_of(evaluation: Evaluation) -> ApprovalPolicy:
    data = loads_strict(evaluation.snapshots[evaluation.report.policy_sha256])
    return ApprovalPolicy.model_validate(data)


@dataclass(frozen=True)
class Readback:
    report: ReadinessReport
    outcome: Outcome  # recomputed from the stored gates, not the recorded decision
    manifest_sha256: str
    verified_evidence: int
    not_retained: list[str] = field(default_factory=list)  # evidence that is gone
    not_historical: int = 0  # repo/ evidence: never retained, never verified here


def read_report(data_root: Path, report_sha256: str, policy: ApprovalPolicy) -> Readback:
    """Verify a stored report against `policy`. Raises `ReadbackError` or `PartialPublication`."""
    relative = f"{ROOT}/reports/{report_sha256}.json"
    try:
        report_bytes = read_regular(data_root, relative)
    except FileMissing as error:
        raise ReadbackError(f"no report {report_sha256}") from error
    except FileProblem as error:
        raise ReadbackError(str(error)) from error
    if sha256_bytes(report_bytes) != report_sha256:
        raise ReadbackError("report bytes do not match their SHA-256")
    try:
        report = ReadinessReport.model_validate(loads_strict(report_bytes), strict=True)
        if serialize_project_json(report, ReadinessReport) != report_bytes:
            raise ReadbackError("report is not in canonical project form")
        outcome = verify_report(report, policy)
    except (ValueError, ReportInvalid) as error:
        if isinstance(error, ReadbackError):
            raise
        raise ReadbackError(f"invalid report: {str(error)[:300]}") from error

    manifest_relative = f"{ROOT}/reports/{report_sha256}.build.json"
    try:
        manifest_bytes = read_regular(data_root, manifest_relative)
    except FileMissing as error:
        raise PartialPublication("report has no build manifest; publish it again") from error
    try:
        manifest = read_manifest_bytes(manifest_bytes)
        expected, _ = _manifest_for(report, report_bytes)
    except ValueError as error:
        raise ReadbackError(f"invalid build manifest: {str(error)[:300]}") from error
    if manifest != expected:
        raise ReadbackError("build manifest does not describe this report")

    verified = 0
    gone: list[str] = []
    not_historical = 0
    seen: set[str] = set()
    for ref in [r for g in report.gates for r in g.evidence]:
        if ref.locator in seen:
            continue
        seen.add(ref.locator)
        if ref.locator.startswith("repo/"):
            not_historical += 1
            continue
        if ref.locator.startswith("snapshot/"):
            path = f"{ROOT}/evidence/{ref.locator.removeprefix('snapshot/')}"
        elif ref.locator.startswith("data/"):
            path = ref.locator.removeprefix("data/")
        else:
            raise ReadbackError(f"evidence locator {ref.locator} has no known root")
        try:
            data = read_regular(data_root, path)
        except FileMissing:
            gone.append(ref.locator)
            continue
        except FileProblem as error:
            raise ReadbackError(f"evidence {ref.locator} is unsafe: {error}") from error
        if sha256_bytes(data) != ref.sha256:
            raise ReadbackError(f"evidence {ref.locator} differs from the recorded SHA-256")
        verified += 1
    policy_path = f"{ROOT}/evidence/{report.policy_sha256}.json"
    try:
        stored_policy = read_regular(data_root, policy_path)
    except FileProblem as error:
        raise ReadbackError(f"policy copy is unavailable: {error}") from error
    if stored_policy != canonical_json_bytes(policy.model_dump(mode="json")):
        raise ReadbackError("stored policy copy differs from the policy used for verification")
    return Readback(report, outcome, sha256_bytes(manifest_bytes), verified, gone, not_historical)


def read_manifest_bytes(content: bytes) -> BuildManifest:
    return BuildManifest.model_validate(loads_strict(content), strict=True)


def compare_current(report: ReadinessReport, repo: Path, data_root: Path) -> dict[str, str]:
    """Informational: how the present checkout relates to what the report recorded."""
    result: dict[str, str] = {}
    candidate = report.candidate

    def same(name: str, now: str | None, then: str | None) -> None:
        result[name] = (
            "unavailable"
            if now is None or then is None
            else ("same" if now == then else "different")
        )

    try:
        same("head", git_state.probe(repo).head, candidate.head_sha)
    except git_state.GitUnavailable:
        result["head"] = "unavailable"
    try:
        same(
            "renv_lock",
            sha256_bytes(read_regular(repo, evidence.RENV_LOCK)),
            candidate.renv_lock_sha256,
        )
    except FileProblem:
        result["renv_lock"] = "unavailable"
    try:
        same(
            "benchmark_release",
            sha256_bytes(read_regular(data_root, evidence.BENCHMARK_RELEASE)),
            candidate.benchmark_release_sha256,
        )
    except FileProblem:
        result["benchmark_release"] = "unavailable"
    return result
