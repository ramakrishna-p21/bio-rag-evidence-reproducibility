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

`read_historical` verifies a report against the policy copy retained with it instead of the
current policy. It answers only whether the report was internally valid under the policy it
recorded. It never answers whether execution is permitted now: that is `guard`, which uses the
current tracked policy and never calls this module's historical reader. A retained policy copy
proves that those bytes were stored, not that they were once the tracked policy file, that
anyone approved their values, or that the checkout that produced the report was authentic.
The result type and its immutability are a statement of intent, not a security boundary against
code that runs in the same process.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
from biorag_repro.readiness.models import Blocker, Decision, ReadinessReport
from biorag_repro.readiness.policy import SLOT_NAMES, ApprovalPolicy, policy_sha256

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


class HistoricalPolicyError(ReadbackError):
    """The policy copy retained for a report is missing, unsafe, altered or invalid."""


class UnsupportedHistoricalPolicy(HistoricalPolicyError):
    """The retained policy is not representable by the current `ApprovalPolicy` schema.

    Reading it would need a separately approved migration; nothing is reinterpreted here.
    """


class HistoricalEvidenceIncomplete(ReadbackError):
    """Retained `data/` or `snapshot/` evidence the report cites is gone.

    The report is not verified as a whole. The counts are what could still be checked; no
    readback object is produced, so nothing incomplete can be mistaken for a verified report.
    """

    def __init__(
        self,
        report_sha256: str,
        policy_sha256: str,
        verified_evidence: int,
        not_retained: list[str],
        not_historical: int,
    ) -> None:
        super().__init__(
            f"{len(not_retained)} retained evidence item(s) are missing; "
            f"{verified_evidence} verified, {not_historical} not historical"
        )
        self.report_sha256 = report_sha256
        self.policy_sha256 = policy_sha256
        self.verified_evidence = verified_evidence
        self.not_retained = not_retained
        self.not_historical = not_historical


# Fixed here on purpose: it names the schema this reader can interpret and must not follow
# the version of the policy currently in force.
HISTORICAL_POLICY_VERSIONS = frozenset({"1"})

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class HistoricalReadback:
    """What was verified about a stored report under its own retained policy.

    Only produced when every cited `data/` and `snapshot/` item is present and verified, so
    `not_retained` is always empty here. `repo/` evidence is not retained, so it is counted in
    `not_historical` and not verified.
    The statuses inside the report are the evaluator's original claims. Nothing here is
    authority to run anything.
    """

    report_sha256: str
    policy_sha256: str
    decision: Decision  # recomputed from the stored gates
    blockers: tuple[Blocker, ...]
    binding_sha256: str
    manifest_sha256: str
    verified_evidence: int
    not_retained: list[str]
    not_historical: int
    authorizes: Literal[False] = field(default=False, init=False)


def _retained_policy(data_root: Path, claimed: str) -> ApprovalPolicy:
    relative = f"{ROOT}/evidence/{claimed}.json"
    try:
        content = read_regular(data_root, relative)
    except FileMissing as error:
        raise HistoricalPolicyError(f"retained policy {claimed} is missing") from error
    except FileProblem as error:
        raise HistoricalPolicyError(f"retained policy {claimed} is unsafe: {error}") from error
    if sha256_bytes(content) != claimed:
        raise HistoricalPolicyError("retained policy bytes do not match the recorded SHA-256")
    try:
        value = loads_strict(content)
    except FileProblem as error:
        raise HistoricalPolicyError(f"retained policy is not valid JSON: {error}") from error
    version = value.get("policy_version") if isinstance(value, dict) else None
    if not isinstance(version, str) or version not in HISTORICAL_POLICY_VERSIONS:
        raise UnsupportedHistoricalPolicy(
            f"retained policy version {version!r} is not a supported historical version"
        )
    slots = value.get("slots")
    # a missing slot would otherwise default to unset and read as "nothing approved"
    if not isinstance(slots, dict) or set(SLOT_NAMES) - set(slots):
        raise UnsupportedHistoricalPolicy("retained policy does not have the current schema")
    try:
        policy = ApprovalPolicy.model_validate(value, strict=True)
    except ValueError as error:
        raise UnsupportedHistoricalPolicy(
            f"retained policy does not have the current schema: {str(error)[:300]}"
        ) from error
    if canonical_json_bytes(policy.model_dump(mode="json")) != content:
        raise HistoricalPolicyError("retained policy is not in canonical form")
    return policy


def read_historical(data_root: Path, report_sha256: str) -> HistoricalReadback:
    """Verify a stored report against the policy retained when it was published.

    The report names its policy by digest; every field read before `read_report` succeeds is
    untrusted. Raises `ReadbackError` (including `HistoricalPolicyError`) or
    `PartialPublication`; `HistoricalEvidenceIncomplete` when cited retained evidence is missing.
    Writes nothing.
    """
    if not isinstance(report_sha256, str) or not _SHA256.fullmatch(report_sha256):
        raise ReadbackError("the report SHA-256 must be 64 lowercase hex digits")
    try:
        report_bytes = read_regular(data_root, f"{ROOT}/reports/{report_sha256}.json")
    except FileMissing as error:
        raise ReadbackError(f"no report {report_sha256}") from error
    except FileProblem as error:
        raise ReadbackError(str(error)) from error
    if sha256_bytes(report_bytes) != report_sha256:
        raise ReadbackError("report bytes do not match their SHA-256")
    try:
        claimed = loads_strict(report_bytes).get("policy_sha256")
    except (FileProblem, AttributeError) as error:
        raise ReadbackError(f"report is not readable: {str(error)[:300]}") from error
    if not isinstance(claimed, str) or not _SHA256.fullmatch(claimed):
        raise ReadbackError("the report does not name a policy by SHA-256")

    policy = _retained_policy(data_root, claimed)
    stored = read_report(data_root, report_sha256, policy)
    if stored.not_retained:
        raise HistoricalEvidenceIncomplete(
            report_sha256,
            policy_sha256(policy),
            stored.verified_evidence,
            stored.not_retained,
            stored.not_historical,
        )
    return HistoricalReadback(
        report_sha256=report_sha256,
        policy_sha256=policy_sha256(policy),
        decision=stored.outcome.decision,
        blockers=stored.outcome.blockers,
        binding_sha256=stored.report.binding_sha256,
        manifest_sha256=stored.manifest_sha256,
        verified_evidence=stored.verified_evidence,
        not_retained=stored.not_retained,
        not_historical=stored.not_historical,
    )


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
