"""G13 against the final-test root: synthetic positives under a fixture policy, real policy blocked."""

import ast
import inspect
from pathlib import Path

import pytest
from finaltest_support import authorization
from readiness_support import REAL_POLICY
from readiness_world import build_world

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.finaltest import inventory, layout
from biorag_repro.readiness import evidence, publish
from biorag_repro.readiness.decide import verify_report
from biorag_repro.readiness.evaluate import evaluate, gather_all
from biorag_repro.readiness.gather import to_result
from biorag_repro.readiness.policy import ApprovalPolicy, ApprovalSlots

G13 = "G13_final_test_history_integrity"
NO_AUTHORITY = "final_test_inventory_authority_not_established"


@pytest.fixture
def world(tmp_path, monkeypatch):
    return build_world(tmp_path, monkeypatch)


class Setup:
    def __init__(self, world, base: Path):
        self.world, self.base = world, base
        base.mkdir()
        release, data = evidence.read_benchmark_record(world.ctx)
        self.scope = {
            "corpus_content_sha256": release.corpus_content_sha256,
            "benchmark_release_sha256": sha256_bytes(data),
        }
        self.anchors = world.data / evidence.ROOT_ANCHORS

    def approve(self, kind="final_test", **changes):
        auth = authorization(self.base, kind, **(self.scope | changes))
        digest = layout.authorization_sha256(auth)
        folder = self.world.data / evidence.STORAGE_APPROVALS
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{digest}.json").write_bytes(canonical_json_bytes(auth.model_dump(mode="json")))
        policy = ApprovalPolicy(
            policy_version="fixture-g11", slots=ApprovalSlots(storage_authorization=digest)
        )
        return auth, policy

    def initialize(self, auth):
        inventory._initialize(Path(auth.authorized_root), auth, self.anchors, inventory.utc_now)

    def g13(self, policy):
        return evidence.history_integrity(self.world.ctx, policy)


@pytest.fixture
def setup(world, tmp_path):
    return Setup(world, tmp_path / "storage")


def test_real_policy_keeps_g13_unavailable_whatever_exists(setup):
    auth, _ = setup.approve()
    setup.initialize(auth)  # a perfect synthetic final_test root and approval artifact exist
    for policy in (REAL_POLICY, None):
        g = setup.g13(policy)
        assert (g.state, g.reason) == ("unavailable", NO_AUTHORITY)
    assert evidence.history_integrity().state == "unavailable"


def test_real_evaluation_reports_g13_unavailable(setup, world):
    auth, _ = setup.approve()
    setup.initialize(auth)
    got = {g.gate_id: g for g in evaluate(world.ctx, REAL_POLICY).report.gates}[G13]
    assert (got.status, got.reason_code) == ("unavailable", NO_AUTHORITY)


def test_verified_empty_final_test_root_passes_under_a_fixture_policy(setup, world):
    auth, policy = setup.approve()
    setup.initialize(auth)
    gathered = setup.g13(policy)
    assert (gathered.state, gathered.reason) == ("ok", "no_prior_final_test_execution")
    result = to_result(G13, gathered, policy)
    assert (result.status, result.reason_code, result.policy_expectation) == (
        "passed",
        "no_prior_final_test_execution",
        None,
    )
    kinds = {r.kind for r in result.evidence}
    assert kinds == {"storage_authorization", "final_test_root_anchor", "final_test_inventory"}


def test_g11_uses_the_existing_approved_storage_subject(setup, world):
    _, policy = setup.approve()
    gathered = evidence.storage_authorization(world.ctx, policy)
    result = to_result("G11_storage_authorization", gathered, policy)
    assert gathered.state == "ok"
    assert result.status == "passed"
    assert {ref.kind for ref in result.evidence} == {"storage_authorization", "observed_subject"}


def test_passing_g13_report_publishes_and_reads_back(setup, world):
    auth, policy = setup.approve()
    setup.initialize(auth)
    report = evaluate(world.ctx, policy)
    assert {g.gate_id: g for g in report.report.gates}[G13].status == "passed"
    published = publish.publish_report(report, world.data)
    again = publish.read_report(world.data, published.report_sha256, policy)
    assert {g.gate_id: g for g in again.report.gates}[G13].status == "passed"
    assert verify_report(again.report, policy).decision == "NOT_READY"  # other gates still open


def test_the_decision_with_a_passing_g13_is_still_not_ready(setup, world):
    auth, policy = setup.approve()
    setup.initialize(auth)
    assert evaluate(world.ctx, policy).report.decision == "NOT_READY"


def test_rehearsal_root_never_satisfies_g13(setup):
    auth, policy = setup.approve(kind="rehearsal")
    inventory.initialize_root(Path(auth.authorized_root), auth, setup.anchors)
    g = setup.g13(policy)
    assert (g.state, g.reason, g.detail) == ("unavailable", NO_AUTHORITY, "root_kind")


def test_final_test_approval_over_a_rehearsal_root_is_refused(setup):
    rehearsal = authorization(setup.base, "rehearsal", **setup.scope)
    inventory.initialize_root(Path(rehearsal.authorized_root), rehearsal, setup.anchors)
    _, policy = setup.approve()  # approves a final_test root at the same location
    g = setup.g13(policy)
    assert (g.state, g.reason) == ("unavailable", NO_AUTHORITY)


def test_missing_approval_artifact_is_not_authority(setup):
    auth, policy = setup.approve()
    setup.initialize(auth)
    for path in (setup.world.data / evidence.STORAGE_APPROVALS).iterdir():
        path.unlink()
    assert setup.g13(policy).detail == "storage_approval_unusable"


def test_approval_artifact_must_match_the_slot_and_be_canonical(setup):
    auth, policy = setup.approve()
    setup.initialize(auth)
    path = next((setup.world.data / evidence.STORAGE_APPROVALS).iterdir())
    path.write_bytes(path.read_bytes() + b"\n")
    assert setup.g13(policy).detail == "storage_approval_unusable"
    other = ApprovalPolicy(
        policy_version="x", slots=ApprovalSlots(storage_authorization=sha256_bytes(b"other"))
    )
    path.write_bytes(canonical_json_bytes(auth.model_dump(mode="json")))
    assert setup.g13(other).detail == "storage_approval_unusable"


@pytest.mark.parametrize("field", ["corpus_content_sha256", "benchmark_release_sha256"])
def test_approval_for_another_corpus_or_benchmark_is_refused(setup, field):
    auth, policy = setup.approve(**{field: "a" * 64})
    setup.initialize(auth)
    assert setup.g13(policy).detail == "scope"


def test_the_root_location_comes_from_the_approval_not_the_caller(setup, tmp_path):
    auth, policy = setup.approve()
    setup.initialize(auth)
    forged = tmp_path / "forged" / "final_test_v1"
    forged.parent.mkdir()
    # an unrelated populated directory elsewhere cannot be offered: no parameter accepts it
    assert list(inspect.signature(evidence.history_integrity).parameters) == ["ctx", "policy"]
    assert setup.g13(policy).state == "ok"


def test_uninitialized_or_empty_approved_location_is_unavailable(setup):
    auth, policy = setup.approve()
    assert setup.g13(policy).detail == "not_initialized"
    Path(auth.authorized_root).mkdir(parents=True)
    assert setup.g13(policy).detail == "not_initialized"


def test_partial_initialization_maps_to_the_incomplete_category(setup):
    auth, policy = setup.approve()
    setup.initialize(auth)
    next(setup.anchors.iterdir()).unlink()
    g = setup.g13(policy)
    assert (g.state, g.reason) == ("unavailable", "history_inventory_incomplete")
    assert to_result(G13, g, policy).status == "unavailable"


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        (lambda r: (r / "extra").mkdir(), "unauthorized_final_test_artifact"),
        (
            lambda r: (
                (r / "registry" / "000001.json").chmod(0o644)
                or (r / "registry" / "000001.json").write_bytes(b"{")
            ),
            "unverified_final_test_artifact",
        ),
    ],
    ids=["unexpected-directory", "corrupted-genesis-event"],
)
def test_artifact_problems_fail_the_gate_with_the_history_codes(setup, damage, reason):
    auth, policy = setup.approve()
    setup.initialize(auth)
    damage(Path(auth.authorized_root))
    g = setup.g13(policy)
    assert (g.state, g.reason) == ("failed", reason)
    assert to_result(G13, g, policy).status == "failed"


def test_conflicting_genesis_fails_with_the_conflict_code(setup):
    auth, policy = setup.approve()
    setup.initialize(auth)
    root = Path(auth.authorized_root)
    first = (root / "registry" / "000001.json").read_bytes()
    event = layout.make_event(
        root_id=layout.root_id_for(auth),
        sequence=2,
        event_type="root_initialized",
        previous_event_sha256=sha256_bytes(first),
        evidence=[],
        created_utc="2026-01-01T00:00:00Z",
    )
    (root / "registry" / "000002.json").write_bytes(layout.serialize(event))
    assert setup.g13(policy).reason == "conflicting_final_test_artifact"


def test_g13_state_does_not_depend_on_other_policy_slots(setup):
    auth, policy = setup.approve()
    setup.initialize(auth)
    slots = policy.slots.model_dump() | {"corpus_identity": sha256_bytes(b"x")}
    wider = ApprovalPolicy(policy_version="w", slots=ApprovalSlots(**slots))
    assert setup.g13(wider).state == "ok"


def test_gather_all_without_a_policy_is_blocked(world):
    assert gather_all(world.ctx)[G13].reason == NO_AUTHORITY


def test_no_cli_argument_environment_or_file_selects_a_fixture_policy():
    from biorag_repro.readiness import cli

    package = Path(evidence.__file__).parent
    text = (package / "cli.py").read_text()
    assert list(inspect.signature(cli.main).parameters) == ["argv"]
    for banned in ("--policy", "os.environ", "getenv"):
        assert banned not in text
    source = (package / "evidence.py").read_text()
    for banned in ("os.environ", "getenv", "argparse"):
        assert banned not in source


def test_only_the_final_test_inventory_imports_the_history_classifier():
    root = Path(evidence.__file__).parents[1]
    importers = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                "readiness.history"
            ):
                importers.append(str(path.relative_to(root)))
    assert sorted(set(importers)) == ["finaltest/inventory.py"]


def test_production_code_cannot_initialize_a_final_test_root():
    root = Path(evidence.__file__).parents[1]
    callers = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if "_initialize(" in p.read_text() or "initialize_root(" in p.read_text()
    ]
    assert callers == ["finaltest/inventory.py"]


def test_g13_passes_only_with_a_named_history_class():
    from biorag_repro.readiness.gather import Gathered

    other = Gathered("ok", "checks_passed", "s", subject={"x": 1})
    assert to_result(G13, other, REAL_POLICY).status == "pending"


def test_real_policy_names_the_missing_approval(setup):
    auth, _ = setup.approve()
    setup.initialize(auth)
    assert setup.g13(REAL_POLICY).detail == "storage_approval_unusable"
    assert evidence.history_integrity(setup.world.ctx).detail == "no approval to verify against"


def test_no_execution_module_imports_the_final_test_storage_package():
    root = Path(evidence.__file__).parents[1]
    allowed = {"finaltest", "readiness"}
    offenders = []
    for path in root.rglob("*.py"):
        if path.relative_to(root).parts[0] in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom | ast.Import) and "finaltest" in ast.unparse(node):
                offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_run_phase_still_excludes_final_test():
    from typing import get_args

    from biorag_repro.generation.attempts import RunPhase

    assert "final_test" not in get_args(RunPhase)


def _second_event(root):
    first = (root / "registry" / "000001.json").read_bytes()
    genesis = layout.RegistryEvent.model_validate_json(first)
    cite = layout.EvidenceCitation
    return layout.make_event(
        root_id=genesis.root_id,
        sequence=2,
        event_type="batch_reserved",
        previous_event_sha256=sha256_bytes(first),
        evidence=[cite(role=r, sha256="a" * 64) for r in layout.REQUIRED_ROLES["batch_reserved"]],
        created_utc="2026-01-02T00:00:00Z",
    )


@pytest.mark.parametrize("history", ["registry_event", "batches", "runs"])
def test_any_execution_history_fails_g13(setup, history):
    """No execution path exists: a correctly chained later event or any batch or run entry fails."""
    auth, policy = setup.approve()
    setup.initialize(auth)
    root = Path(auth.authorized_root)
    if history == "registry_event":
        (root / "registry" / "000002.json").write_bytes(layout.serialize(_second_event(root)))
    else:
        (root / history).mkdir()
    verdict = inventory.verify_root(root, auth, setup.anchors)
    assert verdict.code == "unauthorized_final_test_artifact"
    gathered = setup.g13(policy)
    assert gathered.state == "failed" and gathered.reason == "unauthorized_final_test_artifact"
    assert to_result(G13, gathered, policy).status != "passed"
