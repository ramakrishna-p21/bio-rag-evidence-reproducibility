"""Historical read-back: a stored report verified against the policy retained with it."""

import ast
import dataclasses
import hashlib
import inspect
import json
import typing
from pathlib import Path

import pytest
import readiness_support as support
from readiness_support import REAL_POLICY
from readiness_world import build_world
from test_readiness_guard import build

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.readiness import approvals, cli, guard, publish
from biorag_repro.readiness.evaluate import evaluate
from biorag_repro.readiness.guard import AuthorizationRefused, FinalTestAuthorization
from biorag_repro.readiness.policy import ApprovalPolicy, policy_sha256
from biorag_repro.readiness.publish import (
    HistoricalEvidenceIncomplete,
    HistoricalPolicyError,
    HistoricalReadback,
    PartialPublication,
    ReadbackError,
    UnsupportedHistoricalPolicy,
    publish_report,
    read_historical,
)

REPO = Path(__file__).resolve().parent.parent


def tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def policy_bytes(policy: ApprovalPolicy) -> bytes:
    return canonical_json_bytes(policy.model_dump(mode="json"))


def put(root: Path, relative: str, content: bytes) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha256_bytes(content)


def stub_report(root: Path, claimed: str) -> str:
    """A report-shaped file that only names a policy digest. Enough to reach the policy checks."""
    content = canonical_json_bytes({"policy_sha256": claimed})
    digest = put(root, f"readiness/reports/{'0' * 64}.json", content)
    # content addressed: store under its own digest
    (root / f"readiness/reports/{'0' * 64}.json").rename(root / f"readiness/reports/{digest}.json")
    return digest


def retained(root: Path, policy_hash: str) -> Path:
    return root / "readiness/evidence" / f"{policy_hash}.json"


def _rename(root: Path, name: str) -> str:
    path = root / f"readiness/reports/{name}.json"
    digest = sha256_bytes(path.read_bytes())
    path.rename(root / f"readiness/reports/{digest}.json")
    return digest


@pytest.fixture
def fixture_root(tmp_path):
    policy = support.fixture_policy("1")
    ready = publish_report(build(policy), tmp_path).report_sha256
    not_ready = publish_report(
        build(policy, tag="n", blocked={"G03_final_test_configuration_freeze": "pending"}),
        tmp_path,
    ).report_sha256
    return tmp_path, policy, ready, not_ready


@pytest.fixture
def world(tmp_path, monkeypatch):
    return build_world(tmp_path, monkeypatch)


# ---- positive ---------------------------------------------------------------------------


def test_world_not_ready_report_verifies_with_retained_evidence(world):
    result = publish_report(evaluate(world.ctx, REAL_POLICY), world.data)
    historical = read_historical(world.data, result.report_sha256)
    ordinary = publish.read_report(world.data, result.report_sha256, REAL_POLICY)
    assert historical.decision == "NOT_READY"
    assert historical.report_sha256 == result.report_sha256
    assert historical.policy_sha256 == policy_sha256(REAL_POLICY)
    assert historical.manifest_sha256 == result.manifest_sha256
    assert historical.verified_evidence == ordinary.verified_evidence > 100
    assert historical.not_retained == ordinary.not_retained == []
    assert historical.not_historical == ordinary.not_historical > 0
    assert historical.binding_sha256 == ordinary.report.binding_sha256
    assert historical.blockers == ordinary.outcome.blockers


def test_ready_report_verifies_under_its_own_retained_policy(fixture_root):
    root, policy, ready, _ = fixture_root
    historical = read_historical(root, ready)
    assert historical.decision == "READY" and historical.blockers == ()
    assert historical.policy_sha256 == policy_sha256(policy) != policy_sha256(REAL_POLICY)


def test_synthetic_not_ready_report_verifies(fixture_root):
    root, _, _, not_ready = fixture_root
    historical = read_historical(root, not_ready)
    assert historical.decision == "NOT_READY"
    assert [b.gate_id for b in historical.blockers] == ["G03_final_test_configuration_freeze"]


def test_old_report_still_verifies_after_the_current_policy_changes(fixture_root):
    root, _, ready, _ = fixture_root
    current = support.fixture_policy("fixture-policy-1")
    assert policy_sha256(current) != read_historical(root, ready).policy_sha256
    with pytest.raises(AuthorizationRefused) as refusal:
        guard._authorize(
            root, ready, repo=root, policy=current, evaluate_fn=lambda p: build(current)
        )
    assert refusal.value.reason == "report_unverifiable"
    assert read_historical(root, ready).decision == "READY"


def test_each_report_uses_the_policy_it_recorded(tmp_path):
    first, second = support.fixture_policy("1"), support.fixture_policy("1")
    second = second.model_copy(
        update={
            "slots": support.fixture_policy("1").slots.model_copy(
                update={"corpus_identity": sha256_bytes(b"other")}
            )
        }
    )
    a = publish_report(build(first), tmp_path).report_sha256
    b = publish_report(build(second), tmp_path).report_sha256
    assert read_historical(tmp_path, a).policy_sha256 == policy_sha256(first)
    assert read_historical(tmp_path, b).policy_sha256 == policy_sha256(second)


def test_result_is_immutable_and_never_authorizes(fixture_root):
    root, _, ready, _ = fixture_root
    historical = read_historical(root, ready)
    assert historical.authorizes is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        historical.authorizes = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        historical.decision = "NOT_READY"  # type: ignore[misc]
    with pytest.raises(TypeError):
        HistoricalReadback(authorizes=True, **_fields(historical))  # type: ignore[call-arg]
    assert typing.get_type_hints(HistoricalReadback)["authorizes"] == typing.Literal[False]
    assert not isinstance(historical, FinalTestAuthorization)
    assert not issubclass(HistoricalReadback, FinalTestAuthorization)
    assert not hasattr(historical, "token")


def _fields(h):
    return {f.name: getattr(h, f.name) for f in dataclasses.fields(h) if f.init}


def test_repeated_verification_is_deterministic_and_writes_nothing(fixture_root):
    root, _, ready, not_ready = fixture_root
    before = tree(root)
    first = [read_historical(root, s) for s in (ready, not_ready)]
    second = [read_historical(root, s) for s in (ready, not_ready)]
    assert first == second
    assert tree(root) == before


def test_verification_goes_through_read_report_with_the_retained_policy(fixture_root, monkeypatch):
    root, policy, ready, _ = fixture_root
    calls = []
    original = publish.read_report

    def spy(data_root, sha, used):
        calls.append((sha, used))
        return original(data_root, sha, used)

    monkeypatch.setattr(publish, "read_report", spy)
    read_historical(root, ready)
    assert calls == [(ready, policy)]


# ---- report stage -----------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "abc", "A" * 64, "../" + "a" * 61, "g" * 64, "a" * 63])
def test_malformed_report_digest_is_refused(fixture_root, bad):
    with pytest.raises(ReadbackError, match="SHA-256"):
        read_historical(fixture_root[0], bad)


def test_non_string_digest_is_refused(fixture_root):
    with pytest.raises(ReadbackError):
        read_historical(fixture_root[0], None)  # type: ignore[arg-type]


def test_missing_report(fixture_root):
    with pytest.raises(ReadbackError, match="no report") as error:
        read_historical(fixture_root[0], "a" * 64)
    assert not isinstance(error.value, HistoricalPolicyError)


def test_report_bytes_that_differ_from_the_requested_digest(fixture_root):
    root, _, ready, _ = fixture_root
    (root / f"readiness/reports/{ready}.json").write_bytes(b"{}")
    with pytest.raises(ReadbackError, match="do not match"):
        read_historical(root, ready)


def test_noncanonical_report_json(fixture_root):
    root, _, ready, _ = fixture_root
    data = json.loads((root / f"readiness/reports/{ready}.json").read_bytes())
    put(root, "readiness/reports/tmp.json", json.dumps(data, indent=2).encode())
    with pytest.raises(ReadbackError, match="canonical"):
        read_historical(root, _rename(root, "tmp"))


def test_duplicate_keys_in_report(fixture_root):
    root, _, ready, _ = fixture_root
    original = (root / f"readiness/reports/{ready}.json").read_bytes()
    put(root, "readiness/reports/tmp.json", b'{"kind":"readiness_report",' + original[1:])
    with pytest.raises(ReadbackError, match="duplicate"):
        read_historical(root, _rename(root, "tmp"))


@pytest.mark.parametrize("claimed", ["zz", "A" * 64, "a" * 63, 7, None])
def test_report_claiming_a_malformed_policy_digest(fixture_root, claimed):
    root = fixture_root[0]
    put(root, "readiness/reports/tmp.json", canonical_json_bytes({"policy_sha256": claimed}))
    with pytest.raises(ReadbackError, match="policy") as error:
        read_historical(root, _rename(root, "tmp"))
    assert not isinstance(error.value, HistoricalPolicyError)


def test_report_without_a_policy_claim(fixture_root):
    root = fixture_root[0]
    put(root, "readiness/reports/tmp.json", canonical_json_bytes({"kind": "other"}))
    with pytest.raises(ReadbackError, match="policy"):
        read_historical(root, _rename(root, "tmp"))


def test_symlinked_report(fixture_root, tmp_path_factory):
    root, _, ready, _ = fixture_root
    path = root / f"readiness/reports/{ready}.json"
    outside = tmp_path_factory.mktemp("outside") / "r.json"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ReadbackError, match="link"):
        read_historical(root, ready)


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda d: d.update(decision="READY", blockers=[]), "invalid report"),
        (lambda d: d["gates"].pop(), "invalid report"),
        (lambda d: d.update(binding_sha256="1" * 64), "invalid report"),
        (lambda d: d.update(blockers=[]), "invalid report"),
    ],
    ids=["forged_ready", "gate_set", "binding", "blockers"],
)
def test_forged_not_ready_report(fixture_root, change, message):
    root, _, _, not_ready = fixture_root
    data = json.loads((root / f"readiness/reports/{not_ready}.json").read_bytes())
    change(data)
    put(root, "readiness/reports/tmp.json", canonical_json_bytes(data))
    with pytest.raises(ReadbackError, match=message):
        read_historical(root, _rename(root, "tmp"))


# ---- manifest and retained evidence ------------------------------------------------------


def test_missing_manifest_is_a_partial_publication(fixture_root):
    root, _, ready, _ = fixture_root
    (root / f"readiness/reports/{ready}.build.json").unlink()
    with pytest.raises(PartialPublication):
        read_historical(root, ready)


def test_altered_manifest(fixture_root):
    root, _, ready, _ = fixture_root
    path = root / f"readiness/reports/{ready}.build.json"
    data = json.loads(path.read_bytes())
    data["status"] = "failed"
    path.write_bytes(json.dumps(data).encode())
    with pytest.raises(ReadbackError, match="manifest"):
        read_historical(root, ready)


def test_altered_retained_snapshot(world):
    result = publish_report(evaluate(world.ctx, REAL_POLICY), world.data)
    snapshot = next(
        p
        for p in sorted((world.data / "readiness/evidence").iterdir())
        if p.name != f"{policy_sha256(REAL_POLICY)}.json"
    )
    snapshot.write_bytes(snapshot.read_bytes() + b" ")
    with pytest.raises(ReadbackError, match="differs"):
        read_historical(world.data, result.report_sha256)


def test_missing_snapshot_makes_the_historical_result_incomplete(fixture_root):
    root, policy, ready, _ = fixture_root
    body = canonical_json_bytes({"gate": "G01_corpus_sources", "tag": ""})
    (root / f"readiness/evidence/{sha256_bytes(body)}.json").unlink()
    before = tree(root)
    with pytest.raises(HistoricalEvidenceIncomplete) as incomplete:
        read_historical(root, ready)
    ordinary = publish.read_report(root, ready, policy)
    error = incomplete.value
    assert isinstance(error, ReadbackError)
    assert error.not_retained == ordinary.not_retained and len(error.not_retained) == 1
    assert error.verified_evidence == ordinary.verified_evidence
    assert error.not_historical == ordinary.not_historical
    assert error.report_sha256 == ready and error.policy_sha256 == policy_sha256(policy)
    assert tree(root) == before  # nothing recreated or repaired


def test_missing_data_evidence_makes_the_historical_result_incomplete(world):
    result = publish_report(evaluate(world.ctx, REAL_POLICY), world.data)
    report = publish.read_report(world.data, result.report_sha256, REAL_POLICY).report
    locator = next(
        r.locator for g in report.gates for r in g.evidence if r.locator.startswith("data/")
    )
    (world.data / locator.removeprefix("data/")).unlink()
    before = tree(world.data)
    with pytest.raises(HistoricalEvidenceIncomplete) as incomplete:
        read_historical(world.data, result.report_sha256)
    assert locator in incomplete.value.not_retained
    assert incomplete.value.not_historical > 0  # repo/ items are reported apart from missing ones
    assert tree(world.data) == before


def test_altered_evidence_is_an_integrity_failure_not_incompleteness(fixture_root):
    root, _, ready, _ = fixture_root
    body = canonical_json_bytes({"gate": "G01_corpus_sources", "tag": ""})
    path = root / f"readiness/evidence/{sha256_bytes(body)}.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ReadbackError, match="differs") as error:
        read_historical(root, ready)
    assert not isinstance(error.value, HistoricalEvidenceIncomplete)


def test_incomplete_error_is_not_a_readback_and_never_authorizes(fixture_root):
    root, _, ready, _ = fixture_root
    body = canonical_json_bytes({"gate": "G01_corpus_sources", "tag": ""})
    (root / f"readiness/evidence/{sha256_bytes(body)}.json").unlink()
    with pytest.raises(HistoricalEvidenceIncomplete) as incomplete:
        read_historical(root, ready)
    assert not isinstance(incomplete.value, HistoricalReadback | FinalTestAuthorization)
    assert not hasattr(incomplete.value, "authorizes")


def test_complete_historical_result_has_nothing_missing(fixture_root):
    root, _, ready, _ = fixture_root
    assert read_historical(root, ready).not_retained == []


def test_altered_data_evidence(world):
    result = publish_report(evaluate(world.ctx, REAL_POLICY), world.data)
    report = publish.read_report(world.data, result.report_sha256, REAL_POLICY).report
    locator = next(
        r.locator for g in report.gates for r in g.evidence if r.locator.startswith("data/")
    )
    target = world.data / locator.removeprefix("data/")
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ReadbackError, match="differs"):
        read_historical(world.data, result.report_sha256)


# ---- retained policy --------------------------------------------------------------------


def test_missing_retained_policy(fixture_root):
    root, policy, ready, _ = fixture_root
    retained(root, policy_sha256(policy)).unlink()
    with pytest.raises(HistoricalPolicyError, match="is missing"):
        read_historical(root, ready)


def test_retained_policy_with_other_bytes(fixture_root):
    root, policy, ready, _ = fixture_root
    retained(root, policy_sha256(policy)).write_bytes(policy_bytes(support.fixture_policy("x")))
    with pytest.raises(HistoricalPolicyError, match="do not match"):
        read_historical(root, ready)


def test_invented_policy_stored_under_the_requested_digest(tmp_path):
    claimed = "c" * 64
    stub = stub_report(tmp_path, claimed)
    put(tmp_path, f"readiness/evidence/{claimed}.json", policy_bytes(support.fixture_policy()))
    with pytest.raises(HistoricalPolicyError, match="do not match"):
        read_historical(tmp_path, stub)


def test_noncanonical_policy_even_when_its_bytes_hash_to_the_claim(tmp_path):
    pretty = json.dumps(support.fixture_policy("1").model_dump(mode="json"), indent=2).encode()
    claimed = sha256_bytes(pretty)
    stub = stub_report(tmp_path, claimed)
    put(tmp_path, f"readiness/evidence/{claimed}.json", pretty)
    with pytest.raises(HistoricalPolicyError, match="canonical"):
        read_historical(tmp_path, stub)


def test_duplicate_keys_in_policy(tmp_path):
    content = b'{"policy_version":"1","policy_version":"2","slots":{}}'
    claimed = sha256_bytes(content)
    stub = stub_report(tmp_path, claimed)
    put(tmp_path, f"readiness/evidence/{claimed}.json", content)
    with pytest.raises(HistoricalPolicyError, match="duplicate"):
        read_historical(tmp_path, stub)


def _policy_dict(**changes):
    data = support.fixture_policy("1").model_dump(mode="json")
    data.update(changes)
    return data


def _without_version(data):
    del data["policy_version"]
    return data


def _drop_slot(data):
    del data["slots"]["gpu_authorization"]
    return data


def _extra_slot(data):
    data["slots"]["new_slot"] = None
    return data


@pytest.mark.parametrize(
    "data",
    [
        _policy_dict(extra_field=1),
        _drop_slot(_policy_dict()),
        _extra_slot(_policy_dict()),
        _policy_dict(policy_version=""),
        _policy_dict(policy_version=1),
        _policy_dict(policy_version="2"),
        _policy_dict(policy_version=None),
        _without_version(_policy_dict()),
        {"policy_version": "0", "approvals": []},
        [],
    ],
    ids=[
        "extra_field",
        "missing_slot",
        "extra_slot",
        "blank_version",
        "int_version",
        "version_2",
        "null_version",
        "missing_version",
        "old",
        "list",
    ],
)
def test_policy_the_current_schema_cannot_represent_is_unsupported(tmp_path, data):
    content = canonical_json_bytes(data)
    claimed = sha256_bytes(content)
    stub = stub_report(tmp_path, claimed)
    put(tmp_path, f"readiness/evidence/{claimed}.json", content)
    with pytest.raises(UnsupportedHistoricalPolicy, match="schema|version"):
        read_historical(tmp_path, stub)


def test_unknown_version_is_refused_at_the_version_boundary(tmp_path):
    content = canonical_json_bytes(_policy_dict(policy_version="2"))  # structure otherwise valid
    claimed = sha256_bytes(content)
    stub = stub_report(tmp_path, claimed)
    put(tmp_path, f"readiness/evidence/{claimed}.json", content)
    with pytest.raises(UnsupportedHistoricalPolicy, match="version"):
        read_historical(tmp_path, stub)


def test_version_1_with_an_unsupported_structure_is_refused(tmp_path):
    content = canonical_json_bytes(_policy_dict(policy_version="1", extra_field=1))
    claimed = sha256_bytes(content)
    stub = stub_report(tmp_path, claimed)
    put(tmp_path, f"readiness/evidence/{claimed}.json", content)
    with pytest.raises(UnsupportedHistoricalPolicy, match="schema"):
        read_historical(tmp_path, stub)


def test_historical_version_1_does_not_follow_the_current_policy_version(fixture_root, monkeypatch):
    root, _, ready, _ = fixture_root
    monkeypatch.setattr(approvals, "POLICY_VERSION", "2")
    assert read_historical(root, ready).decision == "READY"
    assert publish.HISTORICAL_POLICY_VERSIONS == frozenset({"1"})


def test_symlinked_retained_policy(fixture_root, tmp_path_factory):
    root, policy, ready, _ = fixture_root
    path = retained(root, policy_sha256(policy))
    outside = tmp_path_factory.mktemp("outside") / "p.json"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(HistoricalPolicyError, match="link"):
        read_historical(root, ready)


def test_symlinked_evidence_directory(fixture_root, tmp_path_factory):
    root, _, ready, _ = fixture_root
    moved = tmp_path_factory.mktemp("outside") / "evidence"
    (root / "readiness/evidence").rename(moved)
    (root / "readiness/evidence").symlink_to(moved)
    with pytest.raises(HistoricalPolicyError, match="link"):
        read_historical(root, ready)


def test_caller_policy_cannot_replace_the_retained_one():
    parameters = inspect.signature(read_historical).parameters
    assert list(parameters) == ["data_root", "report_sha256"]


def test_stub_policy_stage_failures_write_nothing(tmp_path):
    claimed = "d" * 64
    stub = stub_report(tmp_path, claimed)
    before = tree(tmp_path)
    with pytest.raises(HistoricalPolicyError):
        read_historical(tmp_path, stub)
    assert tree(tmp_path) == before


def test_a_retained_copy_says_nothing_about_the_tracked_policy(fixture_root):
    root, policy, ready, _ = fixture_root
    historical = read_historical(root, ready)
    assert policy_sha256(policy) != policy_sha256(REAL_POLICY)
    assert not any(hasattr(historical, n) for n in ("approved", "tracked", "approver", "authority"))
    assert "tracked" in (publish.__doc__ or "") and "approv" in (publish.__doc__ or "")


# ---- current authorization stays separate -------------------------------------------------


def test_real_guard_refuses_a_historically_ready_report_under_the_production_policy(fixture_root):
    root, _, ready, _ = fixture_root
    assert read_historical(root, ready).decision == "READY"
    with pytest.raises(AuthorizationRefused) as refusal:
        guard.authorize_final_test(root, ready, repo=REPO)
    assert refusal.value.reason == "report_unverifiable"


def test_historical_result_is_not_accepted_by_the_guard(fixture_root):
    root, policy, ready, _ = fixture_root
    historical = read_historical(root, ready)
    with pytest.raises((AuthorizationRefused, OSError)):  # never a FinalTestAuthorization
        guard._authorize(
            root,
            historical,  # type: ignore[arg-type]
            repo=root,
            policy=policy,
            evaluate_fn=lambda p: build(policy),
        )


@pytest.mark.parametrize("module", [guard, approvals, cli])
def test_production_modules_never_reference_the_historical_reader(module):
    names = {
        n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else n.name
        for n in ast.walk(ast.parse(inspect.getsource(module)))
        if isinstance(n, ast.Name | ast.Attribute | ast.alias)
    }
    assert not {n for n in names if "Historical" in n or "read_historical" in n}


def test_policy_loading_has_no_fallback_to_retained_policies(fixture_root, tmp_path):
    root = fixture_root[0]
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(approvals.PolicyError):
        approvals.load_policy(repo)
    assert any((root / "readiness/evidence").iterdir())


def test_publish_does_not_call_the_historical_reader():
    source = inspect.getsource(publish)
    tree_ = ast.parse(source)
    for function in ast.walk(tree_):
        if isinstance(function, ast.FunctionDef) and function.name != "read_historical":
            called = {
                n.func.id
                for n in ast.walk(function)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            assert "read_historical" not in called
