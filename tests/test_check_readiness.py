"""Decision rule of the candidate readiness check (scripts/check_readiness.py)."""

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from scripts.check_readiness import CHECKS, evaluate


def checks(failing=()):
    def make(gate):
        def check(_inputs):
            if gate in failing:
                raise ValueError("evidence does not verify")
            return {"gate": gate}

        return check

    return {gate: (slot, make(gate)) for gate, (slot, _) in CHECKS.items()}


def approved():
    return {
        slot: sha256_bytes(canonical_json_bytes({"gate": gate}))
        for gate, (slot, _) in CHECKS.items()
        if slot is not None
    }


def test_unset_slots_are_pending_and_the_decision_is_not_ready():
    report = evaluate({}, {}, checks())
    assert report["decision"] == "NOT_READY" and report["authoritative"] is False
    statuses = {g: r["status"] for g, r in report["gates"].items()}
    assert statuses.pop("G13_final_test_history_integrity") == "passed"
    assert set(statuses.values()) == {"pending"}
    assert len(report["gates"]) == 13


def test_ready_needs_every_gate_and_a_failed_or_mismatched_one_blocks():
    assert evaluate({}, approved(), checks())["decision"] == "READY"
    failed = evaluate({}, approved(), checks(failing={"G06_python_environment"}))
    assert failed["decision"] == "NOT_READY"
    assert failed["gates"]["G06_python_environment"]["status"] == "failed"
    history = evaluate({}, approved(), checks(failing={"G13_final_test_history_integrity"}))
    assert history["decision"] == "NOT_READY"
    wrong = approved() | {"seed_approvals": "0" * 64}
    assert evaluate({}, wrong, checks())["gates"]["G08_seed_registry"]["status"] == "failed"
