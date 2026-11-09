"""The environment verification is cited by the attempt events of an official run.

All values are synthetic. The guarantee tested is integrity and recorded-event ordering in a
hash chain; the timestamps are written by the producing process and prove nothing about the
real clock.
"""

# ruff: noqa: F811  (fixtures imported from the official-run tests)

import json

import pytest
from synthetic_environment import LOCK, evidence
from test_generation_attempts import coordinates, environment
from test_generation_official import (  # noqa: F401  (fixtures)
    anchored,
    build_world,
    recognised,
    verify,
    view,
)
from test_generation_runner import make_store

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.generation.attempts import (
    AttemptEvent,
    CorruptRecord,
    NonOfficialBackend,
    RetryPolicy,
    RunRecord,
)
from biorag_repro.generation.environment_gate import (
    EnvironmentEvidence,
    publish_environment_evidence,
)
from biorag_repro.generation.receipt import (
    ExecutionFacts,
    ExecutionReceipt,
    GenerationSettings,
    token_ids_sha256,
)

RENDERED = "f" * 64


def h(char):
    return char * 64


def running_event(**changes):
    values = dict(  # noqa: C408
        run_id="1" * 32,
        run_record_sha256=h("a"),
        observation_key="obs_" + h("b"),
        sequence=2,
        previous_event_sha256=h("c"),
        state="running",
        at_utc="2026-01-01T00:00:00Z",
        attempt_id="2" * 32,
        attempt_number=1,
        backend_kind="mock",
        environment_sha256=h("d"),
        rendered_input_sha256=h("e"),
    )
    return AttemptEvent(**(values | changes))


def digest(model, kind):
    return sha256_bytes(serialize_project_json(model, kind))


# ---- artifacts written before this change keep their bytes and hashes ---------------------


def test_a_running_event_without_a_citation_keeps_its_historical_hash():
    # Value computed with the code of the previous commit.
    assert (
        digest(running_event(), AttemptEvent)
        == "ae856d36c8e540efc013e69737f9d42fcbace40072a4d54543281346c64160fd"
    )
    assert b"environment_verification" not in serialize_project_json(running_event(), AttemptEvent)


def test_a_receipt_without_measurements_keeps_its_historical_hash():
    facts = ExecutionFacts(
        input_token_ids_sha256=h("1"),
        input_token_count=5,
        generation=GenerationSettings(
            max_new_tokens=3, do_sample=False, temperature=None, top_p=None, top_k=None,
            eos_token_ids=[9], pad_token_id=None, seed=None, effective={},
        ),
        output_token_ids=[4, 9],
        stop_reason="eos",
    )  # fmt: skip
    receipt = ExecutionReceipt(
        run_id="1" * 32,
        observation_key="obs_" + h("b"),
        attempt_id="2" * 32,
        environment_sha256=h("d"),
        input_record_sha256=h("f"),
        rendered_input_sha256=h("e"),
        raw_output_sha256=h("7"),
        output_token_ids_sha256=token_ids_sha256([4, 9]),
        facts=facts,
    )
    assert (
        digest(receipt, ExecutionReceipt)
        == "85fc1b702c7c62df9a03f0627bcdf934e24c5e08f881a9ce417d925026747e0b"
    )


def test_a_development_run_record_keeps_its_historical_hash():
    record = RunRecord(
        run_id="1" * 32,
        created_utc="2026-01-01T00:00:00Z",
        phase="development",
        benchmark_version="b",
        resolved_config_sha256=h("a"),
        raw_config_sha256=h("b"),
        purpose="synthetic_test",
        matrix_arm="confirmatory",
        backend_kind="mock",
        retry=RetryPolicy(max_attempts=2, retryable_failures=["out_of_memory"]),
    )
    assert (
        digest(record, RunRecord)
        == "614c9b8c37f8538a6e3658d69c6c088f01e61bb4135f8c1b194f42b2e3b60f24"
    )


# ---- the event model -----------------------------------------------------------------------


def test_a_citation_is_a_hash_on_a_running_event_of_an_official_backend_only():
    cited = running_event(backend_kind="official", environment_verification_sha256=h("9"))
    assert b'"environment_verification_sha256"' in serialize_project_json(cited, AttemptEvent)
    with pytest.raises(ValueError, match="official"):
        running_event(backend_kind="mock", environment_verification_sha256=h("9"))
    with pytest.raises(ValueError, match="running"):
        running_event(
            state="interrupted",
            backend_kind=None,
            environment_sha256=None,
            rendered_input_sha256=None,
            environment_verification_sha256=h("9"),
        )


# ---- append ---------------------------------------------------------------------------------


def official_store(tmp_path):
    store, _ = make_store(tmp_path, "official")
    key = store.plan(coordinates())
    return store, key


def start(store, key):
    return store.begin_attempt(
        key, environment=environment("official"), rendered_input_sha256=RENDERED
    )


def verification_path(store):
    return store.run_dir / "environment_evidence" / "verification.json"


def test_an_official_attempt_cannot_start_without_stored_verification(tmp_path):
    store, key = official_store(tmp_path)
    with pytest.raises(NonOfficialBackend, match="environment verification"):
        start(store, key)
    assert store.observation(key).status == "planned"  # no event was written


def test_the_first_attempt_cites_the_stored_verification(tmp_path):
    store, key = official_store(tmp_path)
    publish_environment_evidence(store.run_dir, evidence())
    start(store, key)
    running = [e for e in store.observation(key).events if e.state == "running"]
    assert running[0].environment_verification_sha256 == sha256_bytes(
        verification_path(store).read_bytes()
    )


def test_evidence_stored_late_cannot_be_cited_by_attempts_that_began_without_it(tmp_path):
    store, key = official_store(tmp_path)
    attempt = None
    with pytest.raises(NonOfficialBackend):
        attempt = start(store, key)
    assert attempt is None
    publish_environment_evidence(store.run_dir, evidence())
    start(store, key)  # the first attempt is the first event that cites it
    assert store.observation(key).events[1].environment_verification_sha256


def test_a_mock_attempt_cites_nothing(tmp_path):
    store, _ = make_store(tmp_path)
    key = store.plan(coordinates())
    store.begin_attempt(key, environment=environment("mock"), rendered_input_sha256=RENDERED)
    assert store.observation(key).events[1].environment_verification_sha256 is None


# ---- replay ----------------------------------------------------------------------------------


def test_a_replaced_verification_is_found_on_replay(tmp_path):
    store, key = official_store(tmp_path)
    publish_environment_evidence(store.run_dir, evidence())
    start(store, key)
    other = evidence(python_version="3.11.10")
    path = verification_path(store)
    path.write_bytes(serialize_project_json(other.report, type(other.report)))
    with pytest.raises(CorruptRecord, match="verification"):
        store.verify()


def test_a_removed_verification_is_found_on_replay(tmp_path):
    store, key = official_store(tmp_path)
    publish_environment_evidence(store.run_dir, evidence())
    start(store, key)
    verification_path(store).unlink()
    with pytest.raises(CorruptRecord, match="verification"):
        store.verify()


def test_an_official_running_event_without_a_citation_is_refused_on_replay(tmp_path):
    store, key = official_store(tmp_path)
    publish_environment_evidence(store.run_dir, evidence())
    start(store, key)
    state = store.observation(key)
    stripped = state.events[1].model_copy(update={"environment_verification_sha256": None})
    with pytest.raises(CorruptRecord, match="verification"):
        store._check_event(key, stripped, state.events[:1], state.events[1].previous_event_sha256)


def test_resume_cites_the_first_verification_not_a_fresh_capture(tmp_path):
    store, key = official_store(tmp_path)
    publish_environment_evidence(store.run_dir, evidence())
    first = sha256_bytes(verification_path(store).read_bytes())
    store.fail(key, start(store, key), kind="out_of_memory", detail="x", retryable=True)
    fresh = evidence()
    fresh = EnvironmentEvidence(
        lock_bytes=LOCK, report=fresh.report.model_copy(update={"inventory_sha256": h("9")})
    )
    publish_environment_evidence(store.run_dir, fresh)
    start(store, key)
    cited = {
        e.environment_verification_sha256
        for e in store.observation(key).events
        if e.state == "running"
    }
    assert cited == {first}


# ---- official verification -------------------------------------------------------------------


def test_every_running_event_of_a_verified_official_run_cites_its_verification(
    tmp_path, view, tiny, recognised
):
    world = build_world(tmp_path, view, tiny)
    verified = verify(world, anchored(world))
    cited = {
        e.environment_verification_sha256
        for key in world.store.observation_keys()
        for e in world.store.observation(key).events
        if e.state == "running"
    }
    assert verified.environment.report.outcome == "matches_approved_lock"
    assert cited == {sha256_bytes(verification_path(world.store).read_bytes())}


def test_substituting_other_matching_evidence_after_the_run_is_rejected(
    tmp_path, view, tiny, recognised
):
    world = build_world(tmp_path, view, tiny)
    anchor = anchored(world)
    path = verification_path(world.store)
    data = json.loads(path.read_text())
    data["inventory_sha256"] = h("9")
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    with pytest.raises(CorruptRecord):
        verify(world, anchor)
