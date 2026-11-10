"""The guard-connected start of a rehearsal batch, end to end, on synthetic data.

Every test runs on a disposable rehearsal root with fixture readiness evidence, the mock
backend and invented protected inputs. Negative tests assert the exception type or code of the
intended boundary, and most compare a full snapshot of the storage before and after.
"""

import json
import os
import re
from pathlib import Path

import pytest
from finaltest_audit_support import CONFIGS, ITEMS, PLAN
from finaltest_campaign_support import CANARY, oracle_key, population, sha
from finaltest_start_support import (
    ARM,
    FakeRenderer,
    MockBackend,
    SyntheticInputs,
    make_world,
    mock_environment,
)
from pydantic import ValidationError

from biorag_repro.finaltest import inventory, layout, registry
from biorag_repro.finaltest.audit import AuditContext, audit_run
from biorag_repro.finaltest.batches import AuthorizationCitation, BatchManifest, citation_sha256
from biorag_repro.finaltest.execution import run_batch
from biorag_repro.finaltest.inputs import BatchInputBinding, ConditionRoster, binding_sha256
from biorag_repro.finaltest.inventory import verify_root
from biorag_repro.finaltest.protected import ProtectedInputError
from biorag_repro.finaltest.registry import ProtocolRefused
from biorag_repro.finaltest.runrecord import FinalTestRunRecord
from biorag_repro.finaltest.session import (
    LockOrderViolation,
    SessionBusy,
    SessionLost,
    open_session,
)
from biorag_repro.finaltest.start import StartRefused, _start_with
from biorag_repro.generation.attempts import Coordinates, CorruptRecord, InvalidTransition
from biorag_repro.generation.finaltest_store import GateRefused, StoreOperationRefused
from biorag_repro.readiness.guard import AuthorizationRefused

NOTE = inventory.HEAD_NOT_ESTABLISHED_TEXT


@pytest.fixture
def world(tmp_path):
    return make_world(tmp_path)


def start(w, session, **changes):
    request = w.with_request(**changes) if changes else w.request
    return _start_with(request, w.assess, session, now=w.clock)


def read_records(w, started):
    base = w.root / "batches" / started.batch_id
    manifest = BatchManifest.model_validate_json((base / "manifest.json").read_bytes())
    binding = BatchInputBinding.model_validate_json((base / "inputs/binding.json").read_bytes())
    rosters = {}
    for path in sorted((base / "inputs/rosters").glob("*.json")):
        roster = ConditionRoster.model_validate_json(path.read_bytes())
        rosters[roster.condition] = roster
    citation = AuthorizationCitation.model_validate_json(
        (base / "citations/000001.json").read_bytes()
    )
    return manifest, binding, rosters, citation


def audit(w, started):
    manifest, binding, rosters, citation = read_records(w, started)
    context = AuditContext(
        root_id=layout.root_id_for(w.auth),
        root_kind="rehearsal",
        plan=PLAN,
        item_papers=dict(w.request.population.item_papers),
        manifest=manifest,
        binding=binding,
        rosters=rosters,
        citations=[citation],
    )
    return audit_run(w.root, started.run_id, context)


class Counting(MockBackend):
    def generate(self, rendered, *, seed, max_new_tokens):
        return super().generate(rendered, seed=seed, max_new_tokens=max_new_tokens)


# ---- the complete synthetic lifecycle ---------------------------------------------------------


def test_rehearsal_lifecycle_runs_end_to_end_and_audits_without_authorizing(world):
    w = world
    backend = MockBackend()
    with open_session(w.root) as session:
        started = start(w, session)
        summary = run_batch(started, backend, capture_environment=mock_environment)
    assert (summary.complete, summary.terminal_failures, summary.attempts) == (10, 0, 10)
    assert len(backend.calls) == 10 and w.guard_calls == 1

    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    assert verdict.problems == (NOTE,)
    assert verdict.batches[started.batch_id].state == "authorized"
    assert verdict.head_authority == "not_established" and verdict.registry_head_sequence == 5
    types = [
        json.loads(p.read_bytes())["event_type"] for p in sorted((w.root / "registry").glob("0*"))
    ]
    assert types == [
        "root_initialized",
        "batch_reserved",
        "batch_materialized",
        "batch_inputs_bound",
        "batch_authorized",
    ]

    before = w.tree()
    result = audit(w, started)
    assert w.tree() == before  # the audit writes nothing
    assert result.structure_valid and result.claims_consistent and result.complete
    assert result.authorization == "not_established" and result.ordering == "unverified"
    record = FinalTestRunRecord.model_validate_json(
        (w.root / "runs" / started.run_id / "run.json").read_bytes()
    )
    assert record.official is False
    assert (record.root_kind, record.backend_kind, record.purpose) == (
        "rehearsal",
        "mock",
        "synthetic_test",
    )
    keys = [s.key for s in result.observed.states]
    assert len(keys) == 10 and all(s.status == "complete" for s in result.observed.states)
    for key in keys:
        assert (
            w.root / "runs" / started.run_id / "observations" / key / "final.build.json"
        ).exists()


def test_closed_book_arm_runs_three_repeats_with_no_retrieval_identities(tmp_path):
    w = make_world(tmp_path, arm="closed_book")
    with open_session(w.root) as session:
        started = start(w, session)
        summary = run_batch(started, MockBackend(), capture_environment=mock_environment)
    assert summary.complete == 15
    _, binding, rosters, _ = read_records(w, started)
    assert list(rosters) == ["CLOSED_BOOK"]
    for row in rosters["CLOSED_BOOK"].rows:
        assert (row.source_identity_sha256, row.context_sha256, row.evidence_sequence_sha256) == (
            None,
            None,
            None,
        )
    assert binding.retrieval_config_sha256 is None


def test_observation_keys_match_an_independent_oracle(world):
    w = world
    with open_session(w.root) as session:
        started = start(w, session)
    _, _, rosters, _ = read_records(w, started)
    config = CONFIGS[ARM].configuration_sha256
    expected = {oracle_key(i, ARM, c, 1, config) for i in ITEMS for c in ("JATS", "PDF_DOCLING")}
    assert {r.observation_key for ro in rosters.values() for r in ro.rows} == expected
    assert set(started.rows) == expected


# ---- identities and the acyclic hash graph -----------------------------------------------------


def test_hash_dependency_graph_is_acyclic_and_exact(world):
    w = world
    with open_session(w.root) as session:
        started = start(w, session)
    base = w.root / "batches" / started.batch_id
    files = {
        "roster": sorted((base / "inputs/rosters").glob("*.json")),
        "binding": base / "inputs/binding.json",
        "citation": base / "citations/000001.json",
        "run": w.root / "runs" / started.run_id / "run.json",
        "manifest": base / "manifest.json",
    }
    digest = lambda p: sha256_file(p)
    roster_digests = {digest(p) for p in files["roster"]}
    binding_digest, citation_digest = digest(files["binding"]), digest(files["citation"])
    run_digest = digest(files["run"])
    # a file name is the hash of its content
    assert {p.stem for p in files["roster"]} == roster_digests
    binding_text = files["binding"].read_text()
    citation_text = files["citation"].read_text()
    run_text = files["run"].read_text()
    # downward edges exist: binding names rosters, citation names binding, run names both
    assert all(d in binding_text for d in roster_digests)
    assert binding_digest in citation_text
    assert binding_digest in run_text and citation_digest in run_text
    # and no upward edge: no artifact contains the digest of anything built after it
    for path in files["roster"]:
        assert binding_digest not in path.read_text() and run_digest not in path.read_text()
    assert citation_digest not in binding_text and run_digest not in binding_text
    assert run_digest not in citation_text and citation_digest not in citation_text
    assert run_digest not in run_text and binding_digest not in binding_text
    # the registry events cite exactly these digests, in the pinned order
    events = [json.loads(p.read_bytes()) for p in sorted((w.root / "registry").glob("0*"))]
    bound, authorized = events[3], events[4]
    assert {c["role"]: c["sha256"] for c in bound["evidence"]}["inputs_binding"] == binding_digest
    cited = {c["role"]: c["sha256"] for c in authorized["evidence"]}
    assert cited["authorization_citation"] == citation_digest
    assert cited["run_record"] == run_digest
    assert cited["readiness_report"] == started.report_sha256 == w.fixture.sha
    assert cited["inputs_bound_event"] == sha256_file(w.root / "registry/000004.json")
    # the in-memory identities are the stored ones
    assert (started.binding_sha256, started.citation_sha256, started.run_record_sha256) == (
        binding_digest,
        citation_digest,
        run_digest,
    )
    _, binding, _, citation = read_records(w, started)
    assert (
        binding_sha256(binding) == binding_digest and citation_sha256(citation) == citation_digest
    )
    assert citation.inputs_set_sha256 == binding_digest and citation.kind == "initial"
    assert citation.head_authority == "genesis_anchored" and citation.registry_head_sequence == 1
    assert citation.registry_head_event_sha256 == sha256_file(w.root / "registry/000001.json")


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_identical_inputs_give_identical_rows_in_any_root(tmp_path):
    """Rows are digests of the inputs only; the batch and root identity differ by location."""
    rows = []
    for n in (1, 2):
        w = make_world(tmp_path / str(n))
        with open_session(w.root) as session:
            started = start(w, session)
        _, _, rosters, _ = read_records(w, started)
        rows.append({c: [r.model_dump() for r in ro.rows] for c, ro in rosters.items()})
    assert rows[0] == rows[1]


def test_no_protected_text_reaches_published_records(world):
    w = world
    with open_session(w.root) as session:
        started = start(w, session)
        run_batch(started, MockBackend(), capture_environment=mock_environment)
    allowed_with_text = w.root / "runs" / started.run_id / "inputs"  # the run keeps exact inputs
    for path in w.root.rglob("*"):
        if path.is_file() and allowed_with_text not in path.parents:
            assert CANARY.encode() not in path.read_bytes(), path
    assert CANARY.encode() in next(allowed_with_text.glob("*.json")).read_bytes()


# ---- the guard comes first and is not a token ---------------------------------------------------


def test_the_guard_runs_before_any_write_or_protected_read(world):
    w = world
    seen = {}

    def assess():
        seen["tree"] = w.tree()
        seen["reads"] = w.request.inputs.calls
        return w.assess()

    with open_session(w.root) as session:
        baseline = w.tree()
        _start_with(w.request, assess, session, now=w.clock)
    assert seen["tree"] == baseline and seen["reads"] == 0
    assert w.request.inputs.calls > 0


@pytest.mark.parametrize(
    "arrange, reason",
    [
        (
            lambda w: setattr(
                w.fixture,
                "live",
                __import__("test_readiness_guard").build(
                    w.fixture.policy, blocked={"G09_scientific_source_revision": "failed"}
                ),
            ),
            "live_not_ready",
        ),
        (lambda w: w.fixture.report_path().unlink(), "report_unverifiable"),
    ],
)
def test_a_refused_guard_leaves_storage_untouched_and_reads_nothing(world, arrange, reason):
    w = world
    arrange(w)
    with open_session(w.root) as session:
        baseline = w.tree()
        with pytest.raises(AuthorizationRefused) as caught:
            start(w, session)
        assert caught.value.reason == reason
        assert w.tree() == baseline
    assert w.request.inputs.calls == 0


def test_stored_not_ready_is_refused_before_the_live_evaluation(world):
    import test_readiness_guard as g

    from biorag_repro.readiness import publish

    w = world
    stored = g.build(w.fixture.policy, blocked={"G03_final_test_configuration_freeze": "pending"})
    w.fixture.sha = publish.publish_report(stored, w.fixture.data).report_sha256
    with open_session(w.root) as session:
        baseline = w.tree()
        with pytest.raises(AuthorizationRefused) as caught:
            start(w, session)
        assert caught.value.reason == "stored_not_ready" and w.fixture.calls == 0
        assert w.tree() == baseline


def test_wrong_policy_identity_is_refused(world):
    w = world
    other = w.fixture.policy.model_copy(update={"policy_version": "another"})

    def assess():
        from biorag_repro.finaltest.start import Assessed

        return Assessed(w.fixture.authorize(policy=other), w.fixture.live.report)

    with open_session(w.root) as session, pytest.raises(AuthorizationRefused) as caught:
        _start_with(w.request, assess, session, now=w.clock)
    assert caught.value.reason == "report_unverifiable"


def test_the_registry_moving_during_the_guard_is_refused(world):
    w = world

    def assess():
        # another writer appends while the guard runs; it cannot take the lock, so it writes raw
        registry._ensure_genesis_witness(w.root, w.witnesses, layout.root_id_for(w.auth), w.clock)
        registry._append(
            w.root,
            w.witnesses,
            layout.root_id_for(w.auth),
            "batch_resumed",
            [layout.EvidenceCitation(role="x", sha256=sha("x"))],
            w.clock,
        )
        return w.assess()

    with open_session(w.root) as session, pytest.raises(StartRefused) as caught:
        _start_with(w.request, assess, session, now=w.clock)
    assert caught.value.code == "head_changed"
    assert not (w.root / "batches").exists() and not (w.root / "runs").exists()


def test_a_report_that_does_not_match_the_guard_result_is_refused(world):
    import test_readiness_guard as g

    from biorag_repro.finaltest.start import Assessed

    w = world
    other = g.build(w.fixture.policy, tag="other").report

    def assess():
        return Assessed(w.fixture.authorize(), other)

    with open_session(w.root) as session, pytest.raises(StartRefused) as caught:
        _start_with(w.request, assess, session, now=w.clock)
    assert caught.value.code == "guard_report_mismatch"
    assert not (w.root / "batches").exists()


# ---- production stays closed ---------------------------------------------------------------------


def test_the_public_start_refuses_a_fixture_ready_report_in_the_synthetic_world(
    tmp_path, monkeypatch
):
    import test_readiness_guard as g
    from readiness_world import build_world

    from biorag_repro.finaltest.start import start_batch
    from biorag_repro.readiness import publish

    real = build_world(tmp_path, monkeypatch)
    sha_ = publish.publish_report(g.build(g.support.fixture_policy()), real.data).report_sha256
    before = {str(p): p.read_bytes() for p in real.data.rglob("*") if p.is_file()}
    with pytest.raises(AuthorizationRefused) as caught:
        start_batch(real.data, sha_, repo=real.repo, matrix_arm=ARM)
    assert caught.value.reason == "report_unverifiable"
    assert {str(p): p.read_bytes() for p in real.data.rglob("*") if p.is_file()} == before


def test_even_a_guard_that_accepts_ends_in_a_production_refusal_before_any_write(
    world, monkeypatch, tmp_path
):
    from biorag_repro.finaltest import start as start_module
    from biorag_repro.finaltest.start import ProductionStartRefused, start_batch

    w = world
    monkeypatch.setattr(start_module, "_production_assessed", lambda *_: w.assess())
    with open_session(w.root):
        baseline = w.tree()
    with pytest.raises(ProductionStartRefused) as caught:
        start_batch(tmp_path, w.fixture.sha, repo=tmp_path, matrix_arm=ARM)
    assert caught.value.code == "production_disabled"
    monkeypatch.setattr(start_module, "PRODUCTION_START", True)
    with pytest.raises(ProductionStartRefused) as caught:
        start_batch(tmp_path, w.fixture.sha, repo=tmp_path, matrix_arm=ARM)
    assert caught.value.code == "no_production_request"
    assert w.request.inputs.calls == 0
    with open_session(w.root):
        assert w.tree() == baseline


def test_the_fixture_route_refuses_a_final_test_root_before_calling_the_guard(tmp_path):
    from finaltest_support import authorization

    w = make_world(tmp_path)
    final = authorization(tmp_path / "other", "final_test")
    with open_session(w.root) as session:
        baseline = w.tree()
        with pytest.raises(StartRefused) as caught:
            _start_with(w.with_request(authorization=final), w.assess, session)
        assert caught.value.code == "root_kind"
        assert w.tree() == baseline
    assert w.guard_calls == 0 and w.request.inputs.calls == 0


def test_publishers_refuse_a_final_test_root_on_their_own(tmp_path):
    from finaltest_support import authorization
    from test_finaltest_g13 import world as _unused  # noqa: F401

    base = tmp_path / "final"
    base.mkdir()
    auth = authorization(base, "final_test")
    anchors = base / "anchors"
    inventory._initialize(Path(auth.authorized_root), auth, anchors, inventory.utc_now)
    root = Path(auth.authorized_root)
    before = sorted(p.name for p in (root / "registry").iterdir())
    with inventory.registry_lock(root) as held:
        for call in (
            lambda: registry.append_event_locked(
                held,
                root,
                auth,
                base / "w",
                "batch_inputs_bound",
                [
                    layout.EvidenceCitation(role=r, sha256=sha(r))
                    for r in layout.REHEARSAL_ROLES["batch_inputs_bound"]
                ],
            ),
            lambda: registry.publish_batch_file(
                held, root, auth, f"batches/{'a' * 64}/inputs/binding.json", b"x"
            ),
            lambda: registry.materialize_locked(held, root, auth, anchors, base / "w", "a" * 64),
        ):
            with pytest.raises(ProtocolRefused, match="rehearsal"):
                call()
    assert sorted(p.name for p in (root / "registry").iterdir()) == before
    assert not (root / "batches").exists()


def test_start_batch_has_no_policy_evaluator_flag_or_authorization_parameter():
    import inspect

    from biorag_repro.finaltest.start import start_batch

    assert list(inspect.signature(start_batch).parameters) == [
        "data_root",
        "report_sha256",
        "repo",
        "matrix_arm",
    ]


# ---- the arm is claimed once -------------------------------------------------------------------


def test_an_arm_is_never_started_twice(world):
    w = world
    with open_session(w.root) as session:
        start(w, session)
        before = w.tree()
        with pytest.raises(StartRefused) as caught:
            start(w, session)
        assert caught.value.code == "arm_already_reserved"
        assert w.tree() == before


def test_a_second_arm_can_be_rehearsed_but_its_citation_is_not_anchored(tmp_path):
    w = make_world(tmp_path)
    with open_session(w.root) as session:
        first = start(w, session)
    second_world = w
    second_world.request = w.with_request(
        matrix_arm="closed_book",
        config_toml=__import__("test_generation_config")
        .to_toml(__import__("finaltest_audit_support").settings_for("closed_book"))
        .encode(),
        renderer=FakeRenderer(CONFIGS["closed_book"]),
        inputs=SyntheticInputs("closed_book"),
    )
    with open_session(w.root) as session:
        second = start(w, session)
    _, _, _, citation = read_records(w, second)
    assert citation.head_authority == "not_established" and citation.registry_head_sequence == 5
    assert first.batch_id != second.batch_id
    assert verify_root(w.root, w.auth, w.anchors, w.witnesses).problems == (NOTE,)


# ---- scientific input regressions ----------------------------------------------------------------


def refused_inputs(w, code, **changes):
    with open_session(w.root) as session:
        baseline = w.tree()
        with pytest.raises(StartRefused) as caught:
            start(w, session, **changes)
        assert caught.value.code == code, str(caught.value)
        assert w.tree() == baseline  # nothing written: the arm stays free
    return caught.value


def test_a_missing_item_is_refused(world):
    inputs = SyntheticInputs(ARM, change=lambda kind, v: v[:-1] if kind == "items" else v)
    refused_inputs(world, "inputs", inputs=inputs)


def test_an_item_of_another_paper_is_refused(world):
    from biorag_repro.finaltest.protected import ProtectedItem

    def change(kind, value):
        if kind != "items":
            return value
        first = value[0]
        return [ProtectedItem(first.item_id, "p9", first.question, first.item_sha256), *value[1:]]

    refused_inputs(world, "inputs", inputs=SyntheticInputs(ARM, change=change))


def test_jats_and_pdf_contexts_must_stay_distinguishable(world):
    from dataclasses import replace

    def change(kind, value):
        if kind != "context":
            return value
        item, _condition, found = value
        return replace(found, source_identity_sha256=sha(f"same-{item}"))

    error = refused_inputs(world, "inputs_invalid", inputs=SyntheticInputs(ARM, change=change))
    assert "representations_not_distinguishable" in str(error)


def test_a_missing_context_is_refused(world):
    def change(kind, value):
        if kind == "context" and value[1] == "PDF_DOCLING":
            return None
        return value[2] if kind == "context" else value

    refused_inputs(world, "inputs", inputs=SyntheticInputs(ARM, change=change))


def test_a_closed_book_context_is_refused(tmp_path):
    from biorag_repro.finaltest.protected import ProtectedContext

    leak = ProtectedContext("leak", 1, ("c",), sha("s"), None)
    w = make_world(
        tmp_path,
        arm="closed_book",
        inputs=SyntheticInputs(
            "closed_book", change=lambda kind, v: leak if kind == "context" else v
        ),
    )
    error = refused_inputs(w, "inputs")
    assert "closed-book" in str(error)


def test_another_arms_configuration_or_prompt_is_refused(world):
    w = world
    other = CONFIGS["confirmatory"]
    refused_inputs(w, "configuration", renderer=FakeRenderer(other))
    wrong_toml = (
        __import__("test_generation_config")
        .to_toml(__import__("finaltest_audit_support").settings_for("closed_book"))
        .encode()
    )
    refused_inputs(w, "configuration", config_toml=wrong_toml)
    changed_prompt = __import__("finaltest_audit_support").settings_for(ARM) | {
        "prompt_sha256": sha("x")
    }
    refused_inputs(
        w,
        "configuration",
        config_toml=__import__("test_generation_config").to_toml(changed_prompt).encode(),
    )


def test_another_benchmark_or_unknown_arm_is_refused(world):
    w = world
    refused_inputs(w, "benchmark_identity", population=population(split_sha256=sha("other")))
    refused_inputs(w, "arm", matrix_arm="not_an_arm")


def test_a_sampling_configuration_needs_its_exact_seed_registry(tmp_path):
    from finaltest_audit_support import settings_for
    from test_generation_config import resolve, to_toml

    from biorag_repro.finaltest.campaign import build_plan
    from biorag_repro.generation.seeds import (
        GenerationSeed,
        GenerationSeedRegistry,
        seed_registry_sha256,
    )

    registry_ = GenerationSeedRegistry(
        registry_version="synthetic-seeds-0",
        entries=[
            GenerationSeed(name=f"generation_repeat_{n}", seed=10 + n, consumer="x")
            for n in (1, 2, 3)
        ],
    )
    settings = settings_for("dense_robustness") | {
        "decoding": {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 20},
        "seed_registry_sha256": seed_registry_sha256(registry_),
    }
    config = resolve(settings)
    plan = build_plan(
        population(),
        {a: c.configuration_sha256 for a, c in CONFIGS.items()}
        | {"dense_robustness": config.configuration_sha256},
        config.settings.seed_registry_sha256,
    )
    w = make_world(tmp_path, plan=plan, config_toml=to_toml(settings).encode(),
                   renderer=FakeRenderer(config), seeds=registry_)  # fmt: skip
    with open_session(w.root) as session:
        started = start(w, session)
    _, binding, rosters, _ = read_records(w, started)
    assert [(s.repeat_index, s.seed) for s in binding.seeds] == [(1, 11), (2, 12), (3, 13)][:1]
    assert {r.seed for ro in rosters.values() for r in ro.rows} == {11}
    # a registry that is not the one the configuration names is refused before anything is read
    wrong = registry_.model_copy(
        update={"entries": [e.model_copy(update={"seed": e.seed + 1}) for e in registry_.entries]}
    )
    w2 = make_world(tmp_path / "w2", plan=plan, config_toml=to_toml(settings).encode(),
                    renderer=FakeRenderer(config), seeds=wrong)  # fmt: skip
    refused_inputs(w2, "seeds")
    w3 = make_world(tmp_path / "w3", plan=plan, config_toml=to_toml(settings).encode(),
                    renderer=FakeRenderer(config), seeds=None)  # fmt: skip
    refused_inputs(w3, "seeds")


def test_scientific_repeat_zero_is_rejected_at_the_row(world, monkeypatch):
    from pydantic import ValidationError

    from biorag_repro.finaltest import protected

    monkeypatch.setitem(protected.FROZEN_MATRIX, ARM, (("JATS", "PDF_DOCLING"), (0, 1)))
    with open_session(world.root) as session, pytest.raises(ValidationError):
        start(world, session)
    assert not (world.root / "batches").exists()


def test_error_messages_carry_no_protected_text(world):
    inputs = SyntheticInputs(ARM, change=lambda kind, v: v[:-1] if kind == "items" else v)
    error = refused_inputs(world, "inputs", inputs=inputs)
    assert CANARY not in str(error)
    assert not isinstance(error.__cause__, KeyError)
    assert isinstance(error.__cause__, ProtectedInputError)


# ---- the session --------------------------------------------------------------------------------


def test_a_second_session_is_refused_in_the_same_and_in_another_process(world):
    import multiprocessing

    w = world
    with open_session(w.root):
        with pytest.raises(SessionBusy), open_session(w.root):
            pass
        ctx = multiprocessing.get_context("fork")

        def child():
            try:
                with open_session(w.root):
                    os._exit(1)
            except SessionBusy:
                os._exit(0)

        process = ctx.Process(target=child)
        process.start()
        process.join(30)
        assert process.exitcode == 0
    with open_session(w.root):  # released on exit
        pass


def test_the_session_lock_comes_before_the_registry_lock(world):
    w = world
    with inventory.registry_lock(w.root), pytest.raises(LockOrderViolation), open_session(w.root):
        pass
    with open_session(w.root) as session, session.registry():
        pass


def test_a_lock_file_that_exists_is_not_ownership(world):
    w = world
    with open_session(w.root) as session:
        session.assert_held()
        lock = w.root / layout.SESSION_LOCK_FILE
        # the file is replaced by another one: same name, not our lock
        lock.unlink()
        lock.write_bytes(b"")
        with pytest.raises(SessionLost, match="replaced"):
            session.assert_held()
    with open_session(w.root) as session:
        os.close(session._descriptor)  # the descriptor is gone but the file is still there
        with pytest.raises(SessionLost):
            session.assert_held()
        session._descriptor = None


def test_a_closed_session_cannot_start_or_run(world):
    w = world
    with open_session(w.root) as session:
        started = start(w, session)
    assert not session.active
    with pytest.raises(SessionLost):
        run_batch(started, MockBackend(), capture_environment=mock_environment)
    with pytest.raises(SessionLost):
        start(w, session)
    assert (
        not any((w.root / "runs" / started.run_id / "observations").glob("*"))
        if (w.root / "runs" / started.run_id / "observations").exists()
        else True
    )


def test_the_old_authorization_cannot_be_reused_by_a_new_session(world):
    w = world
    backend = MockBackend()

    class Stop(BaseException):
        pass

    class Dying(MockBackend):
        def generate(self, rendered, *, seed, max_new_tokens):
            raise Stop

    with open_session(w.root) as session:
        started = start(w, session)
        with pytest.raises(Stop):
            run_batch(started, Dying(), capture_environment=mock_environment)
    key = next(iter(started.rows))
    # the attempt is left running; nothing can recover it in this version
    states = {s.status for s in started.store.verify()}
    assert "running" in states
    with pytest.raises(StoreOperationRefused):
        started.store.recover_interrupted(previous_worker_stopped=True)
    with open_session(w.root) as session:
        assert session is not started.session
        before = w.tree()
        with pytest.raises(StartRefused) as caught:
            start(w, session)
        assert caught.value.code == "arm_already_reserved"
        with pytest.raises(GateRefused):  # the old store's gate still points at the closed session
            started.store.plan(Coordinates(item_id="i1", paper_id="p2", matrix_arm=ARM,
                                           representation_or_condition="JATS", repeat_index=1))  # fmt: skip
        with pytest.raises(SessionLost):
            run_batch(started, backend, capture_environment=mock_environment)
        assert w.tree() == before
    assert backend.calls == [] and key


# ---- the guarded store --------------------------------------------------------------------------


def coordinates_of(started, index=0):
    row = list(started.rows.values())[index]
    return row, Coordinates(
        item_id=row.item_id,
        paper_id=row.paper_id,
        matrix_arm=row.matrix_arm,
        representation_or_condition=row.condition,
        repeat_index=row.repeat_index,
        seed=row.seed,
    )


def alter(path: Path, data: bytes) -> bytes:
    original = path.read_bytes()
    path.chmod(0o644)
    path.write_bytes(data)
    return original


@pytest.mark.parametrize(
    "target",
    ["binding", "roster", "citation", "run", "manifest", "event"],
)
def test_a_changed_bound_record_stops_the_run_before_any_backend_call(world, target):
    w = world
    with open_session(w.root) as session:
        started = start(w, session)
        base = w.root / "batches" / started.batch_id
        path = {
            "binding": base / "inputs/binding.json",
            "roster": next((base / "inputs/rosters").glob("*.json")),
            "citation": base / "citations/000001.json",
            "run": w.root / "runs" / started.run_id / "run.json",
            "manifest": base / "manifest.json",
            "event": w.root / "registry/000004.json",
        }[target]
        alter(path, path.read_bytes() + b" ")
        backend = MockBackend()
        with pytest.raises(GateRefused, match="differs"):
            run_batch(started, backend, capture_environment=mock_environment)
    assert backend.calls == []
    assert not (w.root / "runs" / started.run_id / "observations").exists()


def test_a_moved_registry_head_stops_the_run(world):
    w = world
    with open_session(w.root) as session:
        started = start(w, session)
        w.request = w.with_request(
            matrix_arm="closed_book",
            config_toml=__import__("test_generation_config")
            .to_toml(__import__("finaltest_audit_support").settings_for("closed_book"))
            .encode(),
            renderer=FakeRenderer(CONFIGS["closed_book"]),
        )
        registry.reserve_batch(w.root, w.auth, w.anchors, w.witnesses, PLAN, "closed_book")
        backend = MockBackend()
        with pytest.raises(GateRefused, match="head"):
            run_batch(started, backend, capture_environment=mock_environment)
    assert backend.calls == []


def test_the_gate_checks_ownership_and_the_bound_input_before_a_begin(world):
    w = world
    with open_session(w.root) as session:
        started = start(w, session)
        store = started.store
        row, coords = coordinates_of(started)
        key = store.plan(coords)
        assert key == row.observation_key
        events_before = w.tree()
        with pytest.raises(GateRefused, match="differs from the bound roster"):
            store.begin_attempt(key, environment=mock_environment(), rendered_input_sha256="0" * 64)
        foreign = Coordinates(item_id="zz", paper_id="p2", matrix_arm=ARM,
                              representation_or_condition="JATS", repeat_index=1)  # fmt: skip
        with pytest.raises(GateRefused, match="not an observation of this batch"):
            store.plan(foreign)
        assert w.tree() == events_before


def test_completion_and_failure_belong_to_an_attempt_started_in_this_session(world):
    w = world
    with open_session(w.root) as session:
        started = start(w, session)
        store = started.store
        row, coords = coordinates_of(started)
        key = store.plan(coords)
        with pytest.raises(GateRefused, match="not started by this store"):
            store.complete(key, "a" * 32, '{"answer": "x", "abstained": false}')
        with pytest.raises(GateRefused, match="not started by this store"):
            store.fail(key, "a" * 32, kind="out_of_memory", detail="x", retryable=True)
        attempt = store.begin_attempt(
            key, environment=mock_environment(), rendered_input_sha256=row.rendered_input_sha256
        )
        session.close()
        with pytest.raises(GateRefused, match="session"):
            store.complete(key, attempt, '{"answer": "x", "abstained": false}')
        with pytest.raises(GateRefused):
            store.fail(key, attempt, kind="out_of_memory", detail="x", retryable=True)


def test_direct_append_and_seal_paths_are_gated(world):
    w = world
    with open_session(w.root) as session:
        started = start(w, session)
        store = started.store
        _, coords = coordinates_of(started)
        key = store.key_for(coords)
        session.close()
        before = (w.root / "runs" / started.run_id).exists()
        with pytest.raises(GateRefused):
            store._append(key, None, state="planned", coordinates=coords)
        state_like = type("S", (), {"key": key})()
        with pytest.raises(GateRefused):
            store._seal(state_like)
        assert before and not (w.root / "runs" / started.run_id / "observations").exists()


def test_a_final_test_store_cannot_be_created_or_reopened_generically(world):
    from biorag_repro.generation.attempts import RunStore
    from biorag_repro.generation.finaltest_store import FinalTestStore

    w = world
    with open_session(w.root) as session:
        started = start(w, session)
    for call in (
        lambda: FinalTestStore.create(w.root),
        lambda: FinalTestStore.open(w.root, started.run_id),
    ):
        with pytest.raises(StoreOperationRefused):
            call()
    with pytest.raises(ValidationError):  # phase "final_test" is not a RunPhase
        RunStore.create(
            w.root,
            config=CONFIGS[ARM],
            phase="final_test",
            benchmark_version="b",
            backend_kind="mock",
        )
    with pytest.raises(CorruptRecord):  # and a stored final-test record is not a RunRecord
        RunStore.open(w.root, started.run_id, config=CONFIGS[ARM], backend_kind="mock")


def test_a_completed_observation_is_never_replayed(world):
    w = world
    backend = MockBackend()
    with open_session(w.root) as session:
        started = start(w, session)
        run_batch(started, backend, capture_environment=mock_environment)
        assert len(backend.calls) == 10
        before = w.tree()
        again = run_batch(started, backend, capture_environment=mock_environment)
        assert again.attempts == 0 and len(backend.calls) == 10
        assert w.tree() == before
        row, _ = coordinates_of(started)
        with pytest.raises(InvalidTransition, match="never regenerated"):
            started.store.begin_attempt(
                row.observation_key,
                environment=mock_environment(),
                rendered_input_sha256=row.rendered_input_sha256,
            )


def test_audit_of_the_finished_rehearsal_finds_tampering_and_never_authorizes(world):
    w = world
    with open_session(w.root) as session:
        started = start(w, session)
        run_batch(started, MockBackend(), capture_environment=mock_environment)
    run_dir = w.root / "runs" / started.run_id
    key = next(iter(started.rows))
    raw = next((run_dir / "observations" / key / "raw").glob("*.txt"))
    raw.chmod(0o644)
    raw.write_text("altered")
    result = audit(w, started)
    assert not result.structure_valid and result.authorization == "not_established"
    assert any(f.code == "observation_corrupt" for f in result.findings)


def test_a_run_made_without_the_start_is_unbound_in_the_audit(tmp_path):
    from finaltest_audit_support import build_run

    built = build_run(tmp_path)
    result = audit_run(built.base, built.record.run_id, AuditContext(
        root_id=built.context.root_id, root_kind="rehearsal", plan=PLAN,
        item_papers=built.context.item_papers, manifest=None, binding=None, rosters={}))  # fmt: skip
    codes = {f.code for f in result.findings}
    assert {"binding_missing", "citation_missing", "manifest_missing"} <= codes
    assert result.authorization == "not_established"


# ---- interruption at every publication boundary ---------------------------------------------------


class Fault(OSError):
    pass


def count_publications(tmp_path_factory) -> list[str]:
    """The sequence of publications of one complete start, taken from a dry run."""
    seen: list[str] = []
    w = make_world(tmp_path_factory.mktemp("dry"))

    def watch(module, original):
        def wrapper(target, data, *args, **kwargs):
            seen.append(Path(target).name)
            return original(target, data, *args, **kwargs)

        return wrapper

    from biorag_repro.generation import finaltest_store

    patch = pytest.MonkeyPatch()
    patch.setattr(registry, "publish_bytes", watch(registry, registry.publish_bytes))
    patch.setattr(
        finaltest_store, "publish_bytes", watch(finaltest_store, finaltest_store.publish_bytes)
    )
    try:
        with open_session(w.root) as session:
            start(w, session)
    finally:
        patch.undo()
    return seen


@pytest.fixture(scope="module")
def publications(tmp_path_factory):
    return count_publications(tmp_path_factory)


def test_the_publication_sequence_is_the_documented_one(publications):
    names = [re.sub(r"[0-9a-f]{64}", "<sha>", n) for n in publications]
    assert names == [
        "<root>.000001.witness.json".replace("<root>", names[0].split(".")[0]),
        "000002.json",
        names[2],  # witness 2
        "manifest.json",
        "000003.json",
        names[5],  # witness 3
        "000004.json",
        names[7],  # witness 4
        "<sha>.json",
        "<sha>.json",
        "binding.json",
        "000005.json",
        names[12],  # witness 5
        "000001.json",
        "config.canonical.json",
        "config.toml",
        "run.json",
    ]


@pytest.mark.parametrize("index", range(17))
def test_interruption_at_every_publication(tmp_path, publications, index):
    from biorag_repro.generation import finaltest_store

    assert len(publications) == 17
    w = make_world(tmp_path)

    def failing(original):
        counter = {"n": 0}

        def wrapper(target, data, *args, **kwargs):
            counter["n"] += 1
            if counter["n"] == index + 1:
                raise Fault(f"injected at publication {index + 1}")
            return original(target, data, *args, **kwargs)

        return wrapper

    shared = {"n": 0}
    originals = (registry.publish_bytes, finaltest_store.publish_bytes)

    def make(original):
        def wrapper(target, data, *args, **kwargs):
            shared["n"] += 1
            if shared["n"] == index + 1:
                raise Fault(f"injected at publication {index + 1}")
            return original(target, data, *args, **kwargs)

        return wrapper

    patch = pytest.MonkeyPatch()
    patch.setattr(registry, "publish_bytes", make(originals[0]))
    patch.setattr(finaltest_store, "publish_bytes", make(originals[1]))
    backend = MockBackend()
    try:
        with open_session(w.root) as session, pytest.raises(Fault):
            start(w, session)
    finally:
        patch.undo()
    assert backend.calls == []
    # nothing ran: no observation exists whatever survived
    observations = (
        list((w.root / "runs").glob("*/observations")) if (w.root / "runs").exists() else []
    )
    assert observations == []

    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    left = [p for p in verdict.problems if p != NOTE]
    assert all(batch.state != "authorized" for batch in verdict.batches.values())
    outcome = FATE.get(index, "unverified")
    assert bool(left) == (outcome == "unverified"), (index, left)
    surviving = w.tree()
    with open_session(w.root) as session:
        surviving = w.tree()
        if outcome == "free":
            # only the genesis witness or nothing at all was written: the same start can run
            assert start(w, session).batch_id
            return
        with pytest.raises(StartRefused) as caught:
            start(w, session)
        assert caught.value.code == (
            "arm_already_reserved" if outcome == "claimed" else "root_unverified"
        )
        assert w.tree() == surviving  # nothing repaired, adopted or overwritten
    with pytest.raises(ProtocolRefused):  # the key claim survives
        registry.reserve_batch(w.root, w.auth, w.anchors, w.witnesses, PLAN, ARM)


# what the root looks like after the nth publication fails (see the sequence test above):
# "free": nothing but the genesis witness exists; "claimed": a clean but unfinished batch
# whose arm stays reserved; anything else does not verify and blocks every further start
FATE = {0: "free", 1: "free", 6: "claimed", 11: "claimed"}


def test_a_temporary_file_left_by_an_interrupted_publication_blocks_the_next_start(world):
    w = world
    with open_session(w.root) as session:
        started = start(w, session)
    leftover = w.root / "batches" / started.batch_id / "inputs" / ".binding.json.x.tmp"
    leftover.write_bytes(b"partial")
    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    assert any("unfinished publication" in p for p in verdict.problems)
    assert verdict.code == "history_inventory_incomplete"


def test_a_start_that_fails_after_the_guard_and_before_publication_leaves_the_arm_free(world):
    w = world
    inputs = SyntheticInputs(ARM, change=lambda kind, v: v[:-1] if kind == "items" else v)
    refused_inputs(w, "inputs", inputs=inputs)
    assert w.guard_calls == 1  # the guard did run
    with open_session(w.root) as session:
        started = start(w, session)  # and the arm can still be started
    assert started.batch_id


def test_two_workers_starting_the_same_arm_leave_exactly_one_batch(tmp_path):
    import multiprocessing

    w = make_world(tmp_path)
    ctx = multiprocessing.get_context("fork")
    results = ctx.Queue()

    def worker():
        try:
            with open_session(w.root) as session:
                _start_with(w.request, w.assess, session, now=w.clock)
            results.put("started")
        except SessionBusy:
            results.put("busy")
        except StartRefused as error:
            results.put(error.code)

    processes = [ctx.Process(target=worker) for _ in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join(120)
    outcomes = sorted(results.get(timeout=5) for _ in processes)
    # exactly one start succeeds; each other worker either found the session busy or the arm taken
    assert outcomes.count("started") == 1
    assert set(outcomes) <= {"started", "busy", "arm_already_reserved"}
    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    assert verdict.problems == (NOTE,) and len(verdict.batches) == 1
    assert sum(1 for _ in (w.root / "runs").iterdir()) == 1


def test_the_store_cannot_be_created_for_a_final_test_root(world):
    from biorag_repro.generation.finaltest_store import create_final_test_store

    w = world
    with open_session(w.root) as session:
        started = start(w, session)
    record = FinalTestRunRecord.model_validate_json(
        (w.root / "runs" / started.run_id / "run.json").read_bytes()
    )
    official = FinalTestRunRecord.model_validate(
        record.model_dump()
        | {
            "run_id": "f" * 32,
            "root_kind": "final_test",
            "backend_kind": "official",
            "purpose": "development_candidate",
        }
    )
    before = w.tree()
    with pytest.raises(StoreOperationRefused, match="only rehearsal runs"):
        create_final_test_store(w.root, official, CONFIGS[ARM], lambda *_: None)
    assert w.tree() == before and not (w.root / "runs" / ("f" * 32)).exists()


# ---- checks that mutation testing showed were not yet pinned down -----------------------------


def test_a_guard_result_for_another_scope_is_refused(world):
    import dataclasses

    from biorag_repro.finaltest.start import Assessed

    w = world

    def assess():
        authorization = dataclasses.replace(w.fixture.authorize(), scope="something_else")
        return Assessed(authorization, w.fixture.live.report)

    with open_session(w.root) as session:
        baseline = w.tree()
        with pytest.raises(StartRefused) as caught:
            _start_with(w.request, assess, session, now=w.clock)
        assert caught.value.code == "guard_scope" and w.tree() == baseline


def test_the_root_must_be_the_authorized_location(tmp_path):
    w = make_world(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "locks").mkdir(parents=True)
    (elsewhere / layout.SESSION_LOCK_FILE).write_bytes(b"")
    with open_session(w.root) as session:
        with pytest.raises(StartRefused) as caught:
            start(w, session, root=elsewhere)
        assert caught.value.code == "root_location"
    with open_session(elsewhere) as other:  # a session of another root cannot start this one
        with pytest.raises(StartRefused) as caught:
            start(w, other)
        assert caught.value.code == "root_location"
    assert w.guard_calls == 0 and not (w.root / "batches").exists()


def test_a_root_approved_for_another_benchmark_is_refused_before_the_guard(tmp_path):
    from finaltest_support import authorization

    from biorag_repro.finaltest.inventory import initialize_root

    base = tmp_path / "other"
    base.mkdir()
    auth = authorization(base, "rehearsal", benchmark_release_sha256=sha("another-release"))
    initialize_root(Path(auth.authorized_root), auth, base / "anchors")
    w = make_world(tmp_path / "w")
    request = w.with_request(
        root=Path(auth.authorized_root), authorization=auth, anchors=base / "anchors"
    )
    with open_session(request.root) as session:
        with pytest.raises(StartRefused) as caught:
            _start_with(request, w.assess, session, now=w.clock)
        assert caught.value.code == "benchmark_identity"
    assert w.guard_calls == 0 and request.inputs.calls == 0


def test_an_unlocked_but_open_session_descriptor_is_not_ownership(world):
    import fcntl

    w = world
    with open_session(w.root) as session:
        session.assert_held()
        fcntl.flock(session._descriptor, fcntl.LOCK_UN)  # still open, no longer locked
        with pytest.raises(SessionLost, match="not held"):
            session.assert_held()
        fcntl.flock(session._descriptor, fcntl.LOCK_EX)


def test_a_valid_but_different_binding_or_citation_does_not_match_its_event(tmp_path):
    from biorag_repro.finaltest.layout import serialize

    for name in ("binding", "citation"):
        w = make_world(tmp_path / name)
        with open_session(w.root) as session:
            started = start(w, session)
        base = w.root / "batches" / started.batch_id
        _manifest, binding, _rosters, citation = read_records(w, started)
        path = base / ("inputs/binding.json" if name == "binding" else "citations/000001.json")
        record = binding if name == "binding" else citation
        other = (
            record.model_copy(update={"prompt_sha256": sha("other")})
            if name == "binding"
            else record.model_copy(update={"created_utc": "2027-01-01T00:00:00Z"})
        )
        path.chmod(0o644)
        path.write_bytes(serialize(other))
        problems = verify_root(w.root, w.auth, w.anchors, w.witnesses).problems
        assert any("differs from its event" in p for p in problems), (name, problems)


def test_a_citation_about_other_inputs_is_not_an_authorization(tmp_path, monkeypatch):
    from biorag_repro.finaltest import start as start_module

    original = start_module.AuthorizationCitation

    def wrong(**values):
        return original(**(values | {"inputs_set_sha256": sha("not-the-binding")}))

    monkeypatch.setattr(start_module, "AuthorizationCitation", wrong)
    w = make_world(tmp_path)
    with open_session(w.root) as session:
        started = start(w, session)
    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    assert verdict.batches[started.batch_id].state == "inputs_bound"
    assert any("citation of batch" in p and "other evidence" in p for p in verdict.problems)
