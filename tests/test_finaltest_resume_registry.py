"""The verifier's view of resume records, and interruption at each publication.

Every tampering test starts from a real synthetic resume and changes one stored fact, so the
expected refusal is the one check that fact is protected by.
"""

import hashlib
from pathlib import Path

import pytest
from finaltest_resume_support import assess, interrupted, renew, resume, root_problems
from finaltest_start_support import make_world

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.finaltest import inventory, layout, registry
from biorag_repro.finaltest.batches import AuthorizationCitation, citation_sha256
from biorag_repro.finaltest.continuity import ContinuityAssessment
from biorag_repro.finaltest.inventory import _Artifacts, _check_authorization, verify_root
from biorag_repro.finaltest.layout import EvidenceCitation
from biorag_repro.finaltest.resume import ResumeRefused
from biorag_repro.finaltest.runrecord import FinalTestRunRecord
from biorag_repro.finaltest.session import open_session


@pytest.fixture
def resumed(tmp_path):
    """A crashed batch that was resumed once (session closed, nothing run after the resume)."""
    w, started = interrupted(tmp_path)
    renew(w)
    with open_session(w.root) as session:
        resume(w, session)
    assert root_problems(w) == []
    return w, started


def batch_dir(w, started):
    return w.root / "batches" / started.batch_id / "citations"


def raw_text(problems):
    return " | ".join(problems)


# ---- the C2C survivor: the citation must name the batch it is checked for --------------------


def test_a_citation_naming_another_batch_is_refused_by_its_batch_id_alone(tmp_path):
    """Every other check passes, so only `citation.batch_id != batch` can reject this.

    The citation file, its event digest, the binding, the report and the run record all agree
    with each other; they are consistent with a different batch id than the one being checked.
    """
    w, started = interrupted(tmp_path)
    base = w.root / "batches" / started.batch_id
    good = AuthorizationCitation.model_validate_json((base / "citations/000001.json").read_bytes())
    run_dir = w.root / "runs" / started.run_id
    run = FinalTestRunRecord.model_validate_json((run_dir / "run.json").read_bytes())
    other_batch = "e" * 64

    def check(citation_batch):
        citation = good.model_copy(update={"batch_id": citation_batch})
        citation_bytes = layout.serialize(citation)
        digest = sha256_bytes(citation_bytes)
        record = run.model_copy(update={"authorization_citation_sha256": digest})
        run_bytes = layout.serialize(record)
        artifacts = _Artifacts()
        artifacts.citations[started.batch_id] = {layout.CITATION_NAME: citation_bytes}
        artifacts.runs[run.run_id] = run_bytes
        cited = {
            "authorization_citation": digest,
            "readiness_report": good.readiness_report_sha256,
            "run_record": sha256_bytes(run_bytes),
        }
        info = {"binding": good.inputs_set_sha256}
        problems: list[tuple[str, str]] = []
        accepted = _check_authorization(started.batch_id, cited, info, artifacts, set(), problems)
        return accepted, problems

    accepted, problems = check(started.batch_id)
    assert accepted and problems == []  # the control: the same construction with the right id
    accepted, problems = check(other_batch)
    assert not accepted
    assert problems == [
        ("conflicting", f"citation of batch {started.batch_id[:12]} is about other evidence")
    ]


# ---- tampering with the resume records --------------------------------------------------------


def test_a_resume_adds_one_event_and_two_files_and_leaves_the_rest_alone(resumed):
    w, started = resumed
    names = sorted(p.name for p in batch_dir(w, started).iterdir())
    assert names == ["000001.json", "000002.continuity.json", "000002.json"]
    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    record = verdict.batches[started.batch_id]
    assert (record.state, record.resumes) == ("authorized", 1)
    assert (
        record.authorization_event_sequence == 6
    )  # reserved, materialized, bound, authorized, resumed


def test_an_altered_resume_citation_is_conflicting(resumed):
    w, started = resumed
    path = batch_dir(w, started) / "000002.json"
    path.chmod(0o644)  # published files are read-only
    path.write_bytes(path.read_bytes() + b" ")
    found = assess(w, started.batch_id)
    assert found.classification == "contradictory" and found.code == "root_unverified"
    assert (
        "differ" in found.detail
        or "not in canonical" in found.detail
        or "conflicting" in found.detail
    )


def test_a_missing_continuity_assessment_leaves_an_incomplete_publication(resumed):
    w, started = resumed
    (batch_dir(w, started) / "000002.continuity.json").unlink()
    found = assess(w, started.batch_id)
    # the citation file now has no accepted event: the batch is not resumed again either way
    assert found.classification in ("structurally_incomplete", "contradictory")
    assert found.code == "root_unverified" and "lacks its records" in found.detail


def test_an_orphan_citation_without_an_event_is_not_adopted(resumed):
    w, started = resumed
    folder = batch_dir(w, started)
    (folder / "000003.json").write_bytes((folder / "000002.json").read_bytes())
    found = assess(w, started.batch_id)
    assert (found.classification, found.code) == ("contradictory", "root_unverified")
    assert "000003.json" in found.detail and "not authorized" in found.detail


def test_a_second_resume_event_citing_a_stale_authorizing_event_is_conflicting(resumed):
    w, started = resumed
    registry_dir = w.root / "registry"
    events = sorted(registry_dir.glob("0*.json"))
    import json

    last = json.loads(events[-1].read_bytes())
    evidence = [EvidenceCitation(**c) for c in last["evidence"]]
    stale = [c for c in evidence if c.role != "previous_authorization_event"]
    authorized = json.loads(events[4].read_bytes())  # the batch_authorized event
    assert authorized["event_type"] == "batch_authorized"
    stale.insert(
        1,
        EvidenceCitation(
            role="previous_authorization_event",
            sha256=hashlib.sha256(events[4].read_bytes()).hexdigest(),
        ),
    )
    with inventory.registry_lock(w.root) as held:
        registry.append_event_locked(
            held, w.root, w.auth, w.witnesses, "batch_resumed", stale, now=w.clock
        )
    found = assess(w, started.batch_id)
    assert found.classification == "contradictory"
    assert "cites another authorizing event" in found.detail


def test_a_resume_event_in_a_final_test_root_is_an_unauthorized_artifact(tmp_path):
    from finaltest_support import authorization

    base = tmp_path / "final"
    base.mkdir()
    auth = authorization(base, "final_test")
    anchors, witnesses = base / "anchors", base / "witnesses"
    inventory._initialize(Path(auth.authorized_root), auth, anchors, inventory.utc_now)
    root, root_id = Path(auth.authorized_root), layout.root_id_for(auth)
    registry._ensure_genesis_witness(root, witnesses, root_id, inventory.utc_now)
    cited = [
        EvidenceCitation(role=r, sha256="a" * 64) for r in layout.REHEARSAL_ROLES["batch_resumed"]
    ]
    registry._append(root, witnesses, root_id, "batch_resumed", cited, inventory.utc_now)
    (root / layout.HOLDER_FILE).write_bytes(b"{}")
    verdict = verify_root(root, auth, anchors, witnesses)
    assert any("unauthorized type batch_resumed" in p for p in verdict.problems)
    assert any(layout.HOLDER_FILE in p for p in verdict.problems)  # unexpected outside rehearsal


def test_registry_rollback_of_the_resume_event_makes_its_records_orphans(resumed):
    w, started = resumed
    events = sorted((w.root / "registry").glob("0*.json"))
    witnesses = sorted(p for p in w.witnesses.rglob("*") if p.is_file())
    events[-1].unlink()
    for path in witnesses:
        if path.name.startswith(events[-1].stem) or events[-1].stem in path.name:
            path.unlink()
    found = assess(w, started.batch_id)
    assert found.classification == "contradictory"
    assert "not authorized" in found.detail  # the resume records have no event any more


def test_a_head_the_registry_cannot_contain_is_refused_by_name(resumed, monkeypatch):
    from types import SimpleNamespace

    from biorag_repro.finaltest import resume as module

    w, started = resumed
    monkeypatch.setattr(
        module,
        "assess_head",
        lambda *_a, **_k: SimpleNamespace(classification="registry_behind_claim"),
    )
    found = assess(w, started.batch_id)
    assert (found.classification, found.code) == (
        "contradictory",
        "registry_head_registry_behind_claim",
    )


# ---- interruption at each publication of a resume ----------------------------------------------


class Fault(Exception):
    pass


# (index of the failing publication, what it was): event, its witness, continuity, citation
PUBLICATIONS = ["event", "witness", "continuity", "citation"]


@pytest.mark.parametrize("index", range(4))
def test_interruption_at_every_resume_publication(tmp_path, index):
    w, started = interrupted(tmp_path)
    renew(w)
    run_dir = w.root / "runs" / started.run_id
    run_before = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    counter = {"n": 0}
    original = registry.publish_bytes

    def failing(target, data, *args, **kwargs):
        counter["n"] += 1
        if counter["n"] == index + 1:
            raise Fault(PUBLICATIONS[index])
        return original(target, data, *args, **kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(registry, "publish_bytes", failing)
    try:
        with open_session(w.root) as session, pytest.raises(Fault):
            resume(w, session)
    finally:
        patch.undo()
    # no attempt was started or recovered, whatever survived
    assert {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()} == run_before
    found = assess(w, started.batch_id)
    surviving = w.tree()
    renew(w, "2" * 32)
    if index == 0:  # nothing was written: the same resume can simply run again
        assert found.classification == "resumable"
        with open_session(w.root) as session:
            assert resume(w, session).batch_id == started.batch_id
        return
    assert found.classification in ("structurally_incomplete", "contradictory")
    surviving = {k: v for k, v in w.tree().items() if not k.endswith("session.holder")}
    with open_session(w.root) as session, pytest.raises(ResumeRefused) as caught:
        resume(w, session)
    assert caught.value.code == found.classification
    assert {k: v for k, v in w.tree().items() if not k.endswith("session.holder")} == surviving
    assert {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()} == run_before


def test_a_resume_citation_never_contains_the_hash_of_anything_built_after_it(resumed):
    w, started = resumed
    folder = batch_dir(w, started)
    event_files = sorted((w.root / "registry").glob("0*.json"))
    resume_event = event_files[-1].read_bytes()
    parts = {
        "assessment": sha256_bytes((folder / "000002.continuity.json").read_bytes()),
        "citation": sha256_bytes((folder / "000002.json").read_bytes()),
        "event": sha256_bytes(resume_event),
    }
    assert (
        parts["assessment"].encode() in resume_event and parts["citation"].encode() in resume_event
    )
    assert parts["event"].encode() not in (folder / "000002.json").read_bytes()
    assert parts["event"].encode() not in (folder / "000002.continuity.json").read_bytes()
    assert parts["citation"].encode() not in (folder / "000002.continuity.json").read_bytes()
    citation = AuthorizationCitation.model_validate_json((folder / "000002.json").read_bytes())
    assert parts["assessment"] == citation.continuity_assessment_sha256
    assert citation.previous_citation_sha256 == sha256_bytes((folder / "000001.json").read_bytes())
    assert citation_sha256(citation) == parts["citation"]


def test_two_resumes_make_a_chain_of_three_citations(tmp_path):
    w, started = interrupted(tmp_path)
    for nonce in ("1" * 32, "2" * 32):
        renew(w, nonce)
        with open_session(w.root) as session:
            resume(w, session)
    found = assess(w, started.batch_id)
    assert [c.kind for c in found.citations] == ["initial", "resume", "resume"]
    assert root_problems(w) == []
    assert len({c.readiness_report_sha256 for c in found.citations}) == 3


# ---- states other than "authorized" ---------------------------------------------------------


def test_a_batch_closed_incomplete_is_terminal(tmp_path):
    from finaltest_audit_support import ARM, PLAN

    w = make_world(tmp_path)
    batch = registry.reserve_batch(w.root, w.auth, w.anchors, w.witnesses, PLAN, ARM)
    registry.close_incomplete(w.root, w.auth, w.anchors, w.witnesses, batch)
    found = assess(w, batch)
    assert (found.classification, found.code) == ("terminally_closed", "batch_closed_incomplete")


def test_a_materialized_batch_is_not_resumable(tmp_path):
    from finaltest_audit_support import ARM, PLAN

    w = make_world(tmp_path)
    batch = registry.reserve_batch(w.root, w.auth, w.anchors, w.witnesses, PLAN, ARM)
    registry.materialize_batch(w.root, w.auth, w.anchors, w.witnesses, batch)
    found = assess(w, batch)
    assert (found.classification, found.code) == ("structurally_incomplete", "batch_materialized")


# ---- a resume whose records disagree is caught before anything runs -----------------------------


def tampered_resume(tmp_path, monkeypatch, patch):
    from biorag_repro.finaltest import resume as module

    w, started = interrupted(tmp_path)
    renew(w)
    run_dir = w.root / "runs" / started.run_id
    run_before = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    patch(monkeypatch, module, w)
    with open_session(w.root) as session, pytest.raises(ResumeRefused) as caught:
        resume(w, session)
    assert caught.value.code == "publication_unverified"
    # the store was never reopened: no attempt, no recovery
    assert {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()} == run_before
    return w, started


def test_a_resume_event_citing_another_run_record_is_conflicting(tmp_path, monkeypatch):
    def patch(mp, module, _w):
        mp.setattr(module, "run_record_sha256", lambda _r: "9" * 64)

    w, _ = tampered_resume(tmp_path, monkeypatch, patch)
    assert any("about other evidence" in p for p in root_problems(w))


def test_a_citation_naming_a_head_that_is_not_the_previous_event_is_conflicting(
    tmp_path, monkeypatch
):
    def patch(mp, module, w):
        real = module.registry.registry_head
        calls = {"n": 0}

        def stale_head(root):
            calls["n"] += 1
            sequence, digest = real(root)
            if calls["n"] <= 2:  # the value the citation records, and the guard comparison
                return sequence - 1, digest
            return sequence, digest

        mp.setattr(module.registry, "registry_head", stale_head)

    w, _ = tampered_resume(tmp_path, monkeypatch, patch)
    assert any("about other evidence" in p for p in root_problems(w))


@pytest.mark.parametrize(
    "field",
    [
        "inputs_set_sha256",
        "readiness_report_sha256",
        "continuity_assessment_sha256",
        "scientific_identity_sha256",
        "initial_report_sha256",
    ],
)
def test_a_citation_that_disagrees_with_its_event_or_binding_is_conflicting(
    tmp_path, monkeypatch, field
):
    def patch(mp, module, _w):
        real = module.AuthorizationCitation

        def altered(**kwargs):
            return real(**(kwargs | {field: "9" * 64}))

        altered.model_validate_json = real.model_validate_json  # type: ignore[attr-defined]
        mp.setattr(module, "AuthorizationCitation", altered)
        # the in-memory chain check would refuse some of these first; this test is about the
        # verifier, so let the orchestrator reach the publication
        mp.setattr(module, "verify_citation_chain", lambda *_a, **_k: None)

    w, _ = tampered_resume(tmp_path, monkeypatch, patch)
    assert root_problems(w)


def test_a_canonical_but_different_resume_record_is_not_the_one_the_event_cites(resumed):
    w, started = resumed
    folder = batch_dir(w, started)
    for name, model in (
        ("000002.json", AuthorizationCitation),
        ("000002.continuity.json", ContinuityAssessment),
    ):
        path = folder / name
        original = path.read_bytes()
        if model is AuthorizationCitation:
            record = model.model_validate_json(original)
            other = record.model_copy(update={"created_utc": "2030-01-01T00:00:00Z"})
        else:  # same verdict, one more (identical) finding
            record = model.model_validate_json(original)
            other = record.model_copy(update={"findings": [*record.findings, record.findings[0]]})
        path.chmod(0o644)
        path.write_bytes(layout.serialize(other))
        found = assess(w, started.batch_id)
        assert found.classification == "contradictory", name
        assert "differ from it" in found.detail, name
        path.write_bytes(original)
        assert assess(w, started.batch_id).classification == "resumable"


def test_a_resume_event_on_a_batch_that_was_never_authorized_is_conflicting(tmp_path):
    from finaltest_audit_support import ARM, PLAN

    w = make_world(tmp_path)
    batch = registry.reserve_batch(w.root, w.auth, w.anchors, w.witnesses, PLAN, ARM)
    registry.materialize_batch(w.root, w.auth, w.anchors, w.witnesses, batch)
    evidence = [
        EvidenceCitation(role=role, sha256=batch if role == "batch_id" else "a" * 64)
        for role in layout.REHEARSAL_ROLES["batch_resumed"]
    ]
    with inventory.registry_lock(w.root) as held:
        registry.append_event_locked(
            held, w.root, w.auth, w.witnesses, "batch_resumed", evidence, now=w.clock
        )
    problems = root_problems(w)
    assert any("follows batch" in p and "materialized" in p for p in problems), problems


def test_a_claim_finding_of_the_audit_makes_the_batch_contradictory(resumed, monkeypatch):
    from biorag_repro.finaltest import resume as module
    from biorag_repro.finaltest.audit import audit_run
    from biorag_repro.finaltest.inputs import Finding

    w, started = resumed

    def audited(*args, **kwargs):
        result = audit_run(*args, **kwargs)
        return type(result)(
            result.run_id,
            (*result.findings, Finding("claim", "binding_missing", started.batch_id)),
            result.observed,
        )

    monkeypatch.setattr(module, "audit_run", audited)
    found = assess(w, started.batch_id)
    assert (found.classification, found.code) == ("contradictory", "binding_missing")
