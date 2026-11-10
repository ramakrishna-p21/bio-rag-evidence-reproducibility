"""The verifier's view of `batch_completed`, its record and its authorization.

Each tampering test starts from a real synthetic completion and changes one stored fact, so the
expected problem is the one check that fact is protected by. Field-level checks call the
verifier's own fold with the stored records, as the C3 tests do.
"""

from pathlib import Path

import pytest
from finaltest_completion_support import complete_fresh, finished
from finaltest_support import authorization

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.finaltest import inventory, layout, registry
from biorag_repro.finaltest.batches import AuthorizationCitation, citation_sha256
from biorag_repro.finaltest.completion import BatchCompletion, PublicationAuthorization
from biorag_repro.finaltest.continuity import ContinuityAssessment
from biorag_repro.finaltest.inventory import _Artifacts, _check_completion, verify_root
from biorag_repro.finaltest.layout import EvidenceCitation

NOTE = inventory.HEAD_NOT_ESTABLISHED_TEXT
UNAUTH = "unauthorized_final_test_artifact"


def problems(w):
    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    return [p for p in verdict.problems if p != NOTE]


@pytest.fixture
def done(tmp_path):
    """A batch completed by a new session: the world, the started batch and the record."""
    w, started = finished(tmp_path)
    result = complete_fresh(w)
    return w, started, result.completion


def base_of(w, started):
    return w.root / "batches" / started.batch_id


def rewrite(path: Path, data: bytes):
    path.chmod(0o644)
    path.write_bytes(data)
    path.chmod(0o444)


def event_files(w):
    return sorted((w.root / "registry").glob("0*.json"))


# ---- a verified completion --------------------------------------------------------------------


def test_a_completed_batch_verifies_with_only_the_head_note(done):
    w, started, record = done
    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    assert verdict.problems == (NOTE,)
    info = verdict.batches[started.batch_id]
    assert info.state == "completed"
    assert info.completion_sha256 == sha256_bytes(layout.serialize(record))
    entries = {e["path"] for e in verdict.inventory["entries"]}
    assert f"batches/{started.batch_id}/{layout.COMPLETION_NAME}" in entries
    assert f"batches/{started.batch_id}/{layout.COMPLETION_CONTINUITY_NAME}" in entries


# ---- tampered or inconsistent stored facts ---------------------------------------------------


def test_an_altered_completion_file_differs_from_its_event(done):
    w, started, _ = done
    path = base_of(w, started) / layout.COMPLETION_NAME
    rewrite(path, path.read_bytes() + b" ")
    assert any("differs from its event" in p for p in problems(w))


def test_a_completion_that_is_canonical_but_not_the_one_the_event_cites_is_refused(done):
    w, started, record = done
    other = record.model_copy(update={"created_utc": "2026-11-09T00:00:00Z"})
    rewrite(base_of(w, started) / layout.COMPLETION_NAME, layout.serialize(other))
    assert any("differs from its event" in p for p in problems(w))


def test_a_deleted_completion_file_is_an_unfinished_publication(done):
    w, started, _ = done
    (base_of(w, started) / layout.COMPLETION_NAME).unlink()
    assert any("has no completion record" in p for p in problems(w))
    assert verify_root(w.root, w.auth, w.anchors, w.witnesses).batches[started.batch_id].state == (
        "authorized"
    )


def test_a_deleted_continuity_file_is_an_unfinished_publication(done):
    w, started, _ = done
    (base_of(w, started) / layout.COMPLETION_CONTINUITY_NAME).unlink()
    assert any("lacks its continuity record" in p for p in problems(w))


def test_an_altered_continuity_file_misstates_the_authorization(done):
    w, started, _ = done
    path = base_of(w, started) / layout.COMPLETION_CONTINUITY_NAME
    rewrite(path, path.read_bytes() + b" ")
    assert any("misstates its authorization" in p for p in problems(w))


def test_a_completion_file_without_its_event_is_an_unexpected_artifact(tmp_path):
    w, started = finished(tmp_path)
    with inventory.registry_lock(w.root) as held:
        registry.publish_batch_file(
            held, w.root, w.auth, f"batches/{started.batch_id}/{layout.COMPLETION_NAME}", b"{}"
        )
    assert any("has no event" in p for p in problems(w))


def test_a_completion_file_for_an_unreserved_batch_is_unexpected(tmp_path):
    w, _ = finished(tmp_path)
    other = "e" * 64
    folder = w.root / "batches" / other
    folder.mkdir()
    (folder / layout.COMPLETION_NAME).write_bytes(b"{}")
    assert any(other[:12] in p for p in problems(w))


@pytest.mark.parametrize(
    "event_type", ["batch_resumed", "batch_completed", "batch_authorized", "batch_inputs_bound"]
)
def test_no_event_may_follow_a_completion(done, event_type):
    w, started, _ = done
    evidence = [
        EvidenceCitation(role=role, sha256=started.batch_id if role == "batch_id" else "a" * 64)
        for role in layout.REHEARSAL_ROLES[event_type]
    ]
    registry._append(w.root, w.witnesses, layout.root_id_for(w.auth), event_type, evidence, w.clock)
    assert any("completed" in p and "follows batch" in p for p in problems(w))


def test_a_completion_event_that_cites_another_authorizing_event_is_a_conflict(tmp_path):
    w, started = finished(tmp_path)
    evidence = [
        EvidenceCitation(role=role, sha256=started.batch_id if role == "batch_id" else "a" * 64)
        for role in layout.REHEARSAL_ROLES["batch_completed"]
    ]
    registry._append(
        w.root, w.witnesses, layout.root_id_for(w.auth), "batch_completed", evidence, w.clock
    )
    assert any("cites another authorizing event" in p for p in problems(w))


def test_a_completion_event_for_an_unreserved_batch_is_a_conflict(tmp_path):
    w, _ = finished(tmp_path)
    evidence = [
        EvidenceCitation(role=role, sha256="b" * 64)
        for role in layout.REHEARSAL_ROLES["batch_completed"]
    ]
    registry._append(
        w.root, w.witnesses, layout.root_id_for(w.auth), "batch_completed", evidence, w.clock
    )
    assert any("unreserved batch" in p for p in problems(w))


def test_a_rolled_back_completion_event_leaves_a_witness_without_its_event(done):
    w, _, _ = done
    last = event_files(w)[-1]
    last.chmod(0o644)
    last.unlink()
    found = problems(w)
    assert any("witnesses exist beyond the registry head" in p for p in found)
    assert any("has no event" in p for p in found)  # the completion file is now an orphan


# ---- field-level checks of the record against the batch's history ----------------------------


def fold_inputs(w, started, record):
    """The values the verifier folds for a completed batch, rebuilt from the stored files."""
    base = base_of(w, started)
    files = event_files(w)
    number = int(files[-1].name[:6])
    event = layout.RegistryEvent.model_validate_json(files[-1].read_bytes())
    cited = {c.role: c.sha256 for c in event.evidence}
    chain = [
        AuthorizationCitation.model_validate_json(p.read_bytes())
        for p in sorted((base / "citations").glob("0*.json"))
        if "continuity" not in p.name
    ]
    authorizing = next(
        sha256_bytes(p.read_bytes())
        for p in reversed(files)
        if layout.RegistryEvent.model_validate_json(p.read_bytes()).event_type
        in ("batch_authorized", "batch_resumed")
    )
    info = {
        "state": "authorized",
        "run_id": started.run_id,
        "plan": record.campaign_plan_sha256,
        "manifest": record.manifest_sha256,
        "ownership": record.ownership_sha256,
        "binding": record.inputs_set_sha256,
        "run_record": started.run_record_sha256,
        "chain": chain,
        "last_citation": citation_sha256(chain[-1]),
        "auth_event": authorizing,
        "resumes": len(chain) - 1,
    }
    artifacts = _Artifacts()
    artifacts.completions[started.batch_id] = {
        name: (base / name).read_bytes()
        for name in (layout.COMPLETION_NAME, layout.COMPLETION_CONTINUITY_NAME)
        if (base / name).exists()
    }
    hashes = {
        f"registry/{n - 1:06d}.json": sha256_bytes(files[n - 2].read_bytes()) for n in (number,)
    }
    return number, cited, info, artifacts, hashes


def check(w, started, record, mutate=None, *, drop=()):
    number, cited, info, artifacts, hashes = fold_inputs(w, started, record)
    base = base_of(w, started)
    if mutate is not None:
        changed = mutate(record)
        data = layout.serialize(changed)
        artifacts.completions[started.batch_id][layout.COMPLETION_NAME] = data
        cited = cited | {"completion_record": sha256_bytes(data)}
    for name in drop:
        artifacts.completions[started.batch_id].pop(name, None)
    found: list = []
    ok = _check_completion(number, started.batch_id, cited, info, artifacts, hashes, found)
    assert base.exists()
    return ok, [text for _, text in found]


def test_the_stored_completion_passes_the_field_level_check(done):
    w, started, record = done
    assert check(w, started, record) == (True, [])


def auth(record, **changes):
    return record.model_copy(
        update={"authorization": record.authorization.model_copy(update=changes)}
    )


OTHER = "c" * 64


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_id": OTHER},
        {"run_id": "d" * 32},
        {"campaign_plan_sha256": OTHER},
        {"manifest_sha256": OTHER},
        {"ownership_sha256": OTHER},
        {"inputs_set_sha256": OTHER},
        {"run_record_sha256": OTHER},
        {"citation_count": 2},
        {"first_citation_sha256": OTHER},
        {"last_citation_sha256": OTHER},
        {"authorizing_event_sha256": OTHER},
        {"registry_head_sequence": 1},
        {"registry_head_event_sha256": OTHER},
    ],
)
def test_every_binding_of_the_record_to_the_batch_history_is_checked(done, changes):
    w, started, record = done
    ok, found = check(w, started, record, lambda r: r.model_copy(update=changes))
    assert not ok and any("about other evidence" in t for t in found), found


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: auth(r, report_sha256=OTHER),
        lambda r: auth(r, basis_sha256=OTHER),
        lambda r: auth(r, scientific_identity_sha256=OTHER),
    ],
)
def test_the_publication_authorization_must_match_the_event_and_the_batch(done, mutation):
    w, started, record = done
    ok, found = check(w, started, record, mutation)
    assert not ok and any("about other evidence" in t for t in found), found


def test_a_same_session_record_may_not_carry_a_continuity_file(done):
    w, started, record = done
    chain_last = fold_inputs(w, started, record)[2]["chain"][-1]
    claim = auth(
        record,
        mode="same_session",
        basis_sha256=citation_sha256(chain_last),
        report_sha256=chain_last.readiness_report_sha256,
    )
    number, cited, info, artifacts, hashes = fold_inputs(w, started, record)
    data = layout.serialize(claim)
    artifacts.completions[started.batch_id][layout.COMPLETION_NAME] = data
    cited = cited | {
        "completion_record": sha256_bytes(data),
        "authorization_basis": citation_sha256(chain_last),
        "readiness_report": chain_last.readiness_report_sha256,
    }
    found: list = []
    ok = _check_completion(number, started.batch_id, cited, info, artifacts, hashes, found)
    assert not ok and any("misstates its authorization" in t for _, t in found)


def test_a_same_session_record_must_name_the_latest_citation_as_its_basis(done):
    w, started, record = done
    chain_last = fold_inputs(w, started, record)[2]["chain"][-1]
    claim = auth(
        record,
        mode="same_session",
        basis_sha256=OTHER,
        report_sha256=chain_last.readiness_report_sha256,
    )
    number, cited, info, artifacts, hashes = fold_inputs(w, started, record)
    data = layout.serialize(claim)
    artifacts.completions[started.batch_id] = {layout.COMPLETION_NAME: data}
    cited = cited | {
        "completion_record": sha256_bytes(data),
        "authorization_basis": OTHER,
        "readiness_report": chain_last.readiness_report_sha256,
    }
    found: list = []
    assert not _check_completion(number, started.batch_id, cited, info, artifacts, hashes, found)
    assert any("misstates its authorization" in t for _, t in found)


def test_a_fresh_record_whose_report_was_already_cited_is_refused(done):
    w, started, record = done
    chain_last = fold_inputs(w, started, record)[2]["chain"][-1]
    claim = auth(record, report_sha256=chain_last.readiness_report_sha256)
    number, cited, info, artifacts, hashes = fold_inputs(w, started, record)
    data = layout.serialize(claim)
    artifacts.completions[started.batch_id][layout.COMPLETION_NAME] = data
    cited = cited | {
        "completion_record": sha256_bytes(data),
        "readiness_report": chain_last.readiness_report_sha256,
    }
    found: list = []
    assert not _check_completion(number, started.batch_id, cited, info, artifacts, hashes, found)
    assert any("reuses a cited report" in t for _, t in found)


def test_a_blocked_continuity_assessment_cannot_authorize(done):
    w, started, record = done
    number, cited, info, artifacts, hashes = fold_inputs(w, started, record)
    stored = ContinuityAssessment.model_validate_json(
        artifacts.completions[started.batch_id][layout.COMPLETION_CONTINUITY_NAME]
    )
    blocked = stored.model_copy(
        update={
            "verdict": "blocked",
            "findings": [
                stored.findings[0].model_copy(update={"verdict": "blocking"}),
                *stored.findings[1:],
            ],
        }
    )
    data = layout.serialize(blocked)
    digest = sha256_bytes(data)
    claim = auth(record, basis_sha256=digest)
    record_bytes = layout.serialize(claim)
    artifacts.completions[started.batch_id] = {
        layout.COMPLETION_NAME: record_bytes,
        layout.COMPLETION_CONTINUITY_NAME: data,
    }
    cited = cited | {"completion_record": sha256_bytes(record_bytes), "authorization_basis": digest}
    found: list = []
    assert not _check_completion(number, started.batch_id, cited, info, artifacts, hashes, found)
    assert found  # either unreadable as a record or not continuous: never accepted


def test_a_missing_continuity_file_is_incomplete_not_accepted(done):
    w, started, record = done
    ok, found = check(w, started, record, drop=(layout.COMPLETION_CONTINUITY_NAME,))
    assert not ok and any("lacks its continuity record" in t for t in found)


def test_an_unknown_completion_file_is_unexpected(done):
    w, started, record = done
    number, cited, info, artifacts, hashes = fold_inputs(w, started, record)
    artifacts.completions[started.batch_id]["extra.json"] = b"{}"
    found: list = []
    assert not _check_completion(number, started.batch_id, cited, info, artifacts, hashes, found)
    assert any(kind == "unexpected" for kind, _ in found)


# ---- completion model invariants --------------------------------------------------------------


def test_the_record_refuses_inconsistent_counts_outcome_and_order(done):
    _, _, record = done
    data = record.model_dump()
    from pydantic import ValidationError

    for bad in (
        {"complete_count": 9, "missing_count": 1},
        {"outcome": "complete_with_recorded_missing"},
        {"observations": list(reversed(record.observations))},
        {"observations": []},
    ):
        with pytest.raises(ValidationError):
            BatchCompletion.model_validate(data | bad)
    failed = record.observations[0].model_copy(update={"status": "terminal_failure"})
    with pytest.raises(ValidationError):
        type(failed).model_validate(failed.model_dump())


# ---- final-test roots stay as they were --------------------------------------------------------


def test_a_final_test_root_still_treats_a_completion_event_as_unauthorized(tmp_path):
    base = tmp_path / "final"
    base.mkdir()
    auth_ = authorization(base, "final_test")
    anchors, witnesses = base / "anchors", base / "witnesses"
    inventory._initialize(Path(auth_.authorized_root), auth_, anchors, inventory.utc_now)
    root = Path(auth_.authorized_root)
    root_id = layout.root_id_for(auth_)
    registry._ensure_genesis_witness(root, witnesses, root_id, inventory.utc_now)
    evidence = [
        EvidenceCitation(role=r, sha256="a" * 64) for r in layout.REHEARSAL_ROLES["batch_completed"]
    ]
    registry._append(root, witnesses, root_id, "batch_completed", evidence, inventory.utc_now)
    (root / "batches").mkdir()
    (root / "batches" / ("a" * 64)).mkdir()
    (root / "batches" / ("a" * 64) / layout.COMPLETION_NAME).write_bytes(b"{}")
    verdict = verify_root(root, auth_, anchors, witnesses)
    assert verdict.code == UNAUTH
    assert any("unauthorized type batch_completed" in p for p in verdict.problems)
    assert any(layout.COMPLETION_NAME in p for p in verdict.problems)
    with inventory.registry_lock(root) as held, pytest.raises(registry.ProtocolRefused):
        registry.publish_batch_file(
            held, root, auth_, f"batches/{'a' * 64}/{layout.COMPLETION_NAME}", b"{}"
        )
    assert PublicationAuthorization  # the model is part of the public record set


def test_an_observation_end_must_match_how_its_last_attempt_ended(done):
    from pydantic import ValidationError

    from biorag_repro.finaltest.completion import ObservationEnd

    _, _, record = done
    base = record.observations[0].model_dump()  # a complete observation
    refused_cases = [
        {"status": "terminal_failure"},  # it did not end in a terminal failure
        {"status": "retries_exhausted"},  # a completed attempt does not exhaust retries
        {"failure_kind": "k", "failure_detail": "d"},  # a completed attempt has no failure kind
        {"final_state": "interrupted"},  # a complete observation does not end interrupted
        {"failure_kind": "k"},  # a failure carries its kind and its detail together
    ]
    for change in refused_cases:
        with pytest.raises(ValidationError):
            ObservationEnd.model_validate(base | change)
    terminal = {"status": "terminal_failure", "final_state": "terminal_failure"}
    exhausted = {"status": "retries_exhausted"}
    with pytest.raises(ValidationError):  # a terminal failure names its kind
        ObservationEnd.model_validate(base | terminal)
    with pytest.raises(ValidationError):  # a failed last attempt names its kind
        ObservationEnd.model_validate(base | exhausted | {"final_state": "retryable_failure"})
    with pytest.raises(ValidationError):  # an interruption names no failure kind
        ObservationEnd.model_validate(
            base
            | exhausted
            | {"final_state": "interrupted", "failure_kind": "k", "failure_detail": "d"}
        )
    # and the legitimate endings are accepted
    detail = {"failure_kind": "k", "failure_detail": "d"}
    ObservationEnd.model_validate(base | terminal | detail)
    ObservationEnd.model_validate(base | exhausted | {"final_state": "interrupted"})
    ObservationEnd.model_validate(base | exhausted | {"final_state": "retryable_failure"} | detail)
