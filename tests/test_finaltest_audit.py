"""Read-only audit of hand-built final-test-shaped runs. Nothing is executed or repaired."""

import dataclasses
import inspect
import os
import types
import typing
from pathlib import Path

import pytest
from finaltest_audit_support import (
    ARM,
    CONFIGS,
    NOW,
    PLAN,
    RUN_ID,
    build_run,
    roster_for,
    row_for,
    snapshot,
)
from finaltest_campaign_support import CANARY, oracle_key, sha
from pydantic import ValidationError
from test_finaltest_citation import initial_citation
from test_generation_attempts import RENDERED, environment, new_store
from test_generation_attempts import coordinates as dev_coordinates

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.finaltest.audit import (
    AuditContext,
    RunAudit,
    audit_run,
)
from biorag_repro.finaltest.batches import citation_sha256, make_manifest
from biorag_repro.finaltest.inputs import binding_sha256, retry_policy_sha256
from biorag_repro.finaltest.runrecord import FinalTestRunRecord
from biorag_repro.generation.attempts import (
    AttemptEvent,
    RetryPolicy,
    RunRecord,
    RunStore,
    retry_policy_of,
)
from biorag_repro.generation.readonly import ReadOnlyRunView, ReadOnlyViolation

OTHER_RUN = sha("second-run")[:32]


def codes(audit, kind=None):
    return {f.code for f in audit.findings if kind is None or f.kind == kind}


def audit(built, run_id=RUN_ID, **changes):
    context = dataclasses.replace(built.context, **changes)
    return audit_run(built.base, run_id, context)


def no_writes(built, **changes):
    before = snapshot(built.base)
    result = audit(built, **changes)
    assert snapshot(built.base) == before, "the audit changed the stored records"
    return result


def test_clean_run_verifies_but_authorization_stays_not_established(tmp_path):
    built = build_run(tmp_path)
    result = no_writes(built)
    assert result.structure_valid and result.claims_consistent and result.complete
    assert codes(result) == {"ordering_unverified", "authority_not_established"}
    assert result.authorization == "not_established"
    assert result.ordering == "unverified"
    assert len(result.observed.states) == 10 and {s.status for s in result.observed.states} == {
        "complete"
    }


def test_no_verdict_can_say_authorized():
    hints = typing.get_type_hints(RunAudit)
    assert typing.get_args(hints["authorization"]) == ("not_established",)
    assert typing.get_args(hints["ordering"]) == ("unverified",)


def test_a_consistent_citation_is_only_a_historical_claim(tmp_path):
    built = build_run(tmp_path)
    result = no_writes(built)
    assert result.claims_consistent
    assert any(f.code == "authority_not_established" for f in result.of_kind("note"))


# --- no-write guarantee ----------------------------------------------------------------


def test_missing_seal_is_reported_and_never_created(tmp_path):
    built = build_run(tmp_path, default="complete_unsealed")
    result = no_writes(built)
    assert [f.code for f in result.findings if f.code == "seal_missing"] == ["seal_missing"] * 10
    assert not result.structure_valid
    assert not list(built.run_dir.rglob("final.build.json"))


def test_interrupted_and_unfinished_observations_are_classified_not_recovered(tmp_path):
    built = build_run(
        tmp_path,
        statuses={"i1": "interrupted", "i2": "planned", "i3": "terminal_failure"},
    )
    result = no_writes(built)
    assert result.structure_valid
    assert not result.complete
    by_code = {f.code: 0 for f in result.of_kind("incomplete")}
    assert by_code.keys() == {"interrupted_attempt", "never_attempted", "recorded_terminal_failure"}
    assert {s.status for s in result.observed.states} >= {"running", "planned", "terminal_failure"}


def test_read_only_filesystem_cannot_be_written_by_the_audit(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores file modes")
    built = build_run(tmp_path, statuses={"i1": "interrupted"}, default="complete_unsealed")
    for current, directories, files in os.walk(tmp_path):
        for name in files:
            (Path(current) / name).chmod(0o444)
        for name in directories:
            (Path(current) / name).chmod(0o555)
    tmp_path.chmod(0o555)
    try:
        result = audit(built)
    finally:
        for current, directories, files in os.walk(tmp_path, topdown=False):
            for name in directories:
                (Path(current) / name).chmod(0o755)
        tmp_path.chmod(0o755)
    assert "seal_missing" in codes(result) and "interrupted_attempt" in codes(result)


def test_malformed_records_still_leave_the_tree_untouched(tmp_path):
    built = build_run(tmp_path)
    key = built.keys[("i1", "JATS", 1)]
    built.event_path(key, 2).write_bytes(b"{not json")
    result = no_writes(built)
    assert "observation_unreadable" in codes(result)
    assert not result.structure_valid


def test_every_mutating_run_store_method_is_refused_by_the_view():
    reads = {"observation", "observation_keys", "verify", "pending", "key_for", "receipt_of"}
    names = {
        n
        for n, member in inspect.getmembers(RunStore)
        if not n.startswith("__") and callable(member) and n not in reads
    }
    writers = {n for n in names if not n.startswith("_check") and n in vars(ReadOnlyRunView)}
    assert {
        "create", "open", "plan", "begin_attempt", "complete", "fail",
        "recover_interrupted", "publish_input", "_append", "_seal",
    } <= writers  # fmt: skip
    # a new public RunStore method must be classified here, as read or as refused
    public = {n for n in names if not n.startswith("_")}
    assert public <= writers, public - writers
    with pytest.raises(ReadOnlyViolation):
        ReadOnlyRunView.create()
    with pytest.raises(ReadOnlyViolation):
        ReadOnlyRunView.open()


# --- unsafe storage ---------------------------------------------------------------------


def test_symlinks_are_reported_and_not_followed(tmp_path):
    built = build_run(tmp_path)
    key = built.keys[("i1", "JATS", 1)]
    target = tmp_path / "elsewhere.json"
    target.write_bytes(built.event_path(key, 1).read_bytes())
    built.event_path(key, 1).unlink()
    built.event_path(key, 1).symlink_to(target)
    result = no_writes(built)
    assert codes(result, "structure") == {"symlink_in_run"}
    assert result.observed is None


def test_a_symlinked_run_directory_is_not_a_run(tmp_path):
    built = build_run(tmp_path)
    real = tmp_path / "moved"
    built.run_dir.rename(real)
    built.run_dir.symlink_to(real)
    assert codes(no_writes(built), "structure") == {"run_missing"}


def test_temporary_files_and_unknown_runs(tmp_path):
    built = build_run(tmp_path)
    (built.run_dir / ".leftover.tmp").write_bytes(b"x")
    assert "temporary_file_in_run" in codes(no_writes(built))
    assert codes(audit_run(tmp_path, sha("nope")[:32], built.context)) == {"run_missing"}


def test_run_record_must_be_canonical_and_strict(tmp_path):
    built = build_run(tmp_path)
    path = built.run_dir / "run.json"
    path.write_bytes(path.read_bytes() + b"\n")
    assert codes(no_writes(built), "structure") == {"run_record_not_canonical"}
    path.write_bytes(b'{"schema_version": "1"}')
    assert codes(no_writes(built), "structure") == {"run_record_invalid"}


# --- identity of the batch, run and observations ---------------------------------------


def test_foreign_batch_root_and_plan(tmp_path):
    built = build_run(tmp_path)
    foreign = make_manifest(sha("another-root"), PLAN, ARM, NOW)
    assert "foreign_batch" in codes(no_writes(built, manifest=foreign))
    assert "foreign_root" in codes(no_writes(built, root_id=sha("another-root")))
    assert "foreign_root" in codes(no_writes(built, root_kind="final_test"))


def test_run_record_is_checked_against_its_configuration_copies(tmp_path):
    built = build_run(tmp_path)
    (built.run_dir / "config.toml").write_bytes(CONFIGS[ARM].raw_toml + b"# edited\n")
    assert codes(no_writes(built), "structure") == {"config_mismatch"}
    other = build_run(tmp_path / "b", raw_config_sha256=sha("other"))
    assert codes(no_writes(other), "structure") == {"config_mismatch"}


def test_run_record_retry_policy_must_be_the_configuration_policy(tmp_path):
    other = RetryPolicy(max_attempts=9, retryable_failures=["transport_error"])
    built = build_run(tmp_path, retry=other, retry_policy_sha256=retry_policy_sha256(other))
    assert codes(no_writes(built), "structure") == {"config_mismatch"}


def test_foreign_and_unbound_observation(tmp_path):
    stranger = row_for(
        ARM,
        "JATS",
        "i1",
        1,
        item_id="zz",
        paper_id="p9",
        observation_key=oracle_key("zz", ARM, "JATS", 1, CONFIGS[ARM].configuration_sha256),
    )
    built = build_run(tmp_path, extra_rows=(stranger,))
    result = no_writes(built)
    assert {"foreign_observation_key", "unbound_observation"} <= codes(result, "structure")


def test_wrong_condition_repeat_paper_or_seed_in_the_roster(tmp_path):
    built = build_run(tmp_path)
    original = built.context.rosters["JATS"]
    key = original.rows[0].observation_key
    wrong = row_for(ARM, "JATS", "i1", 1, paper_id="p3", observation_key=key)
    rosters = dict(built.context.rosters) | {
        "JATS": roster_for(ARM, "JATS", rows=[wrong, *original.rows[1:]])
    }
    result = no_writes(built, rosters=rosters)
    assert [f.subject for f in result.findings if f.code == "roster_coordinates_mismatch"] == [key]


def test_incorrect_rendered_input_hash(tmp_path):
    built = build_run(tmp_path)
    original = built.context.rosters["PDF_DOCLING"]
    changed = row_for(ARM, "PDF_DOCLING", "i1", 1, rendered_input_sha256=sha("other"))
    rosters = dict(built.context.rosters) | {
        "PDF_DOCLING": roster_for(ARM, "PDF_DOCLING", rows=[changed, *original.rows[1:]])
    }
    result = no_writes(built, rosters=rosters)
    assert [f.subject for f in result.findings if f.code == "rendered_input_mismatch"] == [
        changed.observation_key
    ]


def test_planned_only_observations_have_no_rendered_input_to_compare(tmp_path):
    built = build_run(tmp_path, default="planned")
    result = no_writes(built)
    assert result.structure_valid and codes(result, "incomplete") == {"never_attempted"}


def test_missing_rosters_and_binding(tmp_path):
    built = build_run(tmp_path)
    assert "rosters_missing" in codes(no_writes(built, rosters={}), "incomplete")
    result = no_writes(built, binding=None)
    assert "binding_missing" in codes(result, "claim") and not result.claims_consistent
    assert result.structure_valid


def test_expected_observations_that_were_never_planned_are_absent(tmp_path):
    built = build_run(tmp_path)
    key = built.keys[("i1", "JATS", 1)]
    import shutil

    shutil.rmtree(built.run_dir / "observations" / key)
    result = no_writes(built)
    assert [f.subject for f in result.findings if f.code == "observation_absent"] == [key]


# --- tampering --------------------------------------------------------------------------


def test_altered_raw_output_is_detected(tmp_path):
    built = build_run(tmp_path)
    key = built.keys[("i2", "JATS", 1)]
    raw = next((built.run_dir / "observations" / key / "raw").glob("*.txt"))
    raw.write_text('{"answer": "forged", "abstained": false}')
    result = no_writes(built)
    assert [f.subject for f in result.findings if f.code == "observation_corrupt"] == [key]


def test_altered_completion_event_breaks_canonical_form_or_seal(tmp_path):
    built = build_run(tmp_path)
    key = built.keys[("i3", "PDF_DOCLING", 1)]
    path = built.event_path(key, 3)
    text = path.read_text()
    assert '"answer": "42.5 mg/L"' in text
    path.write_text(text.replace('"answer": "42.5 mg/L"', '"answer": "0.0 mg/L"'))
    result = no_writes(built)
    assert any(f.subject == key for f in result.of_kind("structure"))
    assert not result.structure_valid


def test_hash_chain_break(tmp_path):
    built = build_run(tmp_path)
    key = built.keys[("i4", "JATS", 1)]
    event = AttemptEvent.model_validate_json(built.event_path(key, 2).read_bytes())
    forged = event.model_copy(update={"previous_event_sha256": sha("forged")})
    built.event_path(key, 2).write_bytes(serialize_project_json(forged, AttemptEvent))
    result = no_writes(built)
    assert [f.subject for f in result.findings if f.code == "observation_corrupt"] == [key]


def test_second_completion_is_a_conflict(tmp_path):
    built = build_run(tmp_path)
    key = built.keys[("i5", "JATS", 1)]
    last = AttemptEvent.model_validate_json(built.event_path(key, 3).read_bytes())
    previous = sha256_bytes(built.event_path(key, 3).read_bytes())
    again = last.model_copy(update={"sequence": 4, "previous_event_sha256": previous})
    built.event_path(key, 4).write_bytes(serialize_project_json(again, AttemptEvent))
    result = no_writes(built)
    assert [f.subject for f in result.findings if f.code == "duplicate_completion"] == [key]


def test_same_observation_completed_in_two_runs(tmp_path):
    built = build_run(tmp_path)
    build_run(tmp_path, run_id=OTHER_RUN)
    assert "duplicate_completion_across_runs" in codes(no_writes(built))


def test_event_that_is_not_canonical_bytes(tmp_path):
    built = build_run(tmp_path)
    key = built.keys[("i1", "JATS", 1)]
    path = built.event_path(key, 1)
    path.write_bytes(path.read_bytes() + b"\n")
    assert "event_not_canonical" in codes(no_writes(built))


# --- citations --------------------------------------------------------------------------


def test_missing_and_forged_citation_references(tmp_path):
    built = build_run(tmp_path)
    assert "citation_missing" in codes(no_writes(built, citations=[]), "claim")
    forged = initial_citation(batch_id=built.record.batch_id, inputs_set_sha256=sha("forged"))
    result = no_writes(built, citations=[forged])
    assert codes(result, "claim") == {"citation_reference_unknown"}
    assert not result.claims_consistent


def test_citation_for_another_batch_and_broken_chain(tmp_path):
    built = build_run(tmp_path)
    own = built.context.citations[0]
    stranger = initial_citation(batch_id=sha("other-batch"))
    result = no_writes(built, citations=[stranger, own])
    assert "citation_chain_invalid" in codes(result, "claim")
    other = build_run(tmp_path / "b", authorization_citation_sha256=citation_sha256(stranger))
    result = no_writes(other, citations=[stranger])
    assert {"citation_other_batch", "citation_inputs_mismatch"} <= codes(result, "claim")


def test_inputs_binding_reference_must_match_the_binding(tmp_path):
    built = build_run(tmp_path)
    other_inputs = sha("other-inputs")
    citation = initial_citation(batch_id=built.record.batch_id, inputs_set_sha256=other_inputs)
    run = build_run(
        tmp_path / "b",
        inputs_set_sha256=other_inputs,
        authorization_citation_sha256=citation_sha256(citation),
    )
    result = no_writes(run, citations=[citation])
    assert codes(result, "claim") == {"inputs_binding_mismatch"}
    assert binding_sha256(run.context.binding) != other_inputs


# --- final-test run record -------------------------------------------------------------


def test_run_record_contract(tmp_path):
    record = build_run(tmp_path).record
    values = record.model_dump(mode="json")
    with pytest.raises(ValidationError, match="official runs only"):
        FinalTestRunRecord.model_validate(values | {"root_kind": "final_test"})
    with pytest.raises(ValidationError, match="batch_id does not match"):
        FinalTestRunRecord.model_validate(values | {"root_id": sha("other")})
    with pytest.raises(ValidationError, match="retry_policy_sha256"):
        FinalTestRunRecord.model_validate(values | {"retry_policy_sha256": sha("other")})
    with pytest.raises(ValidationError):
        FinalTestRunRecord.model_validate(values | {"phase": "development"})
    with pytest.raises(ValidationError):
        FinalTestRunRecord.model_validate(values | {"approved": True})
    official = values | {
        "root_kind": "final_test",
        "backend_kind": "official",
        "purpose": "development_candidate",
    }
    assert FinalTestRunRecord.model_validate(official).official
    with pytest.raises(ValidationError, match="development-candidate"):
        FinalTestRunRecord.model_validate(official | {"purpose": "synthetic_test"})


def test_record_retry_policy_is_the_configuration_policy(tmp_path):
    record = build_run(tmp_path).record
    assert record.retry == retry_policy_of(CONFIGS[ARM].settings)


# --- protected content ------------------------------------------------------------------


def test_findings_and_reports_never_contain_protected_text(tmp_path):
    built = build_run(tmp_path, statuses={"i1": "interrupted"}, default="complete_unsealed")
    result = audit(built, binding=None)
    text = " ".join(f"{f}" for f in result.findings)
    stored = b"".join(p.read_bytes() for p in built.run_dir.rglob("*.json"))
    assert CANARY not in text and CANARY.encode() not in stored


# --- development runs are unaffected ---------------------------------------------------


def test_development_runs_keep_their_phase_and_are_not_final_test_runs(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(dev_coordinates())
    attempt = store.begin_attempt(
        key, environment=environment("mock"), rendered_input_sha256=RENDERED
    )
    before = snapshot(tmp_path)
    context = AuditContext(
        root_id=sha("r"), root_kind="rehearsal", plan=PLAN, item_papers={}, manifest=None,
        binding=None, rosters={},
    )  # fmt: skip
    result = audit_run(tmp_path, store.run_id, context)
    assert codes(result, "structure") == {"run_record_invalid"}
    assert snapshot(tmp_path) == before
    store.complete(key, attempt, '{"answer": "1", "abstained": false}')  # still works afterwards
    assert typing.get_args(RunRecord.model_fields["phase"].annotation) == (
        "development",
        "qualification",
    )
    with pytest.raises(ValidationError):
        RunStore.create(
            tmp_path / "x", config=CONFIGS[ARM], phase="final_test", benchmark_version="b",
            backend_kind="mock",
        )  # type: ignore[arg-type]  # fmt: skip


def test_stored_repeat_index_zero_is_reported(tmp_path):
    """Coordinates accepts repeat 0; the audit must not."""
    zero = types.SimpleNamespace(
        item_id="i1",
        paper_id="p2",
        matrix_arm=ARM,
        condition="JATS",
        repeat_index=0,
        seed=None,
        rendered_input_sha256=sha("rendered-zero"),
        observation_key=oracle_key("i1", ARM, "JATS", 0, CONFIGS[ARM].configuration_sha256),
    )
    built = build_run(tmp_path, extra_rows=(zero,))
    result = no_writes(built)
    assert {"repeat_zero", "foreign_observation_key"} <= codes(result, "structure")
