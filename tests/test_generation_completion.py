"""Completed-run manifest and the official-evidence boundary. Everything is synthetic.

Runs here use scripted mock backends. Where a test needs a run that claims to be official,
the claim is the test's own declaration; none of it is evidence about a real Qwen3-8B run.
"""

import json
import shutil

import pytest
from test_generation_roster import publish_view
from test_generation_runner import (
    Crash,
    MockBackend,
    go,
    good,
    make_roster,
    make_store,
    reopen,
)

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.evaluation.stage_a import item_digest
from biorag_repro.evaluation.stage_c_models import GenerationResult
from biorag_repro.generation import official
from biorag_repro.generation.attempts import (
    CorruptRecord,
    DuplicateCompletion,
    NonOfficialBackend,
)
from biorag_repro.generation.completion import (
    RunIncomplete,
    complete_run,
    publish_roster,
    verify_completed_run,
)
from biorag_repro.generation.official import (
    NotVerifiedOfficial,
    check_official_lineage,
    verify_official_run,
)
from biorag_repro.generation.roster import RosterRecord, RosterRow, roster_sha256
from biorag_repro.generation.runner import (
    BackendFailure,
    RosterMismatch,
    decided_results,
    nonofficial_results,
)

SHA = "a" * 64


def record_for(config, roster, *, sha_of=lambda entry: SHA, development_items=None):
    rows = [
        RosterRow(
            item_id=e.item_id,
            item_sha256=sha_of(e),
            paper_id=e.paper_id,
            condition=e.representation_or_condition,
            repeat_index=e.repeat_index,
            rendered_input_sha256=e.rendered.rendered_sha256,
            input_tokens=e.rendered.input_tokens,
            context=None
            if e.rendered.context is None
            else ArtifactRef(artifact_id="ctx", logical_ref="ctx.json", sha256=SHA),
            context_sha256=e.rendered.context_sha256,
        )
        for e in sorted(
            roster, key=lambda e: (e.item_id, e.representation_or_condition, e.repeat_index)
        )
    ]
    return RosterRecord(
        phase="development",
        benchmark_version="benchmark_v1",
        corpus_version="corpus_v1",
        development_items=development_items
        or ArtifactRef(artifact_id="items", logical_ref="items.jsonl", sha256=SHA),
        scope_sha256=SHA,
        resolved_config_sha256=config.configuration_sha256,
        matrix_arm=config.settings.matrix_arm,
        rows=rows,
    )


def finished_run(tmp_path, script=None, repeats=(1,)):
    store, config = make_store(tmp_path)
    roster = make_roster(config, repeats=repeats)
    go(store, config, roster, MockBackend(roster, script))
    return store, config, roster, record_for(config, roster)


def read_back(store, config, anchor, kind="mock"):
    return verify_completed_run(store.root, store.run_id, anchor, config=config, backend_kind=kind)


# ---- publication -------------------------------------------------------------------------


def test_complete_run_publishes_and_reads_back_exactly(tmp_path):
    store, config, roster, record = finished_run(tmp_path)
    anchor = complete_run(store, record)
    done = read_back(store, config, anchor)
    assert done.run_id == store.run_id and done.roster == record
    assert len(done.states) == len(roster) == 6
    assert {s.status for s in done.states} == {"complete"}
    completion = done.completion
    assert completion.roster_sha256 == roster_sha256(record)
    assert completion.resolved_config_sha256 == config.configuration_sha256
    assert completion.backend_kind == "mock"
    assert [o.observation_key for o in completion.observations] == sorted(
        o.observation_key for o in completion.observations
    )
    assert all(o.raw_output is not None for o in completion.observations)
    assert completion.environments  # the environment identity of every attempt is bound


def test_publication_is_idempotent_and_never_rewrites(tmp_path):
    store, _config, _, record = finished_run(tmp_path)
    first = complete_run(store, record)
    snapshot = {p: p.read_bytes() for p in store.run_dir.rglob("*") if p.is_file()}
    assert complete_run(store, record) == first
    assert snapshot == {p: p.read_bytes() for p in store.run_dir.rglob("*") if p.is_file()}


def test_terminal_failures_and_exhausted_retries_are_decided_outcomes(tmp_path):
    script = {
        ("q_a", "JATS"): [BackendFailure("bad_request", "refused")],
        ("q_b", "JATS"): [BackendFailure("out_of_memory", "oom")] * 3,
    }
    store, config, _, record = finished_run(tmp_path, script)
    done = read_back(store, config, complete_run(store, record))
    by_status = {}
    for obs in done.completion.observations:
        by_status.setdefault(obs.status, []).append(obs)
    assert {"terminal_failure", "retries_exhausted", "complete"} == set(by_status)
    assert by_status["retries_exhausted"][0].attempts == 3
    assert by_status["terminal_failure"][0].raw_output is None


def test_retried_observation_keeps_its_attempt_count(tmp_path):
    script = {("q_c", "PDF_DOCLING"): [BackendFailure("transport_error", "reset")]}
    store, config, _, record = finished_run(tmp_path, script)
    done = read_back(store, config, complete_run(store, record))
    retried = [o for o in done.completion.observations if o.attempts == 2]
    assert len(retried) == 1 and retried[0].status == "complete"


def test_completed_observations_are_not_regenerated_after_completion(tmp_path):
    store, config, roster, record = finished_run(tmp_path)
    anchor = complete_run(store, record)
    again = MockBackend(roster)
    resumed = reopen(store, config)
    go(resumed, config, roster, again)
    assert again.calls == []
    read_back(store, config, anchor)


# ---- completion requirements ----------------------------------------------------------------


def test_planned_only_run_is_incomplete_and_publishes_nothing(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    for entry in roster:
        from biorag_repro.generation.attempts import Coordinates

        store.plan(
            Coordinates(
                item_id=entry.item_id,
                paper_id=entry.paper_id,
                matrix_arm="confirmatory",
                representation_or_condition=entry.representation_or_condition,
                repeat_index=entry.repeat_index,
            )
        )
    with pytest.raises(RunIncomplete, match="planned"):
        complete_run(store, record_for(config, roster))
    assert not list(store.run_dir.glob("completion*"))


def test_interrupted_run_is_not_complete_until_it_is_recovered_and_finished(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    with pytest.raises(Crash):
        go(store, config, roster, MockBackend(roster, {("q_b", "JATS"): [Crash()]}))
    record = record_for(config, roster)
    resumed = reopen(store, config)
    with pytest.raises(RunIncomplete, match="running|planned"):
        complete_run(resumed, record)
    resumed.recover_interrupted(previous_worker_stopped=True)
    with pytest.raises(RunIncomplete):
        complete_run(resumed, record)
    go(resumed, config, roster, MockBackend(roster))
    read_back(resumed, config, complete_run(resumed, record))


def test_roster_row_that_was_never_planned_is_incomplete(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    go(store, config, roster[:-1], MockBackend(roster))
    with pytest.raises(RunIncomplete, match="never planned"):
        complete_run(store, record_for(config, roster))


def test_stored_observation_outside_the_roster_is_refused(tmp_path):
    store, config, roster, _ = finished_run(tmp_path)
    with pytest.raises(RosterMismatch, match="not in the roster"):
        complete_run(store, record_for(config, roster[:-1]))


def test_rendered_input_must_be_the_rostered_input(tmp_path):
    store, _config, _roster, record = finished_run(tmp_path)
    row = record.rows[0].model_copy(update={"rendered_input_sha256": "b" * 64})
    forged = record.model_copy(update={"rows": [row, *record.rows[1:]]})
    with pytest.raises(RosterMismatch, match="rendered input"):
        complete_run(store, forged)


def test_roster_must_belong_to_the_run_configuration(tmp_path):
    store, _config, _, record = finished_run(tmp_path)
    with pytest.raises(RosterMismatch, match="configuration"):
        complete_run(store, record.model_copy(update={"resolved_config_sha256": "c" * 64}))


def test_same_observation_completed_in_two_runs_is_refused(tmp_path):
    first, _config, _, record = finished_run(tmp_path)
    complete_run(first, record)
    second, _config2, _, record2 = finished_run(tmp_path)
    with pytest.raises(DuplicateCompletion):
        complete_run(second, record2)
    assert not list(second.run_dir.glob("completion*"))


def test_leftover_temporary_file_blocks_completion(tmp_path):
    store, _config, _, record = finished_run(tmp_path)
    (store.run_dir / "observations" / ".tmp-leftover").write_bytes(b"x")
    with pytest.raises(RunIncomplete, match="temporary"):
        complete_run(store, record)


# ---- read-back and tamper detection ---------------------------------------------------------


def final_files(store):
    key = store.observation_keys()[0]
    events = sorted((store.run_dir / "observations" / key / "events").glob("0*.json"))
    return key, events[-1], store.run_dir / "observations" / key / "final.build.json"


def sealed(tmp_path):
    store, config, _, record = finished_run(tmp_path)
    return store, config, complete_run(store, record)


def unlock(path):
    path.chmod(0o644)
    return path


def test_deleting_a_final_event_is_detected(tmp_path):
    store, config, anchor = sealed(tmp_path)
    _, event, _ = final_files(store)
    unlock(event).unlink()
    with pytest.raises(CorruptRecord):
        read_back(store, config, anchor)


def test_deleting_a_final_event_and_its_seal_is_detected_through_the_manifest(tmp_path):
    store, config, anchor = sealed(tmp_path)
    _, event, seal = final_files(store)
    unlock(event).unlink()
    unlock(seal).unlink()
    # The log now looks like an attempt that is still running; only the completed-run
    # manifest says it was finished.
    with pytest.raises(CorruptRecord):
        read_back(store, config, anchor)


def test_replacing_a_final_event_is_detected(tmp_path):
    store, config, anchor = sealed(tmp_path)
    _, event, _ = final_files(store)
    data = json.loads(event.read_text())
    data["at_utc"] = "2030-01-01T00:00:00.000000Z"
    unlock(event).write_text(json.dumps(data))
    with pytest.raises(CorruptRecord):
        read_back(store, config, anchor)


def test_altering_raw_output_is_detected(tmp_path):
    store, config, anchor = sealed(tmp_path)
    raw = next(store.run_dir.glob("observations/*/raw/*.txt"))
    unlock(raw).write_text(good("0.99"))
    with pytest.raises(CorruptRecord):
        read_back(store, config, anchor)


@pytest.mark.parametrize("name", ["roster.json", "completion.json", "completion.build.json"])
def test_editing_a_completion_record_is_detected(tmp_path, name):
    store, config, anchor = sealed(tmp_path)
    path = unlock(store.run_dir / name)
    path.write_bytes(path.read_bytes().replace(b"1", b"2", 1))
    with pytest.raises(CorruptRecord):
        read_back(store, config, anchor)


@pytest.mark.parametrize("name", ["roster.json", "completion.json", "completion.build.json"])
def test_deleting_a_completion_record_is_detected(tmp_path, name):
    store, config, anchor = sealed(tmp_path)
    unlock(store.run_dir / name).unlink()
    with pytest.raises(CorruptRecord):
        read_back(store, config, anchor)


def test_an_extra_file_in_the_run_directory_is_detected(tmp_path):
    store, config, anchor = sealed(tmp_path)
    (store.run_dir / "notes.txt").write_text("late addition")
    with pytest.raises(CorruptRecord, match="inventory"):
        read_back(store, config, anchor)


def test_a_wrong_anchor_is_refused(tmp_path):
    store, config, anchor = sealed(tmp_path)
    with pytest.raises(CorruptRecord):
        read_back(store, config, "0" * 64)
    assert anchor != "0" * 64


def test_a_forger_who_replaces_the_manifest_defeats_a_reader_without_the_anchor(tmp_path):
    """Characterization of the stated limit: integrity needs an anchor kept elsewhere.

    The attacker deletes a final event and its seal, then rebuilds the roster, completion
    record and manifest around the shortened run. A reader holding the original anchor
    rejects it; a reader that trusts whatever manifest is on disk would not.
    """
    store, config, anchor = sealed(tmp_path)
    key, _, _ = final_files(store)
    shutil.rmtree(store.run_dir / "observations" / key)
    for name in ("completion.json", "completion.build.json"):
        unlock(store.run_dir / name).unlink()
    roster = unlock(store.run_dir / "roster.json")
    full = RosterRecord.model_validate_json(roster.read_text())
    keep = [r for r in full.rows if store.key_for(_coords(r, config)) != key]
    roster.unlink()
    forged = full.model_copy(update={"rows": keep})
    forged_anchor = complete_run(reopen(store, config), forged)
    assert forged_anchor != anchor
    with pytest.raises(CorruptRecord):
        read_back(store, config, anchor)  # the retained anchor still catches it
    read_back(store, config, forged_anchor)  # without it, the forgery is self-consistent


def _coords(row, config):
    from biorag_repro.generation.attempts import Coordinates

    return Coordinates(
        item_id=row.item_id,
        paper_id=row.paper_id,
        matrix_arm=config.settings.matrix_arm,
        representation_or_condition=row.condition,
        repeat_index=row.repeat_index,
    )


def test_read_back_does_not_repair_a_deleted_seal(tmp_path):
    store, config, anchor = sealed(tmp_path)
    _, _, seal = final_files(store)
    unlock(seal).unlink()
    with pytest.raises(CorruptRecord, match="seal"):
        read_back(store, config, anchor)
    assert not seal.exists()


def test_roster_is_published_once_and_must_not_change(tmp_path):
    store, _config, _roster, record = finished_run(tmp_path)
    publish_roster(store, record)
    publish_roster(store, record)  # identical content is accepted
    other = record.model_copy(update={"scope_sha256": "d" * 64})
    with pytest.raises(CorruptRecord, match="roster"):
        publish_roster(store, other)


# ---- official boundary --------------------------------------------------------------------------


@pytest.fixture
def view(tmp_path):
    directory, record_sha = publish_view(tmp_path / "view")
    return directory, record_sha


def official_world(tmp_path, view, *, kind="official", script=None, mutate_rows=None, rename=None):
    """A run over two real view items that declares itself to be on backend `kind`."""
    directory, record_sha = view
    view_record, items = read_development_view(directory, record_sha)
    by_id = {i.item_id: i for i in items}
    store, config = make_store(tmp_path / "run", kind=kind)
    roster = [e for e in make_roster(config, conditions=("JATS",)) if e.item_id in ("q_a", "q_b")]
    mapping = {"q_a": "P001", "q_b": "P002"}
    roster = [e.model_copy(update={"paper_id": mapping[e.item_id]}) for e in roster]
    sha = {e.item_id: item_digest(by_id[e.item_id]) for e in roster}
    roster = [
        e.model_copy(update={"item_id": (rename or {}).get(e.item_id, e.item_id)}) for e in roster
    ]
    sha |= {new: sha[old] for old, new in (rename or {}).items()}
    record = record_for(
        config,
        roster,
        sha_of=lambda e: sha[e.item_id],
        development_items=view_record.output_items,
    )
    if mutate_rows:
        record = mutate_rows(record)
    backend = MockBackend(roster, script)
    backend.identity = backend.identity.model_copy(update={"kind": kind})
    go(store, config, roster, backend, kind=kind)
    return store, config, record


def verify(store, config, view, anchor):
    return verify_official_run(
        store.root,
        store.run_id,
        anchor,
        config=config,
        view_directory=view[0],
        view_record_sha256=view[1],
    )


@pytest.fixture
def recognised(monkeypatch):
    monkeypatch.setattr(official, "OFFICIAL_BACKENDS", frozenset({"synthetic"}))


@pytest.mark.parametrize("kind", ["mock", "smoke"])
def test_mock_and_smoke_runs_never_qualify(tmp_path, view, recognised, kind):
    store, config, record = official_world(tmp_path, view, kind=kind)
    anchor = complete_run(store, record)
    with pytest.raises(NonOfficialBackend, match=f"{kind} run"):
        verify(store, config, view, anchor)


def test_no_backend_is_recognised_until_the_real_one_exists(tmp_path, view):
    store, config, record = official_world(tmp_path, view)
    anchor = complete_run(store, record)
    assert official.OFFICIAL_BACKENDS == frozenset()
    with pytest.raises(NonOfficialBackend, match="recognised"):
        verify(store, config, view, anchor)


def test_verified_official_run_yields_exactly_the_decided_results(tmp_path, view, recognised):
    store, config, record = official_world(tmp_path, view)
    anchor = complete_run(store, record)
    verified = verify(store, config, view, anchor)
    assert verified.run_id == store.run_id and verified.manifest_sha256 == anchor
    assert verified.results == decided_results(store)
    check_official_lineage(verified.results, verified)


def test_failed_attempts_stay_operational_missingness(tmp_path, view, recognised):
    script = {("q_a", "JATS"): [BackendFailure("bad_request", "refused")]}
    store, config, record = official_world(tmp_path, view, script=script)
    verified = verify(store, config, view, complete_run(store, record))
    failed = [r for r in verified.results if r.status == "operational_failure"]
    assert len(failed) == 1 and failed[0].item_id == "q_a"


def test_run_must_be_complete_before_it_can_be_official(tmp_path, view, recognised):
    store, config, _record = official_world(tmp_path, view)
    with pytest.raises(CorruptRecord):
        verify(store, config, view, "0" * 64)


def test_roster_must_cite_the_development_view(tmp_path, view, recognised):
    def other_items(record):
        return record.model_copy(
            update={
                "development_items": ArtifactRef(
                    artifact_id="other", logical_ref="other.jsonl", sha256="e" * 64
                )
            }
        )

    store, config, record = official_world(tmp_path, view, mutate_rows=other_items)
    anchor = complete_run(store, record)
    with pytest.raises(NonOfficialBackend, match="development view"):
        verify(store, config, view, anchor)


def test_rostered_item_must_match_the_view_record(tmp_path, view, recognised):
    def wrong_hash(record):
        row = record.rows[0].model_copy(update={"item_sha256": "f" * 64})
        return record.model_copy(update={"rows": [row, *record.rows[1:]]})

    store, config, record = official_world(tmp_path, view, mutate_rows=wrong_hash)
    anchor = complete_run(store, record)
    with pytest.raises(NonOfficialBackend, match="view"):
        verify(store, config, view, anchor)


def test_rostered_item_outside_the_view_is_refused(tmp_path, view, recognised):
    store, config, record = official_world(tmp_path, view, rename={"q_a": "q_zzz"})
    anchor = complete_run(store, record)
    with pytest.raises(NonOfficialBackend, match="view"):
        verify(store, config, view, anchor)


def test_unapproved_tokenizer_pin_is_refused(tmp_path, view, recognised):
    tokenizer = {
        "repository": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "file_sha256": {"tokenizer.json": "1" * 64},
    }
    directory, record_sha = view
    store, config = make_store(tmp_path / "run", kind="official", tokenizer=tokenizer)
    roster = [e for e in make_roster(config, conditions=("JATS",)) if e.item_id == "q_a"]
    roster = [e.model_copy(update={"paper_id": "P001"}) for e in roster]
    view_record, items = read_development_view(directory, record_sha)
    digest = item_digest(next(i for i in items if i.item_id == "q_a"))
    record = record_for(
        config, roster, sha_of=lambda e: digest, development_items=view_record.output_items
    )
    backend = MockBackend(roster)
    backend.identity = backend.identity.model_copy(update={"kind": "official"})
    go(store, config, roster, backend, kind="official")
    anchor = complete_run(store, record)
    with pytest.raises(NonOfficialBackend, match="tokenizer"):
        verify(store, config, view, anchor)


def test_standalone_results_are_not_official_evidence(tmp_path, view, recognised):
    store, config, record = official_world(tmp_path, view)
    verified = verify(store, config, view, complete_run(store, record))
    genuine = verified.results[0]
    forged = GenerationResult.model_validate(
        genuine.model_dump() | {"answer": "0.99", "abstained": False}
    )
    for impostor in (forged, genuine.model_copy(update={"run_id": "f" * 32})):
        with pytest.raises(NotVerifiedOfficial):
            check_official_lineage([impostor], verified)


def test_mock_run_results_do_not_pass_an_official_lineage_check(tmp_path, view, recognised):
    official_store, config, record = official_world(tmp_path / "o", view)
    verified = verify(official_store, config, view, complete_run(official_store, record))
    mock_store, _, _, _ = finished_run(tmp_path / "m")
    for result in nonofficial_results(mock_store).results:
        with pytest.raises(NotVerifiedOfficial):
            check_official_lineage([result], verified)


def test_stage_c_accepts_verified_results_unchanged(tmp_path, view, recognised):
    from biorag_repro.evaluation import stage_c

    store, config, record = official_world(tmp_path, view)
    verified = verify(store, config, view, complete_run(store, record))
    items = {i.item_id: i for i in read_development_view(*view)[1]}
    for result in verified.results:
        assert stage_c.score(items[result.item_id], result).output == result.output


def test_a_manifest_citing_other_final_events_than_the_completion_is_refused(tmp_path):
    from biorag_repro.core.hashing import sha256_bytes
    from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
    from biorag_repro.core.manifest import BuildManifest, DerivedOutput
    from biorag_repro.core.storage import serialize_manifest

    store, config, _, record = finished_run(tmp_path)
    anchor = complete_run(store, record)
    path = store.run_dir / "completion.build.json"
    manifest = BuildManifest.model_validate_json(path.read_bytes())
    inputs = [i for i in manifest.inputs if i is not manifest.inputs[-1]]  # forget one final event
    out = manifest.outputs[0]
    lineage = [ArtifactInput(i.role, i.artifact.artifact_id, i.artifact.sha256) for i in inputs]
    new_id = derived_artifact_id(
        artifact_kind=out.artifact_kind,
        inputs=lineage,
        transformation_id=manifest.transformation_id,
        transformation_settings_sha256=manifest.transformation_settings_sha256,
        output_sha256=out.artifact.sha256,
        output_format=out.output_format,
        serialization_version=out.serialization_version,
    )
    forged = manifest.model_copy(
        update={
            "inputs": inputs,
            "outputs": [
                DerivedOutput(
                    artifact=out.artifact.model_copy(update={"artifact_id": new_id}),
                    artifact_kind=out.artifact_kind,
                    output_format=out.output_format,
                    serialization_version=out.serialization_version,
                    identity_version=out.identity_version,
                )
            ],
        }
    )
    content = serialize_manifest(BuildManifest.model_validate(forged.model_dump()))
    unlock(path).write_bytes(content)
    with pytest.raises(CorruptRecord, match="does not describe"):
        read_back(store, config, sha256_bytes(content))
    assert anchor != sha256_bytes(content)
