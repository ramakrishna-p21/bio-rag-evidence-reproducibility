"""The official-evidence boundary: what a run must prove before its results count.

A recognised backend name proves nothing here. Eligibility needs the approved checkpoint
inventory, the recorded exact inputs, an execution receipt for every completed observation,
and a completed-run anchor. Backends are scripted except in the one test that drives the
real Hugging Face backend over a stub model. Nothing is a scientific result.
"""

import json
from types import SimpleNamespace

import pytest
from synthetic_environment import approved_locks, run_probes
from synthetic_environment import evidence as synthetic_evidence
from test_generation_attempts import Clock
from test_generation_environment import cpu_probes
from test_generation_hf_backend import REV, fake_loader, make_renderer
from test_generation_prompt import ROOT, needs_tokenizer
from test_generation_receipts import facts_for
from test_generation_roster import publish_view

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.evaluation import stage_c
from biorag_repro.evaluation.stage_a import item_digest
from biorag_repro.evaluation.stage_c_models import GenerationResult
from biorag_repro.generation import config as generator_config
from biorag_repro.generation import hf_backend, official
from biorag_repro.generation.attempts import (
    Coordinates,
    CorruptRecord,
    NonOfficialBackend,
    RunStore,
)
from biorag_repro.generation.completion import complete_run, verify_completed_run
from biorag_repro.generation.environment import (
    BackendIdentity,
    ExecutionRecord,
    GpuDevice,
    GpuReading,
    capture_environment,
)
from biorag_repro.generation.environment_gate import publish_environment_evidence
from biorag_repro.generation.hf_backend import HF_BACKEND_NAME, HuggingFaceBackend
from biorag_repro.generation.inputs import build_input_record, input_record_ref
from biorag_repro.generation.model_snapshot import inventory_snapshot
from biorag_repro.generation.official import (
    NoExecutionReceipt,
    NotVerifiedOfficial,
    check_official_lineage,
    recorded_generator_input,
    simulated_generator_input,
    verify_official_run,
)
from biorag_repro.generation.roster import RosterRecord, RosterRow
from biorag_repro.generation.runner import (
    BackendFailure,
    GenerationOutput,
    RosterEntry,
    RosterMismatch,
    decided_results,
    nonofficial_results,
    run_roster,
)

pytestmark = needs_tokenizer

SCRIPT = "scripted-official"
SHA = "a" * 64
GOOD = json.dumps({"answer": "0.05", "abstained": False})
# One JATS-like and one PDF-derived (table-like) context per item, for two independent papers.
CONTEXTS = {
    ("q_a", "JATS"): "The reported value was 0.05 in the treated group.",
    ("q_a", "PDF_DOCLING"): "| Group | Value |\n| --- | --- |\n| treated | 0.05 |",
    ("q_b", "JATS"): "The protein TP53 was named as the regulator.",
    ("q_b", "PDF_DOCLING"): "Regulator\nTP53\n(named in the results paragraph)",
}


def context_ref(item_id, condition):
    text = CONTEXTS[(item_id, condition)]
    return ArtifactRef(
        artifact_id=f"ctx-{item_id}-{condition}",
        logical_ref=f"contexts/{item_id}-{condition}.json",
        sha256=sha256_bytes(text.encode()),
    )


@pytest.fixture
def view(tmp_path):
    return publish_view(tmp_path / "view")


@pytest.fixture
def recognised(monkeypatch, tiny):
    manifest = inventory_snapshot(tiny, REV).manifest_sha256
    monkeypatch.setattr(generator_config, "APPROVED_ENVIRONMENT_LOCKS", approved_locks())
    monkeypatch.setattr(official, "OFFICIAL_BACKENDS", frozenset({HF_BACKEND_NAME, SCRIPT}))
    monkeypatch.setattr(
        generator_config, "APPROVED_CHECKPOINTS", {("Qwen/Qwen3-8B", REV): manifest}
    )


def gpu_probes(status="ok"):
    devices = (
        [GpuDevice(index=0, name="Synthetic GPU", memory_total_mib=24576, driver_version="0")]
        if status == "ok"
        else []
    )
    return run_probes(cpu_probes(gpus=lambda: GpuReading(status=status, devices=devices)))


class Scripted:
    """Stands in for a backend: it reports whatever its test says it did."""

    def __init__(self, renderer, tiny_dir, kind, *, name=SCRIPT, execution=None, gpu="ok",
                 fail=(), receipts=True, with_execution=True, output_ids=None, eos_ids=(151645,), measure=None):  # fmt: skip
        self.identity = BackendIdentity(kind=kind, name=name, version=None)
        settings = renderer.config.settings
        self.config, self.gpu, self.fail, self.receipts = renderer.config, gpu, set(fail), receipts
        self.eos_ids = list(eos_ids)
        self.measure = measure  # a callable giving the measurements of each call, or None
        # token ids a real backend would report for GOOD: its decoding, then the stop token
        self.output_ids = output_ids or [
            *renderer.tokenizer(GOOD, add_special_tokens=False)["input_ids"],
            151645,
        ]
        fields = {
            "snapshot": inventory_snapshot(tiny_dir, REV),
            "tokenizer_file_sha256": dict(settings.tokenizer.file_sha256),
            "device": "cuda:0",
            "device_name": "Synthetic GPU",
            "dtype": "bfloat16",
            "max_position_embeddings": 8192,
        }
        self.execution = ExecutionRecord(**(fields | (execution or {}))) if with_execution else None

    def environment(self):
        settings = self.config.settings
        return capture_environment(
            backend=self.identity,
            model=(settings.model.repository, settings.model.revision),
            tokenizer=(settings.tokenizer.repository, settings.tokenizer.revision),
            packages=("torch",),
            probes=gpu_probes(self.gpu),
            execution=self.execution,
        )

    def generate(self, rendered, *, seed, max_new_tokens):
        if rendered.context in self.fail:
            raise BackendFailure("bad_request", "refused")
        if not self.receipts:
            return GOOD
        facts = facts_for(
            SimpleNamespace(rendered=rendered),
            self.config,
            output=self.output_ids,
            generation={"eos_token_ids": self.eos_ids},
        )
        if self.measure is not None:
            facts = facts.model_copy(update={"measurements": self.measure()})
        return GenerationOutput(GOOD, facts)


def named_device(record):
    """The CPU stand-in has no device name; give it the synthetic GPU the lock names."""
    execution = record.execution.model_copy(update={"device_name": "Synthetic GPU"})
    return record.model_copy(update={"execution": execution})


def build_world(tmp_path, view, tiny_dir, kind="official", *, row_edit=None, entry_edit=None,
                no_execution=False, run=True, evidence=True, **backend_options):  # fmt: skip
    directory, record_sha = view
    view_record, items = read_development_view(directory, record_sha)
    by_id = {i.item_id: i for i in items}
    purpose = "development_candidate" if kind == "official" else "synthetic_test"
    renderer = make_renderer(purpose=purpose, repository="Qwen/Qwen3-8B", new_tokens=64)
    config = renderer.config
    store = RunStore.create(
        tmp_path / "run",
        config=config,
        phase="development",
        benchmark_version=view_record.benchmark_version,
        backend_kind=kind,
        now=Clock(),
    )
    if evidence and kind == "official":
        publish_environment_evidence(store.run_dir, synthetic_evidence())
    entries, rows = [], []
    for (item_id, condition), text in sorted(CONTEXTS.items()):
        item = by_id[item_id]
        rendered = renderer.render(item.question, text, renderer.counter(text))
        coordinates = Coordinates(
            item_id=item_id,
            paper_id=item.paper_id,
            matrix_arm="confirmatory",
            representation_or_condition=condition,
            repeat_index=1,
        )
        ref = context_ref(item_id, condition)
        stored = build_input_record(
            renderer, rendered, observation_key=store.key_for(coordinates), context=ref
        )
        if entry_edit:
            rendered, stored = entry_edit(item_id, rendered, stored)
        entries.append(
            RosterEntry(
                item_id=item_id,
                paper_id=item.paper_id,
                representation_or_condition=condition,
                repeat_index=1,
                rendered=rendered,
                input_record=stored,
            )
        )
        rows.append(
            RosterRow(
                item_id=item_id,
                item_sha256=item_digest(item),
                paper_id=item.paper_id,
                condition=condition,
                repeat_index=1,
                rendered_input_sha256=rendered.rendered_sha256,
                input_tokens=rendered.input_tokens,
                context=ref,
                context_sha256=rendered.context_sha256,
                input_record=input_record_ref(stored),
            )
        )
    if row_edit:
        rows = row_edit(rows)
    roster = RosterRecord(
        phase="development",
        benchmark_version=view_record.benchmark_version,
        corpus_version=view_record.corpus_version,
        development_items=view_record.output_items,
        scope_sha256=SHA,
        resolved_config_sha256=config.configuration_sha256,
        matrix_arm="confirmatory",
        rows=rows,
    )
    backend = Scripted(renderer, tiny_dir, kind, with_execution=not no_execution, **backend_options)
    if run:
        run_roster(store, entries, backend, capture_environment=backend.environment)
    return SimpleNamespace(
        store=store, config=config, roster=roster, entries=entries, backend=backend, view=view,
        renderer=renderer,
    )  # fmt: skip


def verify(world, anchor, **changes):
    arguments = {
        "config": world.config,
        "view_directory": world.view[0],
        "view_record_sha256": world.view[1],
        "tokenizer_root": ROOT,
    }
    return verify_official_run(
        world.store.root, world.store.run_id, anchor, **(arguments | changes)
    )


def anchored(world):
    return complete_run(world.store, world.roster)


# ---- the successful path ------------------------------------------------------------------------


def test_verified_official_run_yields_exactly_the_decided_results(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny)
    anchor = anchored(world)
    verified = verify(world, anchor)
    assert verified.run_id == world.store.run_id and verified.manifest_sha256 == anchor
    assert verified.results == decided_results(world.store)
    assert {r.status for r in verified.results} == {"completed"}
    check_official_lineage(verified.results, verified)


def test_failed_attempts_stay_operational_missingness(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny, fail=[CONTEXTS[("q_a", "PDF_DOCLING")]])
    verified = verify(world, anchored(world))
    failed = [r for r in verified.results if r.status == "operational_failure"]
    assert [(r.item_id, r.representation_or_condition) for r in failed] == [("q_a", "PDF_DOCLING")]
    assert len(verified.results) == 4


def test_stage_c_accepts_verified_results_unchanged(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny)
    verified = verify(world, anchored(world))
    items = {i.item_id: i for i in read_development_view(*view)[1]}
    for result in verified.results:
        assert stage_c.score(items[result.item_id], result).output == result.output


def test_the_real_backend_over_a_tiny_model_can_be_official_only_when_approved(
    tmp_path,
    view,
    tiny,
    monkeypatch,
):
    """Every approval is patched in; the weights are random and the text is noise."""
    manifest = inventory_snapshot(tiny, REV).manifest_sha256
    monkeypatch.setattr(
        generator_config, "APPROVED_CHECKPOINTS", {("Qwen/Qwen3-8B", REV): manifest}
    )
    monkeypatch.setattr(generator_config, "APPROVED_ENVIRONMENT_LOCKS", approved_locks())
    monkeypatch.setattr(hf_backend, "is_official_device", lambda device: True)
    base = build_world(tmp_path / "w", view, tiny, run=False)
    backend = HuggingFaceBackend(
        base.config,
        model_dir=tiny,
        tokenizer_root=ROOT,
        device="cpu",
        load_model=fake_loader(tail=[11, 12, 151645]),
    )
    assert (
        backend.identity.kind == "official" and backend.identity.name in official.OFFICIAL_BACKENDS
    )
    run_roster(
        base.store,
        base.entries,
        backend,
        capture_environment=lambda: named_device(backend.environment(gpu_probes())),
    )
    verified = verify(base, anchored(base))
    assert len(verified.results) == 4


# ---- what keeps a run from qualifying ---------------------------------------------------------


def test_nothing_is_approved_and_only_the_hugging_face_backend_is_recognised_by_default():
    assert generator_config.APPROVED_CHECKPOINTS == {}
    assert official.OFFICIAL_BACKENDS == frozenset({HF_BACKEND_NAME})


@pytest.mark.parametrize("kind", ["mock", "smoke"])
def test_mock_and_smoke_runs_never_qualify(tmp_path, view, tiny, recognised, kind):
    world = build_world(tmp_path, view, tiny, kind=kind)
    with pytest.raises(NonOfficialBackend, match=f"{kind} run"):
        verify(world, anchored(world))


def test_a_recognised_backend_name_alone_is_not_enough(tmp_path, view, tiny, monkeypatch):
    monkeypatch.setattr(official, "OFFICIAL_BACKENDS", frozenset({SCRIPT}))
    world = build_world(tmp_path, view, tiny)
    with pytest.raises(NonOfficialBackend, match="is not approved"):
        verify(world, anchored(world))


def test_an_unrecognised_backend_name_is_refused(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny, name="some-other-backend")
    with pytest.raises(NonOfficialBackend, match="recognised"):
        verify(world, anchored(world))


def test_a_checkpoint_manifest_other_than_the_approved_one_is_refused(
    tmp_path, view, tiny, recognised, monkeypatch
):
    world = build_world(tmp_path, view, tiny)
    anchor = anchored(world)
    monkeypatch.setattr(
        generator_config, "APPROVED_CHECKPOINTS", {("Qwen/Qwen3-8B", REV): "0" * 64}
    )
    with pytest.raises(NonOfficialBackend, match="manifest"):
        verify(world, anchor)


def test_a_substituted_weight_inventory_is_refused(
    tmp_path, view, tiny, recognised, tmp_path_factory
):
    other = tmp_path_factory.mktemp("altered") / REV
    other.mkdir()
    for source in tiny.iterdir():
        (other / source.name).write_bytes(source.read_bytes())
    (other / "model.safetensors").write_bytes((tiny / "model.safetensors").read_bytes() + b"\0")
    world = build_world(tmp_path, view, other)  # the run records the altered inventory
    with pytest.raises(NonOfficialBackend, match="manifest"):
        verify(world, anchored(world))


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"execution": {"device": "cpu"}}, "device"),
        ({"execution": {"device": "cuda"}}, "device"),
        ({"execution": {"dtype": "float32"}}, "dtype"),
        ({"execution": {"max_position_embeddings": 4096}}, "context window"),
        ({"execution": {"tokenizer_file_sha256": {"tokenizer.json": "1" * 64}}}, "tokenizer"),
        ({"gpu": "no_nvidia_smi"}, "GPU"),
        ({"gpu": "failed"}, "GPU"),
    ],
)
def test_the_execution_environment_must_be_the_official_condition(
    tmp_path, view, tiny, recognised, options, match
):
    world = build_world(tmp_path, view, tiny, **options)
    with pytest.raises(NonOfficialBackend, match=match):
        verify(world, anchored(world))


def test_a_backend_that_records_no_loaded_model_is_refused(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny, no_execution=True)
    with pytest.raises(NonOfficialBackend, match="loaded model"):
        verify(world, anchored(world))


def unproven_world(tmp_path, view, tiny):
    """Completed through the store without any receipt: the runner itself would refuse this."""
    world = build_world(tmp_path, view, tiny, run=False)
    for entry in world.entries:
        store = world.store
        store.publish_input(entry.input_record)
        key = store.plan(
            Coordinates(
                item_id=entry.item_id,
                paper_id=entry.paper_id,
                matrix_arm="confirmatory",
                representation_or_condition=entry.representation_or_condition,
                repeat_index=1,
            )
        )
        attempt = store.begin_attempt(
            key,
            environment=world.backend.environment(),
            rendered_input_sha256=entry.rendered.rendered_sha256,
        )
        store.complete(key, attempt, GOOD)
    return world


def test_a_completed_observation_without_a_receipt_is_refused(tmp_path, view, tiny, recognised):
    world = unproven_world(tmp_path, view, tiny)
    with pytest.raises(NonOfficialBackend, match="receipt"):
        verify(world, anchored(world))


def test_a_roster_row_without_its_stored_input_is_refused(tmp_path, view, tiny, recognised):
    def drop(rows):
        return [rows[0].model_copy(update={"input_record": None}), *rows[1:]]

    world = build_world(tmp_path, view, tiny, row_edit=drop)
    with pytest.raises(NonOfficialBackend, match="stored input"):
        verify(world, anchored(world))


def test_a_roster_row_citing_another_input_than_the_receipt_is_refused(
    tmp_path, view, tiny, recognised
):
    def swap(rows):
        other = rows[1].input_record
        return [rows[0].model_copy(update={"input_record": other}), *rows[1:]]

    world = build_world(tmp_path, view, tiny, row_edit=swap)
    with pytest.raises(NonOfficialBackend):
        verify(world, anchored(world))


def test_a_tampered_stored_input_is_found_through_the_anchor(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny)
    anchor = anchored(world)
    path = world.store.run_dir / world.roster.rows[0].input_record.logical_ref
    path.chmod(0o644)
    path.write_bytes(path.read_bytes().replace(b"reported", b"reportex", 1))
    with pytest.raises(CorruptRecord):
        verify(world, anchor)


def test_token_ids_edited_together_with_their_hashes_fail_the_tokenizer_check(
    tmp_path, view, tiny, recognised
):
    """A well-formed forged input passes every hash; only re-tokenizing the text exposes it."""
    from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes

    def forge(item_id, rendered, stored):
        ids = list(stored.input_token_ids)
        ids[2] += 1
        digest = sha256_bytes(canonical_json_bytes(ids))
        rendered = rendered.model_copy(update={"input_token_ids_sha256": digest})
        data = stored.model_dump(mode="json") | {"input_token_ids": ids}
        data["rendered"] = rendered.model_dump(mode="json")
        return rendered, type(stored).model_validate(data)

    world = build_world(tmp_path, view, tiny, entry_edit=forge)
    with pytest.raises(NonOfficialBackend, match="tokeniz"):
        verify(world, anchored(world))


def test_the_tokenizer_pin_must_be_the_approved_one(tmp_path, view, tiny, recognised, monkeypatch):
    world = build_world(tmp_path, view, tiny)
    anchor = anchored(world)
    monkeypatch.setattr(official.phase5, "ASSETS", {"tokenizer.json": "1" * 64})
    with pytest.raises(NonOfficialBackend, match="tokenizer"):
        verify(world, anchor)


def test_run_must_be_complete_before_it_can_be_official(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny)
    with pytest.raises(CorruptRecord):
        verify(world, "0" * 64)


def test_roster_must_cite_the_development_view(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny)
    other = ArtifactRef(artifact_id="other", logical_ref="other.jsonl", sha256="e" * 64)
    world.roster = world.roster.model_copy(update={"development_items": other})
    with pytest.raises(NonOfficialBackend, match="development view"):
        verify(world, anchored(world))


def test_rostered_item_must_match_the_view_record(tmp_path, view, tiny, recognised):
    def wrong(rows):
        return [rows[0].model_copy(update={"item_sha256": "f" * 64}), *rows[1:]]

    world = build_world(tmp_path, view, tiny, row_edit=wrong)
    with pytest.raises(NonOfficialBackend, match="view"):
        verify(world, anchored(world))


# ---- results and lineage --------------------------------------------------------------------------


def test_standalone_results_are_not_official_evidence(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny)
    verified = verify(world, anchored(world))
    genuine = verified.results[0]
    forged = GenerationResult.model_validate(
        genuine.model_dump() | {"answer": "0.99", "abstained": False}
    )
    for impostor in (forged, genuine.model_copy(update={"run_id": "f" * 32})):
        with pytest.raises(NotVerifiedOfficial):
            check_official_lineage([impostor], verified)


def test_mock_run_results_do_not_pass_an_official_lineage_check(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path / "o", view, tiny)
    verified = verify(world, anchored(world))
    mock = build_world(tmp_path / "m", view, tiny, kind="mock")
    for result in nonofficial_results(mock.store).results:
        with pytest.raises(NotVerifiedOfficial):
            check_official_lineage([result], verified)


# ---- the generator-input link -----------------------------------------------------------------------


def test_an_official_run_yields_a_recorded_generator_input(tmp_path, view, tiny, recognised):
    from biorag_repro.core.hashing import sha256_bytes

    world = build_world(tmp_path, view, tiny)
    verified = verify(world, anchored(world))
    for result in verified.results:
        link = recorded_generator_input(verified, result)
        row = next(
            r
            for r in world.roster.rows
            if (r.item_id, r.condition) == (result.item_id, result.representation_or_condition)
        )
        assert link.provenance == "recorded"
        assert (link.observation_key, link.attempt_id) == (
            result.observation_key,
            result.attempt_id,
        )
        assert link.context == row.context
        text = CONTEXTS[(result.item_id, result.representation_or_condition)]
        assert link.exact_context_sha256 == sha256_bytes(text.encode())
        assert link.input.sha256 == row.input_record.sha256
        assert link.input.logical_ref == f"runs/{world.store.run_id}/{row.input_record.logical_ref}"


def test_a_mock_or_smoke_run_only_yields_a_simulated_link(tmp_path, view, tiny, recognised):
    for kind in ("mock", "smoke"):
        world = build_world(tmp_path / kind, view, tiny, kind=kind)
        done = verify_completed_run(
            world.store.root, world.store.run_id, anchored(world), config=world.config,
            backend_kind=kind,
        )  # fmt: skip
        result = nonofficial_results(world.store).results[0]
        assert simulated_generator_input(done, result).provenance == "simulated"


def test_an_official_run_cannot_be_given_a_simulated_link(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny)
    verified = verify(world, anchored(world))
    with pytest.raises(NonOfficialBackend, match="official"):
        simulated_generator_input(verified.completed, verified.results[0])


def test_a_link_needs_an_executed_result_of_the_verified_run(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny, fail=[CONTEXTS[("q_a", "JATS")]])
    verified = verify(world, anchored(world))
    failed = next(r for r in verified.results if r.status == "operational_failure")
    with pytest.raises(ValueError, match="executed"):
        recorded_generator_input(verified, failed)
    genuine = next(r for r in verified.results if r.status == "completed")
    other_attempt = genuine.model_copy(update={"attempt_id": "f" * 32})
    with pytest.raises(NotVerifiedOfficial):
        recorded_generator_input(verified, other_attempt)


def test_without_a_receipt_a_link_cannot_be_made(tmp_path, view, tiny):
    world = build_world(tmp_path, view, tiny, kind="smoke", receipts=False)
    done = verify_completed_run(
        world.store.root, world.store.run_id, anchored(world), config=world.config,
        backend_kind="smoke",
    )  # fmt: skip
    with pytest.raises(NoExecutionReceipt):
        simulated_generator_input(done, nonofficial_results(world.store).results[0])


def test_a_roster_row_citing_an_input_that_was_never_stored_cannot_complete_the_run(
    tmp_path, view, tiny, recognised
):
    def phantom(rows):
        ref = rows[0].input_record.model_copy(
            update={"sha256": "9" * 64, "logical_ref": f"inputs/{'9' * 64}.json"}
        )
        return [rows[0].model_copy(update={"input_record": ref}), *rows[1:]]

    world = build_world(tmp_path, view, tiny, row_edit=phantom)
    with pytest.raises(RosterMismatch, match="stored input"):
        anchored(world)


def second_record(world, row, **changes):
    """A valid, stored input for the same observation and text that is not the one used."""
    entry = next(
        e
        for e in world.entries
        if (e.item_id, e.representation_or_condition) == (row.item_id, row.condition)
    )
    other = entry.input_record.model_copy(
        update={
            "context": ArtifactRef(
                artifact_id="ctx-other", logical_ref="contexts/other.json", sha256="b" * 64
            )
        }
    )
    return world.store.publish_input(other), other


def test_a_row_citing_another_stored_input_than_the_receipt_names_is_refused(
    tmp_path, view, tiny, recognised
):
    world = build_world(tmp_path, view, tiny)
    row = world.roster.rows[0]
    ref, other = second_record(world, row)
    # Make the roster row internally consistent with the second input: only the receipt differs.
    rendered = other.rendered
    edited = row.model_copy(update={"input_record": ref, "context": other.context})
    world.roster = world.roster.model_copy(update={"rows": [edited, *world.roster.rows[1:]]})
    assert rendered.rendered_sha256 == row.rendered_input_sha256
    with pytest.raises(NonOfficialBackend, match="receipt names another"):
        verify(world, anchored(world))


def test_a_row_citing_a_reference_that_does_not_describe_the_stored_input_is_refused(
    tmp_path, view, tiny, recognised
):
    world = build_world(tmp_path, view, tiny)
    row = world.roster.rows[0]
    renamed = row.input_record.model_copy(update={"artifact_id": "someone-elses-id"})
    edited = row.model_copy(update={"input_record": renamed})
    world.roster = world.roster.model_copy(update={"rows": [edited, *world.roster.rows[1:]]})
    with pytest.raises(NonOfficialBackend, match="differs from the one the roster cites"):
        verify(world, anchored(world))


def test_a_recorded_link_is_refused_for_a_forged_result(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny)
    verified = verify(world, anchored(world))
    genuine = next(r for r in verified.results if r.status == "completed")
    forged = GenerationResult.model_validate(genuine.model_dump() | {"answer": "0.99"})
    with pytest.raises(NotVerifiedOfficial):
        recorded_generator_input(verified, forged)


def test_raw_text_that_the_reported_token_ids_do_not_decode_to_is_refused(
    tmp_path, view, tiny, recognised
):
    world = build_world(tmp_path, view, tiny, output_ids=[11, 12, 151645])
    with pytest.raises(NonOfficialBackend, match="output token ids"):
        verify(world, anchored(world))


def test_stop_tokens_that_omit_the_chat_end_token_are_refused(tmp_path, view, tiny, recognised):
    ids = [11, 12, 151643]  # ends on <|endoftext|>, which this stop list names
    world = build_world(tmp_path, view, tiny, output_ids=ids, eos_ids=(151643,))
    with pytest.raises(NonOfficialBackend, match="stop tokens"):
        verify(world, anchored(world))
