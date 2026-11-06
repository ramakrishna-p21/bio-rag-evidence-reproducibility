"""Authoritative roster construction. Items, papers, questions and prompts are synthetic.

The Phase 5 artifacts come from the retrieval test fixtures and are read back through the
production readers, with the pinned Qwen tokenizer counting the evidence.
"""

import json
from dataclasses import replace

import pytest
from test_bm25_publication import inputs
from test_context_publication import prepared
from test_generation_config import resolve, synthetic_settings
from test_generation_prompt import ROOT, artifact, needs_tokenizer
from test_stage_c import SET_GOLD, entity, make_item, quantity

from biorag_repro.benchmark.development_view import publish_development_view, read_development_view
from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.benchmark.split import PaperMeta, PaperSplit, make_split
from biorag_repro.benchmark.storage import encode_jsonl
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.evaluation.analysis_models import PlannedRoster, RosterArm, RosterItem
from biorag_repro.evaluation.analysis_table import roster_items_from_development_view
from biorag_repro.evaluation.stage_a import item_digest
from biorag_repro.generation.prompt import InputRenderer, output_schema_sha256, prompt_sha256
from biorag_repro.generation.roster import ContextSource, RosterRecord, build_roster, roster_sha256
from biorag_repro.generation.runner import RosterMismatch
from biorag_repro.retrieval import bm25, context

pytestmark = needs_tokenizer


def split_with(papers_in_development: list[str]) -> PaperSplit:
    metas = [PaperMeta(paper_id=f"P{n:03d}", primary_domain=f"d{n % 7}") for n in range(1, 51)]
    for seed in range(20000):
        split = make_split(metas, [], seed=seed)
        if set(papers_in_development) <= set(split.development_ids):
            return split
    raise AssertionError("no seed places the papers in development")


SPLIT = split_with(["P001", "P002"])
OTHER_DEV = next(p for p in SPLIT.development_ids if p not in ("P001", "P002"))
FINAL = SPLIT.final_test_ids[0]

QUESTIONS = {"q_a": "alpha", "q_b": "gamma", "q_c": "delta"}


def bench_item(
    item_id: str, paper: str, question: str, split="development", gold=None
) -> BenchmarkItem:
    base = make_item(gold or quantity("0.05"))
    text = base.model_dump_json().replace("P001", paper).replace("q_001", item_id)
    return BenchmarkItem.model_validate({**json.loads(text), "question": question, "split": split})


ITEMS = [
    bench_item("q_a", "P001", QUESTIONS["q_a"]),
    bench_item("q_b", "P002", QUESTIONS["q_b"], gold=entity("TP53", ["p53"])),
    bench_item("q_c", OTHER_DEV, QUESTIONS["q_c"], gold=SET_GOLD),
    bench_item("q_z", FINAL, "protected final test question", split="final_test"),
]


def publish_view(tmp_path, items=ITEMS):
    root = tmp_path / "data"
    (root / "benchmark/releases/v").mkdir(parents=True)
    (root / "benchmark/working").mkdir(parents=True)
    items_bytes = encode_jsonl(items)
    split_bytes = serialize_project_json(SPLIT, PaperSplit)
    (root / "benchmark/releases/v/items.jsonl").write_bytes(items_bytes)
    (root / "benchmark/working/split.json").write_bytes(split_bytes)
    published = publish_development_view(
        data_root=root,
        items_ref=ArtifactRef(
            artifact_id="items",
            logical_ref="benchmark/releases/v/items.jsonl",
            sha256=sha256_bytes(items_bytes),
        ),
        split_ref=ArtifactRef(
            artifact_id="split",
            logical_ref="benchmark/working/split.json",
            sha256=sha256_bytes(split_bytes),
        ),
        release_ref=ArtifactRef(artifact_id="release", logical_ref="release.json", sha256="c" * 64),
        benchmark_version="benchmark_v1",
        corpus_version="corpus_v1",
        expected_source_items=len(items),
        expected_development_items=sum(i.split == "development" for i in items),
    )
    return root / published.directory, published.record_sha256


def make_scope(directory, record_sha, included, *, arm="confirmatory", conditions=("JATS",),
               repeats=(1,), **changes):  # fmt: skip
    view, items = read_development_view(directory, record_sha)
    rows = []
    for row in roster_items_from_development_view(view, items):
        if row.item_id in included:
            rows.append(row)
        else:
            rows.append(RosterItem(item_id=row.item_id, state="excluded", reason="outside scope"))
    fields = {
        "scope": "partial_development",
        "phase": "development",
        "benchmark_version": view.benchmark_version,
        "corpus_version": view.corpus_version,
        "declared_by": "test",
        "basis": "synthetic",
        "arms": [RosterArm(matrix_arm=arm, conditions=list(conditions), repeats=list(repeats))],
        "items": rows,
    }
    return PlannedRoster(**{**fields, **changes})


def generator_config(retrieval_sha, arm="confirmatory", **budgets):
    prompt = artifact()
    settings = synthetic_settings()
    settings["prompt_sha256"] = prompt_sha256(prompt)
    settings["output_schema_sha256"] = output_schema_sha256(prompt)
    settings["budgets"].update(budgets)
    if arm == "closed_book":
        settings["matrix_arm"] = arm
        del settings["retrieval_config_sha256"]
    else:
        settings["matrix_arm"] = arm
        settings["retrieval_config_sha256"] = retrieval_sha
    return resolve(settings), prompt


def make_renderer(retrieval_sha, arm="confirmatory", **budgets):
    config, prompt = generator_config(retrieval_sha, arm, **budgets)
    return InputRenderer(ROOT, config, prompt)


def source_for(prepared, item_id, question, paper, *, scope="designated_paper"):
    selected, cc, root = prepared
    index = bm25.read_index(selected.index, selected.index_manifest, selected.sources, root)
    q = bm25.Query(
        query_id=item_id,
        question=question,
        scope=scope,
        designated_paper_id=paper if scope == "designated_paper" else None,
    )
    ranked = bm25.retrieve(index, selected.sources, q, selected.retrieval_configuration, root)
    rr, rm = bm25.publish_retrieval(
        ranked,
        selected.index,
        selected.index_manifest,
        selected.sources,
        q,
        selected.retrieval_configuration,
        root,
        root,
    )
    sel = context.Selection(
        rr,
        rm,
        selected.index,
        selected.index_manifest,
        selected.sources,
        q,
        selected.retrieval_configuration,
        root,
        root,
        "bm25",
    )
    ctx = context.assemble(sel, cc, root, ROOT)
    cref, cmref = context.publish(ctx, sel, cc, root, ROOT, root)
    return ContextSource(cref, cmref, sel, cc, root, root)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """The Phase 5 chain is slow to build, so it is built once and only ever read."""
    tmp = tmp_path_factory.mktemp("roster")
    patch = pytest.MonkeyPatch()
    try:
        fixture_inputs = inputs._get_wrapped_function()(tmp, patch)
        selected, cc, root = prepared._get_wrapped_function()(fixture_inputs, tmp)
        directory, record_sha = publish_view(tmp)
        sources = [
            source_for((selected, cc, root), "q_a", QUESTIONS["q_a"], "P001"),
            source_for((selected, cc, root), "q_b", QUESTIONS["q_b"], "P002"),
        ]
        yield (
            directory,
            record_sha,
            sources,
            make_renderer(cc.resolved_sha256),
            (selected, cc, root),
        )
    finally:
        patch.undo()


@pytest.fixture
def world(shared):
    return shared


def build(world, scope=None, sources=None, renderer=None):
    directory, record_sha, srcs, rend, _ = world
    scope = scope or make_scope(directory, record_sha, {"q_a", "q_b"})
    return build_roster(
        view_directory=directory,
        view_record_sha256=record_sha,
        scope=scope,
        renderer=renderer or rend,
        sources=srcs if sources is None else sources,
    )


# ---- construction --------------------------------------------------------------------


def test_entries_carry_the_exact_assembled_context_and_item_identity(world):
    built = build(world)
    assert [e.item_id for e in built.entries] == ["q_a", "q_b"]
    by_id = {e.item_id: e for e in built.entries}
    for source in world[2]:
        ctx = context.read(
            source.context,
            source.manifest,
            source.selection,
            source.configuration,
            source.configuration_root,
            ROOT,
            source.root,
        )
        entry = by_id[source.selection.query.query_id]
        assert entry.rendered.context == ctx.exact_context
        assert entry.rendered.context_tokens == ctx.generator_tokens
        assert entry.representation_or_condition == ctx.representation == "JATS"
    assert {e.paper_id for e in built.entries} == {"P001", "P002"}
    rows = {r.item_id: r for r in built.record.rows}
    items = {i.item_id: i for i in ITEMS}
    assert rows["q_a"].item_sha256 == item_digest(items["q_a"])
    assert rows["q_a"].context is not None and rows["q_b"].context is not None
    assert rows["q_a"].context != rows["q_b"].context


def test_identity_is_deterministic_and_independent_of_input_order(world):
    _directory, _record_sha, sources, _, _ = world
    first = build(world)
    again = build(world, sources=list(reversed(sources)))
    assert first.sha256 == again.sha256 == roster_sha256(first.record)
    assert [e.item_id for e in first.entries] == [e.item_id for e in again.entries]
    data = serialize_project_json(first.record, RosterRecord)
    assert RosterRecord.model_validate_json(data) == first.record


def test_repeats_share_one_rendered_input_and_get_their_own_rows(world):
    directory, record_sha, _, _, _ = world
    scope = make_scope(directory, record_sha, {"q_a", "q_b"}, repeats=(1, 2, 3))
    built = build(world, scope=scope)
    assert len(built.entries) == 6
    by_item = {}
    for entry in built.entries:
        by_item.setdefault(entry.item_id, set()).add(entry.rendered.rendered_sha256)
        assert entry.repeat_index in (1, 2, 3)
    assert all(len(hashes) == 1 for hashes in by_item.values())


def test_excluded_items_produce_nothing_and_cannot_have_a_context(world):
    directory, record_sha, sources, _, _ = world
    scope = make_scope(directory, record_sha, {"q_a"})
    with pytest.raises(RosterMismatch, match="not authorized"):
        build(world, scope=scope)  # q_b has a source but is outside the scope
    built = build(world, scope=scope, sources=sources[:1])
    assert [e.item_id for e in built.entries] == ["q_a"]


def test_identity_changes_with_scope_context_and_configuration(world):
    directory, record_sha, sources, _, prepared_ = world
    base = build(world)
    other_scope = make_scope(directory, record_sha, {"q_a", "q_b"}, repeats=(1, 2))
    assert build(world, scope=other_scope).sha256 != base.sha256
    changed = make_renderer(prepared_[1].resolved_sha256, evidence_tokens=2048, max_new_tokens=129)
    assert build(world, renderer=changed).sha256 != base.sha256
    swapped = [
        replace(sources[0], context=sources[1].context, manifest=sources[1].manifest),
        sources[1],
    ]
    with pytest.raises(ValueError):  # a context that is not the selection's output
        build(world, sources=swapped)


# ---- closed book -----------------------------------------------------------------------


def test_closed_book_rosters_have_no_context_and_refuse_sources(world):
    directory, record_sha, sources, _, _ = world
    renderer = make_renderer(None, arm="closed_book")
    scope = make_scope(
        directory, record_sha, {"q_a", "q_b"}, arm="closed_book",
        conditions=("CLOSED_BOOK",), repeats=(1, 2, 3),
    )  # fmt: skip
    built = build(world, scope=scope, sources=[], renderer=renderer)
    assert len(built.entries) == 6
    assert all(e.rendered.context is None for e in built.entries)
    assert all(r.context is None for r in built.record.rows)
    assert {e.representation_or_condition for e in built.entries} == {"CLOSED_BOOK"}
    with pytest.raises(RosterMismatch, match="not authorized"):
        build(world, scope=scope, sources=sources, renderer=renderer)


def test_closed_book_and_retrieval_identities_differ(world):
    directory, record_sha, _, _, _ = world
    renderer = make_renderer(None, arm="closed_book")
    scope = make_scope(
        directory, record_sha, {"q_a"}, arm="closed_book", conditions=("CLOSED_BOOK",)
    )
    closed = build(world, scope=scope, sources=[], renderer=renderer)
    retrieval = build(world, scope=make_scope(directory, record_sha, {"q_a"}), sources=world[2][:1])
    assert closed.sha256 != retrieval.sha256
    assert closed.record.matrix_arm == "closed_book" != retrieval.record.matrix_arm


# ---- scope authority ---------------------------------------------------------------------


def test_scope_must_declare_exactly_the_view_membership(world):
    directory, record_sha, _, _, _ = world
    good = make_scope(directory, record_sha, {"q_a", "q_b"})
    unknown = RosterItem(item_id="q_zzz", state="excluded", reason="not in view")
    with pytest.raises(RosterMismatch, match="not in the development view"):
        build(world, scope=good.model_copy(update={"items": sorted(
            [*good.items, unknown], key=lambda i: i.item_id)}))  # fmt: skip
    missing = good.model_copy(update={"items": [i for i in good.items if i.item_id != "q_c"]})
    with pytest.raises(RosterMismatch, match="does not declare"):
        build(world, scope=missing)


def test_final_test_item_cannot_be_included(world):
    directory, record_sha, _, _, _ = world
    good = make_scope(directory, record_sha, {"q_a", "q_b"})
    protected = RosterItem(item_id="q_z", state="included", paper_id=FINAL, item_sha256="d" * 64)
    scope = good.model_copy(
        update={"items": sorted([*good.items, protected], key=lambda i: i.item_id)}
    )
    with pytest.raises(RosterMismatch, match="not in the development view"):
        build(world, scope=scope)


@pytest.mark.parametrize("field", ["paper_id", "item_sha256"])
def test_included_item_must_match_the_view_record(world, field):
    directory, record_sha, _, _, _ = world
    good = make_scope(directory, record_sha, {"q_a", "q_b"})
    rows = [
        i.model_copy(update={field: "P999" if field == "paper_id" else "e" * 64})
        if i.item_id == "q_a"
        else i
        for i in good.items
    ]
    with pytest.raises(RosterMismatch, match="differs from the view"):
        build(world, scope=good.model_copy(update={"items": rows}))


@pytest.mark.parametrize(
    "change",
    [{"phase": "qualification"}, {"benchmark_version": "benchmark_v2"}, {"corpus_version": "c2"}],
)
def test_scope_must_belong_to_the_view_version_and_development_phase(world, change):
    directory, record_sha, _, _, _ = world
    scope = make_scope(directory, record_sha, {"q_a", "q_b"}, **change)
    with pytest.raises(RosterMismatch, match="phase|version"):
        build(world, scope=scope)


def test_scope_without_the_configured_arm_is_refused(world):
    directory, record_sha, _, _, _ = world
    scope = make_scope(directory, record_sha, {"q_a", "q_b"}, arm="dense_robustness")
    with pytest.raises(RosterMismatch, match="arm"):
        build(world, scope=scope)


def test_view_record_hash_is_verified(world):
    directory, _, _, _, _ = world
    with pytest.raises(ValueError):
        build_roster(
            view_directory=directory,
            view_record_sha256="0" * 64,
            scope=make_scope(directory, world[1], {"q_a", "q_b"}),
            renderer=world[3],
            sources=world[2],
        )


# ---- context sources --------------------------------------------------------------------


def test_missing_context_source_is_refused(world):
    with pytest.raises(RosterMismatch, match="no context"):
        build(world, sources=world[2][:1])


def test_duplicate_context_source_is_refused(world):
    with pytest.raises(RosterMismatch, match="duplicate"):
        build(world, sources=[*world[2], world[2][0]])


def test_source_for_a_condition_outside_the_scope_is_refused(world):
    directory, record_sha, _, _, _ = world
    scope = make_scope(directory, record_sha, {"q_a", "q_b"}, conditions=("PDF_DOCLING",))
    with pytest.raises(RosterMismatch, match="not authorized"):
        build(world, scope=scope)  # the sources are JATS contexts


def test_source_with_another_question_than_the_item_is_refused(world):
    prepared_ = world[4]
    wrong = source_for(prepared_, "q_a", "beta", "P001")
    with pytest.raises(RosterMismatch, match="question"):
        build(world, sources=[wrong, world[2][1]])


def test_source_for_another_papers_designation_is_refused(world):
    prepared_ = world[4]
    wrong = source_for(prepared_, "q_a", QUESTIONS["q_a"], "P002")
    with pytest.raises(RosterMismatch, match="designated"):
        build(world, sources=[wrong, world[2][1]])


def test_source_scope_must_match_the_arm(world):
    directory, record_sha, _sources, _, prepared_ = world
    renderer = make_renderer(prepared_[1].resolved_sha256, arm="scope_robustness")
    scope = make_scope(directory, record_sha, {"q_a", "q_b"}, arm="scope_robustness")
    with pytest.raises(RosterMismatch, match="scope or retriever"):
        build(world, scope=scope, renderer=renderer)  # the sources are designated-paper


def test_context_configuration_must_be_the_one_the_generator_names(world):
    _directory, _record_sha, _sources, _, _ = world
    other = make_renderer("1" * 64)
    with pytest.raises(RosterMismatch, match="retrieval configuration"):
        build(world, renderer=other)


def test_a_source_for_an_unknown_item_is_refused(world):
    prepared_ = world[4]
    stray = source_for(prepared_, "q_unknown", "alpha", "P001")
    with pytest.raises(RosterMismatch, match="not authorized"):
        build(world, sources=[*world[2], stray])


def test_altered_stored_context_is_rejected(world, tmp_path):
    source = world[2][0]
    altered_root = tmp_path / "altered"
    for ref in (source.context, source.manifest):
        target = altered_root / ref.logical_ref
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source.root / ref.logical_ref).read_bytes())
    path = altered_root / source.context.logical_ref
    path.write_bytes(path.read_bytes().replace(b"alpha", b"omega", 1))
    with pytest.raises(ValueError):
        build(world, sources=[replace(source, root=altered_root), world[2][1]])


def test_a_render_refusal_stops_the_roster_with_the_item_identity(world, monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("input of 99999 tokens exceeds the input limit 4096")

    monkeypatch.setattr(world[3], "render", refuse)
    with pytest.raises(RosterMismatch, match=r"q_a.*exceeds the input limit"):
        build(world)


# ---- stored generator inputs ---------------------------------------------------------------


def test_every_entry_carries_the_stored_input_its_row_cites(world):
    from biorag_repro.core.identity import observation_key
    from biorag_repro.generation.inputs import check_tokenization, input_record_ref

    built = build(world)
    renderer = world[3]
    rows = {(r.item_id, r.condition, r.repeat_index): r for r in built.record.rows}
    assert built.entries and all(e.input_record is not None for e in built.entries)
    for entry in built.entries:
        record = entry.input_record
        row = rows[(entry.item_id, entry.representation_or_condition, entry.repeat_index)]
        assert record.rendered == entry.rendered
        assert row.input_record == input_record_ref(record)
        assert record.context == row.context
        assert record.observation_key == observation_key(
            phase="development",
            benchmark_version=built.record.benchmark_version,
            item_id=entry.item_id,
            matrix_arm=built.record.matrix_arm,
            representation_or_condition=entry.representation_or_condition,
            repeat_index=entry.repeat_index,
            resolved_config_sha256=renderer.config.configuration_sha256,
        )
        check_tokenization(record, renderer.tokenizer)


def test_repeats_get_separate_inputs_because_they_are_separate_observations(world):
    directory, record_sha, _, _, _ = world
    scope = make_scope(directory, record_sha, {"q_a"}, repeats=(1, 2, 3))
    built = build(world, scope=scope, sources=world[2][:1])
    refs = {r.input_record.sha256 for r in built.record.rows}
    keys = {e.input_record.observation_key for e in built.entries}
    assert len(refs) == len(keys) == 3
    assert len({e.input_record.rendered.rendered_sha256 for e in built.entries}) == 1


def test_stored_inputs_are_part_of_the_roster_identity(world):
    built = build(world)
    row = built.record.rows[0]
    other = row.model_copy(update={"input_record": built.record.rows[1].input_record})
    changed = built.record.model_copy(update={"rows": [other, *built.record.rows[1:]]})
    assert roster_sha256(changed) != built.sha256


def test_closed_book_inputs_have_no_context_reference(world):
    directory, record_sha, _, _, _ = world
    renderer = make_renderer(None, arm="closed_book")
    scope = make_scope(
        directory, record_sha, {"q_a"}, arm="closed_book", conditions=("CLOSED_BOOK",)
    )
    built = build(world, scope=scope, sources=[], renderer=renderer)
    assert built.entries[0].input_record.context is None
    assert built.record.rows[0].input_record is not None
