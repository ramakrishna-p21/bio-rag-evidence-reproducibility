"""Matrix-arm rehearsal over synthetic fixtures. Nothing here is a scientific observation.

Two layers, kept apart on purpose:

* Execution: closed-book, confirmatory repeats and the robustness arms run through the real
  roster, driver, mock backend and verification. Mock results stay non-official.
* Table: a full-matrix analysis table built from explicitly synthetic evaluation records
  through the validated table interfaces. These records are test inputs; no mock result is
  converted into one, and `official_observations` is not involved.

Only JATS contexts exist in the roster fixture and the dense retriever needs model assets, so
the chunk (192) and dense arms are exercised as interfaces: the arm label, configuration and
retrieval-scope checks are real, the 192-token chunking and BGE ranking are not.
"""

from collections import Counter

import pytest
from generation_support import ROOT, needs_tokenizer
from test_analysis_export import P_A, _roster, deliver, observation, result_for
from test_analysis_export import world as table_world
from test_generation_driver import Killed, make_plan, run_mock
from test_generation_roster import QUESTIONS as QUESTIONS_TEXT
from test_generation_roster import (
    make_renderer,
    make_scope,
)
from test_generation_roster import shared as roster_world  # noqa: F401  (fixture)

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.core.identity import observation_key
from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
from biorag_repro.evaluation.analysis_storage import publish_table, read_table
from biorag_repro.evaluation.analysis_table import (
    build_table,
    parse_csv,
    primary_pairs,
    serialize_csv,
)
from biorag_repro.generation.driver import DriverError, MockBackend, dry_run, verify
from biorag_repro.generation.handoff import official_observations
from biorag_repro.generation.official import NotVerifiedOfficial
from biorag_repro.generation.roster import ContextSource
from biorag_repro.generation.runner import NonOfficialResults
from biorag_repro.retrieval import bm25, context

pytestmark = pytest.mark.usefixtures("driver_clock")


@pytest.fixture
def base(roster_world):  # noqa: F811
    return roster_world


QUESTIONS = ("q_a", "q_b")
needs_world = needs_tokenizer


def full_corpus_sources(prepared):
    """Contexts retrieved over all papers, with their own scope-specific configurations."""
    from test_bm25_publication import config as scoped_config

    from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
    from biorag_repro.core.manifest import ArtifactRef
    from biorag_repro.core.storage import serialize_project_json
    from biorag_repro.retrieval.models import ConfigurationRef, Phase5Configuration
    from biorag_repro.retrieval.validation import load_configuration

    selected, _context_configuration, root = prepared
    retrieval = scoped_config(root, "full_corpus")
    settings = load_configuration(retrieval, root).model_copy(
        update={
            "packing_tokenizer": context.tokenizer_snapshot(),
            "context_rendering": context.POLICY,
        }
    )
    data = serialize_project_json(settings, Phase5Configuration)
    (root / "packed-full.json").write_bytes(data)
    digest = sha256_bytes(canonical_json_bytes(settings.model_dump()))
    packed = ConfigurationRef(
        artifact=ArtifactRef(
            artifact_id="configuration_" + digest,
            logical_ref="packed-full.json",
            sha256=sha256_bytes(data),
        ),
        resolved_sha256=digest,
    )
    index = bm25.read_index(selected.index, selected.index_manifest, selected.sources, root)
    sources = []
    for item_id in QUESTIONS:
        q = bm25.Query(
            query_id=item_id,
            question=QUESTIONS_TEXT[item_id],
            scope="full_corpus",
            designated_paper_id=None,
        )
        ranked = bm25.retrieve(index, selected.sources, q, retrieval, root)
        rr, rm = bm25.publish_retrieval(
            ranked,
            selected.index,
            selected.index_manifest,
            selected.sources,
            q,
            retrieval,
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
            retrieval,
            root,
            root,
            "bm25",
        )
        ctx = context.assemble(sel, packed, root, ROOT)
        cref, cmref = context.publish(ctx, sel, packed, root, ROOT, root)
        sources.append(ContextSource(cref, cmref, sel, packed, root, root))
    return sources, digest


def arm_world(world, arm):
    """The shared fixture with the generator configuration of one matrix arm."""
    directory, record_sha, sources, _renderer, prepared = world
    retrieval = None if arm == "closed_book" else prepared[1].resolved_sha256
    if arm == "closed_book":
        sources = []
    elif arm == "scope_robustness":
        sources, retrieval = full_corpus_sources(prepared)
    return directory, record_sha, sources, make_renderer(retrieval, arm=arm), prepared


def execute_arm(tmp_path, world, arm, conditions, repeats, **kwargs):
    scope = make_scope(
        world[0], world[1], set(QUESTIONS), arm=arm, conditions=conditions, repeats=repeats
    )
    root = tmp_path / arm
    root.mkdir()
    plan = make_plan(root, world, scope=scope)
    return plan, run_mock(plan, world, **kwargs)


def expected_key(world, result):
    """Independent recomputation of the planned-observation key from its coordinates."""
    view = read_development_view(world[0], world[1])[0]
    return observation_key(
        phase="development",
        benchmark_version=view.benchmark_version,
        item_id=result.item_id,
        matrix_arm=result.matrix_arm,
        representation_or_condition=result.representation_or_condition,
        repeat_index=result.repeat_index,
        resolved_config_sha256=world[3].config.configuration_sha256,
    )


# ---- execution through the driver ------------------------------------------------------------


@needs_world
def test_closed_book_runs_three_repeats_per_item_with_no_context(base, tmp_path):
    world = arm_world(base, "closed_book")
    plan, outcome = execute_arm(tmp_path, world, "closed_book", ("CLOSED_BOOK",), (1, 2, 3))
    assert outcome.mode == "mock" and outcome.summary.complete == 6
    verified = verify(
        plan, outcome.run_id, mode="mock", anchor_file_sha256=outcome.anchor_file_sha256
    )
    assert isinstance(verified, NonOfficialResults)
    assert Counter((r.item_id, r.repeat_index) for r in verified.results) == {
        (item, n): 1 for item in QUESTIONS for n in (1, 2, 3)
    }
    assert {r.representation_or_condition for r in verified.results} == {"CLOSED_BOOK"}
    assert {r.matrix_arm for r in verified.results} == {"closed_book"}
    for r in verified.results:
        assert r.observation_key == expected_key(world, r)


@needs_world
def test_confirmatory_repeats_one_to_three_are_separate_observations(base, tmp_path):
    world = arm_world(base, "confirmatory")
    backend = MockBackend()
    plan, outcome = execute_arm(
        tmp_path, world, "confirmatory", ("JATS",), (1, 2, 3), backend=backend
    )
    assert outcome.summary.complete == 6
    verified = verify(
        plan, outcome.run_id, mode="mock", anchor_file_sha256=outcome.anchor_file_sha256
    )
    assert sorted(r.repeat_index for r in verified.results) == [1, 1, 2, 2, 3, 3]
    keys = {r.observation_key for r in verified.results}
    assert len(keys) == 6
    # the context is held fixed across repeats: each item's input is generated three times
    assert sorted(Counter(backend.calls).values()) == [3, 3]


@needs_world
def test_each_arm_has_its_own_configuration_and_observation_keys(base, tmp_path):
    runs = {}
    for arm, conditions, repeats in (
        ("confirmatory", ("JATS",), (1, 2, 3)),
        ("chunk_robustness", ("JATS",), (1,)),
        ("scope_robustness", ("JATS",), (1,)),
        ("closed_book", ("CLOSED_BOOK",), (1, 2, 3)),
    ):
        world = arm_world(base, arm)
        plan, outcome = execute_arm(tmp_path, world, arm, conditions, repeats)
        results = verify(
            plan, outcome.run_id, mode="mock", anchor_file_sha256=outcome.anchor_file_sha256
        ).results
        runs[arm] = (world, results)
    configs = {a: w[3].config.configuration_sha256 for a, (w, _) in runs.items()}
    assert len(set(configs.values())) == len(configs)  # the arm is part of the configuration
    keys = [r.observation_key for _w, results in runs.values() for r in results]
    assert len(keys) == len(set(keys)) == 6 + 2 + 2 + 6
    for w, results in runs.values():
        for r in results:
            assert r.observation_key == expected_key(w, r)
            assert r.resolved_config_sha256 == w[3].config.configuration_sha256


@needs_world
def test_an_arm_refuses_contexts_from_another_retrieval(base, tmp_path):
    directory, record_sha, sources, _renderer, prepared = base
    for arm in ("dense_robustness", "scope_robustness"):  # the fixture sources are BM25 designated
        world = (
            directory,
            record_sha,
            sources,
            make_renderer(prepared[1].resolved_sha256, arm=arm),
            prepared,
        )
        scope = make_scope(directory, record_sha, set(QUESTIONS), arm=arm, conditions=("JATS",))
        root = tmp_path / arm
        root.mkdir()
        plan = make_plan(root, world, scope=scope)
        with pytest.raises(DriverError, match="roster"):
            dry_run(plan, sources, mode="mock")
        assert list(plan.run_root.glob("*")) == []


@needs_world
def test_a_closed_book_arm_refuses_retrieval_contexts(base, tmp_path):
    world = arm_world(base, "closed_book")
    scope = make_scope(
        world[0], world[1], set(QUESTIONS), arm="closed_book", conditions=("CLOSED_BOOK",)
    )
    plan = make_plan(tmp_path, world, scope=scope)
    with pytest.raises(DriverError, match="not authorized"):
        dry_run(plan, base[2], mode="mock")


@needs_world
def test_a_final_test_scope_never_reaches_a_run(base, tmp_path):
    world = arm_world(base, "confirmatory")
    scope = make_scope(world[0], world[1], set(QUESTIONS), phase="final_test")
    plan = make_plan(tmp_path, world, scope=scope)
    with pytest.raises(DriverError):
        dry_run(plan, world[2], mode="mock")
    assert not plan.run_root.exists() and not plan.anchor_dir.exists()


@needs_world
def test_a_killed_multi_repeat_run_resumes_without_regenerating(base, tmp_path):
    world = arm_world(base, "closed_book")

    class KillsOnFourth(MockBackend):
        def generate(self, rendered, *, seed, max_new_tokens):
            if len(self.calls) == 3:
                raise Killed()
            return super().generate(rendered, seed=seed, max_new_tokens=max_new_tokens)

    scope = make_scope(
        world[0], world[1], set(QUESTIONS), arm="closed_book",
        conditions=("CLOSED_BOOK",), repeats=(1, 2, 3),
    )  # fmt: skip
    plan = make_plan(tmp_path, world, scope=scope)
    with pytest.raises(Killed):
        run_mock(plan, world, backend=KillsOnFourth())
    (run_id,) = [p.name for p in (plan.run_root / "runs").iterdir()]
    resumed = MockBackend()
    outcome = run_mock(plan, world, resume=run_id, recover_interrupted=True, backend=resumed)
    assert outcome.run_id == run_id and outcome.summary.complete == 6
    assert len(resumed.calls) == 3  # the three decided observations were kept
    results = verify(
        plan, run_id, mode="mock", anchor_file_sha256=outcome.anchor_file_sha256
    ).results
    assert len({r.observation_key for r in results}) == 6
    assert all(r.observation_key == expected_key(world, r) for r in results)


@needs_world
def test_mock_runs_cannot_be_verified_or_exported_as_official(base, tmp_path):
    world = arm_world(base, "closed_book")
    plan, outcome = execute_arm(tmp_path, world, "closed_book", ("CLOSED_BOOK",), (1,))
    with pytest.raises(DriverError):
        verify(plan, outcome.run_id, mode="official")
    verified = verify(
        plan, outcome.run_id, mode="mock", anchor_file_sha256=outcome.anchor_file_sha256
    )
    with pytest.raises(NotVerifiedOfficial):
        official_observations(verified, [], lambda *_: None)  # type: ignore[arg-type]


# ---- the full matrix as a synthetic analysis table -------------------------------------------

ARM_CONFIG = {arm: f"{n:x}" * 64 for n, arm in enumerate(FROZEN_MATRIX, 1)}


def synthetic_cell(w, case, arm, condition, repeat, config=None):
    """One synthetic, explicitly test-only record for a planned cell."""
    config = config or ARM_CONFIG[arm]
    result = result_for(
        w.item,
        matrix_arm=arm,
        representation_or_condition=condition,
        repeat_index=repeat,
        resolved_config_sha256=config,
        attempt_id=f"a{repeat}",
    )
    judged = case if arm in ("confirmatory", "chunk_robustness") and condition == "JATS" else None
    return observation(
        w,
        case=judged,
        evidence=None if judged is None else ...,
        result=result,
        matrix_arm=arm,
        condition=condition,
        repeat_index=repeat,
        resolved_config_sha256=config,
    )


def full_matrix(w, case):
    return [
        synthetic_cell(w, case, arm, condition, repeat)
        for arm, (conditions, repeats) in FROZEN_MATRIX.items()
        for condition in conditions
        for repeat in repeats
    ]


def matrix_roster(w):
    every_arm = {a: (list(c), list(r)) for a, (c, r) in FROZEN_MATRIX.items()}
    return _roster(every_arm, [w.item], scope="complete_matrix")


def test_the_whole_frozen_matrix_is_fifteen_rows_per_item():
    w = table_world()
    table = build_table(full_matrix(w, deliver(w, [w.chunk(P_A)])), matrix_roster(w))
    assert len(table.rows) == 15  # EXPERIMENT_PLAN section 10: 15 calls per final-test item
    assert Counter(r.matrix_arm for r in table.rows) == {
        "confirmatory": 6,
        "dense_robustness": 2,
        "chunk_robustness": 2,
        "scope_robustness": 2,
        "closed_book": 3,
    }
    assert sum(r.primary_row for r in table.rows) == 9  # repeat 1 of 4 arms x 2 + closed book


def test_arms_keep_distinct_configurations_and_recomputable_keys():
    w = table_world()
    table = build_table(full_matrix(w, deliver(w, [w.chunk(P_A)])), matrix_roster(w))
    assert {r.matrix_arm: r.config_sha256 for r in table.rows} == ARM_CONFIG
    assert len({r.observation_key for r in table.rows}) == 15
    for r in table.rows:
        assert r.observation_key == observation_key(
            phase=r.phase,
            benchmark_version=r.benchmark_version,
            item_id=r.item_id,
            matrix_arm=r.matrix_arm,
            representation_or_condition=r.condition,
            repeat_index=r.repeat_index,
            resolved_config_sha256=r.config_sha256,
        )


def test_only_confirmatory_repeat_one_forms_the_primary_pair():
    w = table_world()
    table = build_table(full_matrix(w, deliver(w, [w.chunk(P_A)])), matrix_roster(w))
    pairs = primary_pairs(table.rows, "development")
    assert list(pairs) == [w.item.item_id]
    jats, pdf = pairs[w.item.item_id]
    assert [(r.matrix_arm, r.condition, r.repeat_index) for r in (jats, pdf)] == [
        ("confirmatory", "JATS", 1),
        ("confirmatory", "PDF_DOCLING", 1),
    ]
    assert primary_pairs(table.rows, "confirmatory") == {}  # development rows, never confirmatory


def test_the_matrix_table_is_order_independent_and_survives_csv_and_publication(tmp_path):
    import random

    w = table_world()
    obs = full_matrix(w, deliver(w, [w.chunk(P_A)]))
    table = build_table(obs, matrix_roster(w))
    shuffled = obs[:]
    random.Random(8).shuffle(shuffled)
    assert serialize_csv(build_table(shuffled, matrix_roster(w)).rows) == serialize_csv(table.rows)
    assert parse_csv(serialize_csv(table.rows)) == table.rows
    _csv, _inventory, build_ref = publish_table(table, tmp_path)
    assert read_table(build_ref, tmp_path) == table.rows


def test_one_arm_cannot_run_under_two_configurations():
    w = table_world()
    case = deliver(w, [w.chunk(P_A)])
    other = "e" * 64
    obs = [
        synthetic_cell(w, case, "confirmatory", "JATS", 1),
        synthetic_cell(w, case, "confirmatory", "JATS", 2, config=other),
    ]
    roster = _roster({"confirmatory": (["JATS"], [1, 2])}, [w.item])
    with pytest.raises(ValueError, match="one matrix arm runs under one resolved configuration"):
        build_table(obs, roster)


def test_the_same_coordinates_in_two_arms_are_not_duplicates():
    w = table_world()
    case = deliver(w, [w.chunk(P_A)])
    obs = [
        synthetic_cell(w, case, "confirmatory", "JATS", 1),
        synthetic_cell(w, case, "chunk_robustness", "JATS", 1),
    ]
    roster = _roster(
        {"confirmatory": (["JATS"], [1]), "chunk_robustness": (["JATS"], [1])}, [w.item]
    )
    table = build_table(obs, roster)
    assert len({r.observation_key for r in table.rows}) == 2


def test_an_arm_outside_the_declared_roster_is_refused():
    w = table_world()
    case = deliver(w, [w.chunk(P_A)])
    obs = [
        synthetic_cell(w, case, "confirmatory", "JATS", 1),
        synthetic_cell(w, case, "dense_robustness", "JATS", 1),
    ]
    with pytest.raises(ValueError):
        build_table(obs, _roster({"confirmatory": (["JATS"], [1])}, [w.item]))
