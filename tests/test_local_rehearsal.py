"""Rehearsal of the local chain with a mock backend: nothing here is an observation.

view -> serialized context sources -> Phase 5 read-back -> roster -> prompt and tokenization
-> persisted inputs -> driver -> mock backend -> attempts and raw output -> completion
manifest and anchor -> verification -> Stage C -> analysis table -> CSV.
"""

import pytest
from generation_support import needs_tokenizer
from test_generation_driver import make_plan, run_mock
from test_generation_roster import make_scope
from test_generation_roster import shared as roster_world  # noqa: F401  (fixture)
from test_generation_sources import describable, with_fixture_counter

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.evaluation import stage_c
from biorag_repro.evaluation.analysis_table import (
    Observation,
    build_table,
    parse_csv,
    serialize_csv,
)
from biorag_repro.generation.driver import MOCK_TEXT, read_anchor, verify
from biorag_repro.generation.handoff import official_observations
from biorag_repro.generation.official import NotVerifiedOfficial
from biorag_repro.generation.runner import NonOfficialResults
from biorag_repro.generation.sources import load_context_sources, spec_from_sources

pytestmark = [needs_tokenizer, pytest.mark.usefixtures("driver_clock")]


@pytest.fixture
def world(roster_world):  # noqa: F811
    return roster_world


def spec_sources(world, tmp_path):
    spec = tmp_path / "context_sources.json"
    spec.write_text(spec_from_sources(describable(world[2]), retriever="bm25").model_dump_json())
    sources = load_context_sources(spec, sha256_bytes(spec.read_bytes()), world[0], world[1])
    return with_fixture_counter(sources)  # fixtures count chunks with a function, not a tokenizer


def test_chain_runs_from_serialized_sources_to_a_labelled_non_official_table(world, tmp_path):
    sources = spec_sources(world, tmp_path)
    plan = make_plan(tmp_path, world)
    outcome = run_mock(plan, (*world[:2], sources, *world[3:]))
    assert outcome.mode == "mock" and outcome.summary.complete == 2

    anchor = read_anchor(plan.anchor_dir, outcome.run_id, outcome.anchor_file_sha256)
    assert anchor.backend_kind == "mock"
    results = verify(
        plan, outcome.run_id, mode="mock", anchor_file_sha256=outcome.anchor_file_sha256
    )
    assert isinstance(results, NonOfficialResults) and results.backend_kind == "mock"
    assert {r.raw_output for r in results.results} == {MOCK_TEXT}

    items = {i.item_id: i for i in read_development_view(world[0], world[1])[1]}
    config = world[3].config
    observations = [
        Observation(
            item=items[r.item_id],
            phase="development",
            matrix_arm=r.matrix_arm,
            condition=r.representation_or_condition,
            repeat_index=r.repeat_index,
            resolved_config_sha256=config.configuration_sha256,
            retrieval_config_sha256=config.settings.retrieval_config_sha256,
            result=r,
            score=stage_c.score(items[r.item_id], r),
        )
        for r in results.results
    ]
    scope = make_scope(world[0], world[1], {"q_a", "q_b"})
    table = build_table(observations, scope)
    assert {r.context_provenance for r in table.rows} == {"assembled_only"}
    assert {r.stage_a_label for r in table.rows} == {None}  # no judgment was invented
    assert parse_csv(serialize_csv(table.rows)) == table.rows
    # The mock answer is not a correct answer for these items and must never look like one.
    assert "correct" not in {r.stage_c_status for r in table.rows}


def test_mock_results_cannot_cross_the_official_export_boundary(world, tmp_path):
    plan = make_plan(tmp_path, world)
    outcome = run_mock(plan, world)
    results = verify(plan, outcome.run_id, mode="mock")
    items = read_development_view(world[0], world[1])[1]
    with pytest.raises(NotVerifiedOfficial):
        official_observations(results, items, lambda item, condition: None)  # type: ignore[arg-type]
