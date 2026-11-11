# ruff: noqa: F811
"""Export of a verified official run into analysis-table observations.

The official run is the scripted one of test_generation_official: simulated approvals,
synthetic answers, no model. Nothing here is a result.
"""

import pytest
from generation_support import needs_tokenizer
from test_generation_official import (  # noqa: F401  (fixtures)
    CONTEXTS,
    anchored,
    build_world,
    recognised,
    verify,
    view,
)
from test_generation_roster import make_scope

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.evaluation.analysis_table import build_table
from biorag_repro.generation.handoff import official_observations
from biorag_repro.generation.official import NotVerifiedOfficial
from biorag_repro.generation.runner import nonofficial_results

pytestmark = needs_tokenizer


def verified_world(tmp_path, view, tiny, **options):
    world = build_world(tmp_path, view, tiny, **options)
    items = read_development_view(*view)[1]
    return world, verify(world, anchored(world)), items


def scope_for(view):
    return make_scope(*view, {"q_a", "q_b"}, conditions=("JATS", "PDF_DOCLING"))


def test_verified_run_becomes_scored_observations_and_a_valid_table(
    tmp_path,
    view,
    tiny,
    recognised,
):
    _world, verified, items = verified_world(tmp_path, view, tiny)
    observations = official_observations(verified, items, lambda item, condition: None)
    assert len(observations) == len(CONTEXTS) == 4
    assert all(o.score is not None and o.result in verified.results for o in observations)
    table = build_table(observations, scope_for(view))
    assert {r.execution_status for r in table.rows} == {"completed"}
    assert {r.context_provenance for r in table.rows} == {"assembled_only"}  # no Stage B given
    assert {r.run_id for r in table.rows} == {verified.run_id}


def test_operational_failures_stay_unscored_missingness(tmp_path, view, tiny, recognised):
    failing = CONTEXTS[("q_a", "PDF_DOCLING")]
    _world, verified, items = verified_world(tmp_path, view, tiny, fail=[failing])
    observations = official_observations(verified, items, lambda item, condition: None)
    failed = [o for o in observations if o.result.status == "operational_failure"]
    assert [(o.item.item_id, o.condition) for o in failed] == [("q_a", "PDF_DOCLING")]
    assert (
        failed[0].score is not None
        and failed[0].score.status == "missing"
        and failed[0].generator_input is None
    )
    table = build_table(observations, scope_for(view))
    assert sorted({r.execution_status for r in table.rows}) == ["completed", "operational_failure"]


def test_only_a_verified_official_run_can_be_exported(tmp_path, view, tiny, recognised):
    _world, verified, items = verified_world(tmp_path, view, tiny)
    with pytest.raises(NotVerifiedOfficial):
        official_observations(verified.results, items, lambda i, c: None)  # type: ignore[arg-type]
    with pytest.raises(NotVerifiedOfficial):
        official_observations(nonofficial_results, items, lambda i, c: None)  # type: ignore[arg-type]


def test_an_item_missing_from_the_supplied_items_is_refused(tmp_path, view, tiny, recognised):
    _world, verified, items = verified_world(tmp_path, view, tiny)
    with pytest.raises(NotVerifiedOfficial, match="q_b"):
        official_observations(
            verified, [i for i in items if i.item_id != "q_b"], lambda item, condition: None
        )
