"""Campaign completeness on hand-built plans and observation states."""

import pytest
from finaltest_campaign_support import CONFIGS, FINAL_ITEMS, population, sha
from test_finaltest_protocol import Env

from biorag_repro.finaltest import completeness as cp
from biorag_repro.finaltest.campaign import ARMS, expected_keys
from biorag_repro.finaltest.completeness import ObservedBatch, check_campaign
from biorag_repro.generation.attempts import Coordinates, ObservationState

ITEMS = sorted(FINAL_ITEMS)
CELLS = {
    "confirmatory": (["JATS", "PDF_DOCLING"], [1, 2, 3]),
    "closed_book": (["CLOSED_BOOK"], [1, 2, 3]),
    "dense_robustness": (["JATS", "PDF_DOCLING"], [1]),
    "chunk_robustness": (["JATS", "PDF_DOCLING"], [1]),
    "scope_robustness": (["JATS", "PDF_DOCLING"], [1]),
}


def key_of(env, arm, item, condition, repeat, config=None):
    from biorag_repro.core.identity import observation_key

    return observation_key(
        phase="final_test",
        benchmark_version="bench_v1",
        item_id=item,
        matrix_arm=arm,
        representation_or_condition=condition,
        repeat_index=repeat,
        resolved_config_sha256=config or CONFIGS[arm],
    )


def state(env, arm, item, condition, repeat, status="complete", config=None, key=None):
    coordinates = Coordinates(
        item_id=item,
        paper_id=FINAL_ITEMS.get(item, "pX"),
        matrix_arm=arm,
        representation_or_condition=condition,
        repeat_index=repeat,
    )
    return ObservationState(
        key=key or key_of(env, arm, item, condition, repeat, config),
        coordinates=coordinates,
        events=[],
        last_event_sha256=sha("last"),
        status=status,
        attempts=1,
        first_event_sha256=sha("first"),
    )


def full_states(env, arm, status="complete"):
    conditions, repeats = CELLS[arm]
    return [state(env, arm, i, c, r, status) for i in ITEMS for c in conditions for r in repeats]


def run(arm, states, **changes):
    values = {
        "phase": "final_test",
        "benchmark_version": "bench_v1",
        "matrix_arm": arm,
        "resolved_config_sha256": CONFIGS[arm],
        "states": states,
    }
    return ObservedBatch(**(values | changes))


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    for arm in ARMS:
        e.reserve_full(arm)
    return e


def check(env, observed, registry=None):
    return check_campaign(env.plan, ITEMS, env.root_id, registry or env.verify(), observed)


def everything(env):
    return {arm: [run(arm, full_states(env, arm))] for arm in ARMS}


def kinds(result):
    return {a.kind for a in result.anomalies}


def test_every_expected_observation_verified_is_complete_within_the_stated_limit(env):
    result = check(env, everything(env))
    assert result.verdict == "complete_all_observed"
    assert not result.anomalies
    assert result.limits == ("the latest registry head is not independently established",)
    assert {a: sum(c.values()) for a, c in result.per_arm.items()} == {
        "confirmatory": 30,
        "closed_book": 15,
        "dense_robustness": 10,
        "chunk_robustness": 10,
        "scope_robustness": 10,
    }
    assert set(result.outcomes) == {k for a in ARMS for k in expected_keys(env.plan, ITEMS, a)}


def test_registry_files_alone_do_not_make_a_campaign_complete(env):
    result = check(env, {})
    assert result.verdict == "incomplete"
    assert set(result.outcomes.values()) == {cp.NEVER_ATTEMPTED}


def test_a_recorded_failure_is_missing_not_complete_and_is_not_regenerated(env):
    observed = everything(env)
    states = observed["closed_book"][0].states
    states[0] = state(env, "closed_book", ITEMS[0], "CLOSED_BOOK", 1, "terminal_failure")
    result = check(env, observed)
    assert result.verdict == "complete_with_recorded_missing"
    assert result.per_arm["closed_book"] == {cp.VERIFIED: 14, cp.MISSING_RECORDED: 1}
    observed["closed_book"][0].states[0] = state(
        env, "closed_book", ITEMS[0], "CLOSED_BOOK", 1, "retries_exhausted"
    )
    assert check(env, observed).per_arm["closed_book"][cp.MISSING_RECORDED] == 1


@pytest.mark.parametrize(
    ("status", "label"),
    [
        ("planned", cp.NEVER_ATTEMPTED),
        ("retryable", cp.INTERRUPTED),
        ("running", cp.INTERRUPTED),
    ],
)
def test_unfinished_observations_keep_the_campaign_incomplete(env, status, label):
    observed = everything(env)
    observed["closed_book"][0].states[0] = state(
        env, "closed_book", ITEMS[0], "CLOSED_BOOK", 1, status
    )
    result = check(env, observed)
    assert result.verdict == "incomplete"
    assert result.per_arm["closed_book"][label] == 1


def test_an_expected_key_that_was_never_attempted_is_reported(env):
    observed = everything(env)
    observed["closed_book"][0] = run("closed_book", full_states(env, "closed_book")[1:])
    result = check(env, observed)
    assert result.verdict == "incomplete"
    assert result.per_arm["closed_book"] == {cp.VERIFIED: 14, cp.NEVER_ATTEMPTED: 1}


def test_a_duplicate_completion_invalidates_the_campaign(env):
    observed = everything(env)
    observed["closed_book"][0].states.append(observed["closed_book"][0].states[0])
    result = check(env, observed)
    assert result.verdict == "invalid"
    assert "duplicate_completion" in kinds(result)


def test_a_duplicate_across_two_runs_of_one_arm_is_also_found(env):
    observed = everything(env)
    observed["closed_book"].append(run("closed_book", full_states(env, "closed_book")[:1]))
    assert "duplicate_completion" in kinds(check(env, observed))


@pytest.mark.parametrize(
    ("expected_kind", "build"),
    [
        (
            "foreign_key",
            lambda e: state(e, "confirmatory", "not-an-item", "JATS", 1),
        ),
        (
            "wrong_condition",
            lambda e: state(e, "confirmatory", ITEMS[0], "CLOSED_BOOK", 1),
        ),
        (
            "wrong_repeat",
            lambda e: state(e, "confirmatory", ITEMS[0], "JATS", 4),
        ),
        (
            "wrong_repeat",
            lambda e: state(e, "confirmatory", ITEMS[0], "JATS", 0),
        ),
        (
            "identity_mismatch",
            lambda e: state(e, "confirmatory", ITEMS[0], "JATS", 1, config=sha("other")),
        ),
        (
            "identity_mismatch",
            lambda e: state(e, "confirmatory", ITEMS[0], "JATS", 1, key="obs_" + "0" * 64),
        ),
    ],
)
def test_an_observation_with_foreign_coordinates_is_invalid(env, expected_kind, build):
    observed = everything(env)
    observed["confirmatory"][0].states.append(build(env))
    result = check(env, observed)
    assert result.verdict == "invalid"
    assert expected_kind in kinds(result)


def test_a_run_of_another_campaign_or_configuration_is_invalid(env):
    for changes, kind in (
        ({"benchmark_version": "bench_v2"}, "wrong_campaign"),
        ({"phase": "development"}, "wrong_campaign"),
        ({"resolved_config_sha256": sha("other")}, "wrong_configuration"),
        ({"matrix_arm": "confirmatory"}, "wrong_arm"),
    ):
        observed = everything(env)
        observed["closed_book"] = [run("closed_book", full_states(env, "closed_book"), **changes)]
        result = check(env, observed)
        assert result.verdict == "invalid"
        assert kind in kinds(result)


def test_observations_for_an_arm_outside_the_plan_are_invalid(env):
    observed = everything(env) | {"extra_arm": []}
    assert "wrong_arm" in kinds(check(env, observed))


def test_unmaterialized_or_closed_batches_keep_a_fully_observed_campaign_incomplete(tmp_path):
    env = Env(tmp_path)
    for arm in ARMS[:-1]:
        env.reserve_full(arm)
    last = env.reserve(ARMS[-1])
    assert check(env, everything(env)).verdict == "incomplete"
    env.close(last)
    result = check(env, everything(env))
    assert result.verdict == "incomplete"
    assert "batch_closed_incomplete" in kinds(result)
    assert result.anomalies[0].subject


def test_a_registry_with_conflicts_makes_the_result_invalid(env):
    (env.root / "runs").mkdir()
    result = check(env, everything(env))
    assert result.verdict == "invalid"
    assert "registry_not_verified" in kinds(result)


def test_population_drift_changes_the_expected_keys_not_the_accounting(env):
    items = ITEMS[:-1]
    result = check_campaign(env.plan, items, env.root_id, env.verify(), everything(env))
    assert result.verdict == "invalid"  # states exist for an item the population no longer has
    assert "foreign_key" in kinds(result)
    assert population().final_test_item_ids == tuple(ITEMS)


def test_a_campaign_without_any_reservation_is_incomplete_even_if_everything_is_observed(
    tmp_path,
):
    env = Env(tmp_path)  # genesis only: structurally clean, no batch reserved
    result = check(env, everything(env))
    assert result.verdict == "incomplete"
    assert not result.anomalies


def test_an_unfinished_publication_keeps_a_fully_observed_campaign_incomplete(env):
    (env.root / "registry" / ".000099.json.abc.tmp").write_bytes(b"{")
    result = check(env, everything(env))
    assert result.verdict == "incomplete"
    assert {a.kind for a in result.anomalies} == {"registry_incomplete"}
