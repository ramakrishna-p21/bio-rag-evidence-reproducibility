import hashlib

import pytest
from finaltest_campaign_support import (
    BENCH,
    CANARY,
    CONFIGS,
    FINAL_ITEMS,
    SEEDS,
    StubSplit,
    full_scale_inputs,
    oracle_all,
    oracle_key,
    population,
    release_fixture,
    sha,
)

from biorag_repro.core.identity import observation_key
from biorag_repro.finaltest import campaign
from biorag_repro.finaltest.campaign import PlanError, build_plan, expected_keys, plan_sha256

ITEMS = sorted(FINAL_ITEMS)


def plan(**changes):
    return build_plan(population(**changes), CONFIGS, SEEDS)


def test_counts_are_derived_and_match_a_hand_calculation():
    p = plan()
    by_arm = {b.matrix_arm: b.expected_count for b in p.batches}
    # 5 items: confirmatory 2 conditions x 3 repeats, closed book 1 x 3, three arms 2 x 1
    assert by_arm == {
        "confirmatory": 30,
        "closed_book": 15,
        "dense_robustness": 10,
        "chunk_robustness": 10,
        "scope_robustness": 10,
    }
    assert p.total_expected == 75 == 15 * len(ITEMS)
    assert sum(by_arm.values()) == p.total_expected


def test_keys_equal_the_independent_oracle_for_every_arm():
    p = plan()
    for arm, oracle in oracle_all(ITEMS).items():
        assert set(expected_keys(p, ITEMS, arm)) == oracle
        batch = next(b for b in p.batches if b.matrix_arm == arm)
        digest = hashlib.sha256("\n".join(sorted(oracle)).encode()).hexdigest()
        assert batch.key_set_sha256 == digest


def test_golden_vector_for_one_coordinate():
    got = observation_key(
        phase="final_test",
        benchmark_version=BENCH,
        item_id="i1",
        matrix_arm="confirmatory",
        representation_or_condition="PDF_DOCLING",
        repeat_index=3,
        resolved_config_sha256=CONFIGS["confirmatory"],
    )
    assert got == oracle_key("i1", "confirmatory", "PDF_DOCLING", 3, CONFIGS["confirmatory"])


def test_repeats_are_one_based_and_zero_is_not_planned():
    p = plan()
    keys = set(expected_keys(p, ITEMS, "confirmatory"))
    assert oracle_key("i1", "confirmatory", "JATS", 0, CONFIGS["confirmatory"]) not in keys
    assert oracle_key("i1", "confirmatory", "JATS", 3, CONFIGS["confirmatory"]) in keys
    assert all(c.repeats == [1, 2, 3] for c in p.batches[0].cells)
    assert all(c.repeats == [1] for c in p.batches[2].cells)
    for bad in ([0, 1, 2], [1, 2, 4], [2, 3, 1]):
        data = p.model_dump()
        data["batches"][0] = _with_repeats(p, bad)
        with pytest.raises(ValueError, match="frozen matrix"):
            campaign.CampaignPlan.model_validate(data)


def _with_repeats(p, repeats):
    batch = p.model_dump()["batches"][0]
    batch["cells"] = [{**c, "repeats": repeats} for c in batch["cells"]]
    return batch


def test_plan_identity_ignores_item_and_arm_input_order():
    shuffled = dict(reversed(list(CONFIGS.items())))
    a = build_plan(population(), CONFIGS, SEEDS)
    b = build_plan(
        population(
            final_test_item_ids=tuple(reversed(ITEMS)),
            final_test_paper_ids=("p3", "p2"),
        ),
        shuffled,
        SEEDS,
    )
    assert plan_sha256(a) == plan_sha256(b)
    assert expected_keys(a, ITEMS, "closed_book") == expected_keys(
        b, list(reversed(ITEMS)), "closed_book"
    )


def test_keys_differ_by_condition_repeat_item_and_configuration():
    p = plan()
    keys = expected_keys(p, ITEMS, "confirmatory")
    assert len(set(keys)) == len(keys) == 30
    other = build_plan(population(), CONFIGS | {"confirmatory": sha("different")}, SEEDS)
    assert not set(keys) & set(expected_keys(other, ITEMS, "confirmatory"))


def test_changed_benchmark_identity_changes_the_plan_and_every_key():
    base = plan()
    for change in (
        {"benchmark_release_sha256": sha("other release")},
        {"corpus_content_sha256": sha("other corpus")},
        {"split_sha256": sha("other split")},
    ):
        assert plan_sha256(plan(**change)) != plan_sha256(base)
    renamed = plan(benchmark_version="bench_v2")
    assert not set(expected_keys(renamed, ITEMS, "closed_book")) & set(
        expected_keys(base, ITEMS, "closed_book")
    )


def test_changed_seed_registry_changes_the_plan():
    assert plan_sha256(build_plan(population(), CONFIGS, sha("s2"))) != plan_sha256(plan())


def test_a_configuration_is_required_for_each_arm_and_none_may_repeat():
    with pytest.raises(PlanError):
        build_plan(population(), {k: v for k, v in CONFIGS.items() if k != "closed_book"}, SEEDS)
    with pytest.raises(PlanError):
        build_plan(population(), CONFIGS | {"extra": sha("x")}, SEEDS)
    with pytest.raises(ValueError, match="own configuration"):
        build_plan(population(), CONFIGS | {"closed_book": CONFIGS["confirmatory"]}, SEEDS)


def test_development_leakage_is_rejected():
    with pytest.raises(PlanError, match="leak"):
        plan(development_paper_ids=("p1", "p2"))
    with pytest.raises(PlanError, match="non-final-test"):
        plan(item_papers={**FINAL_ITEMS, "i1": "p1"})


def test_duplicate_coordinates_are_rejected():
    p = plan()
    with pytest.raises(PlanError, match="duplicate"):
        expected_keys(p, ITEMS + ["i1"], "confirmatory")


def test_a_duplicated_item_in_the_population_is_a_duplicate_coordinate():
    with pytest.raises(PlanError, match="duplicate"):
        plan(final_test_item_ids=(*ITEMS, "i1"))


def test_plan_has_no_authorizing_state():
    p = plan()
    assert p.status == "structural"
    assert not any("approv" in name or "authoriz" in name for name in type(p).model_fields)
    with pytest.raises(ValueError):
        campaign.CampaignPlan.model_validate({**p.model_dump(), "status": "approved"})


def test_check_plan_detects_a_population_that_drifted():
    p = plan()
    campaign.check_plan(p, population())
    with pytest.raises(PlanError):
        campaign.check_plan(
            p,
            population(
                final_test_item_ids=tuple(ITEMS[:-1]),
                item_papers={k: v for k, v in FINAL_ITEMS.items() if k != "i5"},
            ),
        )
    with pytest.raises(PlanError):
        campaign.check_plan(p, population(benchmark_release_sha256=sha("drift")))


def test_batch_cells_with_other_counts_than_the_matrix_are_rejected():
    data = plan().model_dump()
    data["batches"][2]["cells"][0]["observation_count"] += 1
    with pytest.raises(ValueError, match="frozen matrix"):
        campaign.CampaignPlan.model_validate(data)
    data = plan().model_dump()
    data["total_expected"] += 1
    with pytest.raises(ValueError, match="sum"):
        campaign.CampaignPlan.model_validate(data)
    data = plan().model_dump()
    data["batches"] = data["batches"][::-1]
    with pytest.raises(ValueError, match="one batch per matrix arm"):
        campaign.CampaignPlan.model_validate(data)


def test_full_scale_population_is_derived_from_release_and_split():
    dev, final, items = full_scale_inputs()
    release = release_fixture(len(items))
    pop = campaign.population_from(
        release=release,
        release_sha256=sha("rel"),
        split=StubSplit(dev, final),
        split_sha256=sha("split"),
        items=items,
    )
    p = build_plan(pop, CONFIGS, SEEDS)
    assert p.final_test_item_count == 80
    assert p.total_expected == 15 * 80 == 1200
    assert [b.expected_count for b in p.batches] == [480, 160, 160, 160, 240]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda i: i[0].update(split="final_test"),  # development paper labeled final_test
        lambda i: i[-1].update(split="development"),  # final-test paper labeled development
        lambda i: i[-1].update(paper_id="stranger"),  # paper in neither partition
        lambda i: i[-1].update(candidate_status="excluded"),
        lambda i: i.append(dict(i[-1])),  # duplicate item id (also changes the count)
    ],
)
def test_population_rejects_items_that_disagree_with_the_split(mutate):
    dev, final, items = full_scale_inputs()
    count = len(items)
    mutate(items)
    release = release_fixture(count if len(items) == count else len(items))
    with pytest.raises(PlanError):
        campaign.population_from(
            release=release,
            release_sha256=sha("rel"),
            split=StubSplit(dev, final),
            split_sha256=sha("split"),
            items=items,
        )


def test_population_rejects_a_release_with_other_counts():
    dev, final, items = full_scale_inputs()
    with pytest.raises(PlanError, match="item count"):
        campaign.population_from(
            release=release_fixture(len(items) + 1),
            release_sha256=sha("rel"),
            split=StubSplit(dev, final),
            split_sha256=sha("split"),
            items=items,
        )
    with pytest.raises(PlanError, match="corpus version"):
        campaign.population_from(
            release=release_fixture(len(items)),
            release_sha256=sha("rel"),
            split=StubSplit(dev, final, corpus_version="other"),
            split_sha256=sha("split"),
            items=items,
        )


def test_item_reader_keeps_identity_fields_only():
    row = (
        '{"item_id":"i1","paper_id":"p2","split":"final_test","candidate_status":"accepted",'
        f'"question":"{CANARY}","gold":{{"answer":"{CANARY}"}}}}\n'
    )
    kept = campaign._read_items(row.encode())
    assert kept == [
        {"item_id": "i1", "paper_id": "p2", "split": "final_test", "candidate_status": "accepted"}
    ]
    p = plan()
    assert CANARY not in repr(p) and CANARY not in repr(population()) and CANARY not in repr(kept)
