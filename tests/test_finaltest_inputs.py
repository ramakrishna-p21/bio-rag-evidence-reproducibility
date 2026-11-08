"""Input-binding and roster-identity contracts. Synthetic release, digests only."""

import hashlib

import pytest
from finaltest_audit_support import (
    ARM,
    CONFIGS,
    FINAL_ITEMS,
    ITEMS,
    NOW,
    PLAN,
    ROOT_ID,
    binding_for,
    manifest_for,
    roster_for,
    rosters_for,
    row_for,
    settings_for,
)
from finaltest_campaign_support import CANARY, oracle_key, population, sha
from pydantic import ValidationError
from test_generation_config import resolve

from biorag_repro.finaltest.batches import make_manifest
from biorag_repro.finaltest.campaign import build_plan
from biorag_repro.finaltest.inputs import (
    BatchInputBinding,
    ConditionRoster,
    RepeatSeed,
    RosterIdentityRow,
    binding_sha256,
    build_binding,
    ordered_digest,
    roster_sha256,
    verify_binding,
)
from biorag_repro.finaltest.layout import serialize
from biorag_repro.generation.attempts import retry_policy_of
from biorag_repro.generation.seeds import (
    GenerationSeed,
    GenerationSeedRegistry,
    seed_registry_sha256,
)


def run(rosters=None, binding=None, arm=ARM, plan=PLAN, manifest=None, config=None, seeds=None):
    config = config or CONFIGS[arm]
    rosters = rosters or rosters_for(arm)
    return verify_binding(
        binding or binding_for(arm, rosters),
        plan=plan,
        manifest=manifest or manifest_for(arm),
        item_papers=dict(FINAL_ITEMS),
        rosters=rosters,
        resolved=config,
        retry=retry_policy_of(config.settings),
        seeds=seeds,
    )


def codes(findings):
    return {f.code for f in findings}


def test_consistent_records_have_no_findings():
    assert run() == ()


def test_key_set_and_count_match_an_independent_oracle():
    binding = binding_for(ARM)
    config = CONFIGS[ARM].configuration_sha256
    oracle = {oracle_key(i, ARM, c, 1, config) for i in ITEMS for c in ("JATS", "PDF_DOCLING")}
    assert binding.expected_count == len(oracle) == 10
    assert binding.key_set_sha256 == hashlib.sha256("\n".join(sorted(oracle)).encode()).hexdigest()
    assert binding.key_set_sha256 == manifest_for(ARM).key_set_sha256


def test_binding_identity_is_the_digest_of_its_stored_bytes_and_deterministic():
    first, second = binding_for(ARM), binding_for(ARM)
    assert binding_sha256(first) == binding_sha256(second)
    assert binding_sha256(first) == hashlib.sha256(serialize(first)).hexdigest()
    assert b"created_utc" not in serialize(first)


@pytest.mark.parametrize(
    "field,value",
    [
        ("manifest_sha256", sha("other")),
        ("resolved_config_sha256", sha("other")),
        ("prompt_sha256", sha("other")),
        ("model_revision", "other"),
        ("tokenizer_revision", "other"),
        ("retry_policy_sha256", sha("other")),
        ("key_set_sha256", sha("other")),
        ("split_sha256", sha("other")),
        ("benchmark_release_sha256", sha("other")),
    ],
)
def test_every_bound_field_changes_the_identity(field, value):
    binding = binding_for(ARM)
    changed = binding.model_copy(update={field: value})
    assert binding_sha256(changed) != binding_sha256(binding)


def test_roster_identity_changes_with_any_row_field():
    roster = rosters_for(ARM)["JATS"]
    other = roster_for(ARM, "JATS", rows=[row_for(ARM, "JATS", i, 1) for i in ITEMS[:-1]])
    assert roster_sha256(roster) != roster_sha256(other)


# --- scientific identities -------------------------------------------------------------


def test_changed_item_id_is_a_foreign_item_with_a_wrong_key():
    rows = [row_for(ARM, "JATS", i, 1) for i in ITEMS]
    rows[0] = row_for(ARM, "JATS", "i1", 1, item_id="zz")
    rows.sort(key=lambda r: (r.item_id, r.repeat_index))
    rosters = rosters_for(ARM) | {"JATS": roster_for(ARM, "JATS", rows=rows)}
    found = codes(run(rosters))
    assert {"foreign_item", "observation_key_mismatch"} <= found


def test_changed_paper_identity():
    rows = [row_for(ARM, "JATS", i, 1) for i in ITEMS]
    rows[0] = row_for(ARM, "JATS", "i1", 1, paper_id="p3")  # i1 belongs to p2
    rosters = rosters_for(ARM) | {"JATS": roster_for(ARM, "JATS", rows=rows)}
    assert "paper_mismatch" in codes(run(rosters))


def test_wrong_benchmark_identity_in_roster_and_in_binding():
    rosters = rosters_for(ARM) | {"JATS": roster_for(ARM, "JATS", benchmark_version="bench_v2")}
    assert "roster_identity_mismatch" in codes(run(rosters))
    binding = binding_for(ARM).model_copy(update={"benchmark_release_sha256": sha("other")})
    assert "binding_mismatch" in codes(run(binding=binding))


def test_changed_configuration_hash():
    other = sha("other-config")
    with pytest.raises(ValidationError, match="not of this arm, condition and configuration"):
        roster_for(ARM, "JATS", rows=[row_for(ARM, "JATS", "i1", 1, resolved_config_sha256=other)])
    rows = [
        row_for(
            ARM,
            "JATS",
            i,
            1,
            resolved_config_sha256=other,
            observation_key=oracle_key(i, ARM, "JATS", 1, other),
        )
        for i in ITEMS
    ]
    rosters = rosters_for(ARM) | {
        "JATS": roster_for(ARM, "JATS", rows=rows, resolved_config_sha256=other)
    }
    assert {"roster_identity_mismatch", "observation_key_mismatch"} <= codes(run(rosters))
    changed = binding_for(ARM).model_copy(update={"resolved_config_sha256": other})
    assert "binding_mismatch" in codes(run(binding=changed))


def test_a_different_configuration_for_the_arm_is_not_the_planned_one():
    other = resolve(settings_for(ARM, scoring_version="another"))
    assert "manifest_plan_mismatch" not in codes(run())
    assert {f.detail for f in run(config=other) if f.code == "binding_mismatch"} == {
        "configuration_identity"
    }


def test_jats_and_pdf_contexts_must_stay_distinguishable():
    rosters = rosters_for(ARM)
    jats = rosters["JATS"].rows[0]
    rows = [row_for(ARM, "PDF_DOCLING", i, 1) for i in ITEMS]
    rows[0] = row_for(
        ARM,
        "PDF_DOCLING",
        jats.item_id,
        1,
        context_sha256=jats.context_sha256,
        source_identity_sha256=jats.source_identity_sha256,
    )
    broken = rosters | {"PDF_DOCLING": roster_for(ARM, "PDF_DOCLING", rows=rows)}
    found = [f for f in run(broken) if f.code == "representations_not_distinguishable"]
    assert {f.detail for f in found} == {"source_identity_sha256", "context_sha256"}


def test_closed_book_context_leakage_is_refused_by_the_row_contract():
    with pytest.raises(ValidationError, match="closed-book observation has no source"):
        row_for("closed_book", "CLOSED_BOOK", "i1", 1, context_sha256=sha("leak"))
    with pytest.raises(ValidationError, match="retrieval observation needs"):
        row_for(ARM, "JATS", "i1", 1, context_sha256=None)


def test_closed_book_binding_has_no_retrieval_identity():
    binding = binding_for("closed_book")
    assert binding.retrieval_config_sha256 is None
    assert binding.expected_count == 15
    assert [r.condition for r in binding.rosters] == ["CLOSED_BOOK"]
    with pytest.raises(ValidationError, match="closed book has none"):
        BatchInputBinding.model_validate(
            binding.model_dump(mode="json") | {"retrieval_config_sha256": sha("x")}
        )


@pytest.mark.parametrize("repeat", [0, -1])
def test_repeat_index_zero_is_refused(repeat):
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        row_for(ARM, "JATS", "i1", repeat)
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        RepeatSeed(repeat_index=repeat, seed=1)


@pytest.mark.parametrize(
    "arm,repeat", [(ARM, 2), (ARM, 4), ("confirmatory", 4), ("closed_book", 0)]
)
def test_repeat_outside_the_matrix_is_refused(arm, repeat):
    condition = "CLOSED_BOOK" if arm == "closed_book" else "JATS"
    with pytest.raises(ValidationError):
        row_for(arm, condition, "i1", repeat)


def test_condition_outside_the_arm_is_refused():
    with pytest.raises(ValidationError, match="not a condition of"):
        row_for(ARM, "CLOSED_BOOK", "i1", 1)
    with pytest.raises(ValidationError, match="not a matrix arm"):
        row_for(ARM, "JATS", "i1", 1, matrix_arm="made_up")


def test_duplicate_rows_and_unsorted_rows_are_refused():
    rows = [row_for(ARM, "JATS", i, 1) for i in ITEMS]
    with pytest.raises(ValidationError, match="unique and sorted"):
        roster_for(ARM, "JATS", rows=rows + [rows[0]])
    with pytest.raises(ValidationError, match="unique and sorted"):
        roster_for(ARM, "JATS", rows=list(reversed(rows)))
    with pytest.raises(ValidationError, match="at least one row"):
        roster_for(
            ARM,
            "JATS",
            rows=[],
        )


def test_duplicate_observation_key_across_conditions():
    rows = [row_for(ARM, "PDF_DOCLING", i, 1) for i in ITEMS]
    stolen = row_for(ARM, "JATS", "i1", 1).observation_key
    rows[0] = row_for(ARM, "PDF_DOCLING", "i1", 1, observation_key=stolen)
    broken = rosters_for(ARM) | {"PDF_DOCLING": roster_for(ARM, "PDF_DOCLING", rows=rows)}
    assert {"duplicate_observation_key", "observation_key_mismatch"} <= codes(run(broken))


def test_foreign_observation_key_from_another_arm():
    rows = [row_for(ARM, "JATS", i, 1) for i in ITEMS]
    foreign = oracle_key(
        "i1", "chunk_robustness", "JATS", 1, CONFIGS["chunk_robustness"].configuration_sha256
    )
    rows[0] = row_for(ARM, "JATS", "i1", 1, observation_key=foreign)
    broken = rosters_for(ARM) | {"JATS": roster_for(ARM, "JATS", rows=rows)}
    found = codes(run(broken))
    assert {"foreign_observation_key", "observation_key_mismatch", "missing_observation"} <= found


def test_key_set_digest_mismatch_is_reported_twice_over():
    binding = binding_for(ARM).model_copy(update={"key_set_sha256": sha("other")})
    found = codes(run(binding=binding))
    assert {"binding_mismatch", "key_set_digest_mismatch"} <= found


def test_missing_roster_and_unreferenced_roster():
    rosters = rosters_for(ARM)
    binding = binding_for(ARM, rosters)
    assert "roster_missing" in codes(run({"JATS": rosters["JATS"]}, binding=binding))
    only = binding_for(ARM, rosters).model_copy(
        update={"rosters": binding.rosters[:1], "expected_count": 5}
    )
    assert "roster_unreferenced" in codes(run(rosters, binding=only))


def test_roster_hash_and_count_mismatch():
    rosters = rosters_for(ARM)
    binding = binding_for(ARM, rosters)
    shorter = roster_for(ARM, "JATS", rows=[row_for(ARM, "JATS", i, 1) for i in ITEMS[:-1]])
    found = codes(run(rosters | {"JATS": shorter}, binding=binding))
    assert {"roster_hash_mismatch", "roster_count_mismatch", "missing_observation"} <= found


def test_binding_refuses_duplicate_or_misordered_conditions():
    binding = binding_for(ARM)
    refs = [r.model_dump(mode="json") for r in binding.rosters]
    base = binding.model_dump(mode="json")
    for rosters in (refs + [refs[0]], list(reversed(refs)), refs[:1]):
        with pytest.raises(ValidationError, match="conditions of the arm"):
            BatchInputBinding.model_validate(base | {"rosters": rosters})
    with pytest.raises(ValidationError, match="add up"):
        BatchInputBinding.model_validate(base | {"expected_count": 9})


def test_wrong_plan_or_manifest():
    other_plan = build_plan(
        population(final_test_item_ids=("i1",), item_papers={"i1": "p2"}),
        {a: c.configuration_sha256 for a, c in CONFIGS.items()},
        PLAN.generation_seed_registry_sha256,
    )
    assert "manifest_plan_mismatch" in codes(run(plan=other_plan))
    other_manifest = make_manifest(sha("other-root"), PLAN, ARM, NOW)
    assert "binding_mismatch" in codes(run(manifest=other_manifest))


# --- seeds ------------------------------------------------------------------------------


def sampling_world(seed=11):
    registry = GenerationSeedRegistry(
        registry_version="r1",
        entries=[GenerationSeed(name="generation_repeat_1", seed=seed, consumer="set_seed")],
    )
    config = resolve(
        settings_for(
            ARM,
            decoding={"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 20},
            seed_registry_version="r1",
            seed_registry_sha256=seed_registry_sha256(registry),
        )
    )
    hashes = {a: c.configuration_sha256 for a, c in CONFIGS.items()}
    hashes[ARM] = config.configuration_sha256
    plan = build_plan(population(), hashes, seed_registry_sha256(registry))
    manifest = make_manifest(ROOT_ID, plan, ARM, NOW)
    rosters = {
        c: roster_for(
            ARM,
            c,
            batch_id=manifest.batch_id,
            resolved_config_sha256=config.configuration_sha256,
            rows=[
                row_for(
                    ARM,
                    c,
                    i,
                    1,
                    seed=seed,
                    resolved_config_sha256=config.configuration_sha256,
                    observation_key=oracle_key(i, ARM, c, 1, config.configuration_sha256),
                )
                for i in ITEMS
            ],
        )
        for c in ("JATS", "PDF_DOCLING")
    }
    return registry, config, plan, manifest, rosters


def sampling_findings(registry, config, plan, manifest, rosters, bound_registry=None):
    binding = build_binding(
        manifest=manifest,
        plan=plan,
        resolved=config,
        retry=retry_policy_of(config.settings),
        seeds=bound_registry or registry,
        rosters=list(rosters.values()),
    )
    return binding, run(
        rosters, binding=binding, plan=plan, manifest=manifest, config=config, seeds=registry
    )


def test_sampling_batch_binds_the_seed_registry_mapping():
    world = sampling_world()
    binding, found = sampling_findings(*world)
    assert found == ()
    assert [(s.repeat_index, s.seed) for s in binding.seeds] == [(1, 11)]


def test_incorrect_seed_mapping_is_found_in_binding_and_in_rows():
    registry, config, plan, manifest, rosters = sampling_world()
    wrong = GenerationSeedRegistry(
        registry_version="r1",
        entries=[GenerationSeed(name="generation_repeat_1", seed=99, consumer="set_seed")],
    )
    _, found = sampling_findings(registry, config, plan, manifest, rosters, bound_registry=wrong)
    assert "seed_mapping_mismatch" in codes(found)
    other_rows = sampling_world(seed=12)[4]  # rows carry another seed than the registry
    _, found = sampling_findings(registry, config, plan, manifest, other_rows)
    assert {f.detail for f in found if f.code == "seed_mapping_mismatch"} == {"row seed"}


def test_greedy_batch_with_a_bound_seed_is_refused():
    binding = binding_for(ARM)
    with_seed = binding.model_copy(update={"seeds": [RepeatSeed(repeat_index=1, seed=1)]})
    details = {f.detail for f in run(binding=with_seed) if f.code == "seed_mapping_mismatch"}
    assert "seeds bound for greedy decoding" in details
    rows = [row_for(ARM, "JATS", i, 1, seed=5) for i in ITEMS]
    broken = rosters_for(ARM) | {"JATS": roster_for(ARM, "JATS", rows=rows)}
    assert "seed_mapping_mismatch" in codes(run(broken))


# --- protected content ------------------------------------------------------------------


def test_protected_canary_never_reaches_records_or_findings():
    rosters = rosters_for(ARM)
    binding = binding_for(ARM, rosters)
    texts = [serialize(binding).decode(), *(serialize(r).decode() for r in rosters.values())]
    broken = rosters | {
        "JATS": roster_for(ARM, "JATS", rows=[row_for(ARM, "JATS", i, 1) for i in ITEMS[:-1]])
    }
    texts += [f"{f}" for f in run(broken)]
    try:
        row_for(ARM, "JATS", "i1", 0)
    except ValidationError as error:
        texts.append(str(error))
    assert all(CANARY not in text for text in texts)
    assert sha(CANARY + "i1") in serialize(rosters["JATS"]).decode()  # only the digest is kept


def test_row_contract_has_no_free_text_field():
    names = set(RosterIdentityRow.model_fields)
    assert not names & {"question", "gold", "answer", "context", "evidence", "text", "prompt"}


def test_ordered_digest_depends_on_order():
    assert ordered_digest(["a", "b"]) != ordered_digest(["b", "a"])
    assert ordered_digest(["a", "b"]) == ordered_digest(["a", "b"])


def test_roster_is_a_frozen_record():
    roster = rosters_for(ARM)["JATS"]
    assert isinstance(roster, ConditionRoster)
    with pytest.raises(ValidationError):
        roster.condition = "PDF_DOCLING"
