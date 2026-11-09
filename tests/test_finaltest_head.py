"""What a claimed latest registry head does and does not establish. Synthetic rehearsal roots only."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from finaltest_campaign_support import sha
from test_finaltest_protocol import Env

import biorag_repro
from biorag_repro.finaltest import layout
from biorag_repro.finaltest.head import (
    HeadAssessment,
    HeadAttestation,
    _assess_with_fixture_trust,
    assess_head,
    attestation_sha256,
    parse_attestation,
)

SRC = Path(biorag_repro.__file__).parent


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def file_sha(env, number):
    return hashlib.sha256((env.root / "registry" / f"{number:06d}.json").read_bytes()).hexdigest()


def claim(env, number, **changes):
    values = {
        "root_id": env.root_id,
        "sequence": number,
        "event_file_sha256": file_sha(env, number),
        "provenance": "unverified_claim",
    }
    return HeadAttestation(**(values | changes))


def assess(env, attestation=None) -> HeadAssessment:
    return assess_head(env.root, env.auth, env.anchors, env.witnesses, attestation)


def at_head_five(env):
    env.reserve_full("closed_book")
    env.reserve_full("confirmatory")


def snapshot(env, where):
    shutil.copytree(env.root, where / "root")
    shutil.copytree(env.witnesses, where / "witnesses")


def restore(env, where):
    shutil.rmtree(env.root)
    shutil.rmtree(env.witnesses)
    shutil.copytree(where / "root", env.root)
    shutil.copytree(where / "witnesses", env.witnesses)


# ---- the attestation record is a claim, never an approval ----------------------------------


def test_the_record_has_no_field_that_could_pass_for_an_approval():
    names = set(HeadAttestation.model_fields)
    assert not names & {"approved", "authorized", "trusted", "signature", "signed_by"}


def test_attestation_identity_is_the_sha256_of_its_stored_bytes(env):
    env.reserve_full("closed_book")
    attestation = claim(env, 3)
    stored = layout.serialize(attestation)
    assert attestation_sha256(attestation) == hashlib.sha256(stored).hexdigest()
    fields = json.loads(stored)
    assert fields["sequence"] == 3 and fields["event_file_sha256"] == file_sha(env, 3)
    assert fields["provenance"] == "unverified_claim"


def test_parse_round_trips_only_canonical_bytes(env):
    env.reserve_full("closed_book")
    attestation = claim(env, 3)
    stored = layout.serialize(attestation)
    assert parse_attestation(stored) == attestation
    with pytest.raises(ValueError):
        parse_attestation(stored.replace(b"\n  ", b" "))  # same content, not canonical
    with pytest.raises(ValueError):
        parse_attestation(stored + b" ")


@pytest.mark.parametrize(
    "extra",
    [{"approved": True}, {"authorized": True}, {"signature": "x"}, {"trusted": True}],
)
def test_forged_json_with_an_approval_field_is_rejected(env, extra):
    env.reserve_full("closed_book")
    forged = json.loads(layout.serialize(claim(env, 3))) | extra
    with pytest.raises(ValueError):
        parse_attestation(json.dumps(forged, sort_keys=True, indent=2).encode() + b"\n")


@pytest.mark.parametrize("provenance", ["independent_authority", "approved", "", "operator"])
def test_provenance_cannot_name_an_authority(env, provenance):
    env.reserve_full("closed_book")
    with pytest.raises(ValueError):
        claim(env, 3, provenance=provenance)


@pytest.mark.parametrize(
    "changes",
    [
        {"sequence": 0},
        {"previous_sequence": 2},  # a predecessor needs its hash
        {"previous_event_file_sha256": sha("p")},  # and the other way round
        {"previous_sequence": 3, "previous_event_file_sha256": sha("p")},  # not earlier
        {"batch_id": sha("b")},  # a batch scope needs its plan
    ],
)
def test_malformed_attestations_are_rejected(env, changes):
    env.reserve_full("closed_book")
    with pytest.raises(ValueError):
        claim(env, 3, **changes)


# ---- genesis and later heads ---------------------------------------------------------------


def test_a_genesis_only_registry_is_anchored_without_any_claim(env):
    result = assess(env)
    assert (result.classification, result.authority) == ("genesis_anchored", "genesis_anchored")
    assert result.registry_head_sequence == 1 and result.chain_valid
    assert not result.grants_authorization


def test_a_claim_matching_genesis_is_still_just_genesis(env):
    result = assess(env, claim(env, 1))
    assert (result.classification, result.authority) == ("genesis_anchored", "genesis_anchored")


def test_a_later_head_without_a_claim_is_unestablished(env):
    env.reserve_full("closed_book")
    result = assess(env)
    assert result.classification == "head_unestablished"
    assert result.authority == "not_established"
    assert result.chain_valid and result.witnesses_consistent
    assert not result.grants_authorization


def test_a_matching_claim_is_continuity_not_authority(env):
    env.reserve_full("closed_book")
    result = assess(env, claim(env, 3))
    assert result.classification == "claim_matches_head"
    assert result.authority == "not_established"
    assert result.claim_in_registry is True
    assert not result.grants_authorization


def test_a_well_formed_claim_with_fixture_provenance_is_not_authority_on_the_public_path(env):
    env.reserve_full("closed_book")
    result = assess(env, claim(env, 3, provenance="test_fixture"))
    assert result.authority == "not_established"
    assert not result.grants_authorization


def test_the_fixture_path_trusts_only_fixture_provenance_and_still_grants_nothing(env):
    env.reserve_full("closed_book")
    args = (env.root, env.auth, env.anchors, env.witnesses)
    fixture = _assess_with_fixture_trust(*args, claim(env, 3, provenance="test_fixture"))
    assert fixture.authority == "fixture_only" and fixture.classification == "claim_matches_head"
    assert not fixture.grants_authorization
    not_fixture = _assess_with_fixture_trust(*args, claim(env, 3))
    assert not_fixture.authority == "not_established"


def test_the_fixture_path_does_not_trust_a_claim_that_does_not_match(env):
    at_head_five(env)
    args = (env.root, env.auth, env.anchors, env.witnesses)
    stale = _assess_with_fixture_trust(*args, claim(env, 3, provenance="test_fixture"))
    assert stale.classification == "registry_ahead_of_claim"
    assert stale.authority == "not_established"


# ---- wrong claims --------------------------------------------------------------------------


def test_a_changed_sequence_is_not_in_the_registry(env):
    env.reserve_full("closed_book")
    assert assess(env, claim(env, 3, event_file_sha256=file_sha(env, 2))).classification == (
        "claim_not_in_registry"
    )


def test_a_changed_event_hash_is_not_in_the_registry(env):
    env.reserve_full("closed_book")
    result = assess(env, claim(env, 3, event_file_sha256=sha("other")))
    assert result.classification == "claim_not_in_registry"
    assert result.claim_in_registry is False


def test_a_claim_for_another_root_is_refused_as_such(env):
    env.reserve_full("closed_book")
    result = assess(env, claim(env, 3, root_id=sha("another root")))
    assert result.classification == "claim_for_another_root"


def test_a_valid_historical_head_presented_as_current_is_stale(env):
    at_head_five(env)
    historical = claim(env, 3)
    result = assess(env, historical)
    assert result.classification == "registry_ahead_of_claim"
    assert result.claim_in_registry is True  # a true statement about the past, not the present
    assert result.authority == "not_established"


def test_a_claim_beyond_the_registry_means_the_registry_is_behind(env):
    at_head_five(env)
    newer = claim(env, 5)
    (env.root / "registry" / "000005.json").chmod(0o644)
    # simulate a registry that lost its tail but kept consistent witnesses
    for number in (5,):
        (env.root / "registry" / f"{number:06d}.json").unlink()
        (env.witnesses / f"{env.root_id}.{number:06d}.witness.json").unlink()
    result = assess(env, newer)
    assert result.classification == "registry_behind_claim"
    assert result.claim_in_registry is False


def test_a_lineage_that_the_registry_does_not_contain_is_a_conflict(env):
    at_head_five(env)
    good = claim(env, 5, previous_sequence=3, previous_event_file_sha256=file_sha(env, 3))
    assert assess(env, good).classification == "claim_matches_head"
    bad = claim(env, 5, previous_sequence=3, previous_event_file_sha256=sha("not event 3"))
    assert assess(env, bad).classification == "claim_lineage_conflict"


def test_a_scope_must_name_a_known_batch_of_the_same_plan(env):
    batch = env.reserve_full("closed_book")
    plan_sha = env.verify().batches[batch].campaign_plan_sha256
    ok = claim(env, 3, batch_id=batch, campaign_plan_sha256=plan_sha)
    assert assess(env, ok).classification == "claim_matches_head"
    unknown = claim(env, 3, batch_id=sha("no batch"), campaign_plan_sha256=plan_sha)
    assert assess(env, unknown).classification == "claim_scope_conflict"
    other_plan = claim(env, 3, batch_id=batch, campaign_plan_sha256=sha("other plan"))
    assert assess(env, other_plan).classification == "claim_scope_conflict"


# ---- chain and witness defects ---------------------------------------------------------------


def test_a_broken_hash_chain_is_invalid_whatever_the_claim_says(env):
    env.reserve_full("closed_book")
    head_hash = file_sha(env, 3)
    middle = env.root / "registry" / "000002.json"
    middle.chmod(0o644)
    middle.write_bytes(middle.read_bytes().replace(b"batch_reserved", b"batch_reservex"))
    result = assess(env, claim(env, 3, event_file_sha256=head_hash))
    assert result.classification == "chain_invalid"
    assert result.chain_valid is False
    assert not result.grants_authorization


def test_the_correct_expected_head_over_a_corrupt_chain_is_not_enough(env):
    env.reserve_full("closed_book")
    attestation = claim(env, 3)  # the true head, recorded before the damage
    event = env.root / "registry" / "000002.json"
    previous = json.loads(event.read_bytes())["previous_event_sha256"]
    event.chmod(0o644)
    event.write_bytes(event.read_bytes().replace(previous.encode(), sha("forged").encode()))
    result = assess(env, attestation)
    assert result.classification == "chain_invalid"
    assert result.authority == "not_established"


def test_a_head_without_its_witness_is_reported_as_a_witness_defect(env):
    env.reserve_full("closed_book")
    (env.witnesses / f"{env.root_id}.000003.witness.json").unlink()
    result = assess(env, claim(env, 3))
    assert result.classification == "witness_inconsistent"
    assert result.witnesses_consistent is False
    assert result.unwitnessed_head is True
    assert result.chain_valid is True


def test_a_missing_middle_witness_is_a_witness_defect(env):
    at_head_five(env)
    (env.witnesses / f"{env.root_id}.000002.witness.json").unlink()
    result = assess(env, claim(env, 5))
    assert result.classification == "witness_inconsistent"
    assert result.witnesses_consistent is False and result.unwitnessed_head is False


def test_registry_ahead_of_its_witnesses_is_a_witness_defect(env):
    at_head_five(env)
    for number in (4, 5):
        (env.witnesses / f"{env.root_id}.{number:06d}.witness.json").unlink()
    result = assess(env, claim(env, 5))
    assert result.classification == "witness_inconsistent"
    assert result.chain_valid is True


def test_witnesses_ahead_of_the_registry_are_a_witness_defect(env):
    at_head_five(env)
    for number in (4, 5):
        (env.root / "registry" / f"{number:06d}.json").unlink()
    shutil.rmtree(env.root / "batches" / env.batch_id("confirmatory"))  # keep the rest sound
    result = assess(env)
    assert result.classification == "witness_inconsistent"


def test_rolling_back_only_the_registry_is_detected(env, tmp_path):
    env.reserve_full("closed_book")
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    snapshot(env, earlier)
    env.reserve_full("confirmatory")
    shutil.rmtree(env.root)
    shutil.copytree(earlier / "root", env.root)  # witnesses stay at head 5
    assert assess(env).classification == "witness_inconsistent"


def test_rolling_back_only_the_witnesses_is_detected(env, tmp_path):
    env.reserve_full("closed_book")
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    snapshot(env, earlier)
    env.reserve_full("confirmatory")
    shutil.rmtree(env.witnesses)
    shutil.copytree(earlier / "witnesses", env.witnesses)  # registry stays at head 5
    assert assess(env).classification == "witness_inconsistent"


def test_a_diverged_event_with_an_old_witness_is_a_conflict(env):
    env.reserve_full("closed_book")
    before = claim(env, 3)
    path = env.root / "registry" / "000003.json"
    event = layout.RegistryEvent.model_validate_json(path.read_bytes())
    path.chmod(0o644)
    path.write_bytes(
        layout.serialize(event.model_copy(update={"created_utc": "2027-01-01T00:00:00Z"}))
    )
    result = assess(env, before)
    assert result.classification == "witness_inconsistent"
    assert result.claim_in_registry is False  # the registry no longer holds the claimed bytes


# ---- the limit this layer cannot remove ------------------------------------------------------


def test_a_coordinated_rollback_is_invisible_without_an_independently_retained_head(env, tmp_path):
    """Registry, witnesses and manifests share one data root, so an editor can rewind all three."""
    env.reserve_full("closed_book")
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    snapshot(env, earlier)
    before = assess(env)
    env.reserve_full("confirmatory")
    retained = claim(env, 5)  # kept somewhere the editor could not reach
    restore(env, earlier)
    after = assess(env)
    # indistinguishable from the honest registry as it was at head 3
    assert (after.classification, after.registry_head_sequence, after.problems) == (
        before.classification,
        before.registry_head_sequence,
        before.problems,
    )
    assert after.authority == "not_established"
    # only a claim retained independently exposes it, and that claim is still only a claim
    exposed = assess(env, retained)
    assert exposed.classification == "registry_behind_claim"
    assert not exposed.grants_authorization


def test_rolling_back_to_genesis_is_indistinguishable_from_a_fresh_root(env):
    env.reserve_full("closed_book")
    retained = claim(env, 3)
    shutil.rmtree(env.root / "batches")
    for number in (3, 2):
        path = env.root / "registry" / f"{number:06d}.json"
        path.chmod(0o644)
        path.unlink()
    shutil.rmtree(env.witnesses)
    fresh = assess(env)
    assert fresh.classification == "genesis_anchored"
    assert assess(env, retained).classification == "registry_behind_claim"


# ---- nothing here grants anything, and nothing executes it -----------------------------------


def test_no_outcome_ever_grants_authorization(env):
    env.reserve_full("closed_book")
    outcomes = [
        assess(env),
        assess(env, claim(env, 3)),
        assess(env, claim(env, 3, event_file_sha256=sha("x"))),
        _assess_with_fixture_trust(
            env.root, env.auth, env.anchors, env.witnesses, claim(env, 3, provenance="test_fixture")
        ),
    ]
    assert all(o.grants_authorization is False for o in outcomes)


def _functions_named(tree: ast.AST, name: str) -> int:
    return sum(
        1
        for node in ast.walk(tree)
        if (isinstance(node, ast.Name) and node.id == name)
        or (isinstance(node, ast.Attribute) and node.attr == name)
    )


def test_the_fixture_trust_path_has_no_caller_in_the_package():
    users = []
    for path in SRC.rglob("*.py"):
        if path.name == "head.py":
            continue
        if _functions_named(
            ast.parse(path.read_text(encoding="utf-8")), "_assess_with_fixture_trust"
        ):
            users.append(path.name)
    assert users == []


def test_execution_and_readiness_code_do_not_use_the_head_contracts():
    banned = {"HeadAttestation", "assess_head", "parse_attestation", "attestation_sha256"}
    for package in ("generation", "readiness", "evaluation", "retrieval"):
        for path in (SRC / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            used = {n for n in banned if _functions_named(tree, n)}
            assert not used, f"{path.name} uses {used}"
