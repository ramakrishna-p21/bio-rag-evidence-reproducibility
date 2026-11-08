import hashlib
import json
from pathlib import Path

import pytest
from finaltest_support import CORPUS, RELEASE, authorization
from pydantic import ValidationError

from biorag_repro.core.hashing import canonical_json_bytes
from biorag_repro.evaluation.analysis_models import MATRIX_ID
from biorag_repro.finaltest import layout


def test_matrix_id_is_the_frozen_one():
    assert layout.MATRIX_ID == MATRIX_ID


def test_authorization_digest_is_the_canonical_json_digest(tmp_path):
    auth = authorization(tmp_path, "final_test")
    expected = hashlib.sha256(canonical_json_bytes(auth.model_dump(mode="json"))).hexdigest()
    assert layout.authorization_sha256(auth) == expected


def test_root_identity_is_derived_from_the_approved_fields(tmp_path):
    auth = authorization(tmp_path, "final_test")
    payload = {
        "layout_version": "final_test_v1",
        "root_kind": "final_test",
        "authorized_root": auth.authorized_root,
        "corpus_content_sha256": CORPUS,
        "benchmark_release_sha256": RELEASE,
        "matrix_id": "experiment_plan_v1",
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert layout.root_id_for(auth) == hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize(
    "changes",
    [
        {"root_kind": "rehearsal"},
        {"authorized_root": "/elsewhere/final_test_v1"},
        {"corpus_content_sha256": "a" * 64},
        {"benchmark_release_sha256": "b" * 64},
    ],
)
def test_every_approved_scope_field_changes_the_root_identity(tmp_path, changes):
    base = authorization(tmp_path, "final_test")
    assert layout.root_id_for(authorization(tmp_path, "final_test", **changes)) != (
        layout.root_id_for(base)
    )


def test_storage_label_is_approved_but_does_not_name_the_root(tmp_path):
    base = authorization(tmp_path, "final_test")
    other = authorization(tmp_path, "final_test", storage_label="another label")
    assert layout.authorization_sha256(base) != layout.authorization_sha256(other)
    assert layout.root_id_for(base) == layout.root_id_for(other)


@pytest.mark.parametrize(
    "changes",
    [
        {"layout_version": "final_test_v2"},
        {"root_kind": "production"},
        {"matrix_id": "other"},
        {"authorized_root": "relative/final_test_v1"},
        {"authorized_root": "/a/../b/final_test_v1"},
        {"authorized_root": "/a/b/final_test_v1/"},
        {"storage_label": " "},
        {"corpus_content_sha256": "A" * 64},
        {"unexpected": 1},
    ],
)
def test_authorization_rejects_unsupported_values(tmp_path, changes):
    with pytest.raises(ValidationError):
        authorization(tmp_path, "final_test", **changes)


def test_event_identity_excludes_the_informational_timestamp():
    kwargs = {
        "root_id": "1" * 64,
        "sequence": 1,
        "event_type": "root_initialized",
        "previous_event_sha256": None,
        "evidence": [layout.EvidenceCitation(role="root_file", sha256="2" * 64)],
    }
    one = layout.make_event(created_utc="2026-01-01T00:00:00Z", **kwargs)
    two = layout.make_event(created_utc="2030-05-05T10:00:00Z", **kwargs)
    assert one.identity_sha256 == two.identity_sha256
    assert layout.serialize(one) != layout.serialize(two)


def test_event_identity_is_checked_on_validation():
    event = layout.make_event(
        root_id="1" * 64,
        sequence=1,
        event_type="root_initialized",
        previous_event_sha256=None,
        evidence=[],
        created_utc="2026-01-01T00:00:00Z",
    )
    forged = event.model_dump() | {"identity_sha256": "f" * 64}
    with pytest.raises(ValidationError):
        layout.RegistryEvent.model_validate(forged)


@pytest.mark.parametrize(
    ("sequence", "previous"),
    [(1, "3" * 64), (2, None)],
    ids=["genesis-with-previous", "no-previous"],
)
def test_genesis_and_successor_previous_hash_rules(sequence, previous):
    with pytest.raises(ValidationError):
        layout.make_event(
            root_id="1" * 64,
            sequence=sequence,
            event_type="root_initialized",
            previous_event_sha256=previous,
            evidence=[],
            created_utc="2026-01-01T00:00:00Z",
        )


def test_event_types_are_the_operational_and_the_reserved_future_ones():
    assert layout.OPERATIONAL_EVENTS == (
        "root_initialized",
        "batch_reserved",
        "batch_materialized",
        "batch_closed_incomplete",
    )
    assert layout.FUTURE_EVENTS == (
        "batch_inputs_bound",
        "batch_authorized",
        "batch_resumed",
        "batch_completed",
    )
    assert layout.EVENT_TYPES == layout.OPERATIONAL_EVENTS + layout.FUTURE_EVENTS


def test_an_unknown_event_type_is_rejected():
    with pytest.raises(ValidationError):
        layout.make_event(
            root_id="1" * 64,
            sequence=1,
            event_type="batch_approved",
            previous_event_sha256=None,
            evidence=[],
            created_utc="2026-01-01T00:00:00Z",
        )


def test_timestamps_must_be_utc():
    with pytest.raises(ValidationError):
        layout.make_event(
            root_id="1" * 64,
            sequence=1,
            event_type="root_initialized",
            previous_event_sha256=None,
            evidence=[],
            created_utc="2026-01-01T00:00:00+02:00",
        )


def test_layout_constants_are_relative_logical_paths():
    for name in (layout.ROOT_FILE, layout.REGISTRY_DIR, layout.LOCK_FILE):
        assert not Path(name).is_absolute() and ".." not in name
