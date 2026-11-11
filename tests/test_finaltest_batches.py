"""Batch manifest, authorization citation and the boundaries the batch layout must not cross."""

import ast
import hashlib
import json
import typing
from pathlib import Path

import pytest
from finaltest_campaign_support import CONFIGS, SEEDS, population, sha
from pydantic import ValidationError

from biorag_repro.finaltest import batches, layout
from biorag_repro.finaltest.batches import (
    BatchManifest,
    batch_id_for,
    make_manifest,
    observed_state_sha256,
    ownership_sha256,
)
from biorag_repro.finaltest.campaign import build_plan, plan_sha256
from biorag_repro.generation.attempts import RunPhase

ROOT = sha("root")
NOW = "2026-01-01T00:00:00Z"


def plan():
    return build_plan(population(), CONFIGS, SEEDS)


def test_batch_id_is_the_canonical_digest_of_root_plan_and_arm():
    payload = {
        "kind": "final_test_batch",
        "root_id": ROOT,
        "campaign_plan_sha256": sha("p"),
        "matrix_arm": "closed_book",
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert batch_id_for(ROOT, sha("p"), "closed_book") == hashlib.sha256(text.encode()).hexdigest()
    ids = {
        batch_id_for(ROOT, sha("p"), "closed_book"),
        batch_id_for(sha("r2"), sha("p"), "closed_book"),
        batch_id_for(ROOT, sha("p2"), "closed_book"),
        batch_id_for(ROOT, sha("p"), "confirmatory"),
    }
    assert len(ids) == 4


def test_ownership_changes_with_benchmark_arm_and_configuration():
    base = ownership_sha256(sha("b"), "closed_book", sha("c"))
    assert base != ownership_sha256(sha("b2"), "closed_book", sha("c"))
    assert base != ownership_sha256(sha("b"), "confirmatory", sha("c"))
    assert base != ownership_sha256(sha("b"), "closed_book", sha("c2"))


def test_manifest_is_structural_and_binds_the_planned_batch():
    p = plan()
    m = make_manifest(ROOT, p, "confirmatory", NOW)
    batch = p.batches[0]
    assert (m.expected_count, m.key_set_sha256) == (batch.expected_count, batch.key_set_sha256)
    assert m.batch_id == batch_id_for(ROOT, plan_sha256(p), "confirmatory")
    fields = set(BatchManifest.model_fields)
    assert not {f for f in fields if "roster" in f or "readiness" in f or "authoriz" in f}
    assert layout.serialize(m) == layout.serialize(make_manifest(ROOT, p, "confirmatory", NOW))


def test_manifest_identity_is_checked_and_unknown_fields_are_refused():
    m = make_manifest(ROOT, plan(), "closed_book", NOW)
    data = m.model_dump()
    with pytest.raises(ValidationError, match="batch_id"):
        BatchManifest.model_validate(data | {"batch_id": sha("other")})
    with pytest.raises(ValidationError):
        BatchManifest.model_validate(data | {"readiness_report_sha256": sha("r")})
    with pytest.raises(ValidationError):
        BatchManifest.model_validate(data | {"expected_count": 0})


def test_the_manifest_is_a_frozen_value():
    m = make_manifest(ROOT, plan(), "closed_book", NOW)
    with pytest.raises(ValidationError):
        m.expected_count = 1  # type: ignore[misc]


def test_observed_state_digest_distinguishes_manifest_and_materialization():
    values = {
        observed_state_sha256(None, False),
        observed_state_sha256(sha("m"), False),
        observed_state_sha256(sha("m"), True),
    }
    assert len(values) == 3


def test_run_phase_still_refuses_final_test():
    assert typing.get_args(RunPhase) == ("development", "qualification")


def test_no_finaltest_module_creates_a_run_store_or_calls_the_guard():
    forbidden = {"RunStore", "authorize_final_test", "FinalTestAuthorization", "driver"}
    package = Path(batches.__file__).parent
    for path in package.glob("*.py"):
        names = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.ImportFrom | ast.Import):
                for alias in node.names:
                    names.update(alias.name.split("."))
                names.update((getattr(node, "module", None) or "").split("."))
        # start.py is the one module that calls the guard; it still never names a RunStore.
        # store.py subclasses RunStore only to refuse its create, open and recover operations.
        allowed = {
            "start.py": {"authorize_final_test", "FinalTestAuthorization"},
            "store.py": {"RunStore"},
        }.get(path.name, set())
        assert not names & (forbidden - allowed), (path.name, names & (forbidden - allowed))
