"""Known refusals of final-test execution, and the fact that a readiness result unlocks none.

These are behavioural tests of the reviewed entrypoints. The source inventory at the end is
supplemental: it flags a change to the reviewed sites for human review and proves nothing about
routes it does not name. Phase 10 will add the one authorized route, behind
`readiness.guard.authorize_final_test`; until then nothing may accept an authorization.
"""

import ast
import inspect
import typing
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_chunking import build
from test_stage_a import CANONICAL, PROSE_COMPONENTS, make_item
from test_stage_b import P_A, P_B, P_C, prose_xml

import biorag_repro
from biorag_repro.evaluation import analysis_handoff, analysis_table
from biorag_repro.evaluation.stage_a import IneligibleItem, check_item
from biorag_repro.generation import attempts
from biorag_repro.readiness.guard import FinalTestAuthorization
from biorag_repro.readiness.models import Candidate, HostBinding

SRC = Path(biorag_repro.__file__).parent


def authorization() -> FinalTestAuthorization:
    sha = "a" * 64
    return FinalTestAuthorization(
        report_sha256=sha,
        manifest_sha256=sha,
        policy_sha256=sha,
        evaluator_code_sha256=sha,
        report_binding_sha256=sha,
        live_binding_sha256=sha,
        candidate=Candidate(
            head_sha=None,
            scientific_source_identity_sha256=None,
            freeze_record_sha256=None,
            configuration_sha256_by_arm={},
            benchmark_release_sha256=None,
            corpus_content_sha256=None,
            python_lock_sha256=None,
            renv_lock_sha256=None,
            generation_seed_registry_sha256=None,
            benchmark_seed_registry_sha256=None,
        ),
        host=HostBinding(hostname="h", operating_system="Linux", machine="x", gpu_names=[]),
        scope="final_test_batch_start",
        verified_at_utc="2026-11-07T00:00:00.000000Z",
    )


# ---- behavioural refusals ------------------------------------------------------------------


def test_run_phase_type_excludes_final_test():
    assert typing.get_args(attempts.RunPhase) == ("development", "qualification")
    with pytest.raises(ValidationError):
        attempts.RunRecord.model_validate({"phase": "final_test"})


def test_stage_a_refuses_a_final_test_item():
    document, _, _, _, _ = build(prose_xml(P_A, P_B, P_C), counter=lambda s: len(s.split()) * 11)
    item = make_item(document, PROSE_COMPONENTS, split="final_test")
    with pytest.raises(IneligibleItem, match="protected"):
        check_item(item, document, CANONICAL)


def test_analysis_table_refuses_final_test_phase_and_split():
    import test_analysis_export as export

    w = export.world()
    with pytest.raises(ValueError, match="final-test"):
        analysis_table.build_row(export.observation(w, phase="final_test"))


def test_final_test_handoff_is_official_and_names_its_custody_exception():
    field = analysis_handoff.AnalysisHandoff.model_fields["phase"]
    assert typing.get_args(field.annotation) == ("development", "final_test")
    source = inspect.getsource(analysis_handoff.AnalysisHandoff.origin_matches_runs)
    assert "custody_exception is None" in source and "verified_official_run" in source


# ---- a readiness result unlocks nothing ----------------------------------------------------


@pytest.mark.parametrize(
    "function",
    [
        check_item,
        analysis_table.build_row,
        analysis_handoff.build_handoff,
    ],
)
def test_no_protected_entrypoint_accepts_an_authorization(function):
    for name, parameter in inspect.signature(function).parameters.items():
        assert "authoriz" not in name.lower()
        assert "FinalTestAuthorization" not in str(parameter.annotation)


def test_authorization_does_not_change_a_refusal(monkeypatch):
    held = authorization()
    assert held.scope == "final_test_batch_start"
    document, _, _, _, _ = build(prose_xml(P_A, P_B, P_C), counter=lambda s: len(s.split()) * 11)
    item = make_item(document, PROSE_COMPONENTS, split="final_test")
    with pytest.raises(IneligibleItem):
        check_item(item, document, CANONICAL)
    with pytest.raises(ValidationError):
        attempts.RunRecord.model_validate({"phase": "final_test"})


# ---- the Phase 10 boundary -----------------------------------------------------------------


def _modules():
    for path in sorted(SRC.rglob("*.py")):
        yield path, ast.parse(path.read_text())


def test_only_the_readiness_package_refers_to_the_guard():
    users = []
    for path, tree in _modules():
        if "readiness" in path.relative_to(SRC).parts:
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(a.name for a in node.names)]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any("readiness.guard" in n or n == "guard" for n in names):
                users.append(str(path.relative_to(SRC)))
    # No module outside the readiness package uses the guard; any user is an unreviewed
    # integration
    assert users == [], f"{users} use the authorization guard"


REVIEWED_REFERENCES = {
    "evaluation/stage_a.py": {"check_item"},
    "evaluation/analysis_table.py": {"build_row", "roster_items_from_final_test"},
    "evaluation/analysis_models.py": {"analysis_role", "AnalysisRow", "Phase"},
    "evaluation/analysis_handoff.py": {"AnalysisHandoff", "_describe", "derive_accounting"},
}


def _referencing(relative: str) -> set[str]:
    tree = ast.parse((SRC / relative).read_text())
    found = set()
    for node in tree.body:
        if any(isinstance(c, ast.Constant) and c.value == "final_test" for c in ast.walk(node)):
            name = getattr(node, "name", None)
            if name is None and isinstance(node, ast.Assign):
                name = node.targets[0].id  # type: ignore[attr-defined]
            found.add(name)
    return found


@pytest.mark.parametrize("relative", sorted(REVIEWED_REFERENCES))
def test_reviewed_final_test_sites_have_not_changed(relative):
    # Supplemental. A difference means a protected site was added, moved or removed:
    # review the change against the readiness guard before updating this table.
    assert _referencing(relative) == REVIEWED_REFERENCES[relative]


def test_run_types_do_not_name_final_test():
    assert _referencing("generation/attempts.py") == set()
