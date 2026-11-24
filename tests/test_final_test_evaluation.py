"""Final-test evaluation on synthetic records only: scope gate, authorization, Stage B review and
reuse, masked Stage C adjudication, the final-test table, the official handoff and R.

No protected item, gold, answer or context is used. The "official" R proof in the last test is a
test shim standing in for the Python verifier; it exercises R's handling of the final-test origin,
not provenance.
"""

import csv
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from test_analysis_export import failed_for, link, observation, result_for
from test_analysis_handoff_pairing import (
    COMPONENTS,
    FIXTURE_REVIEWER,
    deliver,
    documents,
    pdf_chunk_with_evidence,
)
from test_chunking import build
from test_stage_a import (
    CANONICAL,
    PROSE_COMPONENTS,
    ComponentLocus,
    Placement,
    SetLocus,
    judgment,
    make_item,
    occurrence_ids,
)
from test_stage_b import P_A, P_B, P_C, Case, finish, prose_xml

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import sha256_file
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.evaluation import analysis_handoff as ah
from biorag_repro.evaluation import final_test, stage_b, stage_c
from biorag_repro.evaluation.analysis_models import PlannedRoster, RosterArm, RosterItem
from biorag_repro.evaluation.analysis_storage import publish_table
from biorag_repro.evaluation.analysis_table import Evidence, Table, build_table
from biorag_repro.evaluation.seeds import APPROVED_REGISTRY_SHA256, read_analysis_seed
from biorag_repro.evaluation.stage_a import IneligibleItem, check_item, final_test_scope
from biorag_repro.generation.environment import GitState
from biorag_repro.retrieval.storage import record_ref

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "analysis"
AT = "2026-11-25T00:00:00Z"
needs_r = pytest.mark.skipif(
    shutil.which("Rscript") is None or not (ANALYSIS / "renv" / "library").is_dir(),
    reason="R and the analysis renv library are needed",
)


# ---- scope gate ------------------------------------------------------------------------------


def test_final_test_items_are_admitted_only_inside_the_final_test_scope():
    document, _, _, _, _ = build(prose_xml(P_A, P_B, P_C), counter=lambda s: len(s.split()) * 11)
    final = make_item(document, PROSE_COMPONENTS, split="final_test")
    development = make_item(document, PROSE_COMPONENTS)
    with pytest.raises(IneligibleItem):
        check_item(final, document, CANONICAL)
    check_item(development, document, CANONICAL)
    with final_test_scope():
        check_item(final, document, CANONICAL)
        with pytest.raises(IneligibleItem):  # development items never enter a final-test run
            check_item(development, document, CANONICAL)
    with pytest.raises(IneligibleItem):  # the scope ends with its block
        check_item(final, document, CANONICAL)


# ---- authorization ---------------------------------------------------------------------------


def approved(tmp_path, *, pending=False, **changes):
    repo = tmp_path / "repo"
    texts = {}
    for name in (final_test.PROTOCOL_FILE, final_test.CUSTODY_FILE):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        status = "PENDING RESEARCHER APPROVAL" if pending else "APPROVED"
        path.write_text(f"# text\n\n**Status: {status}.**\n")
        texts[name] = sha256_file(path)
    record = {
        "schema_version": "final_test_evaluation_v1",
        "decision": "approved",
        "approved_by": final_test.RESEARCHER,
        "approved_utc": AT,
        "protocol_sha256": texts[final_test.PROTOCOL_FILE],
        "custody_exception_sha256": texts[final_test.CUSTODY_FILE],
        "campaign_id": "synthetic_campaign",
        "run_id": "run_synthetic",
        "drive_manifest_sha256": "1" * 64,
        "head_request_sha256": "2" * 64,
        "new_head_sha256": "3" * 64,
        "observations": 4575,
        "evaluation_source_revision": "c" * 40,
        "reliability_design": "synthetic",
        "ai_assistance": "synthetic",
    } | changes
    path = repo / final_test.AUTHORIZATION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path, sha256_file(path), repo


def test_authorization_is_required_exact_and_approved(tmp_path):
    path, digest, repo = approved(tmp_path)
    value, ref = final_test.read_authorization(path, digest, repo)
    assert value.run_id == "run_synthetic" and ref.sha256 == digest
    with pytest.raises(final_test.EvaluationRefused, match="no evaluation authorization"):
        final_test.read_authorization(tmp_path / "absent.json", digest, repo)
    with pytest.raises(final_test.EvaluationRefused, match="trusted SHA-256"):
        final_test.read_authorization(path, "0" * 64, repo)
    (repo / final_test.PROTOCOL_FILE).write_text("edited after approval\n")
    with pytest.raises(final_test.EvaluationRefused, match="differs from the approved text"):
        final_test.read_authorization(path, digest, repo)


def test_pending_texts_or_a_non_approval_are_refused(tmp_path):
    path, digest, repo = approved(tmp_path / "a", pending=True)
    with pytest.raises(final_test.EvaluationRefused, match="pending approval"):
        final_test.read_authorization(path, digest, repo)
    path, digest, repo = approved(tmp_path / "b", decision="draft")
    with pytest.raises(final_test.EvaluationRefused, match="not an approval"):
        final_test.read_authorization(path, digest, repo)


# ---- synthetic final-test worlds -------------------------------------------------------------


def final_worlds(item_id):
    """Both representations of one synthetic final-test item (JATS prose, PDF table)."""
    jats, jats_chunks, pdf, pdf_chunks = documents()
    item = make_item(jats, COMPONENTS, item_id=item_id, split="final_test")
    data = item.model_dump()
    data["pdf_refs"][0].update(artifact_id=pdf.source.artifact_id, source_sha256=pdf.source.sha256)
    item = BenchmarkItem.model_validate(data)
    table_node = next(n.node_id for n in pdf.nodes if n.kind == "table")
    placement = Placement(
        container_node_id=table_node,
        components=[
            ComponentLocus(role=r, text=t, occurrence_ids=occurrence_ids(pdf, t))
            for r, t in COMPONENTS
        ],
    )
    reviewed = judgment(
        item,
        pdf,
        "preserved",
        sets=[SetLocus(set_index=0, placements=[placement])],
        reviewer=FIXTURE_REVIEWER,
    )
    return finish(jats, jats_chunks, item), finish(pdf, pdf_chunks, item, stage_a=reviewed)


def pdf_review_record(status="delivered"):
    if status == "unresolved":
        return {"status": "unresolved", "unresolved_reason": "two defensible readings",
                "rationale": "R5", "at_utc": AT}  # fmt: skip
    return {
        "status": "delivered",
        "rationale": "R1, R7: the row states score and entry for ClusterONE",
        "at_utc": AT,
        "citations": [
            {"chunk": "C1", "quote": text, "set_index": 0, "component_index": n}
            for n, (_, text) in enumerate(COMPONENTS)
        ],
    }


def other_arm(case):
    """The same supplied context reached through another retrieval (another arm)."""
    ranked = case.ranked.model_copy(update={"query_id": "q_other_arm"})
    ctx = case.ctx.model_copy(update={"retrieval": record_ref(ranked)})
    return Case(case.world, ranked, ctx, case.chunks)


# ---- Stage B ---------------------------------------------------------------------------------


def test_stage_b_review_record_becomes_a_validated_v2_judgment_and_is_reused_only_when_identical():
    with final_test_scope():
        _, pdf = final_worlds("q_b01")
        case = deliver(pdf, [pdf_chunk_with_evidence(pdf)])
        inputs = case.inputs()
        assert stage_b.needs_review_v2(stage_b.evaluate(inputs), inputs)
        view, handles = final_test.stage_b_view(inputs)
        shown = json.dumps(view)
        assert "PDF_DOCLING" not in shown and "JATS" not in shown and "drv_" not in shown
        assert set(view) == {"question", "evidence_sets", "chunks"}  # no answer, score or arm
        judged = final_test.judgment_from_record(pdf_review_record(), handles, inputs, "Rama")
        assert (judged.status, judged.rules, judged.basis) == ("delivered", "v2", "reviewer")

        second = other_arm(case)
        assert final_test.reuse_key(second.inputs()) == final_test.reuse_key(inputs)
        reused = final_test.apply_review(judged, inputs, second.inputs())
        assert reused.context == record_ref(second.ctx) and reused.citations == judged.citations
        stage_b.validate_judgment(reused, second.inputs())

        different = deliver(pdf, list(pdf.chunks.values())[:2])
        with pytest.raises(final_test.EvaluationRefused, match="contexts differ"):
            final_test.apply_review(judged, inputs, different.inputs())


def test_stage_b_record_with_a_wrong_quote_or_handle_is_refused():
    with final_test_scope():
        _, pdf = final_worlds("q_b02")
        inputs = deliver(pdf, [pdf_chunk_with_evidence(pdf)]).inputs()
        _, handles = final_test.stage_b_view(inputs)
        record = pdf_review_record()
        record["citations"][0]["quote"] = "Clusterone"
        with pytest.raises(ValueError):
            final_test.judgment_from_record(record, handles, inputs, "Rama")
        record = pdf_review_record()
        record["citations"][0]["chunk"] = "C9"
        with pytest.raises(final_test.EvaluationRefused, match="unknown chunk handle"):
            final_test.judgment_from_record(record, handles, inputs, "Rama")


# ---- Stage C ---------------------------------------------------------------------------------


def test_stage_c_queue_is_masked_and_one_decision_covers_every_matching_answer():
    with final_test_scope():
        jats, _ = final_worlds("q_c01")
    item = jats.item
    results = [
        result_for(item, answer="1 genes", phase="final_test", representation_or_condition=c,
                   repeat_index=r)
        for c in ("JATS", "PDF_DOCLING") for r in (1, 2, 3)
    ] + [result_for(item, answer="1", phase="final_test", matrix_arm="chunk_robustness")]  # fmt: skip
    scores = [stage_c.score(item, r, "v2") for r in results]
    assert scores[0].status == "non_evaluable" and scores[-1].status == "correct"
    answers = {r.observation_key: r.answer for r in results}
    entries, mapping = final_test.stage_c_queue(scores, {item.item_id: item}, 7, answers)
    assert len(entries) == 1 and len(mapping["M0001"]) == 6
    assert not {
        "condition",
        "representation",
        "repeat_index",
        "observation_key",
        "matrix_arm",
    } & set(entries[0])
    assert entries[0]["answers"] == ['"1 genes"']
    decided = final_test.apply_decisions(
        scores, mapping, {"M0001": {"status": "correct", "rationale": "count matches; C1",
                                    "at_utc": AT}}, "Rama",
    )  # fmt: skip
    assert [s.status for s in decided] == ["correct"] * 7
    assert all(s.basis == "reviewer" for s in decided[:6]) and decided[-1].basis == "deterministic"
    kept = final_test.apply_decisions(
        scores, mapping, {"M0001": {"status": "non_evaluable", "rationale": "unit unclear; C7"}},
        "Rama",
    )  # fmt: skip
    assert [s.status for s in kept[:6]] == ["non_evaluable"] * 6  # never forced to incorrect
    with pytest.raises(final_test.EvaluationRefused, match="recorded reason"):
        final_test.apply_decisions(scores, mapping, {"M0001": {"status": "non_evaluable"}}, "R")


# ---- end to end: table, official handoff and R ------------------------------------------------

# item -> (paper, JATS (evidence, answer), PDF (evidence, answer)); evidence: "delivered" (JATS
# prose automatic, PDF reviewed), "empty" (nothing supplied), "unresolved" (reviewed); answer:
# "1" correct, "2" incorrect, "1 genes" adjudicated, "fail" operational failure.
CASES = {
    "q_001": ("P001", ("delivered", "1"), ("delivered", "1")),
    "q_002": ("P001", ("delivered", "1"), ("delivered", "1 genes")),
    "q_003": ("P002", ("delivered", "1"), ("delivered", "fail")),
    "q_004": ("P002", ("delivered", "2"), ("unresolved", "2")),
    "q_005": ("P003", ("empty", "1"), ("delivered", "2")),
    "q_006": ("P003", ("delivered", "1 kg"), ("delivered", "1 kg")),
}
DECISIONS = {"1 genes": ("correct", "count matches; C1"), "1 kg": ("non_evaluable", "C7")}


def _evidence(world, condition, kind):
    if kind == "empty":
        case = deliver(world, [])
    elif condition == "JATS":
        case = deliver(world, [world.chunk("ClusterONE")])
    else:
        case = deliver(world, [pdf_chunk_with_evidence(world)])
    inputs = case.inputs()
    candidate = stage_b.evaluate(inputs)
    if stage_b.needs_review_v2(candidate, inputs):
        _, handles = final_test.stage_b_view(inputs)
        status = "unresolved" if kind == "unresolved" else "delivered"
        judged = final_test.judgment_from_record(pdf_review_record(status), handles, inputs, "R")
    else:
        judged = stage_b.deterministic_judgment(candidate, "v2", inputs)
    return case, Evidence(world.document, CANONICAL, world.stage_a, judged, inputs)


def build_final_table():
    observations, papers = [], {}
    for item_id, (paper, *cells) in CASES.items():
        papers[item_id] = paper
        for world, condition, (kind, answer) in zip(
            final_worlds(item_id), ("JATS", "PDF_DOCLING"), cells, strict=True
        ):
            case, evidence = _evidence(world, condition, kind)
            for repeat in (1, 2, 3):
                coords = {"phase": "final_test", "representation_or_condition": condition,
                          "repeat_index": repeat}  # fmt: skip
                failed = answer == "fail" and repeat == 1
                result = (
                    failed_for(world.item, **coords) if failed
                    else result_for(world.item, answer="1" if answer == "fail" else answer, **coords)
                )  # fmt: skip
                gin = None if failed else link(result, case, provenance="recorded")
                obs = observation(
                    world, case=case, result=result, evidence=evidence, gin=gin,
                    phase="final_test", condition=condition, repeat_index=repeat,
                )  # fmt: skip
                observations.append(obs)
    scores = [stage_c.score(o.item, o.result, "v2") for o in observations]
    items = {o.item.item_id: o.item for o in observations}
    answers = {o.result.observation_key: o.result.answer for o in observations}
    entries, mapping = final_test.stage_c_queue(scores, items, 11, answers)
    decisions = {}
    for entry in entries:
        (answer,) = entry["answers"]
        status, rationale = DECISIONS[json.loads(answer)]
        decisions[entry["handle"]] = {"status": status, "rationale": rationale, "at_utc": AT}
    scores = final_test.apply_decisions(scores, mapping, decisions, "Rama")
    observations = [o.__class__(**{**o.__dict__, "score": s}) for o, s in zip(observations, scores)]
    roster = PlannedRoster(
        scope="partial_development",
        phase="final_test",
        benchmark_version="benchmark_v1",
        corpus_version="corpus_v1",
        declared_by="synthetic",
        basis="synthetic final-test scope",
        arms=[
            RosterArm(
                matrix_arm="confirmatory", conditions=["JATS", "PDF_DOCLING"], repeats=[1, 2, 3]
            )
        ],
        items=[
            RosterItem(
                item_id=i.item_id,
                state="included",
                paper_id=i.paper_id,
                item_sha256=stage_c.item_digest(i),
            )
            for i in sorted(items.values(), key=lambda i: i.item_id)
        ],
    )
    table = build_table(observations, roster)
    rows = [r.model_copy(update={"paper_id": papers[r.item_id]}) for r in table.rows]
    items_ = [i.model_copy(update={"paper_id": papers[i.item_id]}) for i in roster.items]
    roster = PlannedRoster.model_validate(roster.model_copy(update={"items": items_}).model_dump())
    return Table(rows, roster, table.inputs, table.stage_c_rules, table.stage_b_rules)


def official_handoff(root):
    with final_test_scope():
        table = build_final_table()
        _, _, build = publish_table(table, root)
        code = GitState(status="ok", commit="a" * 40, dirty=False, dirty_paths=0)
        template, _ = ah._describe(build, root, code)
    authorization = ArtifactRef(
        artifact_id="evaluation_authorization_" + "4" * 64,
        logical_ref=final_test.AUTHORIZATION_FILE,
        sha256="4" * 64,
    )
    source = ah.SourceRun(
        run_id="run_synthetic", completion=authorization, run_record_sha256="5" * 64,
        resolved_config_sha256=template.arms[0].resolved_config_sha256, raw_config_sha256="6" * 64,
        roster_sha256="7" * 64, matrix_arm="confirmatory", backend_kind="official", observations=36,
    )  # fmt: skip
    custody = {
        "status": "custody_gate_not_established_exception_v1",
        "authorization": authorization.model_dump(),
        "amendment_sha256": "8" * 64,
        "run_id": "run_synthetic",
        "drive_manifest_sha256": "9" * 64,
    }
    fields = template.model_dump() | {
        "input_origin": "verified_official_run",
        "source_runs": [source.model_dump()],
        "scoring_reference": template.scoring_reference
        | {"primary_bootstrap_registry_sha256": APPROVED_REGISTRY_SHA256},
        "awaiting_approval": [],
    }
    with pytest.raises(ValueError, match="custody exception"):
        ah.AnalysisHandoff.model_validate(fields)
    handoff = ah.AnalysisHandoff.model_validate(fields | {"custody_exception": custody})
    return table, handoff, ah._store(handoff, root)


def test_final_test_table_keeps_states_separate_and_needs_the_scope(tmp_path):
    with pytest.raises(ValueError):
        build_final_table()
    with final_test_scope():
        table = build_final_table()
    primary = {
        (r.item_id, r.condition): r
        for r in table.rows
        if r.repeat_index == 1 and r.matrix_arm == "confirmatory"
    }
    assert table.stage_c_rules == "v2" and table.stage_b_rules == "v2"
    assert {r.analysis_role for r in table.rows} == {"confirmatory"}
    assert primary["q_002", "PDF_DOCLING"].answer_correct == "yes"  # adjudicated
    assert primary["q_002", "PDF_DOCLING"].stage_c_basis == "reviewer"
    assert primary["q_003", "PDF_DOCLING"].answer_correct == "missing"  # operational failure
    assert primary["q_004", "PDF_DOCLING"].evidence_delivered == "non_evaluable"  # unresolved B
    assert primary["q_004", "PDF_DOCLING"].answer_correct == "no"  # C independent of B
    assert primary["q_005", "JATS"].evidence_delivered == "no"
    assert primary["q_005", "JATS"].answer_correct == "yes"  # correct without evidence
    assert primary["q_006", "JATS"].answer_correct == "non_evaluable"  # not forced
    with pytest.raises(ah.HandoffError, match="only inside the final-test evaluation"):
        with final_test_scope():
            _, _, build = publish_table(build_final_table(), tmp_path)
        ah._describe(build, tmp_path, GitState(status="ok", commit="a" * 40, dirty=False,
                                               dirty_paths=0))  # fmt: skip


def _oracle(rows, outcome):
    """Item-weighted paired difference on complete cases, computed independently of R."""
    cells = {}
    for r in rows:
        if r.matrix_arm == "confirmatory" and r.repeat_index == 1:
            cells.setdefault(r.item_id, {})[r.condition] = getattr(r, outcome)
    pairs = [
        (v["JATS"] == "yes") - (v["PDF_DOCLING"] == "yes")
        for v in cells.values()
        if {v["JATS"], v["PDF_DOCLING"]} <= {"yes", "no"}
    ]
    return len(pairs), sum(pairs) / len(pairs)


@needs_r
def test_r_runs_the_frozen_analysis_on_an_official_final_test_handoff(tmp_path):
    root = tmp_path / "root"
    table, handoff, (href, _) = official_handoff(root)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"authorization": "approvals/final_test_evaluation.json"}))
    proof = {
        "status": "verified_official_final_test_handoff",
        "source_verification": "fresh_python_verification",
        "observations": 4575,
        "custody": "custody_gate_not_established_exception_v1",
        "handoff": href.model_dump(),
        "csv_sha256": handoff.csv.sha256,
        "scientific_bootstrap_seed": {
            "registry_sha256": APPROVED_REGISTRY_SHA256,
            "record": read_analysis_seed(REPO).model_dump(mode="json"),
        },
    }
    shim = tmp_path / "python-shim"  # test stand-in for the Python verifier, not provenance
    shim.write_text(f"#!{sys.executable}\nprint({json.dumps(json.dumps(proof))})\n")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    out = tmp_path / "r"
    out.mkdir()
    seed = str(read_analysis_seed(REPO).value)
    done = subprocess.run(
        ["Rscript", "run_analysis.R", "run", "--handoff", str(root / href.logical_ref),
         "--handoff-sha256", href.sha256, "--csv", str(root / handoff.csv.logical_ref),
         "--lock-sha256", sha256_file(ANALYSIS / "renv.lock"), "--seed", seed,
         "--seed-label", "final_test", "--out-root", str(out),
         "--official-request", str(request), "--python", str(shim)],
        cwd=ANALYSIS, capture_output=True, text=True, check=False, env=os.environ,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert "official final-test analysis" in done.stdout
    (result,) = [p for p in out.iterdir() if p.name.startswith("final_test_")]
    record = json.loads((result / "run_record.json").read_text())
    assert record["purpose"] == "final_test_confirmatory"
    assert record["scientific_authority"] == "frozen_confirmatory_analysis"
    assert record["seed"]["status"] == "registered_analysis_seed"
    with (result / "coprimary_results.csv").open() as f:
        results = {r["outcome"]: r for r in csv.DictReader(f)}
    for outcome in ("evidence_delivered", "answer_correct"):
        n, rd = _oracle(table.rows, outcome)
        assert int(results[outcome]["n_complete"]) == n
        assert float(results[outcome]["rd"]) == pytest.approx(rd)
        assert int(results[outcome]["boot_n_resamples"]) == 10000
    assert int(results["answer_correct"]["n_eligible"]) == 6
    assert int(results["evidence_delivered"]["bound_n_unknown_pdf"]) == 1  # unresolved Stage B
    assert int(results["answer_correct"]["bound_n_unknown_pdf"]) == 1  # non_evaluable only
    assert int(results["answer_correct"]["bound_n_operational_excluded"]) == 1  # q_003 missing
    verify = subprocess.run(
        ["Rscript", "run_analysis.R", "verify", "--result", str(result),
         "--official-request", str(request), "--python", str(shim)],
        cwd=ANALYSIS, capture_output=True, text=True, check=False, env=os.environ,
    )  # fmt: skip
    assert verify.returncode == 0, verify.stderr
    assert "final_test_confirmatory" in verify.stdout


# ---- Stage A final queue and the evaluator's refusals ----------------------------------------


def test_stage_a_final_queue_holds_every_unit_and_only_inside_the_scope(tmp_path):
    from test_stage_a_review import CREATED, make_world

    from scripts import stage_a_review as sar

    items, sources = make_world(tmp_path, count=4)
    items = [i.model_copy(update={"split": "final_test"}) for i in items]
    sources = {k: sar.Source(items[int(k[0][2:])], s.document, s.root, s.canonical)
               for k, s in sources.items()}  # fmt: skip
    args = {"out": tmp_path / "out", "seed": 5, "papers": ("P001",), "created_utc": CREATED,
            "code": "abc", "final_test": True,
            "authorization": {"path": str(tmp_path / "absent.json"), "sha256": "0" * 64}}  # fmt: skip
    with pytest.raises(IneligibleItem):
        sar.build_packet(sources, items, **args)
    with final_test_scope():
        sar.build_packet(sources, items, **args)
    manifest = json.loads((tmp_path / "out/provenance/session_manifest.json").read_text())
    assert (manifest["status"], manifest["units"], manifest["pairs"]) == (sar.FINAL_STATUS, 8, 4)
    with pytest.raises(sar.ReviewError, match="authorization does not verify"):
        sar.scope_for(tmp_path / "out")  # later commands re-verify the authorization
    with pytest.raises(sar.ReviewError, match="complete first-rater queue"), final_test_scope():
        sar.build_packet(sources, items, **(args | {"out": tmp_path / "o2", "pairs": 2}))


def test_evaluator_reads_nothing_protected_without_an_authorization(tmp_path):
    from scripts import evaluate_final_test as evaluator

    request = tmp_path / "request.json"
    request.write_text(json.dumps({"authorization": str(tmp_path / "absent.json"),
                                   "authorization_sha256": "0" * 64}))  # fmt: skip
    with pytest.raises(evaluator.Refused, match="no evaluation authorization"):
        evaluator.load(request)


def test_batch_verification_fails_closed_on_custody_evidence(tmp_path):
    from types import SimpleNamespace

    from biorag_repro.finaltest.drive_publication import publish_drive_batch
    from scripts import evaluate_final_test as evaluator

    source = tmp_path / "local" / "run_synthetic"
    (source / "anchors").mkdir(parents=True)
    (source / "batch.json").write_text(json.dumps({"run_id": "run_synthetic", "identity": {}}))
    seal = {"files": {"batch.json": sha256_file(source / "batch.json")},
            "previous_anchor_sha256": None}  # fmt: skip
    (source / "anchors" / "000001.json").write_text(json.dumps(seal))
    batch = publish_drive_batch(source, tmp_path / "drive")
    manifest = sha256_file(batch / "drive_manifest.json")

    def ctx(**auth):
        values = {"run_id": "run_synthetic", "drive_manifest_sha256": manifest} | auth
        return SimpleNamespace(authorization=SimpleNamespace(**values), path=lambda name: batch)

    with pytest.raises(evaluator.Refused, match="not the authorized run"):
        evaluator.verify_batch(ctx(run_id="run_other"))
    with pytest.raises(evaluator.Refused, match="differs from the custody exception"):
        evaluator.verify_batch(ctx(drive_manifest_sha256="0" * 64))
    (batch / "batch.json").chmod(0o644)
    (batch / "batch.json").write_text("{}")
    with pytest.raises(evaluator.Refused, match="batch integrity"):
        evaluator.verify_batch(ctx())


def _final_world(tmp_path):
    from test_stage_a_review import make_world

    from scripts import stage_a_review as sar

    items, sources = make_world(tmp_path, count=4)
    items = [i.model_copy(update={"split": "final_test"}) for i in items]
    sources = {k: sar.Source(items[int(k[0][2:])], s.document, s.root, s.canonical)
               for k, s in sources.items()}  # fmt: skip
    return items, sources


def _files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file()}  # fmt: skip


def test_streamed_final_packet_equals_the_in_memory_packet_and_resumes(tmp_path):
    from test_stage_a_review import CREATED

    from scripts import stage_a_review as sar

    items, sources = _final_world(tmp_path)
    auth = {"path": "approvals/final_test_evaluation.json", "sha256": "1" * 64}
    common = {"seed": 5, "created_utc": CREATED, "code": "abc", "authorization": auth}
    with final_test_scope():
        sar.build_packet(sources, items, out=tmp_path / "reference", papers=("P001",),
                         final_test=True, **common)  # fmt: skip
        out = tmp_path / "packet"
        assert sar.build_final_packet(sources, items, out=out, stop_after=3, **common) == 5
        assert not out.exists()  # never a partial packet
        staging = tmp_path / "packet.incomplete"
        staged = sorted((staging / "units").glob("*.json"))
        assert len(staged) == 3
        staged[0].write_text('{"payload": {}, "payload_sha256": "0"}')  # damaged record
        with pytest.raises(sar.ReviewError, match="another plan"):
            sar.build_final_packet(sources, items, out=out, **(common | {"seed": 6}))
        assert sar.build_final_packet(sources, items, out=out, **common) == 0
        with pytest.raises(sar.ReviewError, match="already exists"):
            sar.build_final_packet(sources, items, out=out, **common)
    assert not staging.exists()
    assert _files(out) == _files(tmp_path / "reference")
    manifest = json.loads((out / "provenance/session_manifest.json").read_text())
    assert (manifest["units"], manifest["authorization"]) == (8, auth)


def test_evaluation_refuses_any_revision_but_the_approved_clean_one():
    with pytest.raises(final_test.EvaluationRefused, match="approved evaluation source"):
        final_test.check_source(REPO, "0" * 40)
