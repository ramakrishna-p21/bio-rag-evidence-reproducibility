"""Handoff accounting over a genuine JATS / PDF_DOCLING evidence pair, and the NA cell bytes.

Everything is synthetic. The PDF_DOCLING Stage A judgment is a labelled test fixture (reviewer
"synthetic test fixture"); it is not a benchmark judgment and no row here is a result.
"""

import csv
import io
from functools import cache

from test_analysis_export import failed_for, observation, result_for, roster_of
from test_chunking import build as build_jats
from test_chunking import native_table, ref, selection, settings
from test_stage_a import judgment, make_item, occurrence_ids
from test_stage_b import deliver, finish, prose_xml

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.documents.chunking import chunk_document
from biorag_repro.evaluation import analysis_handoff as ah
from biorag_repro.evaluation.analysis_storage import publish_table
from biorag_repro.evaluation.analysis_table import (
    build_table,
    parse_csv,
    primary_pairs,
    serialize_csv,
)
from biorag_repro.evaluation.models import ComponentLocus, Placement, SetLocus
from biorag_repro.generation.environment import GitState
from biorag_repro.retrieval.models import ConfigurationRef

CODE = GitState(status="ok", commit="a" * 40, dirty=False, dirty_paths=0)
COMPONENTS = [("subject", "ClusterONE"), ("relation", "Score"), ("answer", "Entry")]
SENTENCE = "ClusterONE reported a Score of Entry in this analysis."
FIXTURE_REVIEWER = "synthetic test fixture"


@cache
def documents():
    """One JATS and one PDF_DOCLING document of paper P001 that both contain the components."""
    jats, _, _, _, jats_chunks = build_jats(
        prose_xml(SENTENCE), counter=lambda s: len(s.split()) * 11
    )
    pdf = native_table()
    selected = selection().model_copy(update={"representation": "PDF_DOCLING"})
    config = settings().model_copy(update={"chunk_tokens": 192})
    config_ref = ConfigurationRef(
        artifact=ref("configuration"),
        resolved_sha256=sha256_bytes(canonical_json_bytes(config.model_dump())),
    )
    pdf_chunks = chunk_document(pdf, selected, config_ref, config, lambda s: len(s.split()) * 3)
    return jats, jats_chunks, pdf, pdf_chunks


def worlds(item_id):
    """The same benchmark item seen through both representations (JATS world, PDF world)."""
    jats, jats_chunks, pdf, pdf_chunks = documents()
    item = make_item(jats, COMPONENTS, item_id=item_id)
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


def pdf_chunk_with_evidence(pdf_world):
    return next(iter(pdf_world.chunks.values()))  # the first table-row chunk holds all three


def pdf_chunk_without_evidence(pdf_world):
    return list(pdf_world.chunks.values())[1]


def pair_for(item_id, *, pdf="delivered", pdf_answer="1", pdf_failed=False):
    """JATS delivered and correct; the PDF side as asked."""
    jats_world, pdf_world = worlds(item_id)
    item = jats_world.item
    jats = observation(jats_world, case=deliver(jats_world, [jats_world.chunk("ClusterONE")]))
    kwargs = {"condition": "PDF_DOCLING"}
    kwargs["result"] = (
        failed_for(item, representation_or_condition="PDF_DOCLING")
        if pdf_failed
        else result_for(item, answer=pdf_answer, representation_or_condition="PDF_DOCLING")
    )
    if pdf == "no_stage_a_record":
        kwargs["evidence"] = None
    else:
        chunk = (
            pdf_chunk_with_evidence(pdf_world)
            if pdf == "delivered"
            else pdf_chunk_without_evidence(pdf_world)
        )
        kwargs["case"] = deliver(pdf_world, [chunk])
    return [jats, observation(pdf_world, **kwargs)]


def mixed_table_observations():
    return [
        *pair_for("q_001"),
        *pair_for("q_002", pdf="evidence_lost", pdf_answer="2"),
        *pair_for("q_003", pdf_failed=True),
        *pair_for("q_004", pdf="no_stage_a_record"),
    ]


def published(tmp_path, observations):
    table = build_table(observations, roster_of(observations))
    csv_ref, _inventory, build = publish_table(table, tmp_path)
    return table, csv_ref, build


def accounting(handoff, name):
    return next(o for o in handoff.accounting if o.outcome == name)


# ---- genuine pair: both representations carry a real Stage A/B judgment ----------------------


def test_the_pdf_world_is_a_genuine_pdf_docling_judgment_not_a_relabelled_jats_row(tmp_path):
    _, pdf_world = worlds("q_001")
    assert pdf_world.document.representation == "PDF_DOCLING"
    assert pdf_world.stage_a.representation == "PDF_DOCLING"
    assert pdf_world.stage_a.basis == "reviewer" and pdf_world.stage_a.reviewer == FIXTURE_REVIEWER
    table, _, _ = published(tmp_path, pair_for("q_001"))
    jats, pdf = table.rows
    assert (jats.condition, pdf.condition) == ("JATS", "PDF_DOCLING")
    assert (jats.stage_a_label, pdf.stage_a_label) == ("preserved", "preserved")
    assert (jats.stage_a_artifact, jats.stage_b_artifact) != (
        pdf.stage_a_artifact,
        pdf.stage_b_artifact,
    )
    assert (jats.evidence_delivered, pdf.evidence_delivered) == ("yes", "yes")


def test_a_fully_evaluable_pair_is_a_complete_case_for_both_outcomes(tmp_path):
    table, _, build = published(tmp_path, pair_for("q_001"))
    handoff = ah.build_handoff(build, tmp_path, code=CODE, rehearsal=True)
    for name in ("evidence_delivered", "answer_correct"):
        outcome = accounting(handoff, name)
        assert outcome.complete_item_ids == ["q_001"]
        assert outcome.state_counts == {"JATS|yes": 1, "PDF_DOCLING|yes": 1}
        assert [(p.paper_id, p.eligible, p.complete) for p in outcome.papers] == [("P001", 1, 1)]
    assert sorted(primary_pairs(table.rows, "development")) == ["q_001"]


def test_mixed_success_and_failure_and_outcome_specific_missingness(tmp_path):
    table, _, build = published(tmp_path, mixed_table_observations())
    # What each representation must have produced, written out from the construction above.
    by_item = {r.item_id: {} for r in table.rows}
    for r in table.rows:
        by_item[r.item_id][r.condition] = (r.evidence_delivered, r.answer_correct)
    assert by_item == {
        "q_001": {"JATS": ("yes", "yes"), "PDF_DOCLING": ("yes", "yes")},
        "q_002": {"JATS": ("yes", "yes"), "PDF_DOCLING": ("no", "no")},
        "q_003": {"JATS": ("yes", "yes"), "PDF_DOCLING": ("yes", "missing")},
        "q_004": {"JATS": ("yes", "yes"), "PDF_DOCLING": ("non_evaluable", "yes")},
    }
    handoff = ah.build_handoff(build, tmp_path, code=CODE, rehearsal=True)
    delivered = accounting(handoff, "evidence_delivered")
    answer = accounting(handoff, "answer_correct")
    # Outcome-specific: q_003 drops out of the answer outcome only, q_004 out of delivery only.
    assert delivered.complete_item_ids == ["q_001", "q_002", "q_003"]
    assert answer.complete_item_ids == ["q_001", "q_002", "q_004"]
    assert delivered.state_counts == {
        "JATS|yes": 4,
        "PDF_DOCLING|no": 1,
        "PDF_DOCLING|non_evaluable": 1,
        "PDF_DOCLING|yes": 2,
    }
    assert answer.state_counts == {
        "JATS|yes": 4,
        "PDF_DOCLING|missing": 1,
        "PDF_DOCLING|no": 1,
        "PDF_DOCLING|yes": 2,
    }
    assert [(p.eligible, p.complete) for p in delivered.papers] == [(4, 3)]
    assert [(p.eligible, p.complete) for p in answer.papers] == [(4, 3)]
    # The failed PDF attempt stays an operational failure, not an incorrect answer.
    failed = next(i for i in answer.eligible_items if i.item_id == "q_003").pdf_docling
    assert failed is not None and failed.execution_status == "operational_failure"
    # The published pair definition is not the same as either outcome's complete cases.
    assert sorted(primary_pairs(table.rows, "development")) == ["q_001", "q_002", "q_003", "q_004"]


def test_the_mixed_pair_survives_publication_and_read_back(tmp_path):
    table, _, build = published(tmp_path, mixed_table_observations())
    handoff = ah.build_handoff(build, tmp_path, code=CODE, rehearsal=True)
    _, handoff_build = ah.publish_handoff(handoff, tmp_path)
    result = ah.read_handoff(handoff_build, tmp_path)
    assert result.official is False and result.handoff == handoff
    assert result.rows == table.rows


# ---- NA: the missing token and the one data value that equals it -------------------------------


def raw_cells(tmp_path, csv_ref):
    """Cells as an independent csv.reader sees them, keyed by observation_key."""
    text = (tmp_path / csv_ref.logical_ref).read_bytes().decode("utf-8")
    header, *rows = list(csv.reader(io.StringIO(text, newline="")))
    return header, {
        r[header.index("observation_key")]: dict(zip(header, r, strict=True)) for r in rows
    }


def test_na_cells_in_the_published_bytes_follow_the_column_contract(tmp_path):
    table, csv_ref, build = published(tmp_path, mixed_table_observations())
    handoff = ah.build_handoff(build, tmp_path, code=CODE, rehearsal=True)
    contract = handoff.csv_contract
    _, cells = raw_cells(tmp_path, csv_ref)
    nullable = {c.name for c in contract.columns if c.nullable}
    for row in table.rows:
        raw = cells[row.observation_key]
        for name, cell in raw.items():
            value = getattr(row, name)
            if name in nullable:
                # In a nullable column NA is exactly the missing value.
                assert (cell == contract.missing_token) == (value is None), name
            elif name in contract.na_value_columns:
                # The tri-state column is never null; NA is its 'not evaluable' value.
                assert value is not None and cell == value, name
            else:
                assert cell != contract.missing_token, name
    # Both meanings occur in this table, so the comparison above is not vacuous.
    pdf_without_record = next(
        r for r in table.rows if r.item_id == "q_004" and r.condition == "PDF_DOCLING"
    )
    raw = cells[pdf_without_record.observation_key]
    assert (
        raw["correct_without_evidence"] == "NA"
        and pdf_without_record.correct_without_evidence == "NA"
    )
    assert raw["stage_a_artifact"] == "NA" and pdf_without_record.stage_a_artifact is None
    assert any(c["correct_without_evidence"] in {"TRUE", "FALSE"} for c in cells.values())
    assert any(c["stage_a_artifact"] != "NA" for c in cells.values())


def test_a_non_evaluable_na_value_is_not_read_back_as_a_null(tmp_path):
    table, csv_ref, _ = published(tmp_path, mixed_table_observations())
    data = (tmp_path / csv_ref.logical_ref).read_bytes()
    rows = parse_csv(data)
    assert rows == table.rows and serialize_csv(rows) == data
    states = {r.correct_without_evidence for r in rows}
    assert "NA" in states and None not in states
