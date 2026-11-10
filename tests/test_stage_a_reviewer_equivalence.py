"""Human-reviewed Stage A judgments with paraphrased descriptors and mean ± SD table cells."""

import pytest
from pydantic import ValidationError
from test_documents_jats import parse
from test_stage_a import (
    CANONICAL,
    judgment,
    locus,
    make_item,
    node_of,
    with_text,
)

from biorag_repro.core.hashing import HashMismatchError
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.evaluation.models import (
    ComponentLocus,
    Placement,
    SetLocus,
    StageACandidate,
)
from biorag_repro.evaluation.stage_a import _index, automatic_judgment, locate, validate_judgment
from biorag_repro.evaluation.storage import publish_judgment, read_judgment

MEAN_TABLE = (
    "<article><body><p>Unrelated sentence about Collins and 307.8 in prose.</p>"
    "<table-wrap><caption><p>Performance of TOMOC on the original networks. "
    "Results are reported as mean ± SD over 10 runs</p></caption><table>"
    "<thead><tr><th>Dataset</th><th>Precision</th><th>#Predicted</th></tr></thead>"
    "<tbody><tr><th>Yeast</th><td>0.875 ± 0.014</td><td>153.0 ± 6.0</td></tr>"
    "<tr><th>Collins</th><td>0.869 ± 0.016</td><td>307.8 ± 12.9</td></tr></tbody>"
    "</table></table-wrap></body></article>"
)
NO_MEAN_STATEMENT = MEAN_TABLE.replace("mean ± SD", "values ± SD")
DESCRIPTORS = [
    ("subject", "TOMOC on original Collins network"),
    ("relation", "mean predicted-complex count"),
    ("answer", "307.8"),
]
CAPTION = "Performance of TOMOC on the original networks"


def occurrences(document, text, *, cell=True, tag="table", prefix=None):
    """Occurrence IDs of the unit whose text is `text` (cell) or contains it (linked text)."""
    index = _index(document, CANONICAL)
    ids = []
    for unit in index.units[node_of(document, tag)]:
        if cell and unit.cell is not None and unit.text == text:
            ids += unit.ids([(0, len(unit.text))])
        elif not cell and unit.cell is None and text in unit.text:
            start = unit.text.index(text)
            ids += unit.ids([(start, start + len(text))])
        elif prefix and unit.cell is not None and unit.text.startswith(prefix):
            ids += unit.ids([(0, len(prefix))])
    return ids


def component(role, text, ids, **fields):
    return ComponentLocus(role=role, text=text, occurrence_ids=ids, **fields)


def attested(document, *, subject=None, relation=None, answer=None, answer_text="307.8"):
    """Reviewer-equivalent descriptors anchored in the table, its caption, row and header."""
    subject = subject or (
        occurrences(document, CAPTION, cell=False) + occurrences(document, "Collins")
    )
    relation = relation or occurrences(document, "#Predicted")
    answer = answer or occurrences(document, "307.8 ± 12.9", prefix="307.8")
    return [
        component(
            "subject",
            DESCRIPTORS[0][1],
            subject,
            equivalence_note="caption names TOMOC and the original networks; row label is Collins",
        ),
        component(
            "relation",
            DESCRIPTORS[1][1],
            relation,
            equivalence_note="#Predicted column; caption states cells are mean ± SD",
        ),
        component("answer", answer_text, answer, form="mean_sd"),
    ]


def reviewer_set(document, components, tag="table"):
    placement = Placement(container_node_id=node_of(document, tag), components=components)
    return SetLocus(set_index=0, placements=[placement])


@pytest.fixture
def mean_table():
    document = parse(MEAN_TABLE.encode())
    return document, make_item(document, DESCRIPTORS)


def verdict(item, document, components, label="preserved", **fields):
    result = judgment(item, document, label, sets=[reviewer_set(document, components)], **fields)
    validate_judgment(result, item, document, CANONICAL)
    return result


# ---- accepted: reviewer-attested equivalence with auditable anchors ------------------------


def test_reviewer_preserved_with_paraphrased_descriptors_and_linked_table_anchors(mean_table):
    document, item = mean_table
    assert verdict(item, document, attested(document)).label == "preserved"


def test_equivalence_judgment_publishes_immutably_and_reads_back(mean_table, tmp_path):
    document, item = mean_table
    result = judgment(
        item, document, "preserved", sets=[reviewer_set(document, attested(document))]
    )
    reference, _ = publish_judgment(result, item, document, CANONICAL, tmp_path)
    back = read_judgment(reference, tmp_path, item, document, CANONICAL)
    assert back == result
    assert back.reviewer == "Rama Krishna Pudota" and back.at_utc.endswith("Z")
    [subject, relation, answer] = back.sets[0].placements[0].components
    assert subject.equivalence_note and relation.equivalence_note and answer.form == "mean_sd"
    path = tmp_path / reference.logical_ref
    with pytest.raises(FileExistsError):  # a published record is never overwritten
        publish_judgment(result, item, document, CANONICAL, tmp_path)
    path.write_bytes(path.read_bytes().replace(b"mean_sd", b"mean_xx"))
    with pytest.raises(HashMismatchError):
        read_judgment(reference, tmp_path, item, document, CANONICAL)


def test_genuine_alteration_stays_distinct_from_descriptor_equivalence(mean_table):
    document, item = mean_table
    parts = attested(document)
    parts[2] = component(
        "answer",
        "307.8",
        occurrences(document, "307.8 ± 12.9"),
        observed_text="307.8 ± 12.9",
    )
    altered = verdict(item, document, parts, "altered_but_sufficient", alteration="whole cell")
    assert altered.label == "altered_but_sufficient"
    with pytest.raises(ValueError, match="observed text"):
        verdict(item, document, attested(document), "altered_but_sufficient", alteration="x")
    with pytest.raises(ValueError, match="verbatim"):
        verdict(item, document, parts, "preserved")


# ---- component model rules -----------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"role": "subject", "equivalence_note": "  "},
        {"role": "answer", "equivalence_note": "same value"},
        {"role": "subject", "equivalence_note": "n", "observed_text": "x"},
        {"role": "subject", "form": "mean_sd"},
        {"role": "answer", "form": "mean_sd", "observed_text": "x"},
        {"role": "answer", "form": "mean_sd", "equivalence_note": "n"},
    ],
)
def test_component_attestation_fields_are_exclusive_and_bounded(fields):
    with pytest.raises(ValidationError):
        ComponentLocus(text="t", occurrence_ids=["o"], **fields)


def test_unused_attestation_fields_do_not_change_stored_bytes(mean_table):
    document, _ = mean_table
    plain = ComponentLocus(role="answer", text="1", occurrence_ids=["o"])
    assert set(plain.model_dump(mode="json")) == {"role", "text", "occurrence_ids", "observed_text"}
    assert "equivalence_note" in attested(document)[0].model_dump(mode="json")
    assert "form" in attested(document)[2].model_dump(mode="json")


def test_automatic_candidate_and_deterministic_label_cannot_carry_attested_components(mean_table):
    document, item = mean_table
    with pytest.raises(ValidationError):
        StageACandidate(
            benchmark_version=item.benchmark_version,
            corpus_version=item.corpus_version,
            item_id=item.item_id,
            item_sha256="a" * 64,
            paper_id=item.paper_id,
            representation=document.representation,
            canonical=CANONICAL,
            determination="automatic_preserved",
            sets=[reviewer_set(document, attested(document))],
            suggestions=[],
            qualifications=[],
        )
    candidate = locate(item, document, CANONICAL)
    with pytest.raises(ValueError):
        automatic_judgment(candidate, CANONICAL)


# ---- rejected: missing components, unrelated containers, invalid structural links ----------


def test_row_and_column_anchors_are_both_required(mean_table):
    document, item = mean_table
    no_row = attested(document, subject=occurrences(document, CAPTION, cell=False))
    no_column = attested(document, relation=occurrences(document, CAPTION, cell=False))
    for parts in (no_row, no_column):
        with pytest.raises(ValueError, match="row and column"):
            verdict(item, document, parts)


@pytest.mark.parametrize(
    "subject, relation",
    [
        ("Yeast", "#Predicted"),  # another row's label
        ("Collins", "Precision"),  # another column's header
        ("Collins", "Dataset"),  # header of the row-label column, not of the answer column
    ],
)
def test_anchors_must_be_associated_with_the_answer_cell(mean_table, subject, relation):
    document, item = mean_table
    parts = attested(
        document,
        subject=occurrences(document, subject),
        relation=occurrences(document, relation),
    )
    with pytest.raises(ValueError, match="not linked to the answer cell"):
        verdict(item, document, parts)


def test_answer_in_another_row_is_not_tied_to_this_rows_label():
    document = parse(MEAN_TABLE.encode())
    item = make_item(document, with_text(DESCRIPTORS, "answer", "153.0"))
    parts = attested(
        document,
        answer=occurrences(document, "153.0 ± 6.0", prefix="153.0"),
        answer_text="153.0",
    )
    with pytest.raises(ValueError, match="not linked to the answer cell"):
        verdict(item, document, parts)  # row anchor is still Collins


def test_anchors_in_an_unrelated_container_are_rejected(mean_table):
    document, item = mean_table
    prose = _index(document, CANONICAL).units[node_of(document, "p")][0]
    parts = attested(document, subject=prose.ids([(0, 10)]))
    with pytest.raises(ValueError, match="not text in the named container"):
        verdict(item, document, parts)


def test_value_found_only_in_prose_is_not_a_table_answer(mean_table):
    document, item = mean_table
    prose = _index(document, CANONICAL).units[node_of(document, "p")][0]
    start = prose.text.index("307.8")
    parts = attested(document, answer=prose.ids([(start, start + 5)]))
    with pytest.raises(ValueError):
        verdict(item, document, parts)


# ---- mean ± SD cells: exact leading value, no tolerance ------------------------------------


@pytest.mark.parametrize("answer", ["307.85", "1307.8", "12.9", "307", "30.78"])
def test_mean_sd_never_accepts_a_different_number(answer):
    document = parse(MEAN_TABLE.encode())
    item = make_item(document, with_text(DESCRIPTORS, "answer", answer))
    parts = attested(
        document, answer=occurrences(document, "307.8 ± 12.9", prefix="307.8"), answer_text=answer
    )
    with pytest.raises(ValueError):
        verdict(item, document, parts)


def test_mean_sd_requires_the_caption_or_header_to_state_mean():
    document = parse(NO_MEAN_STATEMENT.encode())
    item = make_item(document, DESCRIPTORS)
    with pytest.raises(ValueError, match="mean ±"):
        verdict(item, document, attested(document))


def test_mean_sd_is_not_available_for_an_answer_outside_a_table_cell():
    document = parse(MEAN_TABLE.encode())
    item = make_item(document, DESCRIPTORS)
    prose = _index(document, CANONICAL).units[node_of(document, "p")][0]
    start = prose.text.index("307.8")
    answer = component("answer", "307.8", prose.ids([(start, start + 5)]), form="mean_sd")
    subject, relation = attested(document)[:2]
    with pytest.raises(ValueError):
        verdict(item, document, [subject, relation, answer])


# ---- automatic Stage A is unchanged --------------------------------------------------------


def test_automatic_stage_a_does_not_accept_mean_sd_cells_or_descriptors(mean_table):
    document, item = mean_table
    assert locate(item, document, CANONICAL).determination == "needs_review"
    exact = make_item(
        document,
        [
            ("subject", "Performance of TOMOC on the original networks"),
            ("relation", "#Predicted"),
            ("answer", "307.8"),
            ("row", "Collins"),
            ("column", "#Predicted"),
        ],
    )
    candidate = locate(exact, document, CANONICAL)
    assert candidate.determination == "needs_review" and not candidate.sets


def test_verbatim_reviewer_judgments_are_unchanged():
    from test_stage_a import PROSE, PROSE_COMPONENTS

    document = parse(PROSE.encode())
    item = make_item(document, PROSE_COMPONENTS)
    result = judgment(item, document, "preserved", sets=[locus(document, PROSE_COMPONENTS)])
    validate_judgment(result, item, document, CANONICAL)
    assert b"equivalence_note" not in serialize_project_json(result, type(result))


# ---- answer cells that span several rows or columns -----------------------------------------

ROW_SPAN_TABLE = (
    "<article><body><table-wrap><caption><p>Hazard ratios</p></caption><table>"
    "<thead><tr><th>Group</th><th>HR</th></tr></thead><tbody>"
    '<tr><th>VDD</th><td rowspan="2">1.104</td></tr>'
    "<tr><th>VDA</th></tr>"
    "<tr><th>Other</th><td>9.9</td></tr>"
    "</tbody></table></table-wrap></body></article>"
)
COLUMN_SPAN_TABLE = (
    "<article><body><table-wrap><caption><p>Hazard ratios</p></caption><table>"
    "<thead><tr><th>Group</th><th>Adjusted</th><th>Crude</th><th>Other</th></tr></thead><tbody>"
    '<tr><th>VDD</th><td colspan="2">1.104</td><td>9.9</td></tr>'
    "</tbody></table></table-wrap></body></article>"
)
SPAN_DESCRIPTORS = [
    ("subject", "vitamin D deficiency"),
    ("relation", "hazard ratio"),
    ("answer", "1.104"),
]


def span_verdict(table, row_label, column_label):
    document = parse(table.encode())
    item = make_item(document, SPAN_DESCRIPTORS)
    note = "row and column labels of the answer cell"
    parts = [
        component(
            "subject",
            "vitamin D deficiency",
            occurrences(document, row_label),
            equivalence_note=note,
        ),
        component(
            "relation", "hazard ratio", occurrences(document, column_label), equivalence_note=note
        ),
        component("answer", "1.104", occurrences(document, "1.104")),
    ]
    return verdict(item, document, parts)


@pytest.mark.parametrize("row_label", ["VDD", "VDA"])
def test_row_anchor_may_lie_in_any_row_covered_by_the_answer_cell(row_label):
    assert span_verdict(ROW_SPAN_TABLE, row_label, "HR").label == "preserved"


def test_row_anchor_outside_the_rows_covered_by_the_answer_cell_is_rejected():
    with pytest.raises(ValueError, match="not linked to the answer cell"):
        span_verdict(ROW_SPAN_TABLE, "Other", "HR")


@pytest.mark.parametrize("column_label", ["Adjusted", "Crude"])
def test_column_anchor_may_lie_over_any_column_covered_by_the_answer_cell(column_label):
    assert span_verdict(COLUMN_SPAN_TABLE, "VDD", column_label).label == "preserved"


def test_column_anchor_outside_the_columns_covered_by_the_answer_cell_is_rejected():
    with pytest.raises(ValueError, match="not linked to the answer cell"):
        span_verdict(COLUMN_SPAN_TABLE, "VDD", "Other")


def test_row_anchor_for_a_spanned_answer_must_still_lie_left_of_it():
    with pytest.raises(ValueError, match="not linked to the answer cell"):
        span_verdict(ROW_SPAN_TABLE, "1.104", "HR")  # the answer cell itself is no anchor
