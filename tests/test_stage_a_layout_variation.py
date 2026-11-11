"""Stage A on layout and typography variation: a fact is preserved only when it is verbatim.

Each case keeps the benchmark components fixed and varies only how the source writes them.
Expected outcomes follow the locator contract (only a complete, verbatim, single-context,
exact-mapped fact is preserved; everything else goes to review and is never repaired) and, for
tables, the layout drawn in the fixture. Nothing here normalises text.
"""

import pytest
from test_documents_jats import parse
from test_stage_a import CANONICAL, kinds, make_item
from test_stage_a_pdf_scope import pdf_document

from biorag_repro.evaluation.stage_a import locate

COMPONENTS = [
    ("subject", "treatment"),
    ("relation", "effect"),
    ("answer", "5"),
    ("condition", "condition A"),
]
PAGE_1 = (1, 50, 700, 570, 680)
PAGE_2 = (2, 50, 700, 570, 640)


def pdf_candidate(blocks, components=COMPONENTS):
    document = pdf_document(blocks)
    return locate(make_item(document, components), document, CANONICAL)


def jats_candidate(xml, components):
    document = parse(xml.encode())
    return locate(make_item(document, components), document, CANONICAL)


def test_a_plain_pdf_sentence_is_the_control():
    candidate = pdf_candidate(
        [("paragraph", "Under condition A, treatment effect was 5.", [PAGE_1])]
    )
    assert candidate.determination == "automatic_preserved"


@pytest.mark.parametrize(
    "text",
    [
        "Under condition A, treat- ment effect was 5.",  # line-break hyphen left in the text
        "Under condition A, treat-\nment effect was 5.",
        "Under condition A, treat­ment effect was 5.",  # soft hyphen inside the word
    ],
)
def test_a_hyphenated_word_is_not_rejoined_and_the_fact_goes_to_review(text):
    candidate = pdf_candidate([("paragraph", text, [PAGE_1])])
    assert candidate.determination == "needs_review"
    assert "component_not_found" in kinds(candidate)


@pytest.mark.parametrize(
    "text, components",
    [
        # ligature glyph where the benchmark writes the letters
        ("Under condition A, eﬀect was 5.", COMPONENTS),
        # a superscript flattened to a plain digit is not the benchmark's exponent
        (
            "Under condition A, treatment effect was 2.7 × 104.",
            [*COMPONENTS[:2], ("answer", "2.7 × 10⁴"), COMPONENTS[3]],
        ),
    ],
)
def test_typographic_differences_in_extracted_text_are_not_repaired(text, components):
    candidate = pdf_candidate([("paragraph", text, [PAGE_1])], components)
    assert candidate.determination == "needs_review"
    assert "component_not_found" in kinds(candidate)


def test_a_fact_broken_across_two_pdf_pages_is_never_combined():
    candidate = pdf_candidate(
        [
            ("paragraph", "Under condition A, treatment effect", [PAGE_1]),
            ("paragraph", "was 5.", [PAGE_2]),
        ]
    )
    assert candidate.determination == "needs_review"
    assert "split_across_contexts" in kinds(candidate)


def test_a_unicode_superscript_in_the_source_is_the_control_for_jats_markup():
    components = [*COMPONENTS[:2], ("answer", "−1.2 × 10⁻³"), COMPONENTS[3]]
    xml = "<article><body><p>Under condition A, treatment effect was −1.2 × 10⁻³.</p></body></article>"
    assert jats_candidate(xml, components).determination == "automatic_preserved"


SUPERSCRIPT_XML = (
    "<article><body><p>Under condition A, treatment effect was −1.2 × 10<sup>−3</sup>.</p>"
    "</body></article>"
)


@pytest.mark.parametrize(
    "xml, answer_or_subject",
    [
        (SUPERSCRIPT_XML, ("answer", "−1.2 × 10⁻³")),
        (SUPERSCRIPT_XML, ("answer", "−1.2 × 10−3")),
        (
            "<article><body><p>Under condition A, H<sub>2</sub>O effect was 5.</p></body></article>",
            ("subject", "H2O"),
        ),
    ],
)
def test_inline_superscript_and_subscript_markup_is_never_automatically_preserved(
    xml, answer_or_subject
):
    role, text = answer_or_subject
    components = [(r, text if r == role else t) for r, t in COMPONENTS]
    candidate = jats_candidate(xml, components)
    assert candidate.determination == "needs_review"


# ---- table spans: the row and column named by the benchmark must own the answer cell -----------

TABLE_OPEN = (
    "<article><body><table-wrap><caption><p>Classifier performance</p></caption><table>"
    "<thead><tr><th>Gene</th><th>Set</th><th>AUROC</th></tr></thead><tbody>"
)
TABLE_CLOSE = "</tbody></table></table-wrap></body></article>"
# Row headers span two data rows each: TP53 owns 0.80 and 0.91, KRAS owns 0.70 and 0.40.
SPANNED_ROWS = (
    TABLE_OPEN
    + '<tr><th rowspan="2">TP53</th><td>train</td><td>0.80</td></tr><tr><td>test</td><td>0.91</td></tr>'
    + '<tr><th rowspan="2">KRAS</th><td>train</td><td>0.70</td></tr><tr><td>test</td><td>0.40</td></tr>'
    + TABLE_CLOSE
)


def table_components(row, answer):
    return [
        ("subject", "Classifier performance"),
        ("relation", "AUROC"),
        ("answer", answer),
        ("row", row),
        ("column", "AUROC"),
    ]


@pytest.mark.parametrize(
    "row, answer, expected",
    [
        ("TP53", "0.80", "automatic_preserved"),
        ("TP53", "0.91", "automatic_preserved"),  # second row under the spanned header
        ("KRAS", "0.40", "automatic_preserved"),
        ("TP53", "0.40", "needs_review"),  # the value exists, but under KRAS
        ("KRAS", "0.91", "needs_review"),
    ],
)
def test_a_spanned_row_header_owns_exactly_its_own_rows(row, answer, expected):
    candidate = jats_candidate(SPANNED_ROWS, table_components(row, answer))
    assert candidate.determination == expected
    if expected == "needs_review":
        assert "association_mismatch" in kinds(candidate)
