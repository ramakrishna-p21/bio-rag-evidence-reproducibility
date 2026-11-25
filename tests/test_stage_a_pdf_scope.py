"""A flat PDF_DOCLING heading may anchor scope only under conservative structural checks.

The canonical PDF has no section ancestry. A heading is accepted for one placement only
when it is the nearest preceding title, equals a frozen benchmark section label, and the
page and column geometry of the reading chain contradicts nothing. Anything else fails
closed: the reviewer's judgment is then invalid and the unit stays unresolved.
"""

import pytest
from test_stage_a import CANONICAL, judgment, make_item
from test_stage_a_reviewer_equivalence import component

from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation.models import Placement, SetLocus
from biorag_repro.evaluation.stage_a import _heading_key, _index, validate_judgment

RAW = ArtifactRef(artifact_id="raw_pdf", logical_ref="raw/p.pdf", sha256="b" * 64)
NATIVE = ArtifactRef(artifact_id="drv_native", logical_ref="native/n.json", sha256="c" * 64)
HEADING = "Performance on the original networks"
SENTENCE = "The coefficient of variation of the F1-score remained below 2% across all datasets"
NOTE = "the heading supplies the original-networks qualifier"
REJECTED = "not text in the named container"

# Page boxes are (page, left, top, right, bottom), origin bottom-left, so a larger top is higher.
WIDE = (1, 50, 700, 570, 680)
TOP = (1, 50, 670, 570, 600)
LOW = (1, 50, 500, 570, 440)
LEFT_HIGH = (1, 50, 700, 300, 600)
LEFT_LOW = (1, 50, 300, 300, 150)
RIGHT_HIGH = (1, 320, 700, 570, 600)
RIGHT_LOW = (1, 320, 300, 570, 150)


def mapping(number, text, boxes, origin="BOTTOMLEFT"):
    return {
        "kind": "docling",
        "state": "exact_native",
        "source_artifact_id": RAW.artifact_id,
        "native_artifact": NATIVE.model_dump(),
        "native_ref": f"#/texts/{number}",
        "provenance": [
            {
                "ordinal": k,
                "page_no": page,
                "bbox": {"l": left, "t": top, "r": right, "b": bottom, "coord_origin": origin},
                "charspan": [0, len(text)] if text else None,
                "charspan_state": "meaningful" if text else "unavailable",
                "state": "approximate",
            }
            for k, (page, left, top, right, bottom) in enumerate(boxes)
        ],
    }


def pdf_document(blocks):
    """Flat PDF canonical document: every block is a child of one generic group.

    A block is (kind, text, boxes) or (kind, text, boxes, overrides).
    """
    root = {
        "node_id": "node_0",
        "parent_node_id": None,
        "order": 0,
        "kind": "source_construct",
        "source_tag": "GroupItem",
        "native_label": "unspecified",
        "native_content_layer": "body",
        "mapping": {
            "kind": "docling",
            "state": "exact_native",
            "source_artifact_id": RAW.artifact_id,
            "native_artifact": NATIVE.model_dump(),
            "native_ref": "#/body",
        },
    }
    nodes = [root]
    for number, block in enumerate(blocks, start=1):
        kind, text, boxes, *rest = block
        over = rest[0] if rest else {}
        title = kind == "title"
        node = {
            "node_id": f"node_{number}",
            "parent_node_id": over.get("parent", "node_0"),
            "order": number,
            "kind": kind,
            "source_tag": "SectionHeaderItem" if title else "TextItem",
            "native_label": "section_header" if title else "text",
            "native_content_layer": over.get("layer", "body"),
            "native_order_state": over.get("order_state", "linked"),
            "native_heading_level": 1 if title else None,
            "interpretation_state": "exact_native",
            "segments": (
                [
                    {
                        "text": text,
                        "mapping": mapping(number, text, boxes, over.get("origin", "BOTTOMLEFT")),
                    }
                ]
                if text
                else []
            ),
            "mapping": mapping(number, text, boxes, over.get("origin", "BOTTOMLEFT")),
        }
        nodes.append(node)
    return CanonicalDocument.model_validate(
        {
            "schema_version": "4",
            "paper_id": "P001",
            "representation": "PDF_DOCLING",
            "source": RAW.model_dump(),
            "native_artifact": NATIVE.model_dump(),
            "transformation_id": "t",
            "configuration_sha256": "d" * 64,
            "status": "complete",
            "nodes": nodes,
        }
    )


COMPONENTS = [
    ("subject", "F1-score across the original datasets"),
    ("relation", "coefficient-of-variation upper bound"),
    ("answer", "below 2%"),
]
VERBATIM = [
    ("subject", "F1-score"),
    ("relation", "coefficient of variation"),
    ("answer", "below 2%"),
]


def pdf_refs(section):
    return [
        {
            "artifact_id": RAW.artifact_id,
            "source_sha256": RAW.sha256,
            "paper_id": "P001",
            "page": 1,
            "source_text": "source",
            "section": section,
        }
    ]


def find(document, text):
    """Container and occurrence IDs of the unit containing `text`."""
    index = _index(document, CANONICAL)
    for container, units in index.units.items():
        for unit in units:
            if text in unit.text:
                start = unit.text.index(text)
                return container, unit.ids([(start, start + len(text))])
    raise AssertionError(text)


def scoped(
    document,
    *,
    section=HEADING,
    title=HEADING,
    components=COMPONENTS,
    heading_ids=None,
):
    """Item and a reviewer judgment whose subject cites the heading (attested)."""
    container, sentence = find(document, SENTENCE)
    heading = heading_ids or find(document, title)[1]
    _, bound = find(document, "below 2%")
    parts = [
        component("subject", components[0][1], sentence + heading, equivalence_note=NOTE),
        component(
            "relation", components[1][1], sentence, equivalence_note="the sentence states it"
        ),
        component("answer", "below 2%", bound),
    ]
    item = make_item(document, components, pdf_refs=pdf_refs(section))
    placement = Placement(container_node_id=container, components=parts)
    return item, judgment(
        item, document, "preserved", sets=[SetLocus(set_index=0, placements=[placement])]
    )


def accepted(blocks, **kwargs):
    document = pdf_document(blocks)
    item, result = scoped(document, **kwargs)
    validate_judgment(result, item, document, CANONICAL)


def rejected(blocks, **kwargs):
    document = pdf_document(blocks)
    item, result = scoped(document, **kwargs)
    with pytest.raises(ValueError, match=REJECTED):
        validate_judgment(result, item, document, CANONICAL)


def flow(heading_box, sentence_box, heading=HEADING):
    return [
        ("title", heading, [heading_box]),
        ("paragraph", SENTENCE + ".", [sentence_box]),
    ]


# ---- accepted structures -------------------------------------------------------------


def test_heading_directly_above_the_paragraph_on_one_page():
    accepted(flow(WIDE, TOP))


def test_heading_at_the_foot_of_a_page_and_paragraph_on_the_next():
    accepted(flow((1, 50, 120, 570, 100), (2, 50, 700, 570, 640)))


def test_a_page_with_no_body_text_between_heading_and_paragraph_is_not_a_contradiction():
    blocks = [
        ("title", HEADING, [(1, 50, 120, 570, 100)]),
        ("paragraph", SENTENCE + ".", [(3, 50, 700, 570, 640)]),
    ]
    accepted(blocks)


def test_heading_at_the_foot_of_the_left_column_and_paragraph_at_the_top_of_the_right():
    blocks = [
        ("paragraph", "Earlier left column text.", [LEFT_HIGH]),
        ("title", HEADING, [LEFT_LOW]),
        ("paragraph", SENTENCE + ".", [RIGHT_HIGH]),
        ("paragraph", "Later right column text.", [RIGHT_LOW]),
    ]
    accepted(blocks)


def test_a_paragraph_that_continues_from_the_left_to_the_right_column():
    blocks = [
        ("title", HEADING, [(1, 50, 420, 300, 400)]),
        ("paragraph", SENTENCE + ".", [(1, 50, 380, 300, 100), RIGHT_HIGH]),
        ("paragraph", "Filler on the right.", [RIGHT_LOW]),
    ]
    accepted(blocks)


def test_a_single_column_page_is_not_split_into_columns_by_a_short_heading():
    blocks = [
        ("title", HEADING, [(1, 166, 222, 400, 212)]),
        ("paragraph", SENTENCE + ".", [(1, 166, 205, 561, 70)]),
    ]
    accepted(blocks)


def test_a_full_width_heading_over_a_column_paragraph():
    blocks = [
        ("title", HEADING, [WIDE]),
        ("paragraph", SENTENCE + ".", [(1, 50, 650, 300, 560)]),
        ("paragraph", "Right column filler.", [RIGHT_LOW]),
    ]
    accepted(blocks)


def test_label_equality_ignores_dash_soft_hyphen_ligature_space_and_case():
    heading = "Kaplan-Meier perfor­mance of ﬁbronectin cohorts"
    label = "KAPLAN–MEIER performance of fibronectin cohorts"
    blocks = flow(WIDE, TOP, heading)
    accepted(blocks, section=label, title=heading)


# ---- refusals: heading and label ---------------------------------------------------------


def test_no_heading_node_means_no_scope_and_a_label_cannot_stand_in():
    blocks = [("paragraph", "Some earlier text.", [WIDE]), ("paragraph", SENTENCE + ".", [TOP])]
    rejected(blocks, heading_ids=["drv_the_label_is_not_text_in_the_document"])


def test_missing_benchmark_label_rejects():
    rejected(flow(WIDE, TOP), section=None)


def test_label_that_names_a_different_section_rejects():
    rejected(flow(WIDE, TOP), section="Cohort samples and characteristics")


def test_a_distant_generic_title_with_a_lost_heading_rejects():
    blocks = [("title", "Article", [WIDE])] + [
        ("paragraph", f"Filler {k}.", [(1, 50, 650 - 5 * k, 570, 645 - 5 * k)]) for k in range(20)
    ]
    blocks.append(("paragraph", SENTENCE + ".", [(2, 50, 700, 570, 640)]))
    rejected(blocks, title="Article", section="Improved prediction of eQTL effects")


def test_a_competing_title_between_heading_and_paragraph_rejects():
    blocks = [
        ("title", HEADING, [WIDE]),
        ("title", "Robustness under perturbation", [TOP]),
        ("paragraph", SENTENCE + ".", [LOW]),
    ]
    rejected(blocks)


def test_an_untitled_competing_title_node_also_rejects():
    blocks = [
        ("title", HEADING, [WIDE]),
        ("title", "", [TOP]),
        ("paragraph", SENTENCE + ".", [LOW]),
    ]
    rejected(blocks)


def test_a_heading_that_follows_its_paragraph_rejects():
    blocks = [("paragraph", SENTENCE + ".", [WIDE]), ("title", HEADING, [TOP])]
    rejected(blocks)


def test_a_furniture_layer_title_is_not_a_heading():
    blocks = [
        ("title", HEADING, [WIDE], {"layer": "furniture"}),
        ("paragraph", SENTENCE + ".", [TOP]),
    ]
    rejected(blocks)


def test_a_title_inside_a_figure_is_not_a_heading():
    blocks = [
        ("figure", "", [WIDE]),
        ("title", HEADING, [TOP], {"parent": "node_1"}),
        ("paragraph", SENTENCE + ".", [LOW]),
    ]
    rejected(blocks)


# ---- refusals: reading order and layout --------------------------------------------------


def test_page_regression_rejects():
    rejected(flow((2, 50, 700, 570, 680), (1, 50, 670, 570, 600)))


def test_a_paragraph_above_the_heading_in_the_same_column_rejects():
    rejected(flow(LOW, TOP))


def test_a_left_column_paragraph_after_a_right_column_heading_rejects():
    blocks = [
        ("paragraph", "Left filler.", [LEFT_LOW]),
        ("title", HEADING, [RIGHT_HIGH]),
        ("paragraph", SENTENCE + ".", [LEFT_HIGH]),
    ]
    rejected(blocks)


def test_a_box_that_cannot_be_placed_in_a_column_rejects():
    blocks = [
        ("paragraph", "Left filler.", [LEFT_HIGH]),
        ("paragraph", "Right filler.", [RIGHT_HIGH]),
        ("title", HEADING, [LEFT_LOW]),
        ("paragraph", SENTENCE + ".", [(1, 200, 100, 420, 50)]),
    ]
    rejected(blocks)


def test_a_column_break_inside_a_multi_part_paragraph_must_also_read_forward():
    blocks = [
        ("title", HEADING, [(1, 50, 420, 300, 400)]),
        ("paragraph", SENTENCE + ".", [RIGHT_HIGH, (1, 50, 380, 300, 100)]),
        ("paragraph", "Filler on the right.", [RIGHT_LOW]),
    ]
    rejected(blocks)


def test_missing_provenance_rejects():
    blocks = [("title", HEADING, [WIDE]), ("paragraph", SENTENCE + ".", [])]
    rejected(blocks)


def test_unsupported_coordinate_origin_rejects():
    blocks = [
        ("title", HEADING, [WIDE], {"origin": "TOPLEFT"}),
        ("paragraph", SENTENCE + ".", [TOP], {"origin": "TOPLEFT"}),
    ]
    rejected(blocks)


def test_an_unpositioned_node_in_the_chain_rejects():
    blocks = [
        ("title", HEADING, [WIDE]),
        ("paragraph", "In between.", [TOP], {"order_state": "unpositioned"}),
        ("paragraph", SENTENCE + ".", [LOW]),
    ]
    rejected(blocks)


def test_body_text_on_a_skipped_page_that_the_chain_does_not_pass_through_rejects():
    blocks = [
        ("title", HEADING, [(1, 50, 120, 570, 100)]),
        ("paragraph", SENTENCE + ".", [(3, 50, 700, 570, 640)]),
        ("paragraph", "Page two body text read later.", [(2, 50, 700, 570, 640)]),
    ]
    rejected(blocks)


def test_a_page_break_where_the_next_page_precedes_the_heading_rejects():
    blocks = [
        ("paragraph", "Early page text.", [(2, 50, 700, 570, 640)]),
        ("title", HEADING, [(1, 50, 700, 570, 680)]),
        ("paragraph", SENTENCE + ".", [(2, 50, 600, 570, 540)]),
    ]
    rejected(blocks)


# ---- container ambiguity uses complete placements only -------------------------------------


def test_a_repeated_short_value_in_the_same_section_does_not_reject():
    blocks = [
        ("title", HEADING, [WIDE]),
        ("paragraph", SENTENCE + ".", [TOP]),
        ("paragraph", "Another measure was also below 2%.", [LOW]),
    ]
    accepted(blocks)


def test_a_second_complete_verbatim_placement_in_the_same_section_rejects():
    blocks = [
        ("title", HEADING, [WIDE]),
        ("paragraph", SENTENCE + ".", [TOP]),
        ("paragraph", SENTENCE + " in a replicate.", [LOW]),
    ]
    document = pdf_document(blocks)
    item, result = scoped(document, components=VERBATIM)
    with pytest.raises(ValueError, match=REJECTED):
        validate_judgment(result, item, document, CANONICAL)


def test_a_complete_placement_under_a_different_heading_is_not_ambiguity():
    blocks = [
        ("title", HEADING, [WIDE]),
        ("paragraph", SENTENCE + ".", [TOP]),
        ("title", "Other results", [LOW]),
        ("paragraph", SENTENCE + " in a replicate.", [(2, 50, 700, 570, 640)]),
    ]
    document = pdf_document(blocks)
    item, result = scoped(document, components=VERBATIM)
    validate_judgment(result, item, document, CANONICAL)


# ---- the label never supplies evidence ----------------------------------------------------


def test_a_verbatim_component_cannot_cite_the_heading():
    document = pdf_document(flow(WIDE, TOP))
    item, _ = scoped(document)
    container, sentence = find(document, SENTENCE)
    _, heading = find(document, HEADING)
    _, bound = find(document, "below 2%")
    parts = [
        component("subject", COMPONENTS[0][1], sentence + heading),  # no reviewer attestation
        component("relation", COMPONENTS[1][1], sentence, equivalence_note="stated"),
        component("answer", "below 2%", bound),
    ]
    forged = judgment(
        item,
        document,
        "preserved",
        sets=[
            SetLocus(
                set_index=0, placements=[Placement(container_node_id=container, components=parts)]
            )
        ],
    )
    with pytest.raises(ValueError, match=REJECTED):
        validate_judgment(forged, item, document, CANONICAL)


def test_the_answer_cannot_rest_on_the_heading():
    document = pdf_document(
        [("title", "Result: below 2%", [WIDE]), ("paragraph", "The measure was reported.", [TOP])]
    )
    container, sentence = find(document, "The measure was reported")
    _, heading = find(document, "below 2%")
    with pytest.raises(ValueError, match="never accepted"):
        component("answer", "below 2%", heading, equivalence_note="the heading gives the value")
    parts = [
        component("subject", COMPONENTS[0][1], sentence, equivalence_note="stated"),
        component("relation", COMPONENTS[1][1], sentence, equivalence_note="stated"),
        component("answer", "below 2%", heading),
    ]
    item = make_item(document, COMPONENTS, pdf_refs=pdf_refs("Result: below 2%"))
    forged = judgment(
        item,
        document,
        "preserved",
        sets=[
            SetLocus(
                set_index=0,
                placements=[Placement(container_node_id=container, components=parts)],
            )
        ],
    )
    with pytest.raises(ValueError, match=REJECTED):
        validate_judgment(forged, item, document, CANONICAL)


# ---- heading normalisation --------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Kaplan–Meier", "Kaplan-Meier"),
        ("perfor­mance", "performance"),
        ("ﬁbronectin", "fibronectin"),
        ("A B  C", "A B C"),
        ("Impact", "IMPACT"),
        ("patients’ outcome", "patients' outcome"),
        ("3.2. Primary Outcomes", "3.2. primary outcomes"),
    ],
)
def test_harmless_typography_differences_are_equal(left, right):
    assert _heading_key(left) == _heading_key(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Rate p < 0.05", "Rate p > 0.05"),
        ("TNF-α response", "TNF-β response"),
        ("CD4+ T cells", "CD4- T cells"),
        ("CD4+ T cells", "CD4 T cells"),
        ("Change of −2 units", "Change of 2 units"),
        ("Change of −2 units", "Change of -2 units"),
        ("Model r² fit", "Model r2 fit"),
        ("Imm p4EBP1+ tumours", "Imm p4EBP1- tumours"),
        ("Dose 5 mg", "Dose 50 mg"),
        ("Upregulated genes", "Downregulated genes"),
    ],
)
def test_signs_symbols_and_distinctions_are_not_collapsed(left, right):
    assert _heading_key(left) != _heading_key(right)
