"""A native ancestor section title may anchor a reviewer-attested descriptor; nothing else may."""

import pytest
from test_documents_jats import parse
from test_stage_a import CANONICAL, judgment, make_item, mutate
from test_stage_a_reviewer_equivalence import component

from biorag_repro.core.hashing import HashMismatchError
from biorag_repro.evaluation.models import Placement, SetLocus
from biorag_repro.evaluation.stage_a import _index, locate, validate_judgment
from biorag_repro.evaluation.stage_a_storage import publish_judgment, read_judgment

SCOPED = (
    "<article><body><sec><title>Results</title><sec><title>Performance on the original networks</title>"
    "<p>The coefficient of variation of the F1-score remained below 2% across all three datasets.</p>"
    "</sec><sec><title>Robustness under perturbation</title><p>Other text here.</p></sec></sec>"
    "<sec><title>Discussion</title><p>Closing text.</p></sec></body></article>"
)
COMPONENTS = [
    ("subject", "F1-score across the original datasets"),
    ("relation", "coefficient-of-variation upper bound"),
    ("answer", "below 2%"),
]
SENTENCE = (
    "The coefficient of variation of the F1-score remained below 2% across all three datasets"
)
NOTE = "the section title supplies the original-networks qualifier"


def find(document, text):
    """Container and occurrence IDs of the unit containing `text`."""
    index = _index(document, CANONICAL)
    for container, units in index.units.items():
        for unit in units:
            if text in unit.text:
                start = unit.text.index(text)
                return container, unit.ids([(start, start + len(text))])
    raise AssertionError(text)


def scoped_judgment(
    document, title="Performance on the original networks", *, label="preserved", **changes
):
    container, sentence = find(document, SENTENCE)
    _, heading = find(document, title)
    _, bound = find(document, "below 2%")
    parts = [
        component("subject", COMPONENTS[0][1], sentence + heading, equivalence_note=NOTE),
        component(
            "relation", COMPONENTS[1][1], sentence, equivalence_note="the sentence states it"
        ),
        component("answer", "below 2%", bound),
    ]
    item = make_item(document, COMPONENTS, **changes)
    placement = Placement(container_node_id=container, components=parts)
    result = judgment(item, document, label, sets=[SetLocus(set_index=0, placements=[placement])])
    return item, result


@pytest.fixture
def scoped():
    return parse(SCOPED.encode())


def test_native_ancestor_section_title_anchors_the_qualifier(scoped):
    item, result = scoped_judgment(scoped)
    validate_judgment(result, item, scoped, CANONICAL)


def test_a_higher_native_ancestor_title_is_also_a_native_ancestor(scoped):
    item, result = scoped_judgment(scoped, "Results")
    validate_judgment(result, item, scoped, CANONICAL)


@pytest.mark.parametrize("title", ["Robustness under perturbation", "Discussion"])
def test_sibling_or_unrelated_section_titles_are_rejected(scoped, title):
    item, result = scoped_judgment(scoped, title)
    with pytest.raises(ValueError, match="not text in the named container"):
        validate_judgment(result, item, scoped, CANONICAL)


def test_title_anchor_needs_a_reviewer_attested_component(scoped):
    item, result = scoped_judgment(scoped)
    verbatim = result.model_copy(deep=True)
    subject = verbatim.sets[0].placements[0].components[0]
    verbatim.sets[0].placements[0].components[0] = component(
        "subject", subject.text, subject.occurrence_ids
    )
    with pytest.raises(ValueError):
        validate_judgment(verbatim, item, scoped, CANONICAL)


def test_fabricated_title_occurrence_is_rejected(scoped):
    item, result = scoped_judgment(scoped)
    forged = result.model_copy(deep=True)
    comp = forged.sets[0].placements[0].components[0]
    forged.sets[0].placements[0].components[0] = component(
        "subject",
        comp.text,
        comp.occurrence_ids[:-1] + ["drv_not_a_real_occurrence"],
        equivalence_note=NOTE,
    )
    with pytest.raises(ValueError, match="not text in the named container"):
        validate_judgment(forged, item, scoped, CANONICAL)


def test_title_not_linked_by_native_structure_is_reading_order_only(scoped):
    title_id, _ = find(scoped, "Performance on the original networks")
    body = next(n.node_id for n in scoped.nodes if n.kind == "body")

    def detach(data):  # the heading still precedes the paragraph but is no section child
        next(n for n in data["nodes"] if n["node_id"] == title_id)["parent_node_id"] = body

    flat = mutate(scoped, detach)
    item, result = scoped_judgment(flat)
    with pytest.raises(ValueError, match="not text in the named container"):
        validate_judgment(result, item, flat, CANONICAL)


def test_flat_group_headings_do_not_count_as_a_native_section(scoped):
    paragraph, _ = find(scoped, SENTENCE)
    parent = next(n for n in scoped.nodes if n.node_id == paragraph).parent_node_id

    def flatten(data):  # a generic group, as in the flattened PDF_DOCLING headings
        next(n for n in data["nodes"] if n["node_id"] == parent)["kind"] = "source_construct"

    flat = mutate(scoped, flatten)
    item, result = scoped_judgment(flat)
    with pytest.raises(ValueError, match="not text in the named container"):
        validate_judgment(result, item, flat, CANONICAL)


def labelled_refs(document, section):
    """JATS and PDF references that carry a frozen benchmark section label."""
    return {
        "jats_refs": [
            {
                "artifact_id": document.source.artifact_id,
                "source_sha256": document.source.sha256,
                "paper_id": document.paper_id,
                "xml_path": "/article/body/p[1]",
                "source_text": "source",
                "section": section,
            }
        ],
        "pdf_refs": [
            {
                "artifact_id": "raw_pdf",
                "source_sha256": "b" * 64,
                "paper_id": document.paper_id,
                "page": 1,
                "source_text": "source",
                "section": section,
            }
        ],
    }


def test_a_matching_benchmark_label_does_not_rescue_a_jats_heading_without_native_ancestry(scoped):
    title = "Performance on the original networks"
    paragraph, _ = find(scoped, SENTENCE)
    parent = next(n for n in scoped.nodes if n.node_id == paragraph).parent_node_id

    def flatten(data):
        next(n for n in data["nodes"] if n["node_id"] == parent)["kind"] = "source_construct"

    flat = mutate(scoped, flatten)
    item, result = scoped_judgment(flat, title, **labelled_refs(flat, title))
    with pytest.raises(ValueError, match="not text in the named container"):
        validate_judgment(result, item, flat, CANONICAL)


def test_jats_native_ancestors_do_not_depend_on_the_benchmark_label(scoped):
    for title in ("Results", "Performance on the original networks"):
        item, result = scoped_judgment(scoped, title, **labelled_refs(scoped, "Another section"))
        validate_judgment(result, item, scoped, CANONICAL)


def test_automatic_stage_a_does_not_use_section_titles(scoped):
    candidate = locate(make_item(scoped, COMPONENTS), scoped, CANONICAL)
    assert candidate.determination == "needs_review" and not candidate.sets


def test_scoped_judgment_publishes_immutably_and_reads_back(scoped, tmp_path):
    item, result = scoped_judgment(scoped)
    reference, _ = publish_judgment(result, item, scoped, CANONICAL, tmp_path)
    assert read_judgment(reference, tmp_path, item, scoped, CANONICAL) == result
    with pytest.raises(FileExistsError):
        publish_judgment(result, item, scoped, CANONICAL, tmp_path)
    path = tmp_path / reference.logical_ref
    path.write_bytes(path.read_bytes().replace(b"original", b"origina1"))
    with pytest.raises(HashMismatchError):
        read_judgment(reference, tmp_path, item, scoped, CANONICAL)
