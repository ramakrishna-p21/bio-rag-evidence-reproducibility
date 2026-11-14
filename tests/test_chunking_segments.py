"""Opt-in sentence-boundary segmentation of oversized prose units.

The counters are synthetic (characters or words), so expected boundaries can be written out by
hand. Segmentation may only use the unit, its occurrence texts, the counter and the limit.
"""

import inspect

import pytest
from pydantic import ValidationError
from test_documents_jats import parse
from test_retrieval_models import ref, selection, settings
from test_stage_a import PROSE_COMPONENTS, make_item
from test_stage_b import P_A, P_C, deliver, finish, prose_xml

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.documents.models import (
    CanonicalDocument,
    CanonicalNode,
    SourceMapping,
    TextSegment,
)
from biorag_repro.retrieval.chunking import (
    chunk_document,
    discover_occurrences,
    render_units,
    segment_oversized_prose,
    validate_chunking,
)
from biorag_repro.retrieval.models import ConfigurationRef, Phase5Configuration
from biorag_repro.retrieval.storage import record_ref

SEGMENTS = "complete_units_or_sentence_segments"
CONTEXT = "[section]\nResults\n[/section]\n"


def scaled(text):
    """1.8 per character: one fixture sentence plus context fits 192, two never do."""
    return len(text) * 9 // 5


def config(policy=SEGMENTS, budget=192):
    return settings().model_copy(update={"chunk_tokens": budget, "unit_policy": policy})


def build(doc, policy=SEGMENTS, budget=192, counter=scaled):
    cfg = config(policy, budget)
    cfg_ref = ConfigurationRef(
        artifact=ref("configuration"),
        resolved_sha256=sha256_bytes(canonical_json_bytes(cfg.model_dump())),
    )
    selected = selection().model_copy(update={"representation": doc.representation})
    return selected, cfg, cfg_ref, chunk_document(doc, selected, cfg_ref, cfg, counter)


def jats(*paragraphs):
    return parse(prose_xml(*paragraphs).encode())


def paragraph_unit(doc):
    occurrences = discover_occurrences(doc, ref())
    units = [u for u in render_units(doc, occurrences) if u.kind == "paragraph"]
    assert len(units) == 1
    return units[0], {o.occurrence_id: o for o in occurrences}


def segment(doc, limit=192, counter=scaled):
    unit, occurrences = paragraph_unit(doc)
    return unit, *segment_oversized_prose(unit, occurrences, counter, limit)


def texts(pieces):
    return [p.text for p in pieces]


# Six sentences of 60-70 characters each; any two together plus context exceed 192 characters.
S1 = "Patients were followed for a median of 4.2 years (IQR 2.1–6.0)."
S2 = "Mortality was higher with deficiency, e.g. at one year (HR 1.31)."
S3 = "Effects persisted at three years in Smith et al. (2020) cohorts."
S4 = "The estimate was 1.2 × 10<sup>−5</sup> per day (p &lt; 0.001)."
S5 = "Sensitivity analyses (see Fig. 2. Panel B) gave similar results."
S6 = "No difference was seen for renal decline in any of the strata."
LONG = f"{S1} {S2} {S3} {S4} {S5} {S6}"


def test_default_policy_keeps_complete_units_and_baseline_chunks_unchanged():
    assert settings().unit_policy == "complete_units"
    doc = jats(LONG)
    *_, default = build(doc, policy="complete_units")
    *_, implicit = build(doc, policy=settings().unit_policy)
    assert [record_ref(c).artifact_id for c in default.chunks] == [
        record_ref(c).artifact_id for c in implicit.chunks
    ]
    oversized = [d for d in default.chunk_set.diagnostics if d.code == "oversized"]
    assert len(oversized) == 1
    assert oversized[0].reason == (
        "Complete source unit and required context exceed the construction limit."
    )
    assert not any(LONG[:40] in c.rendered_text for c in default.chunks)


def test_policy_value_is_explicit_and_part_of_configuration_identity():
    base = settings()
    segmented = config(budget=base.chunk_tokens)
    assert sha256_bytes(canonical_json_bytes(base.model_dump())) != sha256_bytes(
        canonical_json_bytes(segmented.model_dump())
    )
    data = base.model_dump()
    data["unit_policy"] = "split_anything"
    with pytest.raises(ValidationError):
        Phase5Configuration.model_validate(data)


def test_segments_reconstruct_the_unit_exactly_in_source_order():
    unit, pieces, reason = segment(jats(LONG))
    assert reason is None and len(pieces) >= 2
    rebuilt, cursor = "", 0
    for piece in pieces:
        start, end = (int(n) for n in piece.unit_id.rsplit(":s", 1)[1].split("-"))
        assert piece.text == unit.text[start:end]
        assert start >= cursor and not unit.text[cursor:start].strip()
        rebuilt += unit.text[cursor:start] + piece.text
        cursor = end
    assert cursor == len(unit.text) and rebuilt == unit.text
    for piece in pieces:
        assert (piece.kind, piece.context, piece.scope) == (unit.kind, unit.context, unit.scope)
        assert piece.source_node_ids == unit.source_node_ids
        assert scaled(piece.context + piece.text) <= 192


def test_boundaries_skip_abbreviations_decimals_notation_citations_and_parentheticals():
    _, pieces, _ = segment(jats(LONG))
    sentences = [s.replace("&lt;", "<") for s in (S1, S2, S3, S4, S5, S6)]
    # Every segment is a run of whole sentences; no split inside "e.g.", "et al.", "4.2",
    # "10<sup>−5</sup>", "p < 0.001" or the parenthesis containing "Fig. 2. Panel B".
    assert texts(pieces) == sentences


def test_lowercase_continuation_and_inline_markup_are_not_boundaries():
    text = (
        "Values were normalised to approx. 1.0 per sample and then log-transformed for analysis. "
        "this lower-case clause continues the same sentence and must stay with it here "
        "alongside a citation<sup>12. 13</sup> that is never split inside the markup at all."
    )
    _, pieces, reason = segment(jats(text))
    assert pieces == [] and reason == "no permitted sentence boundary"


def test_an_occurrence_crossing_a_boundary_is_content_once_then_continuation():
    doc = jats(f"{S1} {S2} <italic>{S3[:-1]}</italic>. {S6}")
    unit, pieces, _ = segment(doc)
    first_slot = next(c for c in unit.contributors if c.usage == "content")
    usages = [
        [c.usage for c in p.contributors if c.occurrence_id == first_slot.occurrence_id]
        for p in pieces
    ]
    # The paragraph's leading text() slot holds S1 and S2, which fall in different segments.
    assert usages[0] == ["content"] and usages[1] == ["content_continuation"]
    content = [c.occurrence_id for p in pieces for c in p.contributors if c.usage == "content"]
    assert sorted(content) == sorted(
        c.occurrence_id for c in unit.contributors if c.usage == "content"
    )


def test_segmented_chunk_set_validates_and_accounts_every_segment_chunk():
    doc = jats(LONG)
    selected, cfg, cfg_ref, result = build(doc)
    validate_chunking(result, doc, selected, cfg_ref, cfg, scaled)
    assert not [d for d in result.chunk_set.diagnostics if d.code == "oversized"]
    unit, _ = paragraph_unit(doc)
    slot = next(c for c in unit.contributors if c.usage == "content")
    holders = [
        record_ref(c).artifact_id
        for c in result.chunks
        if slot.occurrence_id in {k.occurrence_id for k in c.contributors}
    ]
    accounting = {a.occurrence_id: a for a in result.chunk_set.accounting}[slot.occurrence_id]
    assert accounting.status == "included" and accounting.chunk_ids == holders
    # The leading text() slot runs from S1 to the <sup> inside S4, so it spans four segments.
    assert len(holders) == 4


def test_a_single_sentence_over_the_limit_leaves_the_whole_unit_excluded():
    long_sentence = "Exposure " + "and outcome " * 20 + "were measured."
    doc = jats(f"{S1} {long_sentence} {S6}")
    *_, result = build(doc)
    [diagnostic] = [d for d in result.chunk_set.diagnostics if d.code == "oversized"]
    assert diagnostic.reason.endswith("Not segmented: a single sentence exceeds the limit.")
    assert not any(S1 in c.rendered_text or S6 in c.rendered_text for c in result.chunks)


def test_occurrence_text_that_cannot_be_located_is_not_segmented():
    unit, occurrences = paragraph_unit(jats(LONG))
    altered = type(unit)(**{**unit.__dict__, "text": unit.text.replace("Mortality", "Death")})
    pieces, reason = segment_oversized_prose(altered, occurrences, scaled, 192)
    assert pieces == [] and reason == "occurrence text could not be located in order"


def test_formulas_and_table_rows_are_never_segmented():
    unit, occurrences = paragraph_unit(jats(LONG))
    for kind in ("formula", "table_row"):
        other = type(unit)(**{**unit.__dict__, "kind": kind})
        pieces, reason = segment_oversized_prose(other, occurrences, scaled, 192)
        assert pieces == [] and reason == f"{kind} units are not segmented"


def test_units_within_the_limit_are_untouched():
    doc = jats(S1, S2)
    *_, plain = build(doc, policy="complete_units")
    *_, opted = build(doc)
    assert [c.rendered_text for c in plain.chunks] == [c.rendered_text for c in opted.chunks]


def test_segmentation_does_not_depend_on_paper_identity_or_benchmark_inputs():
    assert list(inspect.signature(segment_oversized_prose).parameters) == [
        "unit",
        "occurrences",
        "count_tokens",
        "limit",
    ]
    doc = jats(LONG)
    other = CanonicalDocument.model_validate({**doc.model_dump(), "paper_id": "P042"})
    _, first, _ = segment(doc)
    _, second, _ = segment(other)
    assert [(p.unit_id, p.text) for p in first] == [(p.unit_id, p.text) for p in second]


def test_chunk_identities_are_deterministic():
    doc = jats(LONG, S6)
    *_, one = build(doc)
    *_, two = build(doc)
    assert [record_ref(c).artifact_id for c in one.chunks] == [
        record_ref(c).artifact_id for c in two.chunks
    ]


def pdf(text, kind="paragraph", qualification=False):
    mapping = SourceMapping(
        kind="docling",
        state="exact_native",
        source_artifact_id="raw",
        native_artifact=ref("native"),
        native_ref="#/texts/0",
    )
    segment_mapping = mapping
    if qualification:
        segment_mapping = mapping.model_copy(update={"state": "derived"})
    node = CanonicalNode(
        node_id="n0",
        parent_node_id=None,
        order=0,
        kind=kind,
        source_tag="Synthetic",
        mapping=mapping,
        segments=[TextSegment(text=text, mapping=segment_mapping)],
        native_order_state="linked",
        interpretation_state="exact_native",
        native_content_layer="body",
    )
    return CanonicalDocument(
        paper_id="P001",
        representation="PDF_DOCLING",
        source=ref("raw"),
        native_artifact=ref("native"),
        transformation_id="synthetic",
        configuration_sha256="a" * 64,
        status="complete",
        nodes=[node],
    )


def test_pdf_units_follow_the_same_rule_as_jats():
    plain = LONG.replace("<sup>", "").replace("</sup>", "").replace("&lt;", "<")
    doc = pdf(plain)
    selected, cfg, cfg_ref, result = build(doc)
    validate_chunking(result, doc, selected, cfg_ref, cfg, scaled)
    expected = [
        s.replace("<sup>", "").replace("</sup>", "").replace("&lt;", "<")
        for s in (S1, S2, S3, S4, S5, S6)
    ]
    assert [c.rendered_text for c in result.chunks] == expected


def test_oversized_pdf_formula_stays_excluded_with_its_reason():
    *_, result = build(pdf(LONG.replace("<sup>", "").replace("</sup>", ""), kind="formula"))
    [diagnostic] = result.chunk_set.diagnostics
    assert diagnostic.reason.endswith("Not segmented: formula units are not segmented.")
    assert not result.chunks


def test_unit_level_qualification_line_is_repeated_on_every_segment():
    unit, occurrences = paragraph_unit(jats(LONG))
    prefix = "[qualification charspan=derived]\n"
    qualified = type(unit)(**{**unit.__dict__, "text": prefix + unit.text})
    pieces, reason = segment_oversized_prose(qualified, occurrences, scaled, 260)
    assert reason is None and len(pieces) >= 2
    assert all(p.text.startswith(prefix) for p in pieces)
    assert prefix + " ".join(p.text[len(prefix) :] for p in pieces) == qualified.text


def segmented_world(*paragraphs):
    doc = parse(prose_xml(*paragraphs).encode())
    *_, result = build(doc, counter=lambda s: len(s.split()) * 11)
    return finish(doc, result, make_item(doc, PROSE_COMPONENTS))


def test_stage_b_never_counts_a_continuation_segment_as_delivered_evidence():
    w = segmented_world(f"{P_A} {P_C}")
    holder = w.chunk(P_A)
    continuation = w.chunk(P_C)
    assert holder is not continuation
    assert {c.usage for c in continuation.contributors} >= {"content_continuation"}
    assert deliver(w, [holder]).evaluate().determination == "automatic_delivered"
    only_part = deliver(w, [continuation]).evaluate()
    assert only_part.determination != "automatic_delivered"
