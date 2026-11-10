"""Scientific rendering expectations are literal, source-derived fixture oracles."""

import pytest
from test_documents_jats import parse
from test_retrieval_models import ref, selection, settings

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.documents.chunking import (
    chunk_document,
    discover_occurrences,
    render_units,
    validate_chunking,
)
from biorag_repro.retrieval.models import ConfigurationRef
from biorag_repro.retrieval.storage import record_ref
from biorag_repro.retrieval.validation import validate_occurrences


def build(xml, budget=192, counter=len):
    doc = parse(xml.encode() if isinstance(xml, str) else xml)
    selected = selection()
    config = settings().model_copy(update={"chunk_tokens": budget})
    config_ref = ConfigurationRef(
        artifact=ref("configuration"),
        resolved_sha256=sha256_bytes(canonical_json_bytes(config.model_dump())),
    )
    result = chunk_document(doc, selected, config_ref, config, counter)
    return doc, selected, config_ref, config, result


def test_discovery_uses_source_slots_in_text_tail_order_without_text_deduplication():
    doc = parse(
        b"<article><body><custom>A<italic>B</italic> C<bold>D</bold> E</custom><p>same</p><p>same</p></body></article>"
    )
    occurrences = discover_occurrences(doc, ref())
    text = [o for o in occurrences if o.text is not None]
    assert [o.text for o in text] == ["A", "B", " C", "D", " E", "same", "same"]
    assert [o.mapping.path.rsplit("/", 1)[-1] for o in text[:5]] == [
        "text()",
        "text()",
        "tail()",
        "text()",
        "tail()",
    ]
    assert text[-1].occurrence_id != text[-2].occurrence_id
    assert len({o.occurrence_id for o in occurrences}) == len(occurrences)
    validate_occurrences(occurrences, doc, ref())


def test_plain_and_nested_mixed_content_preserves_exact_scientific_text():
    xml = "<article><body><p>Effect −<italic>1.2e−3</italic>; p&lt;0.05; H<sub>2</sub>O; x<sup>2</sup>; BRCA1 c.68_69delAG was not increased.</p></body></article>"
    doc, selected, cfgref, cfg, result = build(xml, counter=lambda s: len(s.split()))
    assert [c.rendered_text for c in result.chunks] == [
        "Effect −1.2e−3; p<0.05; H<sub>2</sub>O; x<sup>2</sup>; BRCA1 c.68_69delAG was not increased."
    ]
    assert all(a.status in {"included", "nontextual"} for a in result.chunk_set.accounting)
    validate_chunking(result, doc, selected, cfgref, cfg, lambda s: len(s.split()))


def test_generic_wrapper_preserves_interleaving_and_paragraph_is_not_repeated():
    _, _, _, _, result = build(
        "<article><body><custom>before<italic>inside</italic> after<bold>second</bold> end</custom><p>A <italic>B</italic> C</p></body></article>"
    )
    assert result.chunks[0].rendered_text == "beforeinside aftersecond end\n\nA B C"
    assert sum(o.text == "B" for o in result.chunk_set.occurrences) == 1


def test_namespaced_mixed_text_preserves_uri_and_math_structure():
    doc = parse(
        b'<article xmlns:m="urn:math"><body><p>a<m:msup><m:mi>x</m:mi><m:mn>2</m:mn></m:msup>b</p></body></article>'
    )
    occurrences = discover_occurrences(doc, ref())
    units = render_units(doc, occurrences)
    assert (
        units[0].text
        == "a<{urn:math}msup><{urn:math}mi>x</{urn:math}mi><{urn:math}mn>2</{urn:math}mn></{urn:math}msup>b"
    )


def test_sections_lists_captions_and_no_unrelated_section_packing():
    xml = "<article><body><sec><title>First</title><p>A.</p><list><list-item><p>B.</p></list-item></list><fig><caption><p>C.</p></caption></fig></sec><sec><title>Second</title><p>D.</p></sec></body></article>"
    doc, _, _, _, result = build(xml, counter=lambda s: len(s.split()))
    text = [c.rendered_text for c in result.chunks]
    assert any("[section]\nFirst\n[/section]\nA." in t for t in text)
    assert not any("A." in t and "D." in t for t in text)
    assert sum("C." in unit.text for unit in render_units(doc, result.chunk_set.occurrences)) == 1
    assert sum(o.text == "C." for o in result.chunk_set.occurrences) == 1
    assert any(c.usage == "repeated_context" for chunk in result.chunks for c in chunk.contributors)


@pytest.mark.parametrize("budget", [192, 384])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_exact_construction_boundaries_and_continue_after_oversized_unit(budget, offset):
    value = "x" * (budget + offset)
    _, _, _, _, result = build(f"<article><body><p>{value}</p><p>end</p></body></article>", budget)
    if offset <= 0:
        assert result.chunks[0].rendered_text == value
        assert result.chunks[0].construction_tokens == budget + offset
    else:
        assert [c.rendered_text for c in result.chunks] == ["end"]
        diagnostic = next(d for d in result.chunk_set.diagnostics if d.code == "oversized")
        assert (diagnostic.actual_tokens, diagnostic.token_limit) == (budget + 1, budget)
        assert diagnostic.contributors and diagnostic.reason
        assert (
            result.chunk_set.accounting[
                [o.text for o in result.chunk_set.occurrences].index(value)
            ].status
            == "oversized"
        )
    assert all(c.construction_tokens <= budget for c in result.chunks)


def test_repeated_section_context_is_included_in_the_construction_count():
    _, _, _, _, result = build(
        "<article><body><sec><title>"
        + "T" * 180
        + "</title><p>value has a qualifier</p></sec></body></article>"
    )
    assert any(
        d.code == "oversized" and d.actual_tokens > 192 and d.contributors
        for d in result.chunk_set.diagnostics
    )
    assert not any("value has a qualifier" in c.rendered_text for c in result.chunks)


TABLE = """<article><body><table-wrap><caption><p>Trial</p></caption><table><thead><tr><th>Condition</th><th>p</th></tr></thead><tbody><tr><th rowspan="2">A</th><td>0.05</td></tr><tr><td>0.5</td></tr></tbody></table><table-wrap-foot><p>Qualified result.</p></table-wrap-foot></table-wrap></body></article>"""


def test_jats_table_rows_headers_spans_and_supplied_relations():
    doc = parse(TABLE.encode())
    occurrences = discover_occurrences(doc, ref())
    rows = [u for u in render_units(doc, occurrences) if u.kind == "table_row"]
    assert len(rows) == 3
    complete = rows[1].context + rows[1].text
    assert (
        "[table structure=exact_native captions=present footnotes=present orientation=?]"
        in complete
    )
    assert "[caption]\nTrial\n[/caption]" in complete
    assert "[footnote]\nQualified result.\n[/footnote]" in complete
    assert (
        "[r0c0 span=1x1 header=true row_header=? column_header=? row_section=? fillable=?] Condition"
        in complete
    )
    assert (
        "[r1c0 span=2x1 header=true row_header=? column_header=? row_section=? fillable=?] A"
        in rows[1].text
    )
    assert (
        "[r1c1 span=1x1 header=false row_header=? column_header=? row_section=? fillable=?] 0.05"
        in rows[1].text
    )
    assert (
        "[r2c1 span=1x1 header=false row_header=? column_header=? row_section=? fillable=?] 0.5"
        in rows[2].text
    )
    assert "0.05" not in rows[2].text
    a = next(o for o in occurrences if o.text == "A")
    assert any(
        c.occurrence_id == a.occurrence_id and c.usage == "repeated_context"
        for c in rows[2].contributors
    )


def test_two_tables_never_mix_and_oversized_rows_are_not_shortened():
    xml = "<article><body><table><tr><td>first</td></tr></table><table><tr><td>second</td></tr></table></body></article>"
    _, _, _, _, result = build(xml, counter=lambda s: len(s.split()))
    assert len(result.chunks) == 2
    assert not any(
        "first" in c.rendered_text and "second" in c.rendered_text for c in result.chunks
    )
    _, _, _, _, oversized = build(TABLE)
    assert any(
        d.code == "oversized" and d.kind == "table_row" for d in oversized.chunk_set.diagnostics
    )
    assert not any("0.05" in c.rendered_text for c in oversized.chunks)


def test_empty_cell_and_unrelated_nearby_caption_are_not_invented_relations():
    doc = parse(
        b"<article><body><caption>Nearby</caption><table><tr><td/><td>x</td></tr></table></body></article>"
    )
    occurrences = discover_occurrences(doc, ref())
    validate_occurrences(occurrences, doc, ref())
    row = next(u for u in render_units(doc, occurrences) if u.kind == "table_row")
    assert "captions=absent footnotes=absent" in row.context
    assert "Nearby" not in row.text
    assert (
        "[r0c0 span=1x1 header=false row_header=? column_header=? row_section=? fillable=?] "
        in row.text
    )
    assert any(
        o.field == "nontextual"
        and o.owner_node_id == next(n.node_id for n in doc.nodes if n.kind == "table_cell")
        for o in occurrences
    )


def test_empty_or_entirely_excluded_content_remains_accounted_for():
    for xml in [
        "<article><body/></article>",
        "<article><body><disp-formula><graphic href='formula.svg'/></disp-formula></body></article>",
    ]:
        _, _, _, _, result = build(xml)
        assert result.chunks == []
        assert result.chunk_set.occurrences
        assert len(result.chunk_set.accounting) == len(result.chunk_set.occurrences)
        assert all(a.reason for a in result.chunk_set.accounting)


def test_canonical_is_not_reparsed_or_mutated(monkeypatch):
    doc, selected, cfgref, cfg, result = build("<article><body><p>Exact.</p></body></article>")
    before = doc.model_dump_json()
    from biorag_repro.documents import jats

    monkeypatch.setattr(jats.ET, "fromstring", lambda *_: pytest.fail("chunking reparsed XML"))
    again = chunk_document(doc, selected, cfgref, cfg, len)
    assert result == again and doc.model_dump_json() == before
    assert [record_ref(c) for c in result.chunks] == [record_ref(c) for c in again.chunks]


@pytest.mark.parametrize(
    "attack",
    [
        "omit",
        "duplicate",
        "value",
        "contributor",
        "certainty",
        "suppress_diagnostic",
        "tokens",
        "over_limit",
        "order",
        "accounting",
    ],
)
def test_rehashed_scientific_counterfeits_reject_at_the_same_boundary(attack):
    xml = (
        "<article><body><p>p&lt;0.05, not 0.5.</p><p>"
        + "x" * 193
        + "</p><p>last</p></body></article>"
    )
    doc, selected, cfgref, cfg, good = build(xml)
    validate_chunking(good, doc, selected, cfgref, cfg, len)
    from copy import deepcopy

    bad = deepcopy(good)
    if attack == "omit":
        bad.chunk_set.occurrences.pop()
    elif attack == "duplicate":
        bad.chunk_set.occurrences.append(bad.chunk_set.occurrences[0])
    elif attack == "value":
        bad.chunks[0].rendered_text = "p<0.5, not 0.05."
    elif attack == "contributor":
        bad.chunks[0].contributors[0].node_id = doc.nodes[0].node_id
    elif attack == "certainty":
        bad.chunk_set.occurrences[0].mapping.state = "derived"
    elif attack == "suppress_diagnostic":
        bad.chunk_set.diagnostics = []
    elif attack == "tokens":
        bad.chunks[0].construction_tokens = 1
    elif attack == "over_limit":
        bad.chunks[0].rendered_text = "x" * 193
        bad.chunks[0].construction_tokens = 191
    elif attack == "order":
        bad.chunks.reverse()
    else:
        bad.chunk_set.accounting[0].reason = "invented"
    # Renew downstream chunk IDs and JSONL identity before scientific validation.
    from biorag_repro.documents.chunking import chunks_ref

    bad.chunk_set.chunk_ids = [record_ref(c).artifact_id for c in bad.chunks]
    bad.chunk_set.chunks = chunks_ref(bad.chunks, selected, cfgref, "canonical_whole_units_v1")
    with pytest.raises(ValueError):
        validate_chunking(bad, doc, selected, cfgref, cfg, len)


def test_corrupt_hierarchy_is_rejected_without_guessing_source_order():
    doc, selected, cfgref, cfg, _ = build("<article><body><p>value</p></body></article>")
    doc.nodes[0].parent_node_id = doc.nodes[1].node_id
    with pytest.raises(ValueError, match="hierarchy|cycle"):
        chunk_document(doc, selected, cfgref, cfg, len)


def native_table():
    from types import SimpleNamespace

    from test_documents_table_fidelity import _native_data, _typed_native

    from biorag_repro.documents.pdf import canonicalize_native

    raw = _native_data()
    typed = _typed_native(raw)
    extra = dict(raw["tables"][0]["data"]["table_cells"][0])
    extra.update(
        text="", start_row_offset_idx=4, end_row_offset_idx=5, row_header=False, column_header=False
    )
    raw["tables"][0]["data"]["grid"].append([extra])
    raw["tables"][0]["data"]["num_rows"] = 5
    typed.tables[0].data.grid.append([SimpleNamespace(**extra)])
    typed.tables[0].data.num_rows = 5
    return canonicalize_native(typed, "P001", ref("raw"), ref("native"), "a" * 64, saved_native=raw)


def test_native_table_roles_grid_only_cells_and_unavailable_links():
    doc = native_table()
    occurrences = discover_occurrences(doc, ref())
    rows = [u for u in render_units(doc, occurrences) if u.kind == "table_row"]
    assert len(rows) == 5
    complete = rows[0].context + rows[0].text
    assert "captions=unavailable footnotes=unavailable orientation=rot_90" in complete
    assert (
        "[r0c0 span=1x1 header=true row_header=true column_header=false row_section=false fillable=false] ClusterONE"
        in complete
    )
    assert (
        "[r1c0 span=1x1 header=true row_header=false column_header=true row_section=false fillable=false] Score"
        in complete
    )
    assert any(
        o.mapping.native_ref == "#/tables/0/data/grid/4/0" and o.text == "" for o in occurrences
    )
    assert (
        "[r4c0 span=1x1 header=false row_header=false column_header=false row_section=false fillable=false] "
        in rows[4].text
    )


def test_native_unpositioned_order_and_unverified_formula_are_not_upgraded():
    from biorag_repro.documents.models import (
        CanonicalDocument,
        CanonicalNode,
        SourceMapping,
        TextSegment,
    )

    mapping = SourceMapping(
        kind="docling",
        state="approximate",
        source_artifact_id="raw",
        native_artifact=ref("native"),
        native_ref="#/texts/0",
    )
    node = CanonicalNode(
        node_id="f",
        parent_node_id=None,
        order=0,
        kind="formula",
        source_tag="Formula",
        mapping=mapping,
        native_order_state="unpositioned",
        interpretation_state="ambiguous",
        native_orig="x²",
        orig_qualification="unverified_formula",
        content_state="unverified_native_rendition",
    )
    doc = CanonicalDocument(
        paper_id="P001",
        representation="PDF_DOCLING",
        source=ref("raw"),
        native_artifact=ref("native"),
        transformation_id="synthetic",
        configuration_sha256="a" * 64,
        status="complete",
        nodes=[node],
    )
    occurrences = discover_occurrences(doc, ref())
    unit = render_units(doc, occurrences)[0]
    assert (
        unit.text
        == "[qualification content=unverified_native_rendition interpretation=ambiguous mapping=approximate order=unpositioned orig=unverified_formula]\n[native_orig]\nx²\n[/native_orig]"
    )
    assert occurrences[0].native_order_state == "unpositioned"
    changed = node.model_copy(
        update={
            "segments": [TextSegment(text="same", mapping=mapping)],
            "native_orig": "same",
            "orig_qualification": "same_as_text",
            "content_state": "present",
        }
    )
    aliases = discover_occurrences(doc.model_copy(update={"nodes": [changed]}), ref())
    assert [o.field for o in aliases] == ["segment"]


def test_table_inside_paragraph_does_not_flatten_or_duplicate_cell_evidence():
    xml = "<article><body><p>Value <inline-formula><italic>1.2</italic></inline-formula> was observed.<table-wrap><label>Table 1</label><caption><p>Finding</p></caption><table><tr><th>Group</th><th>Value</th></tr><tr><td>A</td><td>1.2</td></tr></table></table-wrap></p></body></article>"
    _, _, _, _, result = build(xml, 384, lambda text: len(text.split()))
    text = [c.rendered_text for c in result.chunks]
    assert any("Value 1.2 was observed." in value for value in text)
    assert not any("GroupValue" in value or "A1.2" in value for value in text)
    from collections import Counter

    primary = Counter(
        c.occurrence_id
        for chunk in result.chunks
        for c in chunk.contributors
        if c.usage == "content"
    )
    assert all(n == 1 for n in primary.values())


def test_prose_crossing_embedded_table_is_explicitly_unsupported_not_reordered():
    xml = "<article><body><p>Before<table><tr><td>x</td></tr></table>after.</p></body></article>"
    _, _, _, _, result = build(xml, counter=lambda text: len(text.split()))
    assert not any("Beforeafter" in c.rendered_text for c in result.chunks)
    assert any(
        d.code == "unsupported" and "order" in d.reason for d in result.chunk_set.diagnostics
    )
    values = {o.occurrence_id: o.text for o in result.chunk_set.occurrences}
    assert all(
        a.status == "unsupported"
        for a in result.chunk_set.accounting
        if values[a.occurrence_id] in {"Before", "after."}
    )


def test_column_spans_and_nested_header_rows_remain_explicit():
    doc = parse(
        b"<article><body><table><thead><tr><th colspan='2'>Treatment</th></tr><tr><th>A</th><th>B</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table></body></article>"
    )
    rows = [u for u in render_units(doc, discover_occurrences(doc, ref())) if u.kind == "table_row"]
    full = rows[2].context + rows[2].text
    assert (
        "[r0c0 span=1x2 header=true row_header=? column_header=? row_section=? fillable=?] Treatment"
        in full
    )
    assert (
        "[r1c0 span=1x1 header=true row_header=? column_header=? row_section=? fillable=?] A"
        in full
    )
    assert (
        "[r1c1 span=1x1 header=true row_header=? column_header=? row_section=? fillable=?] B"
        in full
    )
    assert (
        "[r2c0 span=1x1 header=false row_header=? column_header=? row_section=? fillable=?] 1"
        in full
    )
    assert (
        "[r2c1 span=1x1 header=false row_header=? column_header=? row_section=? fillable=?] 2"
        in full
    )


@pytest.mark.parametrize(
    "old,new",
    [("r1c1", "r2c1"), ("header=true", "header=false"), ("captions=present", "captions=absent")],
)
def test_rehashed_table_relationship_counterfeits_reject(old, new):
    doc, selected, cfgref, cfg, good = build(TABLE, 384, lambda t: len(t.split()))
    counter = lambda t: len(t.split())
    validate_chunking(good, doc, selected, cfgref, cfg, counter)
    from copy import deepcopy

    from biorag_repro.documents.chunking import chunks_ref

    bad = deepcopy(good)
    target = next(chunk for chunk in bad.chunks if old in chunk.rendered_text)
    target.rendered_text = target.rendered_text.replace(old, new)
    bad.chunk_set.chunk_ids = [record_ref(chunk).artifact_id for chunk in bad.chunks]
    bad.chunk_set.chunks = chunks_ref(bad.chunks, selected, cfgref, "canonical_whole_units_v1")
    with pytest.raises(ValueError):
        validate_chunking(bad, doc, selected, cfgref, cfg, counter)


def test_unresolvable_source_slot_is_not_assigned_by_text_similarity():
    doc = parse(b"<article><body><p>same</p><p>same</p></body></article>")
    doc.nodes[-1].segments[0].mapping.path = "/missing/text()"
    with pytest.raises(ValueError, match="slots"):
        discover_occurrences(doc, ref())


def test_whitespace_outside_prose_is_visible_in_layout_diagnostics():
    _, _, _, _, result = build("<article>\n<body>\n<p>A.</p>\n</body>\n</article>")
    assert any(d.code == "layout_only" for d in result.chunk_set.diagnostics)
    assert [c.rendered_text for c in result.chunks] == ["A."]


def test_ambiguous_jats_titles_are_not_silently_selected():
    doc = parse(
        b"<article><body><sec><title>One</title><title>Two</title><p>Finding.</p></sec></body></article>"
    )
    units = render_units(doc, discover_occurrences(doc, ref()))
    paragraph = next(u for u in units if u.kind == "paragraph")
    assert paragraph.context == "[section context=ambiguous]\n"


def test_pdf_section_context_uses_explicit_linked_heading_levels_only():
    from biorag_repro.documents.models import (
        CanonicalDocument,
        CanonicalNode,
        SourceMapping,
        TextSegment,
    )

    nodes = []
    for ordinal, (kind, text, level) in enumerate(
        [
            ("title", "Results", 1),
            ("paragraph", "A.", None),
            ("title", "Methods", 1),
            ("paragraph", "B.", None),
        ]
    ):
        mapping = SourceMapping(
            kind="docling",
            state="exact_native",
            source_artifact_id="raw",
            native_artifact=ref("native"),
            native_ref=f"#/texts/{ordinal}",
        )
        nodes.append(
            CanonicalNode(
                node_id=f"n{ordinal}",
                parent_node_id=None,
                order=ordinal,
                kind=kind,
                source_tag="Synthetic",
                mapping=mapping,
                segments=[TextSegment(text=text, mapping=mapping)],
                native_heading_level=level,
                native_order_state="linked",
                interpretation_state="exact_native",
                native_content_layer="body",
            )
        )
    doc = CanonicalDocument(
        paper_id="P001",
        representation="PDF_DOCLING",
        source=ref("raw"),
        native_artifact=ref("native"),
        transformation_id="synthetic",
        configuration_sha256="a" * 64,
        status="complete",
        nodes=nodes,
    )
    units = render_units(doc, discover_occurrences(doc, ref()))
    assert units[1].context == "[section context=derived_heading_levels]\nResults\n[/section]\n"
    assert units[3].context == "[section context=derived_heading_levels]\nMethods\n[/section]\n"
    assert units[1].scope != units[3].scope
    doc.nodes[0].native_heading_level = None
    unknown = render_units(doc, discover_occurrences(doc, ref()))
    assert "Results" not in unknown[1].context


def test_native_array_order_cannot_override_structured_column_order():
    from test_documents_table_fidelity import _native_data, _typed_native

    from biorag_repro.documents.pdf import canonicalize_native

    raw = _native_data()
    prototype = raw["tables"][0]["data"]["table_cells"][0]
    cells = []
    for row, column, value in [(1, 1, "B"), (0, 1, "Value"), (1, 0, "A"), (0, 0, "Condition")]:
        cell = dict(prototype)
        cell.update(
            text=value,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=column,
            end_col_offset_idx=column + 1,
            row_header=False,
            column_header=row == 0,
            row_section=False,
            fillable=False,
        )
        cells.append(cell)
    data = raw["tables"][0]["data"]
    data.update(
        num_rows=2, num_cols=2, table_cells=cells, grid=[[cells[3], cells[1]], [cells[2], cells[0]]]
    )
    typed = _typed_native(raw)
    saved_cells = typed.tables[0].data.table_cells
    typed.tables[0].data.num_cols = 2
    typed.tables[0].data.grid = [[saved_cells[3], saved_cells[1]], [saved_cells[2], saved_cells[0]]]
    doc = canonicalize_native(typed, "P001", ref("raw"), ref("native"), "a" * 64, saved_native=raw)
    row = [u for u in render_units(doc, discover_occurrences(doc, ref())) if u.kind == "table_row"][
        1
    ]
    assert row.text.index("r1c0") < row.text.index("r1c1")
    assert row.context.index("r0c0") < row.context.index("r0c1")
