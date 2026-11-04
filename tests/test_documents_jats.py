"""Source-faithful JATS ingestion on original synthetic XML."""

import pytest

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import raw_source_artifact_id
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.documents.jats import ingest_jats
from biorag_repro.documents.models import CanonicalDocument

XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE article SYSTEM "https://example.invalid/jats.dtd">
<article xmlns:m="http://www.w3.org/1998/Math/MathML"><front><article-meta><title-group>
<article-title>Study α</article-title></title-group></article-meta></front>
<body><sec id="s1"><title>Results</title><p>Effect −<italic>1.2e-3</italic> and <xref ref-type="bibr" rid="r1">[1]</xref>.</p>
<sec><title>Nested</title><sec><title>Deeper</title><sec><title>Deepest</title><p>p&lt;0.05</p></sec></sec></sec>
<list><list-item><p>first</p></list-item></list>
<table-wrap id="t1"><caption><p>Table caption</p></caption><table><thead><tr><th>A</th><th>B</th><th>C</th></tr></thead>
<tbody><tr><td rowspan="2">x</td><td colspan="2">1</td></tr><tr><td>2</td><td>3</td></tr></tbody></table>
<table-wrap-foot><fn><p>footnote</p></fn></table-wrap-foot></table-wrap>
<fig id="f1"><caption><p>Figure caption</p></caption><graphic href="figure.png"/></fig>
<disp-formula id="eq1"><alternatives><m:math><m:mi>x</m:mi></m:math><tex-math>\\alpha</tex-math><inline-graphic href="eq.png"/></alternatives></disp-formula>
<supplementary-material id="sup1" href="supp.pdf"><caption><p>Supplement</p></caption></supplementary-material>
<boxed-text custom="kept"><p>unusual</p></boxed-text></sec></body>
<back><ref-list><ref id="r1"><mixed-citation>Reference</mixed-citation></ref></ref-list></back></article>""".encode()


def parse(content: bytes = XML) -> CanonicalDocument:
    digest = sha256_bytes(content)
    source_id = raw_source_artifact_id(paper_id="P001", representation="JATS", sha256=digest)
    return ingest_jats(
        content,
        paper_id="P001",
        source_artifact_id=source_id,
        source_sha256=digest,
        source_logical_ref="synthetic/article.xml",
        configuration_sha256="a" * 64,
    )


def nodes(doc: CanonicalDocument, kind: str):
    return [node for node in doc.nodes if node.kind == kind]


def test_doctype_is_allowed_without_fetch_and_mixed_content_is_ordered():
    doc = parse()
    paragraph = next(node for node in nodes(doc, "paragraph") if "Effect " in str(node.segments))
    assert [segment.text for segment in paragraph.segments] == [
        "Effect −",
        "1.2e-3",
        " and ",
        "[1]",
        ".",
    ]
    assert paragraph.mapping.path.endswith("/p[1]")
    assert doc.source.sha256 == sha256_bytes(XML)
    assert doc.source.artifact_id.startswith("raw_")
    assert doc.source_doctype_declared is True


def test_generic_wrapper_retains_direct_text_and_child_tails():
    xml = (
        b"<article><body><custom>before<italic>inside</italic> after"
        b"<bold>second</bold> end</custom></body></article>"
    )
    doc = parse(xml)
    wrapper = next(node for node in doc.nodes if node.source_tag == "custom")
    assert [(part.mapping.path.rsplit("/", 1)[-1], part.text) for part in wrapper.segments] == [
        ("text()", "before"),
        ("tail()", " after"),
        ("tail()", " end"),
    ]


def test_paths_ids_xrefs_sections_and_source_order():
    doc = parse()
    assert [node.source_id for node in nodes(doc, "section")][:1] == ["s1"]
    assert len(nodes(doc, "section")) == 4
    titles = [node for node in nodes(doc, "title") if node.mapping.path.endswith("/title[1]")]
    assert len(titles) >= 4
    xref = nodes(doc, "xref")[0]
    assert xref.rid == "r1" and xref.ref_type == "bibr"
    assert xref.target_node_ids == [nodes(doc, "reference")[0].node_id]
    assert len(nodes(doc, "list_item")) == 1
    assert nodes(doc, "figure")[0].source_id == "f1"
    assert nodes(doc, "supplement")[0].source_id == "sup1"
    assert len({node.node_id for node in doc.nodes}) == len(doc.nodes)
    assert [node.order for node in doc.nodes] == list(range(len(doc.nodes)))


def test_namespaced_paths_and_same_name_ordinals():
    doc = parse(
        b'<article xmlns:m="urn:math"><body><p>x</p><p>y</p><m:math><m:mi>x</m:mi></m:math></body></article>'
    )
    paragraphs = nodes(doc, "paragraph")
    assert paragraphs[0].mapping.path.endswith("/p[1]")
    assert paragraphs[1].mapping.path.endswith("/p[2]")
    assert nodes(doc, "source_construct")[-1].mapping.path.endswith("/{urn:math}mi[1]")
    assert doc.source_doctype_declared is False


def test_table_coordinates_spans_caption_and_footnote():
    doc = parse()
    table = nodes(doc, "table")[0]
    assert table.table is not None
    assert (table.table.row_count, table.table.column_count) == (3, 3)
    assert [
        (cell.row, cell.column, cell.row_span, cell.column_span) for cell in table.table.cells
    ] == [
        (0, 0, 1, 1),
        (0, 1, 1, 1),
        (0, 2, 1, 1),
        (1, 0, 2, 1),
        (1, 1, 1, 2),
        (2, 1, 1, 1),
        (2, 2, 1, 1),
    ]
    assert table.table.caption_node_ids
    assert table.table.footnote_node_ids
    assert table.table.structure_state == "exact_native"
    assert all(cell.mapping.path and cell.node_id for cell in table.table.cells)


def test_formula_alternatives_and_unsupported_are_visible():
    doc = parse()
    formula = nodes(doc, "formula")[0]
    assert formula.source_id == "eq1"
    assert formula.content_state == "present"
    assert any("http://www.w3.org/1998/Math/MathML" in n.mapping.path for n in doc.nodes)
    assert any(n.source_tag == "tex-math" for n in doc.nodes)
    assert next(n for n in doc.nodes if n.source_tag == "tex-math").segments[0].text == "\\alpha"
    assert any(n.source_tag == "inline-graphic" for n in doc.nodes)
    boxed = next(n for n in doc.nodes if n.source_tag == "boxed-text")
    assert boxed.kind == "source_construct" and boxed.content_state == "unsupported"
    assert boxed.attributes["custom"] == "kept"
    assert boxed.mapping.path


def test_graphic_only_formula_keeps_marker_without_claiming_decoded_text():
    doc = parse(
        b'<article><body><disp-formula><graphic href="eq.svg"/></disp-formula></body></article>'
    )
    formula = nodes(doc, "formula")[0]
    assert formula.content_state == "marker_only"
    assert next(n for n in doc.nodes if n.source_tag == "graphic").attributes["href"] == "eq.svg"


def test_unresolved_xref_and_source_mismatch_are_explicit():
    doc = parse(b'<article><body><p><xref rid="missing">see</xref></p></body></article>')
    xref = nodes(doc, "xref")[0]
    assert xref.target_node_ids == [] and xref.relation_state == "unavailable"
    with pytest.raises(ValueError, match="source hash"):
        ingest_jats(
            b"<article/>",
            paper_id="P001",
            source_artifact_id="raw_bad",
            source_sha256="a" * 64,
            source_logical_ref="synthetic/article.xml",
            configuration_sha256="a" * 64,
        )


def test_empty_table_structure_is_visible_and_bad_spans_fail():
    doc = parse(b"<article><body><table-wrap><table/></table-wrap></body></article>")
    assert nodes(doc, "table")[0].table.structure_state == "unavailable"
    with pytest.raises(ValueError, match="row span"):
        parse(b'<article><body><table><tr><td rowspan="2">x</td></tr></table></body></article>')


def test_multi_target_xref_preserves_all_resolved_targets():
    doc = parse(
        b'<article><body><p><xref rid="f1 f2">figures</xref></p>'
        b'<fig id="f1"/><fig id="f2"/></body></article>'
    )
    xref = nodes(doc, "xref")[0]
    assert xref.rid == "f1 f2"
    assert xref.target_node_ids == [node.node_id for node in nodes(doc, "figure")]
    assert xref.relation_state == "exact_native"


@pytest.mark.parametrize(
    "content",
    [
        b"<article>",
        b"<!DOCTYPE article [<!ENTITY x SYSTEM 'file:///etc/passwd'>]><article>&x;</article>",
        b"<!DOCTYPE article [<!ENTITY x 'abc'>]><article>&x;</article>",
    ],
)
def test_malformed_or_entity_dependent_xml_fails(content: bytes):
    with pytest.raises(ValueError):
        parse(content)


def test_identity_and_serialization_are_deterministic_and_source_scoped():
    first, second = parse(), parse()
    assert serialize_project_json(first, CanonicalDocument) == serialize_project_json(
        second, CanonicalDocument
    )
    assert [node.node_id for node in first.nodes] == [node.node_id for node in second.nodes]
    changed = parse(XML.replace(b"Table caption", b"Other caption"))
    assert first.nodes[0].node_id != changed.nodes[0].node_id
