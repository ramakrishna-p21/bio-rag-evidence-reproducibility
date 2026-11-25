"""Header-block defaults and row-local coordinates (canonical_whole_units_v3), a development candidate.

Expected strings are written out by hand from the fixtures. The reconstruction oracle is its own
reader of the v3 grammar and compares every printed cell with the canonical TableCell values.
"""

import re

import pytest
from test_chunking import TABLE, native_table
from test_chunking_v2 import CLINICAL, NAMES, TRI, v2_units
from test_documents_jats import parse
from test_retrieval_models import ref, selection, settings

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.retrieval.chunking import (
    V3_LEGEND,
    chunk_document,
    chunks_ref,
    discover_occurrences,
    render_units,
    validate_chunking,
)
from biorag_repro.retrieval.models import ConfigurationRef
from biorag_repro.retrieval.storage import record_ref

V1 = "canonical_whole_units_v1"
V2 = "canonical_whole_units_v2"
V3 = "canonical_whole_units_v3"

CELL = re.compile(r"^\[(?:r(\d+))?c(\d+)((?: [a-z_]+=\S+?)*)\] ?(.*)$")
PAIRS = re.compile(r" ([a-z_]+)=(\S+?)(?=[ \]])")


def v3_build(doc, budget=192, counter=len, policy="complete_units"):
    selected = selection()
    cfg = settings().model_copy(
        update={"chunk_tokens": budget, "chunking_algorithm": V3, "unit_policy": policy}
    )
    cfg_ref = ConfigurationRef(
        artifact=ref("configuration"),
        resolved_sha256=sha256_bytes(canonical_json_bytes(cfg.model_dump())),
    )
    return selected, cfg, cfg_ref, chunk_document(doc, selected, cfg_ref, cfg, counter)


def v3_units(doc):
    return [
        u for u in render_units(doc, discover_occurrences(doc, ref()), V3) if u.kind == "table_row"
    ]


def read_unit(text):
    """Return {(row, column): (span, attributes, text)} for every cell line of one v3 row unit."""
    lines = text.split("\n")
    defaults_line = [line for line in lines if line.startswith("[cell_defaults ")]
    assert len(defaults_line) == 1
    defaults = dict(PAIRS.findall(defaults_line[0]))
    found = {}
    block, base, row = None, None, None
    for line in lines:
        if line.startswith("[headers"):
            block, base = "headers", {**defaults, **dict(PAIRS.findall(line))}
            continue
        if line == "[/headers]":
            block = None
            continue
        if m := re.fullmatch(r"\[row (\d+)\]", line):
            block, base, row = "row", defaults, int(m[1])
            continue
        m = CELL.match(line)
        if block is None or not m:
            continue
        # Row-local coordinates are only allowed inside a row block.
        if m[1] is None:
            assert block == "row"
        cell_row = int(m[1]) if m[1] is not None else row
        values = {**base, **dict(PAIRS.findall(m[3] + "]"))}
        record = (
            tuple(int(n) for n in values["span"].split("x")),
            {k: TRI[values[k]] for k in NAMES},
            m[4],
        )
        assert found.setdefault((cell_row, int(m[2])), record) == record
    return found


def assert_reconstructs(doc):
    printed = {}
    for unit in v3_units(doc):
        for key, record in read_unit(unit.context + unit.text).items():
            assert printed.setdefault(key, record) == record
    cells = [c for n in doc.nodes if n.table for c in n.table.cells]
    assert len(printed) == len(cells)
    for cell in cells:
        span, attrs, cell_text = printed[(cell.row, cell.column)]
        assert span == (cell.row_span, cell.column_span)
        assert attrs == {k: getattr(cell, a) for k, a in NAMES.items()}
        assert cell_text == cell.text
    return printed


def test_v3_jats_rows_use_row_local_coordinates_and_keep_spanned_cells_absolute():
    rows = v3_units(parse(TABLE.encode()))
    assert len(rows) == 3
    full = rows[1].context + rows[1].text
    assert (
        "[cell_defaults span=1x1 header=true row_header=? column_header=? row_section=? fillable=?]\n"
        f"{V3_LEGEND}\n"
        "[headers]\n[r0c0] Condition\n[r0c1] p\n[/headers]\n"
    ) in full
    assert rows[0].text == "[row 0]\n[c0] Condition\n[c1] p"
    assert rows[1].text == "[row 1]\n[c0 span=2x1] A\n[c1 header=false] 0.05"
    # The two-row label A starts in row 1, so in row 2 it keeps its absolute coordinates.
    assert rows[2].text == "[row 2]\n[r1c0 span=2x1] A\n[c1 header=false] 0.5"


def test_v3_headers_tag_declares_only_values_that_differ_from_cell_defaults():
    rows = v3_units(parse(CLINICAL.encode()))
    full = rows[2].context + rows[2].text
    assert (
        "[cell_defaults span=1x1 header=false row_header=? column_header=? row_section=? fillable=?]"
        in full
    )
    assert (
        "[headers header=true]\n[r0c0 span=2x1] Marker\n[r0c1 span=1x2] Treated\n"
        "[r0c3 span=2x1] p\n[r1c1] Baseline (ng/mL)\n[r1c2] Week 12 (ng/mL)\n[/headers]\n"
    ) in full
    assert (
        rows[2].text
        == "[row 2]\n[c0 header=true] IL-6\n[c1] 12.4 ± 3.1\n[c2] 6.8 ± 2.2\n[c3] <0.001*"
    )


def test_v3_reconstructs_every_cell_of_the_realistic_jats_table():
    printed = assert_reconstructs(parse(CLINICAL.encode()))
    assert printed[(0, 0)][0] == (2, 1)
    assert printed[(0, 1)][0] == (1, 2)
    assert printed[(2, 0)][1]["header"] is True
    assert printed[(2, 1)][1]["header"] is False
    text = "".join(u.context + u.text for u in v3_units(parse(CLINICAL.encode())))
    assert "* p < 0.05 vs baseline; NA, not assessed; LOD, limit of detection." in text
    assert "Table 2. Serum markers (mean ± SD)" in text


def test_v3_reconstructs_native_pdf_table_with_all_roles():
    doc = native_table()
    assert_reconstructs(doc)
    # The native table's header cells differ from the body majority, so the tag is not empty.
    assert any("[headers " in u.context for u in v3_units(doc))


@pytest.mark.parametrize("header_value", [None, False])
def test_v3_unknown_and_false_stay_distinct_inside_the_headers_tag(header_value):
    doc = native_table()
    table = next(n.table for n in doc.nodes if n.table)
    headers = [c for c in table.cells if c.native_column_header]
    assert headers
    for cell in headers:
        cell.native_fillable = header_value
    body = [c for c in table.cells if not c.native_column_header]
    body[0].native_fillable = None if header_value is False else False
    printed = assert_reconstructs(doc)
    for cell in headers:
        assert printed[(cell.row, cell.column)][1]["fillable"] is header_value
    tags = {
        line for u in v3_units(doc) for line in u.context.split("\n") if line.startswith("[headers")
    }
    assert any(f"fillable={'?' if header_value is None else 'false'}" in t for t in tags) or all(
        "fillable" not in t for t in tags
    )


def test_v3_headers_tag_ties_resolve_false_true_unknown():
    doc = native_table()
    table = next(n.table for n in doc.nodes if n.table)
    headers = sorted((c for c in table.cells if c.native_column_header), key=lambda c: c.column)
    values = [True, False, None, True, False, None][: len(headers)]
    for cell, value in zip(headers, values, strict=True):
        cell.native_fillable = value
    assert_reconstructs(doc)


def test_v3_whole_row_at_limit_is_indexed_and_one_token_over_is_excluded_untruncated():
    doc = parse(TABLE.encode())

    def counter_for(row_tokens):
        return lambda s: row_tokens if "[row 1]" in s and "0.05" in s else len(s.split())

    _, _, _, fits = v3_build(doc, 192, counter_for(192))
    assert any("0.05" in c.rendered_text for c in fits.chunks)
    _, _, _, over = v3_build(doc, 192, counter_for(193))
    assert not any("0.05" in c.rendered_text for c in over.chunks)
    diagnostic = next(
        d for d in over.chunk_set.diagnostics if d.kind == "table_row" and "0.05" in d.rendered_text
    )
    assert (diagnostic.code, diagnostic.actual_tokens) == ("oversized", 193)
    assert diagnostic.rendered_text.endswith("[c0 span=2x1] A\n[c1 header=false] 0.05")
    accounted = {a.occurrence_id: a.status for a in over.chunk_set.accounting}
    cell = next(o for o in over.chunk_set.occurrences if o.text == "0.05")
    assert accounted[cell.occurrence_id] == "oversized"


def test_v3_is_deterministic_and_validates_against_canonical():
    doc = parse(CLINICAL.encode())
    selected, cfg, cfg_ref, first = v3_build(doc, 384, lambda s: len(s.split()))
    _, _, _, second = v3_build(doc, 384, lambda s: len(s.split()))
    assert first == second
    assert record_ref(first.chunk_set) == record_ref(second.chunk_set)
    validate_chunking(first, doc, selected, cfg_ref, cfg, lambda s: len(s.split()))


def test_v3_rendering_does_not_depend_on_paper_identity():
    doc = parse(CLINICAL.encode())
    other = doc.model_copy(update={"paper_id": "P049"})
    assert [u.context + u.text for u in v3_units(doc)] == [
        u.context + u.text for u in v3_units(other)
    ]


def test_v3_is_independent_of_the_prose_segmentation_policy_for_table_rows():
    doc = parse(CLINICAL.encode())
    *_, plain = v3_build(doc, 384, lambda s: len(s.split()))
    *_, segmenting = v3_build(
        doc, 384, lambda s: len(s.split()), policy="complete_units_or_sentence_segments"
    )
    rows = lambda r: [c.rendered_text for c in r.chunks if "[row " in c.rendered_text]
    assert rows(plain) == rows(segmenting)


def test_v1_and_v2_rendering_is_unchanged_by_v3():
    # v2 text pinned in test_chunking_v2; here: v2 keeps absolute coordinates and a bare headers tag.
    rows = v2_units(parse(CLINICAL.encode()))
    full = rows[2].context + rows[2].text
    assert "[headers]\n[r0c0 span=2x1 header=true] Marker\n" in full
    assert rows[2].text.startswith("[row 2]\n[r2c0 header=true] IL-6\n[r2c1] 12.4 ± 3.1")
    v1 = [
        u
        for u in render_units(
            parse(CLINICAL.encode()), discover_occurrences(parse(CLINICAL.encode()), ref())
        )
        if u.kind == "table_row"
    ]
    assert (
        "[r2c1 span=1x1 header=false row_header=? column_header=? row_section=? fillable=?] 12.4 ± 3.1"
        in v1[2].text
    )


def test_v3_has_its_own_identity_and_cannot_be_validated_as_v2():
    doc = parse(TABLE.encode())
    selected, v3_cfg, v3_ref, v3 = v3_build(doc, 384)
    v2_cfg = v3_cfg.model_copy(update={"chunking_algorithm": V2})
    v2_ref = ConfigurationRef(
        artifact=v3_ref.artifact,
        resolved_sha256=sha256_bytes(canonical_json_bytes(v2_cfg.model_dump())),
    )
    v2 = chunk_document(doc, selected, v2_ref, v2_cfg, len)
    assert v2_ref.resolved_sha256 != v3_ref.resolved_sha256
    assert v2.chunk_set.chunks.artifact_id != v3.chunk_set.chunks.artifact_id
    assert chunks_ref(v3.chunks, selected, v3_ref, V3) == v3.chunk_set.chunks
    assert chunks_ref(v3.chunks, selected, v3_ref, V2) != v3.chunk_set.chunks
    with pytest.raises(ValueError):
        validate_chunking(v3, doc, selected, v2_ref, v2_cfg, len)


def test_v1_stays_the_configuration_default():
    assert settings().chunking_algorithm == V1
    assert settings().unit_policy == "complete_units"
