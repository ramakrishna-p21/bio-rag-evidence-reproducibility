"""Table cell-defaults rendering (canonical_whole_units_v2) and its separation from v1.

Expected strings are written out by hand from the fixtures. The reconstruction oracle parses the
rendered text with its own small reader and compares it with the canonical TableCell values.
"""

import re
from collections import Counter

import pytest
from test_chunking import TABLE, native_table
from test_documents_jats import parse
from test_retrieval_models import ref, selection, settings

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.retrieval.chunking import (
    chunk_document,
    chunks_ref,
    discover_occurrences,
    render_units,
    validate_chunking,
)
from biorag_repro.retrieval.models import ConfigurationRef, Phase5Configuration
from biorag_repro.retrieval.storage import record_ref

V1 = "canonical_whole_units_v1"
V2 = "canonical_whole_units_v2"
LEGEND = "Omitted cell attributes inherit cell_defaults."

# Multi-level header, a row-header column, a spanning row label, units, qualifiers and missing values.
CLINICAL = (
    "<article><body><table-wrap><caption><p>Table 2. Serum markers (mean ± SD)</p></caption>"
    "<table><thead>"
    "<tr><th rowspan='2'>Marker</th><th colspan='2'>Treated</th><th rowspan='2'>p</th></tr>"
    "<tr><th>Baseline (ng/mL)</th><th>Week 12 (ng/mL)</th></tr></thead><tbody>"
    "<tr><th>IL-6</th><td>12.4 ± 3.1</td><td>6.8 ± 2.2</td><td>&lt;0.001*</td></tr>"
    "<tr><th>TNF-α</th><td>NA</td><td>&lt;LOD</td><td>0.34</td></tr>"
    "<tr><th>CRP</th><td>5.0 ± 1.0</td><td>4.9 ± 1.1</td><td>0.87</td></tr>"
    "</tbody></table>"
    "<table-wrap-foot><p>* p &lt; 0.05 vs baseline; NA, not assessed; LOD, limit of detection.</p>"
    "</table-wrap-foot></table-wrap></body></article>"
)

LINE = re.compile(r"^\[r(\d+)c(\d+)((?: [a-z_]+=\S+)*)\] ?(.*)$")
DEFAULTS = re.compile(r"^\[cell_defaults((?: [a-z_]+=\S+)+)\]$", re.MULTILINE)
ATTRS = (
    "is_header",
    "native_row_header",
    "native_column_header",
    "native_row_section",
    "native_fillable",
)
NAMES = dict(
    zip(("header", "row_header", "column_header", "row_section", "fillable"), ATTRS, strict=True)
)
TRI = {"true": True, "false": False, "?": None}


def v2_settings(budget=192):
    return settings().model_copy(update={"chunk_tokens": budget, "chunking_algorithm": V2})


def v2_build(doc, budget=192, counter=len):
    selected = selection()
    cfg = v2_settings(budget)
    cfg_ref = ConfigurationRef(
        artifact=ref("configuration"),
        resolved_sha256=sha256_bytes(canonical_json_bytes(cfg.model_dump())),
    )
    return selected, cfg, cfg_ref, chunk_document(doc, selected, cfg_ref, cfg, counter)


def v2_units(doc):
    return [
        u for u in render_units(doc, discover_occurrences(doc, ref()), V2) if u.kind == "table_row"
    ]


def reconstruct(doc, units):
    """Read every printed cell back, applying the stated table defaults, keyed by (row, column)."""
    found = {}
    for unit in units:
        text = unit.context + unit.text
        table_defaults = DEFAULTS.findall(text)
        assert len(table_defaults) == 1, "exactly one cell_defaults declaration per unit"
        defaults = dict(pair.split("=") for pair in table_defaults[0].split())
        for line in text.splitlines():
            m = LINE.match(line)
            if not m or line.startswith("[row"):
                continue
            row, col, attrs, cell_text = int(m[1]), int(m[2]), m[3], m[4]
            values = {**defaults, **dict(a.split("=") for a in attrs.split())}
            span = tuple(int(n) for n in values["span"].split("x"))
            record = (
                span,
                {k: TRI[values[k]] for k in NAMES},
                cell_text,
            )
            assert found.setdefault((row, col), record) == record
    return found


def assert_reconstructs(doc):
    units = v2_units(doc)
    printed = reconstruct(doc, units)
    cells = [c for n in doc.nodes if n.table for c in n.table.cells]
    assert len(printed) == len(cells)
    for cell in cells:
        span, attrs, cell_text = printed[(cell.row, cell.column)]
        assert span == (cell.row_span, cell.column_span)
        assert attrs == {k: getattr(cell, a) for k, a in NAMES.items()}
        assert cell_text == cell.text


def test_v2_jats_rows_use_table_defaults_and_print_only_deviations():
    doc = parse(TABLE.encode())
    rows = v2_units(doc)
    assert len(rows) == 3
    full = rows[1].context + rows[1].text
    assert (
        "[table structure=exact_native captions=present footnotes=present orientation=?]\n"
        "[caption]\nTrial\n[/caption]\n[footnote]\nQualified result.\n[/footnote]\n"
        "[cell_defaults span=1x1 header=true row_header=? column_header=? row_section=? fillable=?]\n"
        f"{LEGEND}\n"
    ) in full
    assert rows[1].text == "[row 1]\n[r1c0 span=2x1] A\n[r1c1 header=false] 0.05"
    # The two-row label A is carried into every row it spans, as in v1.
    assert rows[2].text == "[row 2]\n[r1c0 span=2x1] A\n[r2c1 header=false] 0.5"
    assert "[r0c0] Condition" in full
    assert "[r0c1] p" in full
    assert "0.05" not in rows[2].text


def test_v2_reconstructs_every_cell_of_a_realistic_jats_table():
    doc = parse(CLINICAL.encode())
    assert_reconstructs(doc)
    text = "".join(u.context + u.text for u in v2_units(doc))
    for source in ["12.4 ± 3.1", "NA", "<LOD", "<0.001*", "Baseline (ng/mL)", "Week 12 (ng/mL)"]:
        assert source.replace("<", "<") in text
    assert "* p < 0.05 vs baseline; NA, not assessed; LOD, limit of detection." in text
    assert "Table 2. Serum markers (mean ± SD)" in text


def test_v2_spans_and_headers_of_the_realistic_table():
    doc = parse(CLINICAL.encode())
    printed = reconstruct(doc, v2_units(doc))
    assert printed[(0, 0)][0] == (2, 1)  # Marker spans both header rows
    assert printed[(0, 1)][0] == (1, 2)  # Treated spans two columns
    assert printed[(0, 1)][2] == "Treated"
    assert printed[(2, 0)][1]["header"] is True  # row-header th in the body
    assert printed[(2, 1)][1]["header"] is False


def test_v2_reconstructs_native_pdf_table_with_all_roles():
    assert_reconstructs(native_table())


def test_v2_unknown_and_false_stay_distinct_in_one_table():
    doc = native_table()
    table = next(n.table for n in doc.nodes if n.table)
    table.cells[0].native_fillable = None
    table.cells[1].native_fillable = False
    printed = reconstruct(doc, v2_units(doc))
    first, second = table.cells[0], table.cells[1]
    assert printed[(first.row, first.column)][1]["fillable"] is None
    assert printed[(second.row, second.column)][1]["fillable"] is False
    assert_reconstructs(doc)
    text = "".join(u.context + u.text for u in v2_units(doc))
    assert "fillable=?" in text
    assert "fillable=false" in text or "cell_defaults" in text


@pytest.mark.parametrize(
    "values,expected",
    [
        ([False, False, True, True, None], "false"),  # false beats true
        ([True, True, None, None, False], "true"),  # true beats unknown
        ([None, None, False, False, True], "false"),  # false beats unknown
        ([None, None, None, True, False], "?"),  # a clear majority wins regardless of rank
    ],
)
def test_v2_default_is_most_frequent_and_ties_resolve_false_true_unknown(values, expected):
    doc = native_table()
    table = next(n.table for n in doc.nodes if n.table)
    for cell, value in zip(table.cells, values, strict=True):
        cell.native_fillable = value
    assert f"fillable={expected}]" in DEFAULTS.search(v2_units(doc)[0].context)[0] + "]"
    assert_reconstructs(doc)


def test_v2_most_frequent_value_is_the_default():
    doc = parse(CLINICAL.encode())
    cells = [c for n in doc.nodes if n.table for c in n.table.cells]
    counts = Counter(c.is_header for c in cells)
    expected = "true" if counts[True] > counts[False] else "false"
    assert f"header={expected}" in DEFAULTS.search(v2_units(doc)[0].context)[1]


def test_v2_jats_and_pdf_share_one_line_grammar_for_equivalent_cells():
    jats = reconstruct(parse(TABLE.encode()), v2_units(parse(TABLE.encode())))
    native = reconstruct(native_table(), v2_units(native_table()))
    assert all(set(v[1]) == set(NAMES) for v in [*jats.values(), *native.values()])
    # Same renderer, so a header cell and a body cell differ only in printed deviations.
    assert jats[(1, 0)][0] == (2, 1)


def test_v2_whole_row_at_limit_is_indexed_and_one_token_over_is_excluded_untruncated():
    doc = parse(TABLE.encode())

    def counter_for(row_tokens):
        # Only the unit containing row 1 is given a controlled size; everything else is small.
        return lambda s: row_tokens if "[row 1]" in s and "0.05" in s else len(s.split())

    _, _, _, fits = v2_build(doc, 192, counter_for(192))
    assert any("0.05" in c.rendered_text for c in fits.chunks)
    _, _, _, over = v2_build(doc, 192, counter_for(193))
    assert not any("0.05" in c.rendered_text for c in over.chunks)
    diagnostic = next(
        d for d in over.chunk_set.diagnostics if d.kind == "table_row" and "0.05" in d.rendered_text
    )
    assert (diagnostic.code, diagnostic.actual_tokens, diagnostic.token_limit) == (
        "oversized",
        193,
        192,
    )
    assert diagnostic.rendered_text.endswith("[r1c0 span=2x1] A\n[r1c1 header=false] 0.05")
    accounted = {a.occurrence_id: a.status for a in over.chunk_set.accounting}
    cell = next(o for o in over.chunk_set.occurrences if o.text == "0.05")
    assert accounted[cell.occurrence_id] == "oversized"


def test_v2_is_deterministic_and_validates_against_canonical():
    doc = parse(CLINICAL.encode())
    selected, cfg, cfg_ref, first = v2_build(doc, 384, lambda s: len(s.split()))
    _, _, _, second = v2_build(doc, 384, lambda s: len(s.split()))
    assert first == second
    assert record_ref(first.chunk_set) == record_ref(second.chunk_set)
    validate_chunking(first, doc, selected, cfg_ref, cfg, lambda s: len(s.split()))


def test_v2_indexes_a_row_that_v1_excludes_at_the_same_budget():
    doc = parse(CLINICAL.encode())
    counter = lambda s: len(s) // 4  # coarse stand-in: v1 spends ~3 rows more
    v1_cfg = settings().model_copy(update={"chunk_tokens": 192})
    v1_ref = ConfigurationRef(
        artifact=ref("configuration"),
        resolved_sha256=sha256_bytes(canonical_json_bytes(v1_cfg.model_dump())),
    )
    v1 = chunk_document(doc, selection(), v1_ref, v1_cfg, counter)
    _, _, _, v2 = v2_build(doc, 192, counter)
    excluded = lambda r: sum(d.kind == "table_row" for d in r.chunk_set.diagnostics)
    assert (excluded(v1), excluded(v2)) == (3, 0)
    assert any("IL-6" in c.rendered_text for c in v2.chunks)


def test_v1_default_configuration_and_corrected_contributor_identity():
    assert settings().chunking_algorithm == V1
    assert "chunking_algorithm" in settings().model_dump()
    cfg = settings().model_copy(update={"chunk_tokens": 192})
    config_ref = ConfigurationRef(
        artifact=ref("configuration"),
        resolved_sha256=sha256_bytes(canonical_json_bytes(cfg.model_dump())),
    )
    # Contributor correction does not change the scientific configuration.
    assert (
        config_ref.resolved_sha256
        == "bac540837e37ccc7594a323518ede731fc709106e5801cb8d08f0a3e05cb6b21"
    )
    result = chunk_document(parse(TABLE.encode()), selection(), config_ref, cfg, len)
    assert [chunk.rendered_text for chunk in result.chunks] == ["Trial", "Qualified result."]
    assert result.chunk_set.chunks.artifact_id == (
        "drv_e629ab1053fe0f3412a903619b13cc3b6a20c41e124494fc17e3e8dd7eadee3a"
    )
    assert result.chunk_set.chunks.sha256 == (
        "ff8dcc474746c373885e53b9c503a1997f62a74e0dda91fa708c87f1aadae52c"
    )
    assert record_ref(result.chunk_set).artifact_id == (
        "drv_4f0a32b932253d73ec0f6ded0f662992c0b51caa998d840d0bbc957be8f43bc3"
    )
    assert result.chunk_set.chunks_build_manifest.sha256 == (
        "4294faa71f3362070170e469e30d4d7f1e8658b3fa990c19d3551b2cb7192e8f"
    )


def test_v1_and_v2_have_distinct_recorded_identities():
    doc = parse(TABLE.encode())
    v1_cfg = settings().model_copy(update={"chunk_tokens": 384})
    v1_ref = ConfigurationRef(
        artifact=ref("configuration"),
        resolved_sha256=sha256_bytes(canonical_json_bytes(v1_cfg.model_dump())),
    )
    v1 = chunk_document(doc, selection(), v1_ref, v1_cfg, len)
    selected, _, v2_ref, v2 = v2_build(doc, 384)
    assert v1_ref.resolved_sha256 != v2_ref.resolved_sha256
    assert v1.chunk_set.chunks.artifact_id != v2.chunk_set.chunks.artifact_id
    assert chunks_ref(v1.chunks, selected, v1_ref, V1) == v1.chunk_set.chunks
    assert chunks_ref(v2.chunks, selected, v2_ref, V2) == v2.chunk_set.chunks
    # The same bytes and inputs under the other label give a different artifact identity.
    assert chunks_ref(v2.chunks, selected, v2_ref, V1) != v2.chunk_set.chunks


def test_v2_chunks_cannot_be_relabelled_or_validated_as_v1():
    doc = parse(TABLE.encode())
    selected, v2_cfg, v2_ref, v2 = v2_build(doc, 384)
    v1_cfg = v2_cfg.model_copy(update={"chunking_algorithm": V1})
    v1_ref = ConfigurationRef(
        artifact=v2_ref.artifact,
        resolved_sha256=sha256_bytes(canonical_json_bytes(v1_cfg.model_dump())),
    )
    with pytest.raises(ValueError):
        validate_chunking(v2, doc, selected, v1_ref, v1_cfg, len)
    forged = v2.chunk_set.model_copy(deep=True)
    forged.configuration = v1_ref
    with pytest.raises(ValueError):
        validate_chunking(type(v2)(v2.chunks, forged), doc, selected, v1_ref, v1_cfg, len)


def test_configuration_accepts_only_the_named_algorithms():
    base = settings().model_dump()
    assert Phase5Configuration.model_validate({**base, "chunking_algorithm": V2})
    assert Phase5Configuration.model_validate(
        {**base, "chunking_algorithm": "canonical_whole_units_v3"}
    )
    with pytest.raises(ValueError):
        Phase5Configuration.model_validate(
            {**base, "chunking_algorithm": "canonical_whole_units_v4"}
        )
