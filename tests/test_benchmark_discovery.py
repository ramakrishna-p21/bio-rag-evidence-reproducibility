from biorag_repro.benchmark.discovery import discover_jats
from biorag_repro.benchmark.flags import flag_leads
from biorag_repro.benchmark.review import review_queue, worksheet_rows
from biorag_repro.benchmark.validation import resolve_jats_path

XML = """<article><front><article-meta><abstract><p>BRCA1 increased under condition A (p &lt; 1.2e-3).</p></abstract></article-meta></front><body><sec><title>Results</title><p>BRCA1 increased under condition A (p &lt; 1.2e-3). BRCA1 decreased under condition B (p &lt; 1.2e-3).</p><table-wrap id="T1"><table><thead><tr><th>Condition</th><th>Effect</th></tr></thead><tbody><tr><td>A</td><td>−1.2 × 10⁻³</td></tr><tr><td>B</td><td>2.4</td></tr></tbody></table></table-wrap></sec></body></article>""".encode()


def test_discovery_keeps_sections_and_table_associations() -> None:
    leads = discover_jats(XML, "P001")
    assert any(lead.section == "abstract" for lead in leads)
    assert any(lead.section == "Results" and lead.kind == "prose" for lead in leads)
    assert any(
        lead.kind == "table_cell"
        and lead.row_label == "A"
        and lead.column_label == "Effect"
        and "−1.2" in lead.text
        for lead in leads
    )
    assert all(lead.status == "lead_only" for lead in leads)


def test_repeated_values_flagged_without_merging_conditions() -> None:
    leads = discover_jats(XML, "P001")
    flags = flag_leads(leads)
    assert any(flag.code == "repeated_value_context" for flag in flags)
    rows = worksheet_rows(review_queue(leads), flags)
    assert any("repeated_value_context" in row.flags for row in rows)


def test_table_rowspan_retains_full_context() -> None:
    xml = b'<article><body><table-wrap id="T"><table><tr><th>Time</th><th>Outcome</th><th>HR</th></tr><tr><td rowspan="2">5 years</td><td>Mortality</td><td>1.104</td></tr><tr><td>MACE</td><td>1.154</td></tr></table></table-wrap></body></article>'
    leads = discover_jats(xml, "P001")
    target = next(lead for lead in leads if lead.text == "1.154")
    assert target.row_label == "5 years | MACE"
    assert target.column_label == "HR"
    assert resolve_jats_path(xml, target.xml_path) == "1.154"
