"""Supplementary labels: answer identity, pair and stability classes, evidence form and location.

Expected values are written by hand from the definitions in the module docstring, not produced by
the code under test. Synthetic fixtures only; nothing here is a result.
"""

import dataclasses
import json
from pathlib import Path

import pytest
from test_analysis_export import (
    COORDS,
    P_A,
    closed_book,
    deliver,
    observation,
    world,
)
from test_analysis_handoff import rehearsal
from test_stage_a import CANONICAL, TABLE_COMPONENTS
from test_stage_b import table_world
from test_stage_c import make_result, rekey

from biorag_repro.benchmark.models import BenchmarkItem, JatsRef, PdfRef
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.evaluation import analysis_handoff as ah
from biorag_repro.evaluation import supplement as sp
from biorag_repro.evaluation.analysis_table import Evidence, build_row, serialize_csv
from biorag_repro.evaluation.stage_c import score

# ---- helpers --------------------------------------------------------------------------------


def gen(item, raw, answer, *, condition="JATS", repeat=1, arm="confirmatory", attempt=None, **kw):
    """A completed generation with its own raw bytes; the key follows the coordinates."""
    digest = sha256_bytes(raw.encode())
    changes = {
        "matrix_arm": arm,
        "representation_or_condition": condition,
        "repeat_index": repeat,
        "attempt_id": attempt or f"attempt_{arm}_{condition}_{repeat}",
        "output": ArtifactRef(
            artifact_id=f"gen_{digest[:12]}",
            logical_ref=f"generation/gen_{digest[:12]}.json",
            sha256=digest,
        ),
        "raw_output": raw,
        "answer": answer,
    } | kw
    return rekey(make_result(item), **(COORDS | changes))


def raw(answer, pad=""):
    return json.dumps({"answer": answer, "abstained": False}) + pad


def obs_jats(w, result, repeat, arm="confirmatory"):
    return observation(w, case=deliver(w, [w.chunk(P_A)]), result=result, repeat_index=repeat)


def obs_pdf(w, result, repeat):
    return observation(
        w, evidence=None, condition="PDF_DOCLING", result=result, repeat_index=repeat
    )


def resp(state="answer", status="correct", byte="1", norm="1", repeat="1", **kw):
    base = {
        "item_id": "i1",
        "matrix_arm": "confirmatory",
        "condition": "JATS",
        "repeat_index": repeat,
        "answer_state": state,
        "stage_c_status": status,
        "byte_group": byte if state == "answer" else "NA",
        "norm_group": norm if state == "answer" else "NA",
    }
    return base | kw


# ---- normalized answer identity -------------------------------------------------------------


def test_normalization_is_unicode_composition_and_whitespace_only():
    assert sp.normalized_key("  a\t b\n") == sp.normalized_key("a b")
    assert sp.normalized_key("é") == sp.normalized_key("é")  # NFD vs NFC
    assert sp.normalized_key("TP53") != sp.normalized_key("tp53")  # no case folding
    assert sp.normalized_key("TP53") != sp.normalized_key(["TP53"])  # shape is part of identity
    assert sp.normalized_key(["a", "b"]) != sp.normalized_key(["b", "a"])  # order kept
    assert sp.normalized_key(["a", ""]) != sp.normalized_key(["a"])  # nothing dropped
    assert sp.normalized_key("0.05") != sp.normalized_key("5e-2")  # no numeric equivalence here


# ---- pair classes (hand-derived) ------------------------------------------------------------


@pytest.mark.parametrize(
    "jats, pdf, expected",
    [
        (resp(byte="1", norm="1"), resp(byte="1", norm="1"), "identical_bytes"),
        (resp(byte="1", norm="1"), resp(byte="2", norm="1"), "identical_normalized"),
        (resp(byte="1", norm="1"), resp(byte="2", norm="2"), "both_correct"),
        (
            resp(status="incorrect", norm="1", byte="1"),
            resp(status="incorrect", norm="2", byte="2"),
            "differ_both_incorrect_unresolved",
        ),
        (
            resp(status="correct", norm="1", byte="1"),
            resp(status="incorrect", norm="2", byte="2"),
            "differ_correctness_differs",
        ),
        (
            resp(status="incorrect", norm="1", byte="1"),
            resp(status="correct", norm="2", byte="2"),
            "differ_correctness_differs",
        ),
        (
            resp(status="non_evaluable", norm="1", byte="1"),
            resp(status="correct", norm="2", byte="2"),
            "differ_correctness_unresolved",
        ),
        (resp("abstained", "incorrect"), resp(), "not_comparable"),
        (resp(), resp("unparseable", "incorrect"), "not_comparable"),
        (resp("operational_failure", "missing"), resp(), "not_comparable"),
        (resp("not_executed", "NA"), resp(), "not_comparable"),
        (resp(), None, "not_comparable"),
        (None, resp(), "not_comparable"),
        (
            resp(status="incorrect", norm="1", byte="1"),
            resp(status="non_evaluable", norm="2", byte="2"),
            "differ_correctness_unresolved",
        ),
        (
            resp(status="missing", norm="1", byte="1"),
            resp(status="incorrect", norm="2", byte="2"),
            "differ_correctness_unresolved",
        ),
    ],
)
def test_pair_class_oracle(jats, pdf, expected):
    assert sp.pair_class(jats, pdf) == expected


def test_two_correct_answers_are_not_called_equivalent():
    assert "both_correct_equivalent" not in sp.PAIR_CLASSES


def test_absent_or_unanswered_sides_are_never_substantive_disagreement():
    for other in (None, resp("abstained", "incorrect"), resp("operational_failure", "missing")):
        assert sp.pair_class(resp(norm="1"), other) == "not_comparable"


# ---- stability (hand-derived) ---------------------------------------------------------------


def trio(*members):
    return [
        resp(repeat=str(i), **m) if isinstance(m, dict) else m for i, m in enumerate(members, 1)
    ]


def wrong(norm, byte=None):
    return {"status": "incorrect", "norm": norm, "byte": byte or norm}


def right(norm="1", byte=None):
    return {"status": "correct", "norm": norm, "byte": byte or norm}


@pytest.mark.parametrize(
    "group, expected",
    [
        (trio(wrong("1"), wrong("1"), wrong("1")), ("incorrect_identical_normalized", "yes")),
        (
            trio(wrong("1", "a"), wrong("1", "b"), wrong("1", "a")),
            ("incorrect_identical_normalized", "no"),
        ),
        (trio(wrong("1"), wrong("2"), wrong("1")), ("incorrect_varying", "no")),
        (trio(right(), right(), right()), ("all_correct", "yes")),
        (trio(right(), wrong("2"), right()), ("mixed", "no")),
        (
            trio(wrong("1"), {"state": "abstained", "status": "incorrect"}, wrong("1")),
            ("contains_abstention_or_unparseable", "NA"),
        ),
        (
            trio(wrong("1"), wrong("1"), {"state": "unparseable", "status": "incorrect"}),
            ("contains_abstention_or_unparseable", "NA"),
        ),
        (
            trio(wrong("1"), wrong("1"), {"state": "operational_failure", "status": "missing"}),
            ("incomplete", "NA"),
        ),
        (
            trio(wrong("1"), wrong("1"), {"state": "not_executed", "status": "NA"}),
            ("incomplete", "NA"),
        ),
        (trio(wrong("1"), wrong("1")), ("incomplete", "NA")),  # third repeat has no row
        (
            trio(wrong("1"), {"status": "non_evaluable", "norm": "1", "byte": "1"}, wrong("1")),
            ("correctness_unresolved", "yes"),
        ),
    ],
)
def test_stability_oracle(group, expected):
    assert sp.stability_class(group) == expected


def test_three_abstentions_are_not_identical_incorrect_answers():
    abstained = {"state": "abstained", "status": "incorrect"}
    assert sp.stability_class(trio(abstained, abstained, abstained))[0] != (
        "incorrect_identical_normalized"
    )


def test_stability_ignores_order_of_rows():
    group = trio(wrong("1"), wrong("1"), wrong("1"))
    assert sp.stability_class(list(reversed(group))) == sp.stability_class(group)


# ---- evidence location and form from the structure of the original reference ---------------


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/article/front[1]/article-meta[1]/abstract[1]/p[1]", "abstract"),
        ("/article/front[1]/article-meta[1]/title-group[1]", "other"),
        ("/article/body[1]/sec[2]/p[3]", "body"),
        ("/article/body[1]/sec[1]/table-wrap[1]/table[1]/tbody[1]/tr[1]/td[2]", "body"),
        ("/article/back[1]/ack[1]/p[1]", "back"),
        ("/article/floats-group[1]/fig[1]", "other"),
        ("/article/body/p[1]", "unresolved"),  # no index: not the validated format
        ("/article/body[1]/p[0]", "unresolved"),
        ("/article/", "unresolved"),
        ("/article", "unresolved"),
        ("/other/body[1]/p[1]", "unresolved"),
    ],
)
def test_location_is_structural_ancestry(path, expected):
    assert sp.jats_location(path) == expected


def ref(path, **kw):
    data = {
        "artifact_id": "jats",
        "source_sha256": "a" * 64,
        "paper_id": "p1",
        "xml_path": path,
        "source_text": "t",
    } | kw
    return JatsRef.model_validate(data)


@pytest.mark.parametrize(
    "path, kw, expected",
    [
        ("/article/body[1]/sec[1]/p[2]", {}, "prose"),
        ("/article/front[1]/article-meta[1]/abstract[1]/p[1]", {}, "prose"),
        ("/article/body[1]/sec[1]/table-wrap[1]/table[1]/tbody[1]/tr[1]/td[2]", {}, "table"),
        (
            "/article/body[1]/sec[1]/table-wrap[1]/table[1]/tbody[1]/tr[1]/td[2]",
            {"table_id": "T1", "row_label": "r", "column_label": "c"},
            "table",
        ),
        ("/article/body[1]/sec[1]/table-wrap[1]/caption[1]/p[1]", {}, "table"),
        ("/article/body[1]/sec[1]/fig[1]/caption[1]/p[1]", {}, "other"),
        ("/article/body[1]/sec[1]/title[1]", {}, "other"),
        ("/article/back[1]/ref-list[1]/ref[1]/p[1]", {}, "other"),
        # a table id on a path outside any table contradicts itself
        (
            "/article/body[1]/sec[1]/p[2]",
            {"table_id": "T1", "row_label": "r", "column_label": "c"},
            "unresolved",
        ),
        ("not/a/path[1]", {}, None),
    ],
)
def test_reference_form_needs_positive_structure(path, kw, expected):
    if expected is None:
        with pytest.raises(ValueError):  # the benchmark model itself refuses it
            ref(path, **kw)
        return
    assert sp.jats_ref_form(ref(path, **kw)) == expected


def test_a_missing_table_id_is_not_prose_and_a_pdf_never_establishes_prose():
    pdf = PdfRef.model_validate(
        {
            "artifact_id": "pdf",
            "source_sha256": "b" * 64,
            "paper_id": "p1",
            "source_text": "t",
            "page": 3,
        }
    )
    assert sp.pdf_ref_form(pdf) == "unresolved"
    tabled = pdf.model_copy(update={"table_id": "T1", "row_label": "r", "column_label": "c"})
    assert sp.pdf_ref_form(tabled) == "table"


def item_with(jats_refs, pdf_refs=None, components=None):
    w = world()
    data = w.item.model_dump(mode="json")
    data["jats_refs"] = [r.model_dump(mode="json") for r in jats_refs]
    if pdf_refs is not None:
        data["pdf_refs"] = [r.model_dump(mode="json") for r in pdf_refs]
    if components is not None:
        data["evidence_sets"] = components
    return BenchmarkItem.model_validate(data)


PROSE_REF = "/article/body[1]/sec[1]/p[2]"
CELL_REF = "/article/body[1]/sec[1]/table-wrap[1]/table[1]/tbody[1]/tr[1]/td[2]"


def evidence_row(jats_paths, pdf_table=False, table_roles=False):
    base = world().item.jats_refs[0].model_dump(mode="json")
    jats = [
        ref(p, **{k: base[k] for k in ("artifact_id", "source_sha256", "paper_id")})
        for p in jats_paths
    ]
    pdf_data = world().item.pdf_refs[0].model_dump(mode="json")
    if pdf_table:
        pdf_data |= {"table_id": "T1", "row_label": "r", "column_label": "c"}
    sets = None
    if table_roles:
        sets = [
            {"components": [{"role": r, "text": t} for r, t in TABLE_COMPONENTS]},
        ]
    item = item_with(jats, [PdfRef.model_validate(pdf_data)], sets)
    return sp.evidence_item_row(item, "f" * 64)


def test_item_form_prose_table_mixed_and_unresolved():
    assert evidence_row([PROSE_REF])["jats_form"] == "prose"
    assert evidence_row([CELL_REF], pdf_table=True)["jats_form"] == "table"
    assert evidence_row([PROSE_REF, CELL_REF])["jats_form"] == "mixed"
    assert evidence_row(["/article/body[1]/sec[1]/title[1]"])["jats_form"] == "other"
    # one ambiguous reference makes the item unresolved rather than silently prose
    row = evidence_row([PROSE_REF, "/article/body/p"])
    assert row["jats_form"] == "unresolved" and row["item_form"] == "unresolved"


def test_jats_pdf_disagreement_and_set_roles_are_kept_visible():
    agree = evidence_row([CELL_REF], pdf_table=True)
    assert (agree["form_relation"], agree["item_form"]) == ("agree", "table")
    conflict = evidence_row([PROSE_REF], pdf_table=True)
    assert (conflict["form_relation"], conflict["item_form"]) == ("conflict", "unresolved")
    assert conflict["jats_form"] == "prose" and conflict["pdf_form"] == "table"
    pdf_open = evidence_row([CELL_REF])
    assert (pdf_open["form_relation"], pdf_open["item_form"]) == ("pdf_not_established", "table")
    roles = evidence_row([PROSE_REF], table_roles=True)
    assert (roles["form_relation"], roles["item_form"]) == ("set_roles_conflict", "unresolved")
    assert roles["n_sets_with_table_roles"] == "1"
    assert evidence_row(["/article/body[1]/sec[1]/title[1]"])["form_relation"] == (
        "jats_not_established"
    )


def test_location_of_several_references():
    row = evidence_row([PROSE_REF, "/article/back[1]/ack[1]/p[1]"])
    assert row["jats_location"] == "multiple"
    assert evidence_row([PROSE_REF, CELL_REF])["jats_location"] == "body"
    assert evidence_row(["/article/body/p"])["jats_location"] == "unresolved"


# ---- Stage A / B form comes from the canonical document, per representation ----------------


def test_stage_a_and_b_forms_for_prose_and_for_a_table():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    o = observation(w, case=case)
    r = sp.evidence_observation_row(o, build_row(o)[0])
    assert (r["stage_a_form"], r["stage_a_sets_prose"], r["stage_a_sets_table"]) == (
        "prose",
        "1",
        "0",
    )
    assert (r["stage_b_status"], r["stage_b_form"]) == ("delivered", "prose")

    t = table_world()
    o = observation(t, evidence=Evidence(t.document, CANONICAL, t.stage_a))
    r = sp.evidence_observation_row(o, build_row(o)[0])
    assert (r["stage_a_form"], r["stage_a_sets_table"], r["stage_b_form"]) == (
        "table",
        "1",
        "no_stage_b_record",
    )


def test_missing_records_stay_explicit_not_blank():
    w = world()
    o = observation(w, evidence=None, condition="PDF_DOCLING")
    r = sp.evidence_observation_row(o, build_row(o)[0])
    assert r["stage_a_form"] == "no_stage_a_record" and r["stage_b_form"] == "no_stage_b_record"
    assert r["stage_a_label"] == "NA" and r["stage_a_sets_table"] == "NA"


def test_container_kinds_other_than_paragraph_or_table_are_other():
    w = world()
    node = next(n for n in w.document.nodes if n.kind == "paragraph")
    assert sp.container_form(node) == "prose"
    for kind in ("caption", "title", "figure", "formula", "reference"):
        assert sp.container_form(node.model_copy(update={"kind": kind})) == "other"
    assert sp._combine_sets(["table", "prose"]) == "mixed"
    assert sp._combine_sets(["table", "other"]) == "other"
    assert sp._combine_sets(["mixed", "table"]) == "mixed"


# ---- the built supplement -------------------------------------------------------------------


@dataclasses.dataclass
class Built:
    root: Path
    verified: ah.VerifiedHandoff
    observations: list
    supplement: sp.Supplement


def scenario(tmp_path):
    """One item, two conditions, three repeats, the closed-book arm and a robustness-free table.

    JATS repeats: "2" (wrong), "2" with padding (same normalized answer, other bytes), "2".
    PDF repeats: "1" (right), the exact bytes of JATS repeat 2, an abstention.
    """
    w = world()
    item = w.item
    jr = [
        gen(item, raw("2"), "2", repeat=1),
        gen(item, raw("2", " "), "2", repeat=2),
        gen(item, raw("2"), "2", repeat=3, attempt="attempt_jats_3b"),
    ]
    jr[2] = gen(item, raw(" 2 "), " 2 ", repeat=3)
    pr = [
        gen(item, raw("1"), "1", condition="PDF_DOCLING", repeat=1),
        gen(item, raw("2", " "), "2", condition="PDF_DOCLING", repeat=2),
        gen(
            item,
            '{"answer": null, "abstained": true}',
            None,
            condition="PDF_DOCLING",
            repeat=3,
            abstained=True,
        ),
    ]
    observations = [obs_jats(w, r, r.repeat_index) for r in jr]
    observations += [obs_pdf(w, r, r.repeat_index) for r in pr]
    observations.append(
        closed_book(
            w,
            repeat_index=1,
        )
    )
    return w, observations


def build(tmp_path):
    _, observations = scenario(tmp_path)
    _, _, handoff = rehearsal(tmp_path, observations)
    _, build_ref = ah.publish_handoff(handoff, tmp_path)
    verified = ah.read_handoff(build_ref, tmp_path)
    return Built(tmp_path, verified, observations, sp.build_supplement(verified, observations))


def rows_of(built, name):
    columns = sp.TABLES[name]
    return sp.parse_table(columns, built.supplement.tables[name])


def test_response_labels_for_the_scenario(tmp_path):
    b = build(tmp_path)
    resp_rows = {
        (r["matrix_arm"], r["condition"], r["repeat_index"]): r
        for r in rows_of(b, "supplement_response.csv")
    }
    j1, j2, j3 = (resp_rows[("confirmatory", "JATS", str(i))] for i in (1, 2, 3))
    p1, p2, p3 = (resp_rows[("confirmatory", "PDF_DOCLING", str(i))] for i in (1, 2, 3))
    assert [r["answer_state"] for r in (j1, p1, p3)] == ["answer", "answer", "abstained"]
    # groups are numbered in (repeat, condition) order and are local to item and arm
    assert (j1["byte_group"], j1["norm_group"]) == ("1", "1")  # "2"
    assert (p1["byte_group"], p1["norm_group"]) == ("2", "2")  # "1"
    assert (j2["byte_group"], j2["norm_group"]) == ("3", "1")  # padded bytes, same answer
    assert (p2["byte_group"], p2["norm_group"]) == ("3", "1")  # exactly the bytes of j2
    assert (j3["byte_group"], j3["norm_group"]) == ("4", "1")  # " 2 ": other bytes, same norm
    assert p3["byte_group"] == p3["norm_group"] == "NA" and p3["stage_c_status"] == "incorrect"
    assert resp_rows[("closed_book", "CLOSED_BOOK", "1")]["answer_state"] == "answer"


def test_pairs_and_stability_for_the_scenario(tmp_path):
    b = build(tmp_path)
    pairs = {r["repeat_index"]: r for r in rows_of(b, "supplement_pairs.csv")}
    assert set(pairs) == {"1", "2", "3"}  # closed book has no pair
    assert pairs["1"]["pair_class"] == "differ_correctness_differs"
    assert (pairs["1"]["jats_status"], pairs["1"]["pdf_status"]) == ("incorrect", "correct")
    assert pairs["2"]["pair_class"] == "identical_bytes"
    assert pairs["3"]["pair_class"] == "not_comparable"
    assert (pairs["3"]["jats_state"], pairs["3"]["pdf_state"]) == ("answer", "abstained")

    stab = {(r["matrix_arm"], r["condition"]): r for r in rows_of(b, "supplement_stability.csv")}
    jats = stab[("confirmatory", "JATS")]
    assert (jats["stability_class"], jats["byte_identical"], jats["n_answers"]) == (
        "incorrect_identical_normalized",
        "no",
        "3",
    )
    assert stab[("confirmatory", "PDF_DOCLING")]["stability_class"] == (
        "contains_abstention_or_unparseable"
    )
    assert stab[("closed_book", "CLOSED_BOOK")]["stability_class"] == "incomplete"
    assert stab[("closed_book", "CLOSED_BOOK")]["n_rows"] == "1"


def test_evidence_rows_for_the_scenario(tmp_path):
    b = build(tmp_path)
    [item] = rows_of(b, "supplement_evidence_items.csv")
    # The shared fixture item uses "/article/body/p[1]", which is not the indexed path format the
    # benchmark validator resolves, so its form and location stay unresolved (not guessed prose).
    assert (item["jats_form"], item["jats_location"], item["pdf_form"]) == (
        "unresolved",
        "unresolved",
        "unresolved",
    )
    assert (item["form_relation"], item["item_form"]) == ("jats_not_established", "unresolved")
    by = {
        (r["matrix_arm"], r["condition"], r["repeat_index"]): r
        for r in rows_of(b, "supplement_evidence_observations.csv")
    }
    assert by[("confirmatory", "JATS", "1")]["stage_b_form"] == "prose"
    assert by[("confirmatory", "PDF_DOCLING", "1")]["stage_a_form"] == "no_stage_a_record"
    assert by[("closed_book", "CLOSED_BOOK", "1")]["stage_b_form"] == "no_stage_b_record"


def test_tables_hold_labels_only(tmp_path):
    b = build(tmp_path)
    text = b"".join(b.supplement.tables.values()).decode()
    for secret in ("What was reported?", '"answer"', 'abstained":', "TP53", "0.05", P_A[:12]):
        assert secret not in text
    for name, data in b.supplement.tables.items():
        assert sp.serialize_table(sp.TABLES[name], sp.parse_table(sp.TABLES[name], data)) == data


def test_record_links_the_exact_table_and_is_synthetic(tmp_path):
    b = build(tmp_path)
    rec = b.supplement.record
    assert rec.analysis_table_sha256 == sha256_bytes(serialize_csv(b.verified.rows))
    assert rec.handoff_sha256 == sp.handoff_sha256(b.verified.handoff)
    assert (rec.purpose, rec.scientific_authority, rec.final) == (
        "synthetic_rehearsal",
        "none",
        False,
    )
    assert rec.banner == sp.BANNER


def test_build_is_deterministic_and_independent_of_observation_order(tmp_path):
    b = build(tmp_path)
    again = sp.build_supplement(b.verified, list(reversed(b.observations)))
    assert again.tables == b.supplement.tables
    assert again.record == b.supplement.record


def test_publish_read_roundtrip_and_immutability(tmp_path):
    b = build(tmp_path)
    digest = sp.publish_supplement(b.supplement, tmp_path, "supplement_run")
    assert digest == sha256_bytes(sp.serialize_record(b.supplement.record))
    back = sp.read_supplement(tmp_path, "supplement_run", digest, verified=b.verified)
    assert back.tables == b.supplement.tables
    with pytest.raises(FileExistsError):
        sp.publish_supplement(b.supplement, tmp_path, "supplement_run")
    target = tmp_path / "supplement_run" / "supplement_pairs.csv"
    assert not target.stat().st_mode & 0o222


def edit(tmp_path, name, old, new):
    path = tmp_path / "supplement_run" / name
    path.parent.chmod(0o755)
    path.chmod(0o644)
    path.write_bytes(path.read_bytes().replace(old, new))


def test_read_refuses_changed_or_foreign_content(tmp_path):
    b = build(tmp_path)
    digest = sp.publish_supplement(b.supplement, tmp_path, "supplement_run")
    with pytest.raises(sp.ResultError, match="trusted SHA-256"):
        sp.read_supplement(tmp_path, "supplement_run", "0" * 64, verified=b.verified)
    edit(tmp_path, "supplement_pairs.csv", b"identical_bytes", b"both_correct")
    with pytest.raises(sp.ResultError, match="differs from the hash"):
        sp.read_supplement(tmp_path, "supplement_run", digest, verified=b.verified)


def test_read_rederives_labels_even_when_hashes_are_made_to_agree(tmp_path):
    """A consistent forgery of a dependent label is still caught by re-derivation."""
    b = build(tmp_path)
    tables = dict(b.supplement.tables)
    tables["supplement_pairs.csv"] = tables["supplement_pairs.csv"].replace(
        b"identical_bytes", b"both_correct"
    )
    with pytest.raises(sp.SupplementError, match="pair labels"):
        sp.check_tables(tables, b.verified.rows)
    tables = dict(b.supplement.tables)
    tables["supplement_stability.csv"] = tables["supplement_stability.csv"].replace(
        b"incorrect_identical_normalized", b"incorrect_varying"
    )
    with pytest.raises(sp.SupplementError, match="stability labels"):
        sp.check_tables(tables, b.verified.rows)


def test_response_row_must_match_the_table_row(tmp_path):
    b = build(tmp_path)
    tables = dict(b.supplement.tables)
    tables["supplement_response.csv"] = tables["supplement_response.csv"].replace(
        b",incorrect,", b",correct,", 1
    )
    with pytest.raises(sp.SupplementError, match="disagrees with its analysis-table row"):
        sp.check_tables(tables, b.verified.rows)


# ---- provenance and binding -----------------------------------------------------------------


def test_official_or_unverified_origin_is_unavailable(tmp_path):
    b = build(tmp_path)
    official = dataclasses.replace(b.verified, official=True)
    with pytest.raises(sp.OfficialSupplementUnavailable):
        sp.build_supplement(official, b.observations)
    forged = b.verified.handoff.model_copy(update={"input_origin": "verified_official_run"})
    with pytest.raises(sp.OfficialSupplementUnavailable):
        sp.build_supplement(dataclasses.replace(b.verified, handoff=forged), b.observations)
    digest = sp.publish_supplement(b.supplement, tmp_path, "supplement_run")
    with pytest.raises(sp.OfficialSupplementUnavailable):
        sp.read_supplement(tmp_path, "supplement_run", digest, verified=official)


def test_observations_must_be_exactly_the_tables_rows(tmp_path):
    b = build(tmp_path)
    with pytest.raises(sp.SupplementError, match="not exactly the table's rows"):
        sp.build_supplement(b.verified, b.observations[:-1])
    with pytest.raises(sp.SupplementError, match="supplied twice"):
        sp.build_supplement(b.verified, [*b.observations, b.observations[0]])


def test_an_observation_that_differs_from_the_published_row_stops_the_build(tmp_path):
    b = build(tmp_path)
    w = world()
    other = obs_jats(w, gen(w.item, raw("3"), "3", repeat=1), 1)  # same key, other answer
    changed = [
        other if o.condition == "JATS" and o.repeat_index == 1 else o for o in b.observations
    ]
    with pytest.raises(sp.SupplementError, match="differs from the table row"):
        sp.build_supplement(b.verified, changed)


def test_a_score_from_another_attempt_is_refused(tmp_path):
    b = build(tmp_path)
    w = world()
    first = next(o for o in b.observations if o.condition == "JATS" and o.repeat_index == 1)
    foreign = score(w.item, gen(w.item, raw("2"), "2", repeat=1, attempt="attempt_other"))
    swapped = dataclasses.replace(first, score=foreign)
    others = [o for o in b.observations if o is not first]
    with pytest.raises(sp.SupplementError, match="another attempt|does not belong"):
        sp._check_attempt(swapped)  # the supplement's own binding check
    with pytest.raises(ValueError, match="score differs"):
        sp.build_supplement(b.verified, [swapped, *others])  # and the table's rebuild check


def test_the_handoff_must_cite_the_table_the_rows_serialize_to(tmp_path):
    b = build(tmp_path)
    wrong_csv = b.verified.handoff.csv.model_copy(update={"sha256": "0" * 64})
    bad = b.verified.handoff.model_copy(update={"csv": wrong_csv})
    with pytest.raises(sp.SupplementError, match="not the table the handoff cites"):
        sp.build_supplement(dataclasses.replace(b.verified, handoff=bad), b.observations)


# ---- gaps found by mutation testing -------------------------------------------------------


def test_only_arms_with_the_right_structure_enter_pairs_and_stability():
    rows = [
        resp(matrix_arm="dense_robustness"),
        resp(matrix_arm="closed_book", condition="CLOSED_BOOK"),
        resp(condition="PDF_DOCLING"),
        resp(),
    ]
    # robustness arms have one generation, so they have no stability; closed book has no pair
    assert {r["matrix_arm"] for r in sp.stability_rows(rows)} == {"confirmatory", "closed_book"}
    assert {r["matrix_arm"] for r in sp.pair_rows(rows)} == {"confirmatory", "dense_robustness"}
    assert "dense_robustness" not in {r["matrix_arm"] for r in sp.stability_rows(rows)}


def test_the_scenario_has_no_closed_book_pair(tmp_path):
    b = build(tmp_path)
    assert {r["matrix_arm"] for r in rows_of(b, "supplement_pairs.csv")} == {"confirmatory"}
    assert len(rows_of(b, "supplement_pairs.csv")) == 3


def test_attempt_binding_checks_each_identity_of_the_score(tmp_path):
    b = build(tmp_path)
    first = next(o for o in b.observations if o.condition == "JATS" and o.repeat_index == 1)
    sp._check_attempt(first)  # the honest pair passes
    for change in (
        {"generation_sha256": "0" * 64},
        {"attempt_id": "attempt_other"},
        {"output": first.score.output.model_copy(update={"sha256": "1" * 64})},
        {"observation_key": "other_key"},
        {"repeat_index": 3},
    ):
        tampered = dataclasses.replace(first, score=first.score.model_copy(update=change))
        with pytest.raises(sp.SupplementError, match="does not belong"):
            sp._check_attempt(tampered)
    with pytest.raises(sp.SupplementError, match="needs its Stage C score"):
        sp._check_attempt(dataclasses.replace(first, score=None))


def test_reading_refuses_a_forged_official_origin_without_the_official_flag(tmp_path):
    b = build(tmp_path)
    digest = sp.publish_supplement(b.supplement, tmp_path, "supplement_run")
    forged = b.verified.handoff.model_copy(update={"input_origin": "verified_official_run"})
    with pytest.raises(sp.OfficialSupplementUnavailable):
        sp.read_supplement(
            tmp_path,
            "supplement_run",
            digest,
            verified=dataclasses.replace(b.verified, handoff=forged),
        )


def test_reading_refuses_a_consistent_record_that_cites_another_table(tmp_path):
    """The record is self-consistent (its identity is recomputed) but names another table."""
    b = build(tmp_path)
    record = b.supplement.record
    other = "9" * 64
    forged = sp.SupplementRecord(
        benchmark_version=record.benchmark_version,
        corpus_version=record.corpus_version,
        analysis_table_sha256=other,
        handoff_sha256=record.handoff_sha256,
        files=record.files,
        supplement_sha256=sp.supplement_id(
            record.handoff_sha256, other, {n: f.sha256 for n, f in record.files.items()}
        ),
    )
    digest = sp.publish_supplement(
        sp.Supplement(record=forged, tables=b.supplement.tables), tmp_path, "supplement_run"
    )
    with pytest.raises(sp.ResultError, match="does not cite the verified table"):
        sp.read_supplement(tmp_path, "supplement_run", digest, verified=b.verified)
