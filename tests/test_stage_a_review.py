"""Stage A masked-review packets: scheduling, masking, provenance and record validation.

Fixtures are synthetic. Judgment records written here exercise validation only; they are
not scientific judgments of any benchmark item.
"""

import json
from pathlib import Path

import pytest
from test_chunking import native_table
from test_documents_jats import parse
from test_stage_a import make_item

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.documents.models import CanonicalDocument
from scripts import stage_a_review as sar

JATS_XML = (
    "<article><body><sec><title>Methods</title>"
    "<p>Tables were exported as PDF and compared by AUROC in the JATS release.</p></sec>"
    "<table-wrap><caption><p>Classifier performance</p></caption><table>"
    "<thead><tr><th>Gene</th><th>AUROC</th></tr></thead>"
    "<tbody><tr><th>TP53</th><td>0.91</td></tr><tr><th>KRAS</th><td>0.55</td></tr></tbody>"
    "</table></table-wrap></body></article>"
)
COMPONENTS = [
    ("subject", "Classifier performance"),
    ("relation", "AUROC"),
    ("answer", "0.91"),
    ("row", "TP53"),
    ("column", "AUROC"),
]
CREATED = "2026-11-09T12:00:00.000000Z"
ENTITY_GOLD = {"answer_class": "biological_entity", "canonical": "TP53", "aliases": []}


def doc_ref(root: Path, name: str, document: CanonicalDocument) -> ArtifactRef:
    data = serialize_project_json(document, CanonicalDocument)
    path = root / "docs" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return ArtifactRef(
        artifact_id=f"doc_{name}", logical_ref=f"docs/{name}.json", sha256=sha256_bytes(data)
    )


def make_world(tmp_path: Path, count: int = 5):
    """JATS and PDF-style documents that differ scientifically, plus `count` paired items."""
    root = tmp_path / "world"
    jats, pdf = parse(JATS_XML.encode()), native_table()
    refs = {"JATS": doc_ref(root, "jats", jats), "PDF_DOCLING": doc_ref(root, "pdf", pdf)}
    docs = {"JATS": jats, "PDF_DOCLING": pdf}
    items = []
    for n in range(count):
        extra = {"item_id": f"q_{n:03d}"}
        if n % 2:
            extra.update(answer_class="biological_entity", gold=ENTITY_GOLD)
        item = make_item(jats, COMPONENTS, **extra)
        data = item.model_dump()
        data["pdf_refs"][0].update(
            artifact_id=pdf.source.artifact_id, source_sha256=pdf.source.sha256
        )
        items.append(BenchmarkItem.model_validate(data))
    sources = {
        (item.item_id, rep): sar.Source(item, docs[rep], root, refs[rep])
        for item in items
        for rep in sar.REPRESENTATIONS
    }
    return items, sources


def packet(tmp_path: Path, name="out", seed=11, **changes):
    items, sources = make_world(tmp_path)
    out = tmp_path / name
    args = {
        "out": out,
        "seed": seed,
        "papers": ("P001",),
        "pairs": 3,
        "created_utc": CREATED,
        "code": "abc",
    }
    args.update(changes)
    sar.build_packet(sources, items, **args)
    return out


# ---- scheduling ----------------------------------------------------------------------


def test_schedule_is_deterministic_and_depends_on_the_seed():
    ids = [f"q{n:03d}" for n in range(26)]
    assert sar.schedule(ids, 5) == sar.schedule(list(reversed(ids)), 5)
    assert sar.schedule(ids, 5) != sar.schedule(ids, 6)


@pytest.mark.parametrize("pairs", [1, 2, 3, 5, 13, 26, 40, 100])
def test_every_pair_is_separated_for_any_batch_size_and_seed(pairs):
    ids = [f"q{n:03d}" for n in range(pairs)]
    for seed in range(8):
        order = sar.schedule(ids, seed)
        sar.check_separation(order, sar.min_gap(pairs))
        assert len(order) == 2 * pairs


def test_gap_for_the_rehearsal_is_thirty_percent_of_the_list():
    assert sar.min_gap(26) == 16  # ceil(0.3 * 52)


def test_representation_orientation_is_not_systematic():
    ids = [f"q{n:03d}" for n in range(26)]
    firsts = 0
    for seed in range(20):
        seen: set[str] = set()
        for item_id, rep in sar.schedule(ids, seed):
            if item_id not in seen and rep == "JATS":
                firsts += 1
            seen.add(item_id)
    assert 0.35 < firsts / (20 * 26) < 0.65


def test_separation_check_rejects_close_incomplete_and_repeated_units():
    near = [("a", "JATS"), ("a", "PDF_DOCLING"), ("b", "JATS"), ("b", "PDF_DOCLING")]
    with pytest.raises(sar.ReviewError, match="closer"):
        sar.check_separation(near, 2)
    with pytest.raises(sar.ReviewError, match="complete pair"):
        sar.check_separation([("a", "JATS")], 1)
    with pytest.raises(sar.ReviewError, match="repeats"):
        sar.check_separation([("a", "JATS"), ("a", "JATS")], 1)


def test_schedule_rejects_empty_or_duplicate_items():
    with pytest.raises(sar.ReviewError):
        sar.schedule([], 1)
    with pytest.raises(sar.ReviewError):
        sar.schedule(["a", "a"], 1)


# ---- selection -----------------------------------------------------------------------


def test_selection_is_deterministic_development_only_and_refuses_p027(tmp_path):
    items, _ = make_world(tmp_path)
    assert sar.select_items(items, ("P001",), 4, 3) == sar.select_items(items, ("P001",), 4, 3)
    with pytest.raises(sar.ReviewError, match="excluded"):
        sar.select_items(items, ("P001", "P027"), 2, 3)
    final = items[0].model_copy(update={"split": "final_test"})
    with pytest.raises(sar.ReviewError):
        sar.select_items([final], ("P001",), 1, 3)


def test_selection_covers_answer_classes_before_repeating_a_stratum(tmp_path):
    items, _ = make_world(tmp_path)
    for seed in range(5):
        chosen = sar.select_items(items, ("P001",), 2, seed)
        assert {i.answer_class for i in chosen} == {"quantitative_value", "biological_entity"}


def test_selection_rejects_more_pairs_than_items(tmp_path):
    items, _ = make_world(tmp_path)
    with pytest.raises(sar.ReviewError):
        sar.select_items(items, ("P001",), 99, 1)


# ---- packet, masking and provenance --------------------------------------------------


def test_packet_is_byte_identical_for_the_same_seed_and_differs_for_another(tmp_path):
    a, b, c = packet(tmp_path, "a"), packet(tmp_path, "b"), packet(tmp_path, "c", seed=12)
    for name in (
        "reviewer/masked_view.md",
        "reviewer/review.json",
        "provenance/order.json",
        "provenance/unmasked_evidence.json",
        "provenance/session_manifest.json",
    ):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    assert (a / "provenance/order.json").read_bytes() != (c / "provenance/order.json").read_bytes()
    assert json.loads((a / "provenance/order.json").read_bytes())["seed"] == 11


def test_reviewer_directory_holds_no_provenance_and_no_labels(tmp_path):
    out = packet(tmp_path)
    reviewer = {p.name for p in (out / "reviewer").iterdir()}
    assert reviewer == {"masked_view.md", "review.json", "inspections.jsonl"}
    assert all(r["label"] is None for r in sar.read_review(out))
    view = (out / "reviewer/masked_view.md").read_text()
    for forbidden in ("doc_jats", "doc_pdf", "q_00", "P001", "world"):
        assert forbidden not in view


def test_scaffold_has_no_representation_names_or_identifiers_but_source_text_keeps_them(tmp_path):
    out = packet(tmp_path)
    _, units = sar.load_provenance(out)
    jats_handle = next(h for h, u in units.items() if u["representation"] == "JATS")
    analysis = sar.load_analysis(jats_handle, 11, units[jats_handle])
    lines = analysis.render_unit()
    assert sar.scaffold_leaks(lines, analysis.identifiers()) == []
    source = "\n".join(text for line in lines for kind, text in line if kind == "source")
    assert "exported as PDF and compared by AUROC in the JATS release" in source


def test_leak_detector_flags_system_identifiers_but_not_source_words(tmp_path):
    out = packet(tmp_path)
    _, units = sar.load_provenance(out)
    handle = next(iter(units))
    analysis = sar.load_analysis(handle, 11, units[handle])
    node = next(iter(analysis.identifiers() - {analysis.source.item.item_id}))
    assert sar.scaffold_leaks([[sar.S(f"see {node}")]], analysis.identifiers())
    assert sar.scaffold_leaks([[sar.S("representation PDF_DOCLING")]], set())
    assert (
        sar.scaffold_leaks(
            [[sar.S("x"), sar.T("PDF JATS docling " + node)]], analysis.identifiers()
        )
        == []
    )


def test_every_shown_source_text_is_the_canonical_text_byte_for_byte(tmp_path):
    out = packet(tmp_path)
    _, units = sar.load_provenance(out)
    for handle, record in units.items():
        analysis = sar.load_analysis(handle, 11, record)
        rendered = {
            text for line in analysis.render_unit() for kind, text in line if kind == "source"
        }
        for container in analysis.shown():
            for unit in analysis.index.units[container]:
                assert unit.text in rendered
        assert bool(record["shown"]) == (record["representation"] == "JATS")


def test_jats_and_pdf_views_keep_their_own_different_science(tmp_path):
    out = packet(tmp_path)
    _, units = sar.load_provenance(out)
    text = {}
    for handle, record in units.items():
        analysis = sar.load_analysis(handle, 11, record)
        text.setdefault(record["representation"], sar.to_text(analysis.render_unit()))
    assert "0.91" in text["JATS"] and "TP53" in text["JATS"]
    assert "none located" in text["PDF_DOCLING"].split("Candidate contexts:")[1]
    assert "0.91" not in text["PDF_DOCLING"].split("Candidate contexts:")[1]


def test_unmasked_record_hash_is_fixed_in_the_manifest_and_tampering_is_detected(tmp_path):
    out = packet(tmp_path)
    manifest, _ = sar.load_provenance(out)
    data = (out / "provenance/unmasked_evidence.json").read_bytes()
    assert sha256_bytes(data) == manifest["unmasked_evidence_sha256"]
    assert manifest["status"] == sar.STATUS and manifest["excluded_papers"] == ["P027"]
    (out / "provenance/unmasked_evidence.json").write_bytes(data + b" ")
    with pytest.raises(sar.ReviewError, match="differs"):
        sar.load_provenance(out)


def test_handles_map_back_to_the_locator_candidate_occurrences(tmp_path):
    out = packet(tmp_path)
    _, units = sar.load_provenance(out)
    handle = next(h for h, u in units.items() if u["representation"] == "JATS")
    analysis = sar.load_analysis(handle, 11, units[handle])
    assert units[handle]["containers"] == {h: c for c, h in analysis.handles.items()}
    assert (
        units[handle]["candidate"]["artifact_id"] == sar.stage_a_ref(analysis.candidate).artifact_id
    )
    for set_locus in analysis.candidate.sets:
        for placement in set_locus.placements:
            assert placement.container_node_id in analysis.handles


def test_second_rater_and_intra_rater_packets(tmp_path):
    second = packet(tmp_path, "r2", seed=99, pairs=None, fraction=0.2, role="second_rater")
    assert json.loads((second / "provenance/session_manifest.json").read_bytes())["pairs"] == 1
    first = packet(tmp_path, "r1")
    with pytest.raises(sar.ReviewError, match="manifest"):
        packet(tmp_path, "i0", seed=3, role="intra_rater")
    previous = first / "provenance/session_manifest.json"
    with pytest.raises(sar.ReviewError, match="14 days"):
        packet(
            tmp_path,
            "i1",
            seed=3,
            role="intra_rater",
            previous_manifest=previous,
            created_utc="2026-11-20T00:00:00.000000Z",
        )
    with pytest.raises(sar.ReviewError, match="14 days"):
        packet(
            tmp_path,
            "i2",
            seed=11,
            role="intra_rater",
            previous_manifest=previous,
            created_utc="2026-12-01T00:00:00.000000Z",
        )
    later = packet(
        tmp_path,
        "i3",
        seed=3,
        role="intra_rater",
        previous_manifest=previous,
        created_utc="2026-11-24T00:00:00.000000Z",
    )
    assert (
        json.loads((later / "provenance/session_manifest.json").read_bytes())["role"]
        == "intra_rater"
    )


def test_exactly_one_of_pairs_or_fraction(tmp_path):
    with pytest.raises(sar.ReviewError):
        packet(tmp_path, pairs=None)


# ---- timing and records --------------------------------------------------------------


def fill(out: Path, handle: str, **fields):
    records = sar.read_review(out)
    sar.find(records, handle).update(fields)
    sar.save_review(out, records)


def complete_unit(out, handle, label, seconds, guess, *, sets=None, **extra):
    t0 = "2026-11-09T12:00:00.000000Z"
    t1 = f"2026-11-09T12:{seconds // 60:02d}:{seconds % 60:02d}.000000Z"
    sar.start(out, handle, t0)
    fill(
        out,
        handle,
        label=label,
        rationale="inspected the shown context",
        sets=sets or [],
        consulted_original_source=False,
        **extra,
    )
    sar.finish(out, handle, t1)
    sar.guess(out, handle, guess, 3, "2026-11-09T13:00:00.000000Z")


def table_sets(analysis):
    container = next(c for c in analysis.index.units if analysis.nodes[c].table is not None)
    components = [{"role": r, "text": t} for r, t in COMPONENTS]
    return [
        {
            "set_index": 0,
            "placements": [{"container": analysis.handles[container], "components": components}],
        }
    ]


def units_by_arm(out):
    _, units = sar.load_provenance(out)
    return {u["representation"]: h for h, u in units.items()}, units


def test_timing_commands_enforce_order(tmp_path):
    out = packet(tmp_path)
    h = sar.read_review(out)[0]["handle"]
    with pytest.raises(sar.ReviewError):
        sar.finish(out, h, CREATED)
    sar.start(out, h, CREATED)
    with pytest.raises(sar.ReviewError, match="already"):
        sar.start(out, h, CREATED)
    with pytest.raises(sar.ReviewError, match="label"):
        sar.finish(out, h, CREATED)
    with pytest.raises(sar.ReviewError, match="finished"):
        sar.guess(out, h, "JATS", 3, CREATED)
    fill(out, h, label="unresolved", rationale="r")
    sar.finish(out, h, CREATED)
    with pytest.raises(sar.ReviewError):
        sar.guess(out, h, "XML", 3, CREATED)
    with pytest.raises(sar.ReviewError):
        sar.guess(out, h, "JATS", 9, CREATED)
    sar.guess(out, h, "cannot_tell", 1, CREATED)


def complete_packet(tmp_path):
    out = packet(tmp_path, pairs=2)
    _, units = sar.load_provenance(out)
    by_rep: dict[str, list[str]] = {"JATS": [], "PDF_DOCLING": []}
    for h, u in units.items():
        by_rep[u["representation"]].append(h)
    for k, h in enumerate(by_rep["JATS"]):
        analysis = sar.load_analysis(h, 11, units[h])
        complete_unit(out, h, "preserved", 60 + 30 * k, "JATS", sets=table_sets(analysis))
    for k, h in enumerate(by_rep["PDF_DOCLING"]):
        sar.inspect(out, h, None, CREATED)
        complete_unit(out, h, "absent", 120 + 60 * k, "cannot_tell")
    return out, by_rep


def test_valid_records_round_trip_through_the_stage_a_validator(tmp_path):
    out, _ = complete_packet(tmp_path)
    judgments = sar.validate_all(out, "Rama Krishna Pudota")
    assert {j.label for j in judgments.values()} == {"preserved", "absent"}
    written = json.loads((out / "reviewer/judgments_first_rater.json").read_bytes())
    assert written["status"] == sar.STATUS
    assert sorted(p.name for p in (out / "reviewer").iterdir()) == [
        "inspections.jsonl",
        "judgments_first_rater.json",
        "masked_view.md",
        "review.json",
    ]


def test_incomplete_or_missing_records_are_rejected(tmp_path):
    out, by_rep = complete_packet(tmp_path)
    fill(out, by_rep["JATS"][0], representation_guess=None)
    with pytest.raises(sar.ReviewError, match="incomplete"):
        sar.validate_all(out, "r")
    out2 = packet(tmp_path, "o2")
    with pytest.raises(sar.ReviewError, match="incomplete"):
        sar.validate_all(out2, "r")
    records = sar.read_review(out)
    sar.save_review(out, records[:-1])
    with pytest.raises(sar.ReviewError, match="exactly once"):
        sar.validate_all(out, "r")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("label", "mostly_fine", "unknown label"),
        ("finished_utc", "2026-11-09T11:00:00.000000Z", "out of order"),
        ("guessed_utc", "2026-11-09T12:00:01.000000Z", "out of order"),
        ("guess_confidence", 7, "invalid representation guess"),
        ("started_utc", "2026-11-09T12:00:00", "UTC"),
    ],
)
def test_inconsistent_records_are_rejected(tmp_path, field, value, message):
    out, by_rep = complete_packet(tmp_path)
    fill(out, by_rep["JATS"][0], **{field: value})
    with pytest.raises(sar.ReviewError, match=message):
        sar.validate_all(out, "r")


def test_loci_must_name_a_known_container_and_text_it_contains(tmp_path):
    out, by_rep = complete_packet(tmp_path)
    h = by_rep["JATS"][0]
    good = sar.read_review(out)
    sets = sar.find(good, h)["sets"]
    sets[0]["placements"][0]["container"] = "Hnope"
    fill(out, h, sets=sets)
    with pytest.raises(sar.ReviewError, match="unknown container"):
        sar.validate_all(out, "r")
    sets[0]["placements"][0]["container"] = sar.find(sar.read_review(out), h)["sets"][0][
        "placements"
    ][0]["container"]
    _, units = sar.load_provenance(out)
    analysis = sar.load_analysis(h, 11, units[h])
    sets = table_sets(analysis)
    sets[0]["placements"][0]["components"][2]["text"] = "9.99"
    fill(out, h, sets=sets)
    with pytest.raises(sar.ReviewError, match="does not occur"):
        sar.validate_all(out, "r")


def test_existing_validator_rejects_a_label_the_locator_contradicts(tmp_path):
    out, by_rep = complete_packet(tmp_path)
    h = by_rep["JATS"][0]
    sar.inspect(out, h, None, CREATED)
    fill(out, h, label="absent", sets=[])
    with pytest.raises(sar.ReviewError, match="contradicts"):
        sar.validate_all(out, "r")


def test_altered_label_without_alteration_is_rejected_by_the_model(tmp_path):
    out, by_rep = complete_packet(tmp_path)
    fill(out, by_rep["JATS"][0], label="altered_but_sufficient")
    with pytest.raises(sar.ReviewError, match="alteration"):
        sar.validate_all(out, "r")


def test_absent_requires_a_recorded_expanded_inspection(tmp_path):
    out, _ = complete_packet(tmp_path)
    (out / "reviewer/inspections.jsonl").write_text("")
    with pytest.raises(sar.ReviewError, match="inspected"):
        sar.validate_all(out, "r")


def test_inspection_stays_in_the_representation_under_review_and_is_logged(tmp_path):
    out = packet(tmp_path)
    by_arm, units = units_by_arm(out)
    text = sar.inspect(out, by_arm["PDF_DOCLING"], None, CREATED)
    assert "ClusterONE" in text and "TP53" not in text and "Methods" not in text
    jats = sar.inspect(out, by_arm["JATS"], None, CREATED)
    assert "TP53" in jats and "ClusterONE" not in jats
    container = units[by_arm["JATS"]]["shown"][0]
    section = sar.inspect(out, by_arm["JATS"], container, CREATED)
    assert section
    events = [json.loads(x) for x in (out / "reviewer/inspections.jsonl").read_text().splitlines()]
    assert [e["scope"] for e in events] == ["document", "document", "section"]
    with pytest.raises(sar.ReviewError, match="unknown container"):
        sar.inspect(out, by_arm["JATS"], "Hnope", CREATED)


# ---- summary -------------------------------------------------------------------------


def test_summary_matches_a_hand_computed_oracle_and_labels_projection(tmp_path):
    out, _ = complete_packet(tmp_path)
    summary = sar.summarize(out, "r")
    assert summary["units"] == 4
    assert summary["labels"] == {"absent": 2, "preserved": 2}
    assert summary["attribution_collapse"] == {"failed": 2, "usable": 2}
    assert summary["observed_seconds"]["total"] == 60 + 90 + 120 + 180
    assert summary["observed_seconds"]["median"] == 105
    assert summary["observed_seconds_by_representation"]["JATS"]["total"] == 150
    assert summary["masking"]["JATS"] == {"correct": 2, "mean_confidence": 3}
    assert summary["masking"]["PDF_DOCLING"] == {"cannot_tell": 2, "mean_confidence": 3}
    assert summary["units_with_expanded_inspection"] == 2
    projected = summary["projected_not_observed"]
    assert projected["full_benchmark_units"] == 610
    assert projected["first_rater_hours_median"] == pytest.approx(610 * 105 / 3600)
    assert "observed" not in "".join(projected)
    assert sar.dumps(sar.summarize(out, "r")) == sar.dumps(summary)


def test_selection_prefers_rare_strata_within_a_paper(tmp_path):
    items, _ = make_world(tmp_path, count=5)  # entity items (odd ids) are the rarer stratum
    for seed in range(5):
        chosen = sar.select_items(items, ("P001",), 1, seed)
        assert chosen[0].answer_class == "biological_entity"


# ---- balanced orientation ------------------------------------------------------------


def first_arms(order):
    seen: dict[str, str] = {}
    for item_id, rep in order:
        seen.setdefault(item_id, rep)
    return seen


PAPER_SIZES = {"A": 5, "B": 5, "C": 5, "D": 5, "E": 6}
GROUPS = {f"{p}{n}": p for p, size in PAPER_SIZES.items() for n in range(size)}


@pytest.mark.parametrize("seed", range(12))
def test_orientation_is_balanced_overall_and_within_papers(seed):
    order = sar.schedule(sorted(GROUPS), seed, groups=GROUPS)
    firsts = first_arms(order)
    assert sum(r == "JATS" for r in firsts.values()) == 13
    for paper, size in PAPER_SIZES.items():
        jats = sum(firsts[i] == "JATS" for i in GROUPS if GROUPS[i] == paper)
        assert abs(jats - (size - jats)) <= 1
    sar.check_separation(order, sar.min_gap(26))
    assert sar.min_gap(26) == 16


def test_balanced_schedule_is_reproducible_and_still_seed_dependent():
    ids = sorted(GROUPS)
    assert sar.schedule(ids, 4, groups=GROUPS) == sar.schedule(ids, 4, groups=GROUPS)
    assert sar.schedule(ids, 4, groups=GROUPS) != sar.schedule(ids, 5, groups=GROUPS)


@pytest.mark.parametrize("pairs", [1, 2, 3, 7, 26, 41])
def test_orientation_without_groups_differs_by_at_most_one(pairs):
    firsts = first_arms(sar.schedule([f"q{n:03d}" for n in range(pairs)], 3))
    jats = sum(r == "JATS" for r in firsts.values())
    assert abs(jats - (pairs - jats)) <= 1
