"""Stage A evidence preservation: conservative locator, judgments, and read-back."""

import pytest
from pydantic import ValidationError
from test_documents_jats import parse
from test_retrieval_models import ref

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import HashMismatchError
from biorag_repro.documents.chunking import discover_occurrences
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation.models import (
    ComponentLocus,
    Placement,
    SetLocus,
    StageACandidate,
    StageAJudgment,
)
from biorag_repro.evaluation.stage_a import (
    IneligibleItem,
    automatic_judgment,
    item_digest,
    locate,
    validate_judgment,
)
from biorag_repro.evaluation.storage import (
    publish_candidate,
    publish_judgment,
    read_candidate,
    read_judgment,
    stage_a_ref,
)

PROSE = (
    "<article><body><sec><title>Results</title>"
    "<p>Under condition A, treatment effect was −1.2 × 10⁻³ in the cohort.</p>"
    "</sec></body></article>"
)
TABLE = (
    "<article><body><table-wrap><caption><p>Classifier performance</p></caption><table>"
    "<thead><tr><th>Gene</th><th>AUROC</th><th>AUPRC</th></tr></thead>"
    "<tbody><tr><th>TP53</th><td>0.91</td><td>0.62</td></tr>"
    "<tr><th>KRAS</th><td>0.55</td><td>0.88</td></tr></tbody></table></table-wrap></body></article>"
)
PROSE_COMPONENTS = [
    ("subject", "treatment"),
    ("relation", "effect"),
    ("answer", "−1.2 × 10⁻³"),
    ("condition", "condition A"),
]
TABLE_COMPONENTS = [
    ("subject", "Classifier performance"),
    ("relation", "AUROC"),
    ("answer", "0.91"),
    ("row", "TP53"),
    ("column", "AUROC"),
]
CANONICAL = ref("canonical")


def make_item(document: CanonicalDocument, *component_sets, **changes) -> BenchmarkItem:
    jats_hash = document.source.sha256
    data = {
        "schema_version": "1",
        "benchmark_version": "benchmark_v1",
        "corpus_version": "corpus_v1",
        "item_id": "q_001",
        "paper_id": document.paper_id,
        "question": "What was reported?",
        "answer_class": "quantitative_value",
        "gold": {"answer_class": "quantitative_value", "value": "1", "comparator": "eq"},
        "evidence_sets": [
            {"components": [{"role": r, "text": t} for r, t in components]}
            for components in component_sets
        ],
        "jats_refs": [
            {
                "artifact_id": document.source.artifact_id,
                "source_sha256": jats_hash,
                "paper_id": document.paper_id,
                "xml_path": "/article/body/p[1]",
                "source_text": "source",
            }
        ],
        "pdf_refs": [
            {
                "artifact_id": "raw_pdf",
                "source_sha256": "b" * 64,
                "paper_id": document.paper_id,
                "page": 1,
                "source_text": "source",
            }
        ],
        "jats_status": "present_equivalent",
        "pdf_status": "present_equivalent",
        "split": "development",
        "candidate_status": "accepted",
        "review_status": "reviewed",
        "audit_status": "not_selected",
        "decision_ids": ["d1"],
    }
    data.update(changes)
    return BenchmarkItem.model_validate(data)


def with_text(components, role, text):
    return [(r, text if r == role else t) for r, t in components]


def mutate(document: CanonicalDocument, change) -> CanonicalDocument:
    data = document.model_dump()
    change(data)
    return CanonicalDocument.model_validate(data)


def occurrence_ids(document, needle):
    return [
        o.occurrence_id
        for o in discover_occurrences(document, CANONICAL)
        if o.text and needle in o.text
    ]


def node_of(document, tag):
    return next(n for n in document.nodes if n.source_tag == tag).node_id


def judgment(item, document, label, *, sets=(), basis="reviewer", **fields):
    values = {
        "benchmark_version": item.benchmark_version,
        "corpus_version": item.corpus_version,
        "item_id": item.item_id,
        "item_sha256": item_digest(item),
        "paper_id": item.paper_id,
        "representation": document.representation,
        "canonical": CANONICAL,
        "label": label,
        "basis": basis,
        "candidate": None,
        "sets": list(sets),
        "alteration": None,
        "rationale": "inspected original source and canonical content",
        "reviewer": "Rama Krishna Pudota",
        "at_utc": "2026-11-05T12:00:00Z",
    }
    values.update(fields)
    return StageAJudgment.model_validate(values)


def locus(document, items, container_tag="p", observed=None):
    observed = observed or {}
    components = [
        ComponentLocus(
            role=role,
            text=text,
            occurrence_ids=occurrence_ids(document, observed.get(role, text)),
            observed_text=observed.get(role),
        )
        for role, text in items
    ]
    return SetLocus(
        set_index=0,
        placements=[
            Placement(container_node_id=node_of(document, container_tag), components=components)
        ],
    )


@pytest.fixture
def prose():
    document = parse(PROSE.encode())
    return document, make_item(document, PROSE_COMPONENTS)


def kinds(candidate):
    return {s.kind for s in candidate.suggestions}


# ---- preserved: only a complete, verbatim, single-context, exact-mapped fact ----------------


def test_complete_verbatim_prose_fact_is_automatically_preserved(prose):
    document, item = prose
    candidate = locate(item, document, CANONICAL)
    assert candidate.determination == "automatic_preserved"
    [set_locus] = candidate.sets
    [placement] = set_locus.placements
    assert placement.container_node_id == node_of(document, "p")
    assert [c.role for c in placement.components] == [r for r, _ in PROSE_COMPONENTS]
    assert all(c.observed_text is None and c.occurrence_ids for c in placement.components)
    result = automatic_judgment(candidate, stage_a_ref(candidate))
    assert (result.label, result.basis, result.reviewer, result.at_utc) == (
        "preserved",
        "deterministic",
        None,
        None,
    )
    validate_judgment(result, item, document, CANONICAL)


def test_complete_table_fact_requires_verified_row_and_column_association():
    document = parse(TABLE.encode())
    item = make_item(document, TABLE_COMPONENTS)
    candidate = locate(item, document, CANONICAL)
    assert candidate.determination == "automatic_preserved"
    [placement] = candidate.sets[0].placements
    assert placement.container_node_id == node_of(document, "table")


@pytest.mark.parametrize(
    "role, text",
    [("row", "KRAS"), ("column", "AUPRC"), ("row", "Gene"), ("column", "TP53")],
)
def test_table_value_attached_to_wrong_row_or_column_is_never_automatic(role, text):
    document = parse(TABLE.encode())
    item = make_item(document, with_text(TABLE_COMPONENTS, role, text))
    candidate = locate(item, document, CANONICAL)
    assert candidate.determination == "needs_review"
    assert "association_mismatch" in kinds(candidate)
    assert not candidate.sets


def test_table_value_matching_only_a_header_cell_is_not_an_answer_cell():
    document = parse(TABLE.encode())
    item = make_item(document, with_text(TABLE_COMPONENTS, "answer", "TP53"))
    assert locate(item, document, CANONICAL).determination == "needs_review"


def test_table_without_row_and_column_components_cannot_establish_association():
    document = parse(TABLE.encode())
    item = make_item(document, [c for c in TABLE_COMPONENTS if c[0] not in {"row", "column"}])
    assert locate(item, document, CANONICAL).determination == "needs_review"


# ---- failed matching never becomes corruption or absence --------------------------------


@pytest.mark.parametrize(
    "answer, found, suggestion",
    [
        ("0.05", "0.5", "numeric_digit_sequence_match"),
        ("−0.5", "0.5", "sign_difference"),
        ("0.5", "−0.5", "contained_in_longer_token"),
    ],
)
def test_numeric_discrepancies_are_advisory_suggestions_not_labels(answer, found, suggestion):
    document = parse(
        f"<article><body><p>treatment effect was {found} here</p></body></article>".encode()
    )
    item = make_item(
        document,
        [("subject", "treatment"), ("relation", "effect"), ("answer", answer)],
    )
    candidate = locate(item, document, CANONICAL)
    assert candidate.determination == "needs_review"
    assert suggestion in kinds(candidate)
    assert not candidate.sets
    with pytest.raises(ValueError):
        automatic_judgment(candidate, stage_a_ref(candidate))


@pytest.mark.parametrize(
    "role, text, found",
    [
        ("direction", "increased", "decreased"),
        ("answer", "p.R175H", "p.R175L"),
        ("answer", "p.R175H", "p.R175Hfs"),
        ("unit", "mg/L", "mg/dL"),
    ],
)
def test_polarity_variant_and_unit_changes_need_review(role, text, found):
    components = [("subject", "treatment"), ("relation", "effect"), ("answer", "7")]
    if role != "answer":
        components.append((role, text))
    else:
        components[2] = (role, text)
    body = "treatment effect 7 " + found if role != "answer" else "treatment effect " + found
    document = parse(f"<article><body><p>{body}</p></body></article>".encode())
    candidate = locate(make_item(document, components), document, CANONICAL)
    assert candidate.determination == "needs_review"
    assert not candidate.sets


def test_component_missing_everywhere_is_reported_but_not_labelled_absent(prose):
    document, _ = prose
    item = make_item(document, with_text(PROSE_COMPONENTS, "answer", "9.99"))
    candidate = locate(item, document, CANONICAL)
    assert candidate.determination == "needs_review"
    assert "component_not_found" in kinds(candidate)
    assert not hasattr(candidate, "label")


def test_facts_split_across_contexts_are_not_combined():
    document = parse(
        b"<article><body><p>treatment effect</p><p>was 5 under condition A</p></body></article>"
    )
    item = make_item(
        document,
        [
            ("subject", "treatment"),
            ("relation", "effect"),
            ("answer", "5"),
            ("condition", "condition A"),
        ],
    )
    candidate = locate(item, document, CANONICAL)
    assert candidate.determination == "needs_review"
    assert "split_across_contexts" in kinds(candidate)


def test_repeated_value_under_another_condition_is_not_evidence():
    document = parse(
        b"<article><body><p>Under condition A, treatment effect was 5.</p>"
        b"<p>Under condition B, treatment effect was 5.</p></body></article>"
    )
    components = [
        ("subject", "treatment"),
        ("relation", "effect"),
        ("answer", "5"),
        ("condition", "condition A"),
    ]
    candidate = locate(make_item(document, components), document, CANONICAL)
    [placement] = candidate.sets[0].placements
    first = next(n for n in document.nodes if n.source_tag == "p")
    assert placement.container_node_id == first.node_id
    assert len(candidate.sets[0].placements) == 1


def test_identical_complete_facts_in_two_blocks_remain_two_placements():
    document = parse(
        b"<article><body><p>treatment effect was 5.</p><p>treatment effect was 5.</p></body></article>"
    )
    item = make_item(document, [("subject", "treatment"), ("relation", "effect"), ("answer", "5")])
    candidate = locate(item, document, CANONICAL)
    assert len(candidate.sets[0].placements) == 2


def test_alternative_minimum_sufficient_sets_stay_distinct():
    document = parse(
        b"<article><body><p>treatment effect was 5 under condition A</p>"
        b"<p>intervention outcome was 5 at baseline</p></body></article>"
    )
    first = [("subject", "treatment"), ("relation", "effect"), ("answer", "5")]
    second = [("subject", "intervention"), ("relation", "outcome"), ("answer", "5")]
    candidate = locate(make_item(document, first, second), document, CANONICAL)
    assert [s.set_index for s in candidate.sets] == [0, 1]
    only_second = locate(
        make_item(
            document, [("subject", "treatment"), ("relation", "missing"), ("answer", "5")], second
        ),
        document,
        CANONICAL,
    )
    assert candidate.determination == only_second.determination == "automatic_preserved"
    assert [s.set_index for s in only_second.sets] == [1]


@pytest.mark.parametrize(
    "change",
    [
        lambda d: [
            s["mapping"].update(state="approximate") for n in d["nodes"] for s in n["segments"]
        ],
        lambda d: [n.update(content_state="unverified_native_rendition") for n in d["nodes"]],
        lambda d: [n.update(interpretation_state="ambiguous") for n in d["nodes"]],
    ],
)
def test_uncertain_mapping_or_content_never_certifies_preservation(prose, change):
    document, item = prose
    candidate = locate(item, mutate(document, change), CANONICAL)
    assert candidate.determination == "needs_review"
    assert candidate.qualifications
    assert "qualified_mapping" in kinds(candidate)
    assert not candidate.sets


def test_incomplete_canonical_document_is_never_preserved(prose):
    document, item = prose
    candidate = locate(item, mutate(document, lambda d: d.update(status="incomplete")), CANONICAL)
    assert candidate.determination == "needs_review"
    assert "incomplete_canonical" in kinds(candidate)


# ---- eligibility and false provenance ---------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"jats_status": "absent", "candidate_status": "proposed"},
        {"pdf_status": "materially_different", "candidate_status": "proposed"},
        {"jats_status": "unresolved", "candidate_status": "proposed"},
        {"candidate_status": "excluded"},
        {"split": "final_test"},
    ],
)
def test_ineligible_or_protected_items_are_rejected_not_scored(prose, changes):
    document, _ = prose
    item = make_item(document, PROSE_COMPONENTS, **changes)
    with pytest.raises(IneligibleItem):
        locate(item, document, CANONICAL)


def test_item_source_reference_must_name_the_documents_original_source(prose):
    document, _ = prose
    item = make_item(document, PROSE_COMPONENTS)
    data = item.model_dump()
    data["jats_refs"][0]["source_sha256"] = "c" * 64
    with pytest.raises(IneligibleItem):
        locate(BenchmarkItem.model_validate(data), document, CANONICAL)
    data = item.model_dump()
    data["jats_refs"][0]["artifact_id"] = "raw_other"
    with pytest.raises(IneligibleItem):
        locate(BenchmarkItem.model_validate(data), document, CANONICAL)


def test_item_for_another_paper_is_rejected(prose):
    document, _ = prose
    other = mutate(document, lambda d: d.update(paper_id="P002"))
    item = make_item(document, PROSE_COMPONENTS)
    with pytest.raises(IneligibleItem):
        locate(item, other, CANONICAL)


# ---- reviewer judgments: all five labels, each with its own invariants -------------------


def test_reviewer_preserved_requires_complete_verbatim_loci(prose):
    document, item = prose
    good = judgment(item, document, "preserved", sets=[locus(document, PROSE_COMPONENTS)])
    validate_judgment(good, item, document, CANONICAL)
    partial = locus(document, PROSE_COMPONENTS[:3])
    with pytest.raises(ValueError):
        validate_judgment(
            judgment(item, document, "preserved", sets=[partial]), item, document, CANONICAL
        )
    with pytest.raises(ValidationError):
        judgment(item, document, "preserved", sets=[])


def test_altered_but_sufficient_records_the_observed_alteration():
    document = parse(PROSE.encode())
    item = make_item(document, with_text(PROSE_COMPONENTS, "answer", "−1.2×10⁻³"))
    candidate = locate(item, document, CANONICAL)
    assert candidate.determination == "needs_review"
    items = with_text(PROSE_COMPONENTS, "answer", "−1.2×10⁻³")
    sets = [locus(document, items, observed={"answer": "−1.2 × 10⁻³"})]
    good = judgment(
        item, document, "altered_but_sufficient", sets=sets, alteration="spacing around ×"
    )
    validate_judgment(good, item, document, CANONICAL)
    with pytest.raises(ValidationError):
        judgment(item, document, "altered_but_sufficient", sets=sets)
    with pytest.raises(ValidationError):
        judgment(item, document, "altered_but_sufficient", sets=[], alteration="x")
    unseen = [locus(document, items, observed={"answer": "−1.2 × 10⁻³"})]
    unseen[0].placements[0].components[2].observed_text = "not in source"
    with pytest.raises(ValueError):
        validate_judgment(
            judgment(item, document, "altered_but_sufficient", sets=unseen, alteration="x"),
            item,
            document,
            CANONICAL,
        )
    with pytest.raises(ValueError):
        no_change = [locus(document, PROSE_COMPONENTS)]
        validate_judgment(
            judgment(item, document, "altered_but_sufficient", sets=no_change, alteration="x"),
            item,
            document,
            CANONICAL,
        )


def test_observed_text_must_differ_from_the_benchmark_component():
    with pytest.raises(ValidationError):
        ComponentLocus(role="answer", text="5", occurrence_ids=["o"], observed_text="5")


def test_corrupted_requires_alteration_and_reviewer_and_survives_without_loci():
    document = parse(PROSE.replace("−1.2 × 10⁻³", "−1.2 × 10⁻²").encode())
    item = make_item(document, PROSE_COMPONENTS)
    candidate = locate(item, document, CANONICAL)
    assert candidate.determination == "needs_review"
    good = judgment(
        item,
        document,
        "scientifically_corrupted",
        alteration="exponent changed from −3 to −2",
        candidate=stage_a_ref(candidate),
    )
    validate_judgment(good, item, document, CANONICAL)
    with pytest.raises(ValidationError):
        judgment(item, document, "scientifically_corrupted")
    with pytest.raises(ValidationError):
        judgment(item, document, "scientifically_corrupted", alteration="x", reviewer=None)
    with pytest.raises(ValidationError):
        judgment(item, document, "scientifically_corrupted", alteration="x", basis="deterministic")


def test_absent_and_unresolved_judgments_are_reviewer_decisions():
    document = parse(PROSE.encode())
    item = make_item(document, with_text(PROSE_COMPONENTS, "answer", "9.99"))
    absent = judgment(item, document, "absent")
    validate_judgment(absent, item, document, CANONICAL)
    validate_judgment(judgment(item, document, "unresolved"), item, document, CANONICAL)
    with pytest.raises(ValidationError):
        judgment(item, document, "absent", sets=[locus(document, PROSE_COMPONENTS)])
    with pytest.raises(ValidationError):
        judgment(item, document, "absent", rationale=None)


@pytest.mark.parametrize("label", ["absent", "scientifically_corrupted"])
def test_loss_labels_are_rejected_when_a_complete_verbatim_fact_is_present(prose, label):
    document, item = prose
    with pytest.raises(ValueError, match="contradict"):
        validate_judgment(
            judgment(item, document, label, alteration="x"), item, document, CANONICAL
        )


def test_deterministic_basis_is_limited_to_a_reproducible_preserved_candidate():
    document = parse(PROSE.replace("−1.2 × 10⁻³", "−1.2 × 10⁻²").encode())
    item = make_item(document, PROSE_COMPONENTS)
    candidate = locate(item, document, CANONICAL)
    with pytest.raises(ValueError):
        automatic_judgment(candidate, stage_a_ref(candidate))
    forged = judgment(
        item,
        document,
        "preserved",
        basis="deterministic",
        reviewer=None,
        at_utc=None,
        rationale=None,
        candidate=stage_a_ref(candidate),
        sets=[
            SetLocus(
                set_index=0,
                placements=[
                    Placement(
                        container_node_id=node_of(document, "p"),
                        components=[
                            ComponentLocus(
                                role=role,
                                text=text,
                                occurrence_ids=occurrence_ids(document, "effect"),
                            )
                            for role, text in PROSE_COMPONENTS
                        ],
                    )
                ],
            )
        ],
    )
    with pytest.raises(ValueError):
        validate_judgment(forged, item, document, CANONICAL)


def test_reviewer_fields_and_basis_are_structurally_enforced(prose):
    document, item = prose
    sets = [locus(document, PROSE_COMPONENTS)]
    with pytest.raises(ValidationError):
        judgment(item, document, "preserved", sets=sets, reviewer=" ")
    with pytest.raises(ValidationError):
        judgment(item, document, "preserved", sets=sets, at_utc="2026-11-05T12:00:00+01:00")
    with pytest.raises(ValidationError):
        judgment(item, document, "preserved", sets=sets, rationale="  ")
    with pytest.raises(ValidationError):
        judgment(item, document, "preserved", sets=sets, basis="deterministic")


def test_judgment_cannot_be_reused_for_changed_item_or_other_artifacts(prose):
    document, item = prose
    result = judgment(item, document, "preserved", sets=[locus(document, PROSE_COMPONENTS)])
    changed = make_item(document, with_text(PROSE_COMPONENTS, "answer", "0"))
    with pytest.raises(ValueError):
        validate_judgment(result, changed, document, CANONICAL)
    with pytest.raises(ValueError):
        validate_judgment(result, item, document, ref("other_canonical"))
    other_arm = result.model_copy(update={"representation": "PDF_DOCLING"})
    with pytest.raises(ValueError):
        validate_judgment(other_arm, item, document, CANONICAL)


def test_loci_must_name_real_text_occurrences_of_this_canonical_document(prose):
    document, item = prose
    real = locus(document, PROSE_COMPONENTS)
    fake = real.model_copy(deep=True)
    fake.placements[0].components[0].occurrence_ids = ["occ_missing"]
    with pytest.raises(ValueError):
        validate_judgment(
            judgment(item, document, "preserved", sets=[fake]), item, document, CANONICAL
        )
    wrong_container = real.model_copy(deep=True)
    wrong_container.placements[0].container_node_id = "node_missing"
    with pytest.raises(ValueError):
        validate_judgment(
            judgment(item, document, "preserved", sets=[wrong_container]), item, document, CANONICAL
        )
    wrong_text = real.model_copy(deep=True)
    wrong_text.placements[0].components[0].text = "something else"
    with pytest.raises(ValueError):
        validate_judgment(
            judgment(item, document, "preserved", sets=[wrong_text]), item, document, CANONICAL
        )
    with pytest.raises(ValidationError):
        judgment(item, document, "preserved", sets=[real, real])


# ---- identity, publication, and scientific read-back ------------------------------------


def test_stage_a_is_independent_of_retrieval_and_chunk_configuration():
    for model in (StageACandidate, StageAJudgment):
        names = " ".join(model.model_fields)
        assert "chunk" not in names and "retriev" not in names


def test_candidate_and_judgment_publish_immutably_and_read_back(prose, tmp_path):
    document, item = prose
    candidate = locate(item, document, CANONICAL)
    cref, cmanifest = publish_candidate(candidate, item, document, CANONICAL, tmp_path)
    assert cref == stage_a_ref(candidate)
    assert read_candidate(cref, tmp_path, item, document, CANONICAL) == candidate
    result = automatic_judgment(candidate, cref)
    jref, _ = publish_judgment(result, item, document, CANONICAL, tmp_path)
    assert read_judgment(jref, tmp_path, item, document, CANONICAL) == result
    assert jref.artifact_id != cref.artifact_id
    assert cmanifest.artifact_id.startswith("build_")
    with pytest.raises(FileExistsError):
        publish_judgment(result, item, document, CANONICAL, tmp_path)


def test_identity_is_deterministic_and_content_dependent(prose):
    document, item = prose
    sets = [locus(document, PROSE_COMPONENTS)]
    first = judgment(item, document, "preserved", sets=sets)
    again = judgment(item, document, "preserved", sets=sets)
    other = judgment(item, document, "preserved", sets=sets, rationale="different reviewer note")
    assert stage_a_ref(first) == stage_a_ref(again)
    assert stage_a_ref(first).artifact_id != stage_a_ref(other).artifact_id


def test_read_back_rejects_byte_tampering_and_scientific_counterfeits(prose, tmp_path):
    document, item = prose
    result = judgment(item, document, "preserved", sets=[locus(document, PROSE_COMPONENTS)])
    jref, _ = publish_judgment(result, item, document, CANONICAL, tmp_path)
    path = tmp_path / jref.logical_ref
    other_item = make_item(document, with_text(PROSE_COMPONENTS, "answer", "0"))
    with pytest.raises(ValueError):
        read_judgment(jref, tmp_path, other_item, document, CANONICAL)
    path.chmod(0o644)
    path.write_bytes(path.read_bytes().replace(b"inspected", b"inspectEd"))
    with pytest.raises(HashMismatchError):
        read_judgment(jref, tmp_path, item, document, CANONICAL)


def test_publication_validates_before_writing(prose, tmp_path):
    document, item = prose
    bad = judgment(item, document, "preserved", sets=[locus(document, PROSE_COMPONENTS[:3])])
    with pytest.raises(ValueError):
        publish_judgment(bad, item, document, CANONICAL, tmp_path)
    assert not list(tmp_path.rglob("*.json"))


def test_docling_arm_uses_pdf_source_references_and_native_header_roles():
    from test_chunking import native_table

    document = native_table()
    item = make_item(
        document,
        [
            ("subject", "ClusterONE"),
            ("relation", "Score"),
            ("answer", "Entry"),
            ("row", "Section"),
            ("column", "Score"),
        ],
    )
    data = item.model_dump()
    data["pdf_refs"][0].update(
        artifact_id=document.source.artifact_id, source_sha256=document.source.sha256
    )
    item = BenchmarkItem.model_validate(data)
    candidate = locate(item, document, CANONICAL)
    assert candidate.representation == "PDF_DOCLING"
    assert candidate.determination == "needs_review"
    with pytest.raises(IneligibleItem):
        locate(make_item(document, PROSE_COMPONENTS), document, CANONICAL)


def test_rehashed_counterfeit_judgment_is_rejected_by_scientific_read_back(prose, tmp_path):
    from biorag_repro.evaluation.storage import _publish

    document, item = prose
    real = locus(document, PROSE_COMPONENTS)
    forged = real.model_copy(deep=True)
    forged.placements[0].components[2].occurrence_ids = (
        forged.placements[0].components[0].occurrence_ids
    )
    forged.placements[0].components[2].observed_text = "9.99"
    counterfeit = judgment(
        item, document, "altered_but_sufficient", sets=[forged], alteration="invented"
    )
    ref, _ = _publish(counterfeit, tmp_path)
    assert stage_a_ref(counterfeit) == ref
    with pytest.raises(ValueError):
        read_judgment(ref, tmp_path, item, document, CANONICAL)


@pytest.mark.parametrize(
    "found",
    [
        "0.5",
        "0.050",
        "10.05",
        "0.05e3",
        "0.054",
        "-0.05",
        "−0.05",
        "0.05%",
        "0.0 5",
        "0,05",
        "0.05⁻³",
    ],
)
def test_no_numeric_variant_of_the_answer_is_automatically_preserved(found):
    document = parse(
        f"<article><body><p>treatment effect was {found} overall</p></body></article>".encode()
    )
    item = make_item(
        document, [("subject", "treatment"), ("relation", "effect"), ("answer", "0.05")]
    )
    assert locate(item, document, CANONICAL).determination == "needs_review"


@pytest.mark.parametrize("after", [".", ",", ";", ")", " "])
def test_ordinary_punctuation_after_the_exact_answer_remains_preserved(after):
    document = parse(
        f"<article><body><p>treatment effect was 0.05{after} overall</p></body></article>".encode()
    )
    item = make_item(
        document, [("subject", "treatment"), ("relation", "effect"), ("answer", "0.05")]
    )
    assert locate(item, document, CANONICAL).determination == "automatic_preserved"
