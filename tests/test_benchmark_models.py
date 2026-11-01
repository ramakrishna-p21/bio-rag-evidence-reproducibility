import json

import pytest
from pydantic import ValidationError

from biorag_repro.benchmark.models import (
    BenchmarkItem,
    ConclusionGold,
    EntityGold,
    EvidenceComponent,
    EvidenceSet,
    JatsRef,
    PdfRef,
    QuantityGold,
    SetGold,
)
from biorag_repro.benchmark.storage import decode_jsonl, encode_jsonl


def item(gold: object) -> BenchmarkItem:
    return BenchmarkItem.model_validate(
        {
            "schema_version": "1",
            "benchmark_version": "benchmark_v1",
            "corpus_version": "corpus_v1",
            "item_id": "q_001",
            "paper_id": "P001",
            "question": "What was the reported effect under condition A?",
            "answer_class": gold.answer_class,
            "gold": gold.model_dump(),
            "evidence_sets": [
                {
                    "components": [
                        {"role": "subject", "text": "treatment"},
                        {"role": "relation", "text": "effect"},
                        {"role": "answer", "text": "−1.2 × 10⁻³"},
                        {"role": "condition", "text": "condition A"},
                    ]
                }
            ],
            "jats_refs": [
                {
                    "artifact_id": "raw_j",
                    "source_sha256": "a" * 64,
                    "paper_id": "P001",
                    "xml_path": "/article/body/sec[1]/p[1]",
                    "source_text": "effect −1.2 × 10⁻³",
                }
            ],
            "pdf_refs": [
                {
                    "artifact_id": "raw_p",
                    "source_sha256": "b" * 64,
                    "paper_id": "P001",
                    "page": 2,
                    "source_text": "effect −1.2 × 10⁻³",
                }
            ],
            "jats_status": "unresolved",
            "pdf_status": "unresolved",
            "candidate_status": "lead",
            "review_status": "pending",
            "audit_status": "not_selected",
        }
    )


@pytest.mark.parametrize(
    "gold",
    [
        QuantityGold(
            value="-0.0012",
            comparator="eq",
            unit="mg/L",
            reported_precision="2 significant figures",
        ),
        EntityGold(
            answer_class="biological_entity", canonical="BRCA1", aliases=["breast cancer 1"]
        ),
        EntityGold(answer_class="dataset_cohort", canonical="UK Biobank", aliases=["UKB"]),
        SetGold(
            members=[
                EntityGold(answer_class="biological_entity", canonical="BRCA1"),
                EntityGold(answer_class="biological_entity", canonical="BRCA2"),
            ]
        ),
        ConclusionGold(
            subject="treatment",
            relation="associated with",
            direction="decrease",
            condition="condition A",
        ),
    ],
)
def test_gold_classes_round_trip(gold: object) -> None:
    original = item(gold)
    assert decode_jsonl(encode_jsonl([original]), BenchmarkItem) == [original]


def test_quantity_preserves_comparator_and_precision() -> None:
    assert QuantityGold(value="1.2e-3", comparator="lt", unit="mg/L").value == "1.2e-3"
    signed = QuantityGold(value="-1.2e-3", comparator="ge", unit="mg/L")
    assert signed.value.startswith("-") and signed.comparator == "ge"
    assert QuantityGold(
        value="1.2",
        comparator="eq",
        reported_precision="one decimal place",
        rounding_interval=["1.15", "1.25"],
    ).rounding_interval == ["1.15", "1.25"]
    with pytest.raises(ValidationError):
        QuantityGold(value="NaN", comparator="eq")
    with pytest.raises(ValidationError):
        QuantityGold(value="−1.2", comparator="eq")
    with pytest.raises(ValidationError):
        QuantityGold(
            value="1.2",
            comparator="lt",
            reported_precision="one decimal place",
            rounding_interval=["1.15", "1.25"],
        )


def test_quantity_approved_numeric_equivalent_is_explicit_and_typed() -> None:
    gold = QuantityGold(
        value="2.725e-4",
        comparator="eq",
        reported_precision="four significant figures",
        approved_equivalents=["2.7e-4"],
    )
    assert decode_jsonl(encode_jsonl([item(gold)]), BenchmarkItem)[0].gold == gold
    with pytest.raises(ValidationError):
        QuantityGold(value="2.725e-4", comparator="eq", approved_equivalents=["NaN"])
    with pytest.raises(ValidationError):
        QuantityGold(value="2.725e-4", comparator="eq", approved_equivalents=["2.725e-4"])


def test_evidence_requires_complete_core() -> None:
    with pytest.raises(ValidationError):
        EvidenceSet(components=[EvidenceComponent(role="answer", text="1.2")])


def test_references_require_location_and_bounds() -> None:
    with pytest.raises(ValidationError):
        JatsRef(
            artifact_id="raw", source_sha256="a" * 64, paper_id="P001", xml_path="", source_text="x"
        )
    with pytest.raises(ValidationError):
        PdfRef(artifact_id="raw", source_sha256="a" * 64, paper_id="P001", page=0, source_text="x")
    with pytest.raises(ValidationError):
        PdfRef(
            artifact_id="raw",
            source_sha256="a" * 64,
            paper_id="P001",
            page=1,
            source_text="x",
            region=[5, 5, 1, 10],
        )


def test_jsonl_is_stable_and_rejects_bad_bytes() -> None:
    data = encode_jsonl([item(QuantityGold(value="-0.0012", comparator="eq"))])
    assert data.endswith(b"\n") and b"\\u" not in data
    assert json.loads(data) == json.loads(encode_jsonl(decode_jsonl(data, BenchmarkItem)))
    with pytest.raises(ValueError):
        decode_jsonl(data + b"\n", BenchmarkItem)


def test_unicode_source_and_variant_identifier_survive_round_trip() -> None:
    variant = EntityGold(
        answer_class="biological_entity", canonical="BRCA1 c.68_69delAG", aliases=["185delAG"]
    )
    record = item(variant)
    assert "−1.2 × 10⁻³" in encode_jsonl([record]).decode()
    assert decode_jsonl(encode_jsonl([record]), BenchmarkItem)[0].gold == variant


@pytest.mark.parametrize("source_status", ["absent", "materially_different", "unresolved"])
def test_non_equivalent_pdf_cannot_be_accepted(source_status: str) -> None:
    record = item(QuantityGold(value="1.2", comparator="eq"))
    data = record.model_dump()
    data.update(
        candidate_status="accepted",
        review_status="reviewed",
        split="development",
        decision_ids=["e1"],
        jats_status="present_equivalent",
        pdf_status=source_status,
    )
    with pytest.raises(ValidationError):
        BenchmarkItem.model_validate(data)
