"""Development-only benchmark view: filtering, isolation, publication and read-back.

All fixtures are synthetic. The sentinel string stands in for protected final-test
content and must never reach an output, an exception message or captured output.
"""

import json
from pathlib import Path

import pytest

from biorag_repro.benchmark import development_view as dv
from biorag_repro.benchmark.development_view import (
    DerivationError,
    derive_development_items,
    publish_development_view,
    read_development_view,
)
from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.benchmark.split import PaperMeta, PaperSplit, make_split
from biorag_repro.benchmark.storage import encode_jsonl
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.evaluation.analysis_models import PlannedRoster
from biorag_repro.evaluation.analysis_table import roster_items_from_development_view
from biorag_repro.evaluation.stage_a import item_digest

SENTINEL = "PROTECTED-FINAL-TEST-CONTENT"


def make_split_fixture() -> PaperSplit:
    papers = [PaperMeta(paper_id=f"P{n:03d}", primary_domain=f"d{n % 7}") for n in range(1, 51)]
    return make_split(papers, [], seed=7)


SPLIT = make_split_fixture()
DEV = SPLIT.development_ids
FINAL = SPLIT.final_test_ids


def make_item(item_id: str, paper_id: str, split: str, question: str | None = None, **extra):
    data = {
        "schema_version": "1",
        "benchmark_version": "benchmark_v1",
        "corpus_version": "corpus_v1",
        "item_id": item_id,
        "paper_id": paper_id,
        "question": question or f"Question about {item_id}?",
        "answer_class": "biological_entity",
        "gold": {"answer_class": "biological_entity", "canonical": "TP53", "aliases": ["p53"]},
        "evidence_sets": [
            {
                "components": [
                    {"role": "subject", "text": "gene"},
                    {"role": "relation", "text": "is mutated"},
                    {"role": "answer", "text": "TP53"},
                ]
            }
        ],
        "jats_refs": [
            {
                "artifact_id": "raw_j",
                "source_sha256": "a" * 64,
                "paper_id": paper_id,
                "xml_path": "/article/body/p[1]",
                "source_text": "TP53 is mutated",
            }
        ],
        "pdf_refs": [
            {
                "artifact_id": "raw_p",
                "source_sha256": "b" * 64,
                "paper_id": paper_id,
                "page": 1,
                "source_text": "TP53 is mutated",
            }
        ],
        "jats_status": "present_equivalent",
        "pdf_status": "present_equivalent",
        "split": split,
        "candidate_status": "accepted",
        "review_status": "reviewed",
        "audit_status": "not_selected",
        "decision_ids": ["d1"],
    }
    return BenchmarkItem.model_validate(data | extra)


def mixed_items():
    return [
        make_item("q_001", DEV[0], "development"),
        make_item("q_002", FINAL[0], "final_test", question=SENTINEL),
        make_item("q_003", DEV[1], "development"),
        make_item("q_004", FINAL[1], "final_test", question=SENTINEL),
        make_item("q_005", DEV[0], "development"),
    ]


def derive(items, split=SPLIT, *, total=None, development=3):
    return derive_development_items(
        encode_jsonl(items),
        split,
        benchmark_version="benchmark_v1",
        corpus_version="corpus_v1",
        expected_source_items=len(items) if total is None else total,
        expected_development_items=development,
    )


def test_keeps_only_development_records_byte_for_byte_in_source_order():
    items = mixed_items()
    content, selected = derive(items)
    dev_items = [i for i in items if i.split == "development"]
    assert selected == dev_items
    assert content == encode_jsonl(dev_items)
    assert SENTINEL.encode() not in content


def test_final_test_text_never_appears_in_errors():
    bad_final_split = make_item("q_002", FINAL[0], "development", question=SENTINEL)
    with pytest.raises(DerivationError) as raised:
        derive([make_item("q_001", DEV[0], "development"), bad_final_split], development=1)
    assert SENTINEL not in str(raised.value)
    assert raised.value.__suppress_context__ or raised.value.__cause__ is None


def test_paper_outside_split_is_rejected_without_content():
    unknown = make_item("q_009", "P999", "final_test", question=SENTINEL)
    with pytest.raises(DerivationError, match="paper") as raised:
        derive([make_item("q_001", DEV[0], "development"), unknown], development=1)
    assert SENTINEL not in str(raised.value)


def test_development_paper_labelled_final_test_is_rejected():
    conflicting = make_item("q_002", DEV[0], "final_test")
    with pytest.raises(DerivationError, match="split"):
        derive([conflicting], development=0)


def test_duplicate_item_identity_is_rejected_across_splits():
    items = [
        make_item("q_001", DEV[0], "development"),
        make_item("q_001", FINAL[0], "final_test", question=SENTINEL),
    ]
    with pytest.raises(DerivationError, match="repeats an earlier") as raised:
        derive(items, development=1)
    assert "q_001" not in str(raised.value)
    assert SENTINEL not in str(raised.value)


def test_malformed_record_error_carries_no_record_content():
    good = encode_jsonl([make_item("q_001", DEV[0], "development")])
    broken = json.dumps({"paper_id": FINAL[0], "question": SENTINEL}).encode() + b"\n"
    with pytest.raises(DerivationError, match="record 2") as raised:
        derive_development_items(
            good + broken,
            SPLIT,
            benchmark_version="benchmark_v1",
            corpus_version="corpus_v1",
            expected_source_items=2,
            expected_development_items=1,
        )
    assert SENTINEL not in str(raised.value)
    assert SENTINEL not in repr(raised.value.__cause__)


def test_non_json_line_error_carries_no_content():
    with pytest.raises(DerivationError, match="record 1") as raised:
        derive_development_items(
            f"{SENTINEL} not json\n".encode(),
            SPLIT,
            benchmark_version="benchmark_v1",
            corpus_version="corpus_v1",
            expected_source_items=1,
            expected_development_items=0,
        )
    assert SENTINEL not in str(raised.value)


def test_noncanonical_bytes_are_rejected():
    line = encode_jsonl([make_item("q_001", DEV[0], "development")])
    spaced = line.replace(b'"item_id":"q_001"', b'"item_id": "q_001"')
    with pytest.raises(DerivationError, match="canonical"):
        derive_development_items(
            spaced,
            SPLIT,
            benchmark_version="benchmark_v1",
            corpus_version="corpus_v1",
            expected_source_items=1,
            expected_development_items=1,
        )


def test_unaccepted_or_foreign_version_record_is_rejected():
    lead = make_item(
        "q_001",
        DEV[0],
        "development",
        candidate_status="proposed",
    )
    with pytest.raises(DerivationError, match="accepted"):
        derive([lead], development=1)
    other = make_item("q_001", DEV[0], "development", benchmark_version="benchmark_v2")
    with pytest.raises(DerivationError, match="version"):
        derive([other], development=1)


def test_expected_counts_must_match_and_omissions_are_not_silent():
    items = mixed_items()
    with pytest.raises(DerivationError, match="source item count"):
        derive(items, total=6)
    with pytest.raises(DerivationError, match="development item count"):
        derive(items, development=4)


def test_empty_source_is_rejected():
    with pytest.raises(DerivationError):
        derive_development_items(
            b"",
            SPLIT,
            benchmark_version="benchmark_v1",
            corpus_version="corpus_v1",
            expected_source_items=0,
            expected_development_items=0,
        )


def test_split_authority_must_be_a_10_40_split():
    # PaperSplit itself enforces 10/40; the filter relies on its disjoint id sets.
    assert len(set(DEV)) == 10 and len(set(FINAL)) == 40 and not set(DEV) & set(FINAL)


# ---- publication -----------------------------------------------------------------


def write_source(tmp_path: Path, items) -> tuple[Path, Path, str, str]:
    root = tmp_path / "data"
    (root / "benchmark/releases/benchmark_v1").mkdir(parents=True)
    (root / "benchmark/working/pass_a").mkdir(parents=True)
    items_bytes = encode_jsonl(items)
    items_path = root / "benchmark/releases/benchmark_v1/items.jsonl"
    items_path.write_bytes(items_bytes)
    from biorag_repro.core.storage import serialize_project_json

    split_bytes = serialize_project_json(SPLIT, PaperSplit)
    split_path = root / "benchmark/working/pass_a/paper_split.json"
    split_path.write_bytes(split_bytes)
    return root, items_path, sha256_bytes(items_bytes), sha256_bytes(split_bytes)


def publish(tmp_path, items=None, **overrides):
    items = items or mixed_items()
    root, _, items_sha, split_sha = write_source(tmp_path, items)
    release_sha = "c" * 64
    kwargs = {
        "data_root": root,
        "items_ref": ArtifactRef(
            artifact_id="benchmark_v1_items",
            logical_ref="benchmark/releases/benchmark_v1/items.jsonl",
            sha256=items_sha,
        ),
        "split_ref": ArtifactRef(
            artifact_id="benchmark_v1_split",
            logical_ref="benchmark/working/pass_a/paper_split.json",
            sha256=split_sha,
        ),
        "release_ref": ArtifactRef(
            artifact_id="benchmark_v1_release",
            logical_ref="benchmark/releases/benchmark_v1/release_record.json",
            sha256=release_sha,
        ),
        "benchmark_version": "benchmark_v1",
        "corpus_version": "corpus_v1",
        "expected_source_items": len(items),
        "expected_development_items": 3,
    }
    return root, publish_development_view(**(kwargs | overrides))


def test_publication_is_verified_immutable_and_independent_of_source(tmp_path):
    root, published = publish(tmp_path)
    out = root / "benchmark/derived/benchmark_v1_development"
    assert sorted(p.name for p in out.iterdir()) == [
        "build_manifest.json",
        "development_view.json",
        "items.jsonl",
    ]
    view, items = read_development_view(out, published.record_sha256)
    assert view.access_classification == "development_only"
    assert view.development_item_count == len(items) == 3
    assert view.source_item_count == 5
    assert view.contributing_paper_ids == sorted({DEV[0], DEV[1]})
    assert view.development_paper_ids == DEV
    assert sha256_bytes((out / "items.jsonl").read_bytes()) == view.output_items.sha256
    assert SENTINEL not in b"".join(p.read_bytes() for p in out.iterdir()).decode()
    # read-back works with the mixed source gone
    (root / "benchmark/releases/benchmark_v1/items.jsonl").unlink()
    again, again_items = read_development_view(out, published.record_sha256)
    assert again == view and again_items == items


def test_publication_is_deterministic_and_does_not_overwrite(tmp_path):
    root, first = publish(tmp_path)
    out = root / "benchmark/derived/benchmark_v1_development"
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        publish_development_view(**_kwargs_again(root, tmp_path))
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    _, second = publish(tmp_path / "other")
    assert second.record_sha256 == first.record_sha256


def _kwargs_again(root, tmp_path):
    items = mixed_items()
    return {
        "data_root": root,
        "items_ref": ArtifactRef(
            artifact_id="benchmark_v1_items",
            logical_ref="benchmark/releases/benchmark_v1/items.jsonl",
            sha256=sha256_bytes(encode_jsonl(items)),
        ),
        "split_ref": ArtifactRef(
            artifact_id="benchmark_v1_split",
            logical_ref="benchmark/working/pass_a/paper_split.json",
            sha256=sha256_bytes((root / "benchmark/working/pass_a/paper_split.json").read_bytes()),
        ),
        "release_ref": ArtifactRef(
            artifact_id="benchmark_v1_release",
            logical_ref="benchmark/releases/benchmark_v1/release_record.json",
            sha256="c" * 64,
        ),
        "benchmark_version": "benchmark_v1",
        "corpus_version": "corpus_v1",
        "expected_source_items": len(items),
        "expected_development_items": 3,
    }


def test_source_hash_mismatch_stops_before_any_output(tmp_path):
    root = write_source(tmp_path, mixed_items())[0]
    wrong = ArtifactRef(
        artifact_id="benchmark_v1_items",
        logical_ref="benchmark/releases/benchmark_v1/items.jsonl",
        sha256="0" * 64,
    )
    with pytest.raises(ValueError):
        publish_development_view(**(_kwargs_again(root, tmp_path) | {"items_ref": wrong}))
    assert not (root / "benchmark/derived").exists()


def test_failed_derivation_leaves_no_partial_output(tmp_path):
    with pytest.raises(DerivationError):
        publish(tmp_path, expected_development_items=9)
    assert not list((tmp_path / "data").rglob("development_view.json"))
    assert not list((tmp_path / "data").rglob("items.jsonl.tmp"))


def test_tampered_published_items_are_detected_on_read(tmp_path):
    root, published = publish(tmp_path)
    out = root / "benchmark/derived/benchmark_v1_development"
    target = out / "items.jsonl"
    target.chmod(0o644)
    target.write_bytes(target.read_bytes().replace(b"TP53", b"TP54"))
    with pytest.raises(ValueError):
        read_development_view(out, published.record_sha256)


def test_publication_self_check_rejects_a_protected_record_in_output(tmp_path, monkeypatch):
    items = mixed_items()
    real = dv.derive_development_items

    def leaky(*args, **kwargs):
        content, selected = real(*args, **kwargs)
        extra = [i for i in items if i.split == "final_test"][:1]
        return content + encode_jsonl(extra), selected + extra

    monkeypatch.setattr(dv, "derive_development_items", leaky)
    with pytest.raises(DerivationError):
        publish(tmp_path)
    assert not list((tmp_path / "data").rglob("development_view.json"))


# ---- roster authority ------------------------------------------------------------


def test_roster_items_come_from_view_membership_not_caller(tmp_path):
    root, published = publish(tmp_path)
    out = root / "benchmark/derived/benchmark_v1_development"
    view, items = read_development_view(out, published.record_sha256)
    roster_items = roster_items_from_development_view(view, items)
    assert [r.item_id for r in roster_items] == sorted(i.item_id for i in items)
    assert all(r.state == "included" for r in roster_items)
    by_id = {i.item_id: i for i in items}
    for r in roster_items:
        assert r.paper_id == by_id[r.item_id].paper_id
        assert r.item_sha256 == item_digest(by_id[r.item_id])
    roster = PlannedRoster(
        scope="partial_development",
        phase="development",
        benchmark_version="benchmark_v1",
        corpus_version="corpus_v1",
        declared_by="test",
        basis="synthetic development view",
        arms=[{"matrix_arm": "confirmatory", "conditions": ["JATS"], "repeats": [1]}],
        items=roster_items,
    )
    assert len(roster.coordinates()) == len(items)


def test_roster_helper_rejects_items_not_in_view(tmp_path):
    root, published = publish(tmp_path)
    view, items = read_development_view(
        root / "benchmark/derived/benchmark_v1_development", published.record_sha256
    )
    protected = make_item("q_777", FINAL[0], "final_test", question=SENTINEL)
    with pytest.raises(ValueError) as raised:
        roster_items_from_development_view(view, [*items, protected])
    assert SENTINEL not in str(raised.value)
