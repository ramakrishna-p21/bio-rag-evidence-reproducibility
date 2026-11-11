"""Five-arm synthetic table through the real handoff, R analysis and secondary derivation.

Twelve synthetic papers with two items each carry the full planned matrix (confirmatory x3
repeats, closed book x3, and one repeat of each of the dense, chunk and scope cells), 15 rows
per item. The expected values are computed here from the cell specification alone, not from the
table or from R. Everything is synthetic; no value is a scientific result.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from test_analysis_export import _roster, failed_for, observation, result_for
from test_analysis_handoff_pairing import (
    pdf_chunk_with_evidence,
    pdf_chunk_without_evidence,
    worlds,
)
from test_analysis_rehearsal import rehearse
from test_analysis_result import needs_r
from test_matrix_rehearsal import ARM_CONFIG
from test_r_input_fixture import CODE
from test_release_index import derive
from test_stage_b import Case, deliver

from biorag_repro.evaluation import analysis_handoff as ah
from biorag_repro.evaluation import analysis_result as ar
from biorag_repro.evaluation import release_index as ri
from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX, PlannedRoster
from biorag_repro.evaluation.analysis_storage import publish_table
from biorag_repro.evaluation.analysis_table import ARM_RETRIEVAL, Table, build_table
from biorag_repro.retrieval.models import DenseQuery
from biorag_repro.retrieval.storage import record_ref

N_ITEMS = 24
ITEMS = [f"q_{n:03d}" for n in range(1, N_ITEMS + 1)]
PAPER = {item: f"P{(n // 2) + 1:02d}" for n, item in enumerate(ITEMS)}
ARM_OFFSET = {
    "confirmatory": 0,
    "dense_robustness": 1,
    "chunk_robustness": 2,
    "scope_robustness": 3,
    "closed_book": 4,
}
ROBUSTNESS = ("dense_robustness", "chunk_robustness", "scope_robustness")


def delivered(n: int, arm: str, condition: str) -> bool | None:
    """Evidence delivered (True/False) or None when no Stage A/B record exists.

    The context is fixed across an item's repeats, so there is no repeat argument.
    """
    if arm == "confirmatory" and condition == "JATS" and n in (5, 17):
        return None
    if arm == "confirmatory" and condition == "PDF_DOCLING" and n == 12:
        return None
    k = ARM_OFFSET[arm]
    if condition == "JATS":
        return (n * 7 + k) % 5 != 0
    return (n * 3 + k * 2) % 4 != 0


def answer(n: int, arm: str, condition: str, repeat: int) -> str:
    """'right', 'wrong' or 'failed' (operational failure, no scorable answer)."""
    if arm == "confirmatory" and condition == "PDF_DOCLING" and n == 9 and repeat == 1:
        return "failed"
    if arm == "confirmatory" and condition == "JATS" and n == 20 and repeat == 1:
        return "failed"
    k = ARM_OFFSET[arm]
    if condition == "PDF_DOCLING":
        return "wrong" if (n * 2 + k + repeat) % 4 == 0 else "right"
    return "wrong" if (n * 5 + k * 2 + repeat * 3) % 3 == 0 else "right"


def _closed_book(world, answer_kind: str, repeat: int):
    config = ARM_CONFIG["closed_book"]
    coords = {
        "matrix_arm": "closed_book",
        "representation_or_condition": "CLOSED_BOOK",
        "repeat_index": repeat,
        "resolved_config_sha256": config,
        "attempt_id": f"a{repeat}",
    }
    if answer_kind == "failed":
        result = failed_for(world.item, **coords)
    else:
        result = result_for(world.item, answer="1" if answer_kind == "right" else "2", **coords)
    return observation(
        world,
        result=result,
        evidence=None,
        matrix_arm="closed_book",
        condition="CLOSED_BOOK",
        repeat_index=repeat,
        resolved_config_sha256=config,
    )


def _case(world, condition, arm, is_delivered):
    """A real Stage B case, with the retriever and scope the arm uses (see ARM_RETRIEVAL)."""
    if condition == "JATS":
        picks = [world.chunk("ClusterONE")] if is_delivered else []
    else:
        picks = [(pdf_chunk_with_evidence if is_delivered else pdf_chunk_without_evidence)(world)]
    scope, retriever = ARM_RETRIEVAL[arm]
    case = deliver(world, picks, scope=scope)
    ranked = case.ranked.model_copy(
        update={
            "retriever": retriever,
            "scope": scope,
            "designated_paper_id": world.item.paper_id if scope == "designated_paper" else None,
            # synthetic query evidence; the dense arm's real BGE ranking is not exercised here
            "dense_query": (
                DenseQuery(token_count=8, model_input_sha256="d" * 64, vector=[1.0])
                if retriever == "bge"
                else None
            ),
        }
    )
    ctx = case.ctx.model_copy(update={"retrieval": record_ref(ranked)})
    return Case(world, ranked, ctx, case.chunks)


def _retrieval_cell(world, arm, condition, repeat, n):
    kind = answer(n, arm, condition, repeat)
    config = ARM_CONFIG[arm]
    coords = {
        "matrix_arm": arm,
        "representation_or_condition": condition,
        "repeat_index": repeat,
        "resolved_config_sha256": config,
        "attempt_id": f"a{repeat}",
    }
    if kind == "failed":
        result = failed_for(world.item, **coords)
    else:
        result = result_for(world.item, answer="1" if kind == "right" else "2", **coords)
    is_delivered = delivered(n, arm, condition)
    plan = (
        {"evidence": None}
        if is_delivered is None
        else {"case": _case(world, condition, arm, is_delivered)}
    )
    return observation(
        world,
        result=result,
        matrix_arm=arm,
        condition=condition,
        repeat_index=repeat,
        resolved_config_sha256=config,
        **plan,
    )


def five_arm_table() -> Table:
    observations = []
    for n, item_id in enumerate(ITEMS, 1):
        jats_world, pdf_world = worlds(item_id)
        for arm, (conditions, repeats) in FROZEN_MATRIX.items():
            for condition in conditions:
                for repeat in repeats:
                    if arm == "closed_book":
                        observations.append(
                            _closed_book(jats_world, answer(n, arm, "JATS", repeat), repeat)
                        )
                    else:
                        w = jats_world if condition == "JATS" else pdf_world
                        observations.append(_retrieval_cell(w, arm, condition, repeat, n))
    items = list({o.item.item_id: o.item for o in observations}.values())
    roster = _roster(
        {a: (list(c), list(r)) for a, (c, r) in FROZEN_MATRIX.items()},
        items,
        scope="complete_matrix",
    )
    table = build_table(observations, roster)
    rows = [r.model_copy(update={"paper_id": PAPER[r.item_id]}) for r in table.rows]
    roster_items = [i.model_copy(update={"paper_id": PAPER[i.item_id]}) for i in table.roster.items]
    roster = PlannedRoster.model_validate(
        table.roster.model_copy(update={"items": roster_items}).model_dump()
    )
    return Table(rows, roster, table.inputs)


# ---- expected values from the cell specification only ---------------------------------------


def outcome_value(outcome: str, n: int, arm: str, condition: str):
    """1, 0 or None (not evaluable / missing) for one cell at repeat 1."""
    if outcome == "evidence_delivered":
        d = delivered(n, arm, condition)
        return None if d is None else int(d)
    kind = answer(n, arm, condition, 1)
    return None if kind == "failed" else int(kind == "right")


def expected_pairs(outcome: str, arm: str):
    pairs = []
    for n, item in enumerate(ITEMS, 1):
        j = outcome_value(outcome, n, arm, "JATS")
        p = outcome_value(outcome, n, arm, "PDF_DOCLING")
        if j is not None and p is not None:
            pairs.append((item, j, p))
    return pairs


def read_csv(directory: Path, name: str) -> list[dict[str, str]]:
    with (directory / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def built_table():
    """The table is built once per module; both users only read it."""
    return five_arm_table()


@pytest.fixture(scope="module")
def five_arm_run(tmp_path_factory, built_table):
    root = tmp_path_factory.mktemp("five_arm_store")
    table = built_table
    _csv, _inventory, table_build = publish_table(table, root)
    _ref, handoff_build = ah.publish_handoff(
        ah.build_handoff(table_build, root, code=CODE, rehearsal=True), root
    )
    done = rehearse(root, handoff_build)
    secondary, inputs = derive(root, done.preflight_build, done.result_dir, "analysis/r_secondary")
    return table, root, done, secondary, inputs


def test_the_table_has_the_planned_five_arm_structure(built_table):
    table = built_table
    assert len(table.rows) == 15 * N_ITEMS  # 360
    assert Counter(r.matrix_arm for r in table.rows) == {
        "confirmatory": 6 * N_ITEMS,
        "dense_robustness": 2 * N_ITEMS,
        "chunk_robustness": 2 * N_ITEMS,
        "scope_robustness": 2 * N_ITEMS,
        "closed_book": 3 * N_ITEMS,
    }
    assert len({r.observation_key for r in table.rows}) == len(table.rows)
    assert len({r.paper_id for r in table.rows}) == 12
    assert {r.matrix_arm: r.config_sha256 for r in table.rows} == ARM_CONFIG
    by_pair = Counter((r.item_id, r.matrix_arm, r.condition) for r in table.rows)
    assert all(by_pair[(i, "confirmatory", c)] == 3 for i in ITEMS for c in ("JATS", "PDF_DOCLING"))
    assert all(by_pair[(i, "closed_book", "CLOSED_BOOK")] == 3 for i in ITEMS)


@needs_r
def test_the_primary_result_equals_an_independent_calculation(five_arm_run):
    _table, root, done, _secondary, _inputs = five_arm_run
    result = root / done.result_dir
    by_outcome = {r["outcome"]: r for r in read_csv(result, "coprimary_results.csv")}
    assert set(by_outcome) == {"evidence_delivered", "answer_correct"}
    for outcome, row in by_outcome.items():
        pairs = expected_pairs(outcome, "confirmatory")
        n_complete = len(pairs)
        assert int(row["n_eligible"]) == N_ITEMS
        assert int(row["n_complete"]) == n_complete
        assert int(row["n_papers"]) == len({PAPER[i] for i, _j, _p in pairs})
        assert float(row["rd"]) == pytest.approx(sum(j - p for _i, j, p in pairs) / n_complete)
        assert float(row["rate_jats"]) == pytest.approx(sum(j for _i, j, _p in pairs) / n_complete)
        assert float(row["rate_pdf"]) == pytest.approx(sum(p for _i, _j, p in pairs) / n_complete)
        assert row["boot_status"] == "ok"
        assert int(row["boot_n_resamples"]) == 10_000
        assert int(row["boot_n_empty"]) == 0
        assert float(row["boot_ci_lower"]) <= float(row["rd"]) <= float(row["boot_ci_upper"])
        assert row["test_status"] == "ok" and row["holm_status"] == "ok"
        assert float(row["test_df"]) > 0
    # Holm over exactly the two co-primary p-values.
    p = sorted(float(r["test_p"]) for r in by_outcome.values())
    low = min(1.0, 2 * p[0])
    high = min(1.0, max(low, p[1]))
    expected = {p[0]: low, p[1]: high}
    for row in by_outcome.values():
        assert float(row["holm_p"]) == pytest.approx(expected[float(row["test_p"])])


@needs_r
def test_paper_and_state_counts_follow_the_primary_pairs_only(five_arm_run):
    _table, root, done, _secondary, _inputs = five_arm_run
    result = root / done.result_dir
    papers = read_csv(result, "paper_counts.csv")
    for outcome in ("evidence_delivered", "answer_correct"):
        complete = Counter(PAPER[i] for i, _j, _p in expected_pairs(outcome, "confirmatory"))
        got = {r["paper_id"]: r for r in papers if r["outcome"] == outcome}
        assert set(got) == set(PAPER.values())
        assert {p: int(r["eligible"]) for p, r in got.items()} == dict.fromkeys(got, 2)
        assert {p: int(r["complete"]) for p, r in got.items()} == {
            p: complete.get(p, 0) for p in got
        }
    states = read_csv(result, "state_counts.csv")
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for r in states:
        totals[(r["outcome"], r["condition"])] += int(r["n"])
    # one repeat-1 confirmatory row per item and representation; no other arm or repeat enters
    assert set(totals) == {
        (o, c) for o in ("evidence_delivered", "answer_correct") for c in ("JATS", "PDF_DOCLING")
    }
    assert set(totals.values()) == {N_ITEMS}


@needs_r
def test_robustness_arms_are_reported_per_arm_and_never_enter_the_primary(five_arm_run):
    _table, root, done, secondary, _inputs = five_arm_run
    rows = read_csv(root / secondary, "secondary_robustness.csv")
    assert {r["matrix_arm"] for r in rows} == {"confirmatory", *ROBUSTNESS}
    for r in rows:
        pairs = expected_pairs(r["outcome"], r["matrix_arm"])
        assert r["status"] == "ok"
        assert int(r["n_items"]) == N_ITEMS and int(r["n_paired"]) == N_ITEMS
        assert int(r["n_complete"]) == len(pairs)
        assert float(r["rd"]) == pytest.approx(sum(j - p for _i, j, p in pairs) / len(pairs))
    primary = {
        r["outcome"]: float(r["rd"])
        for r in read_csv(root / done.result_dir, "coprimary_results.csv")
    }
    reference = {r["outcome"]: float(r["rd"]) for r in rows if r["matrix_arm"] == "confirmatory"}
    assert reference == pytest.approx(primary)


@needs_r
def test_repeat_and_closed_book_tables_use_the_full_repeat_structure(five_arm_run):
    _table, root, _done, secondary, _inputs = five_arm_run
    outcomes = read_csv(root / secondary, "secondary_repeat_outcomes.csv")
    seen = {(r["matrix_arm"], r["condition"], int(r["repeat_index"])) for r in outcomes}
    assert {(a, c, k) for a, c, k in seen if a == "confirmatory"} == {
        ("confirmatory", c, k) for c in ("JATS", "PDF_DOCLING") for k in (1, 2, 3)
    }
    assert {(a, c, k) for a, c, k in seen if a == "closed_book"} == {
        ("closed_book", "CLOSED_BOOK", k) for k in (1, 2, 3)
    }
    variability = read_csv(root / secondary, "secondary_repeat_variability.csv")
    for r in variability:
        if r["matrix_arm"] == "closed_book":
            assert int(r["n_items"]) == N_ITEMS
            assert int(r["n_all_repeats_evaluable"]) == N_ITEMS  # no closed-book failures planned


@needs_r
def test_the_release_index_cites_the_five_arm_primary_and_secondary(five_arm_run):
    _table, root, done, secondary, inputs = five_arm_run
    index = ri.build_release_index(
        root, primary_dir=done.result_dir, secondary_dir=secondary, inputs=inputs
    )
    assert index.final is False and index.scientific_authority == "none"
    assert index.primary_scientific_result_sha256 == done.scientific_result_sha256
    assert ar.read_result_record(done.record.build, root).record.scientific_result_sha256 == (
        done.scientific_result_sha256
    )
