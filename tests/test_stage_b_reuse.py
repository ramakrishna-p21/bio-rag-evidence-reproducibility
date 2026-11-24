"""stage_b.reuse_checks: command-local reuse of two pure checks, with identical results."""

import gc

import pytest
from test_stage_b import P_A, P_B, P_C, deliver, prose_world

from biorag_repro.evaluation import stage_b
from biorag_repro.retrieval.storage import record_ref


def contexts(world):
    return [
        deliver(world, [world.chunk(P_A)]),
        deliver(world, [world.chunk(P_B)]),
        deliver(world, [world.chunk(P_A), world.chunk(P_C)]),
        deliver(world, [world.chunk(P_A)], scope="full_corpus"),
    ]


def work(world):
    """Everything the queue and ingest do per context, as serialized results."""
    out = []
    for case in contexts(world):
        i = case.inputs()
        candidate = stage_b.evaluate(i)
        review = stage_b.needs_review_v2(candidate, i)
        out.append((candidate.model_dump_json(), review))
        if not review:
            judged = stage_b.deterministic_judgment(candidate, "v2", i)
            stage_b.validate_candidate(candidate, i)
            stage_b.validate_judgment(judged, i)
            out.append(judged.model_dump_json())
    return out


def test_reuse_gives_byte_identical_results():
    w = prose_world(P_A, P_B, P_C)
    plain = work(w)
    with stage_b.reuse_checks():
        assert work(w) == plain


def test_one_judgment_is_read_back_once_across_contexts_and_chunks_are_hashed_once(monkeypatch):
    w = prose_world(P_A, P_B, P_C)
    reads, hashes = [], []
    real_read, real_ref = stage_b.validate_stage_a, stage_b.record_ref
    monkeypatch.setattr(stage_b, "validate_stage_a", lambda *a: (reads.append(1), real_read(*a)))
    monkeypatch.setattr(stage_b, "record_ref", lambda r: (hashes.append(1), real_ref(r))[1])
    work(w)
    plain_reads, plain_hashes = len(reads), len(hashes)
    reads.clear()
    hashes.clear()
    with stage_b.reuse_checks():
        work(w)
    assert len(reads) == 1 < plain_reads
    assert len(hashes) < plain_hashes


def test_nothing_is_remembered_outside_the_scope_or_after_it():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    case.evaluate()
    assert stage_b._reuse.get() is None
    with stage_b.reuse_checks():
        case.evaluate()
        assert stage_b._reuse.get().entries
    assert stage_b._reuse.get() is None


def test_a_failed_read_back_is_not_remembered():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    forged = w.stage_a.model_copy(update={"item_id": "q_999"})
    with stage_b.reuse_checks():
        for _ in range(2):
            with pytest.raises(ValueError):
                case.evaluate(stage_a=forged)
        case.evaluate()  # the genuine judgment still passes


def test_a_judgment_changed_after_it_was_verified_is_checked_again():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    with stage_b.reuse_checks():
        case.evaluate()
        w.stage_a.item_id = "q_999"
        with pytest.raises(ValueError):
            case.evaluate()


def test_a_chunk_changed_after_it_was_hashed_is_checked_again():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    with stage_b.reuse_checks():
        case.evaluate()
        w.chunk(P_A).construction_tokens += 1  # changes only the chunk's recorded identity
        with pytest.raises(ValueError):
            case.evaluate()


def test_a_different_canonical_reference_is_not_answered_from_memory():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    other = w.stage_a.canonical.model_copy(update={"sha256": "e" * 64, "artifact_id": "x"})
    with stage_b.reuse_checks():
        case.evaluate()
        with pytest.raises(ValueError):
            case.evaluate(canonical=other)


def test_context_checks_still_run_for_every_context():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    key = record_ref(w.chunk(P_A)).artifact_id
    swapped = {**case.chunks, key: w.chunk(P_B)}
    with stage_b.reuse_checks():
        case.evaluate()
        with pytest.raises(ValueError):
            case.evaluate(chunks=swapped)
        with pytest.raises(ValueError):
            case.evaluate(context_ref=record_ref(w.chunk(P_A)))


def test_remembered_objects_are_released_with_their_owners():
    with stage_b.reuse_checks():
        w = prose_world(P_A, P_B, P_C)
        deliver(w, [w.chunk(P_A)]).evaluate()
        seen = stage_b._reuse.get()
        assert seen.entries
        del w
        gc.collect()
        assert not [e for e in seen.entries.values() if e[0]() is None]


def test_memory_is_bounded(monkeypatch):
    monkeypatch.setattr(stage_b, "REUSE_LIMIT", 3)
    w = prose_world(P_A, P_B, P_C)
    with stage_b.reuse_checks():
        work(w)
        assert len(stage_b._reuse.get().entries) <= 3
