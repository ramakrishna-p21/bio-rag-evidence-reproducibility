"""Selected multipaper evidence and renewed context counterfeits."""

from dataclasses import replace

import pytest
from test_bm25_publication import inputs, query  # noqa: F401

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import publish_bytes, publish_manifest, serialize_project_json
from biorag_repro.retrieval import bm25, context
from biorag_repro.retrieval.models import ConfigurationRef, ContextRecord, Phase5Configuration
from biorag_repro.retrieval.storage import record_ref
from biorag_repro.retrieval.validation import load_configuration


@pytest.fixture
def prepared(inputs, tmp_path):  # noqa: F811
    sources, cfg = inputs
    index = bm25.build_index(sources)
    ir, im = bm25.publish_index(index, sources, tmp_path)
    q = query()
    ranked = bm25.retrieve(index, sources, q, cfg, tmp_path)
    rr, rm = bm25.publish_retrieval(ranked, ir, im, sources, q, cfg, tmp_path, tmp_path)
    settings = load_configuration(cfg, tmp_path).model_copy(
        update={
            "packing_tokenizer": context.tokenizer_snapshot(),
            "context_rendering": context.POLICY,
        }
    )
    data = serialize_project_json(settings, Phase5Configuration)
    (tmp_path / "context-config.json").write_bytes(data)
    digest = sha256_bytes(canonical_json_bytes(settings.model_dump()))
    cc = ConfigurationRef(
        artifact=ArtifactRef(
            artifact_id="configuration_" + digest,
            logical_ref="context-config.json",
            sha256=sha256_bytes(data),
        ),
        resolved_sha256=digest,
    )
    selected = context.Selection(rr, rm, ir, im, sources, q, cfg, tmp_path, tmp_path, "bm25")
    return selected, cc, tmp_path


def test_scientific_roundtrip_two_roots_and_no_clobber(prepared, tmp_path):
    selected, cfg, root = prepared
    value = context.assemble(selected, cfg, root, None, fixture_counter=len)
    assert [d.rank for d in value.decisions] == [1, 2]
    assert value.stop_reason == "exhausted"
    assert value.retrieval == selected.retrieval
    refs = context.publish(value, selected, cfg, root, None, root / "one", fixture_counter=len)
    again = context.publish(value, selected, cfg, root, None, root / "two", fixture_counter=len)
    assert refs == again
    assert (
        context.read(*refs, selected, cfg, root, None, root / "one", fixture_counter=len) == value
    )
    assert (
        context.read(*again, selected, cfg, root, None, root / "two", fixture_counter=len) == value
    )
    for ref in refs:
        assert (root / "one" / ref.logical_ref).read_bytes() == (
            root / "two" / ref.logical_ref
        ).read_bytes()
    with pytest.raises(FileExistsError):
        context.publish(value, selected, cfg, root, None, root / "one", fixture_counter=len)


def counterfeit(value, root):
    ref, mr = record_ref(value), context.manifest_ref(value)
    target = root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(target, serialize_project_json(value, ContextRecord), ref.sha256)
    publish_manifest(
        context.build_manifest(value), root / mr.logical_ref, {ref.logical_ref: target}
    )
    return ref, mr


@pytest.mark.parametrize(
    "attack",
    [
        "text",
        "count",
        "tokenizer",
        "suppress",
        "reorder",
        "paper",
        "skip",
        "policy",
        "omit",
        "merge",
    ],
)
def test_renewed_counterfeit(prepared, attack):
    selected, cfg, root = prepared
    good = context.assemble(selected, cfg, root, None, fixture_counter=len)
    refs = context.publish(good, selected, cfg, root, None, root / "good", fixture_counter=len)
    assert (
        context.read(*refs, selected, cfg, root, None, root / "good", fixture_counter=len) == good
    )
    first_text = good.exact_context.split("\n\n[evidence]", 1)[0]
    if attack == "text":
        bad = good.model_copy(
            update={
                "exact_context": good.exact_context + " altered",
                "generator_tokens": good.generator_tokens + 8,
            }
        )
    elif attack == "count":
        bad = good.model_copy(update={"generator_tokens": 1})
    elif attack == "tokenizer":
        bad = good.model_copy(
            update={
                "packing_tokenizer": good.packing_tokenizer.model_copy(
                    update={"revision": "a" * 40}
                )
            }
        )
    elif attack == "policy":
        bad = good.model_copy(update={"packing_policy": "other"})
    elif attack == "suppress":
        bad = good.model_copy(
            update={
                "retrieval_build_manifest": ArtifactRef(
                    artifact_id="other", logical_ref="other", sha256="a" * 64
                )
            }
        )
    elif attack == "paper":
        bad = good.model_copy(update={"representation": "PDF_DOCLING"})
    elif attack == "omit":
        bad = good.model_copy(
            update={
                "decisions": good.decisions[:1],
                "included_chunk_ids": good.included_chunk_ids[:1],
                "exact_context": first_text,
                "generator_tokens": len(first_text),
            }
        )
    elif attack == "merge":
        ds = [
            good.decisions[0],
            good.decisions[1].model_copy(update={"decision": "duplicate", "duplicate_of_rank": 1}),
        ]
        bad = good.model_copy(
            update={
                "decisions": ds,
                "included_chunk_ids": good.included_chunk_ids[:1],
                "exact_context": first_text,
                "generator_tokens": len(first_text),
            }
        )
    elif attack == "reorder":
        ds = [d.model_copy(update={"rank": i}) for i, d in enumerate(reversed(good.decisions), 1)]
        bad = good.model_copy(
            update={
                "decisions": ds,
                "included_chunk_ids": [d.chunk_id for d in ds],
                "exact_context": "\n\n".join(reversed(good.exact_context.split("\n\n"))),
            }
        )
    else:
        ds = [
            good.decisions[0].model_copy(update={"decision": "nonfitting"}),
            good.decisions[1].model_copy(update={"decision": "not_processed"}),
        ]
        bad = good.model_copy(
            update={
                "decisions": ds,
                "included_chunk_ids": [],
                "exact_context": "",
                "generator_tokens": 0,
                "stop_reason": "first_nonfitting",
            }
        )
    bad = ContextRecord.model_validate(bad.model_dump())
    refs = counterfeit(bad, root / "bad")
    with pytest.raises(ValueError):
        context.read(*refs, selected, cfg, root, None, root / "bad", fixture_counter=len)


def test_interrupted_publication(prepared, monkeypatch):
    selected, cfg, root = prepared
    good = context.assemble(selected, cfg, root, None, fixture_counter=len)

    def fail(*args, **kwargs):
        raise OSError("interrupted")

    monkeypatch.setattr(context, "publish_manifest", fail)
    with pytest.raises(OSError):
        context.publish(good, selected, cfg, root, None, root / "out", fixture_counter=len)
    with pytest.raises(FileNotFoundError):
        context.read(
            record_ref(good),
            context.manifest_ref(good),
            selected,
            cfg,
            root,
            None,
            root / "out",
            fixture_counter=len,
        )


def test_selected_query_cannot_be_substituted(prepared):
    selected, cfg, root = prepared
    with pytest.raises(ValueError):
        context.assemble(
            replace(selected, query=query(text="other")), cfg, root, None, fixture_counter=len
        )


def test_unauthorized_rank21_and_duplicate_reference_shapes(prepared):
    selected, cfg, root = prepared
    good = context.assemble(selected, cfg, root, None, fixture_counter=len)
    # The very same scientific reader's positive control passes above; shapes
    # that cannot form a persistent artifact reject before hashing/publication.
    extra = good.decisions[-1].model_copy(update={"rank": 21})
    with pytest.raises(ValueError):
        ContextRecord.model_validate(
            good.model_copy(update={"decisions": good.decisions + [extra]}).model_dump()
        )
    first = good.decisions[0]
    second = good.decisions[1].model_copy(update={"decision": "duplicate", "duplicate_of_rank": 2})
    with pytest.raises(ValueError):
        ContextRecord.model_validate(
            good.model_copy(
                update={
                    "decisions": [first, second],
                    "included_chunk_ids": good.included_chunk_ids[:1],
                }
            ).model_dump()
        )


def test_upstream_unknown_chunk_and_changed_source_reject(prepared):
    selected, cfg, root = prepared
    source = selected.sources[0]
    absent = ArtifactRef(artifact_id="unknown", logical_ref="unknown", sha256="a" * 64)
    with pytest.raises((ValueError, FileNotFoundError)):
        context.assemble(
            replace(selected, sources=[replace(source, chunk_set=absent)]),
            cfg,
            root,
            None,
            fixture_counter=len,
        )


def test_multipaper_full_scope_preserves_paper_occurrences(inputs, tmp_path):  # noqa: F811
    from test_bm25_publication import config

    sources, _index_cfg = inputs
    index = bm25.build_index(sources)
    ir, im = bm25.publish_index(index, sources, tmp_path)
    full_cfg = config(tmp_path, "full_corpus")
    q = query(scope="full_corpus", paper=None)
    ranked = bm25.retrieve(index, sources, q, full_cfg, tmp_path)
    rr, rm = bm25.publish_retrieval(ranked, ir, im, sources, q, full_cfg, tmp_path, tmp_path)
    settings = load_configuration(full_cfg, tmp_path).model_copy(
        update={
            "packing_tokenizer": context.tokenizer_snapshot(),
            "context_rendering": context.POLICY,
        }
    )
    data = serialize_project_json(settings, Phase5Configuration)
    (tmp_path / "packed-full.json").write_bytes(data)
    digest = sha256_bytes(canonical_json_bytes(settings.model_dump()))
    cfg = ConfigurationRef(
        artifact=ArtifactRef(
            artifact_id="configuration_" + digest,
            logical_ref="packed-full.json",
            sha256=sha256_bytes(data),
        ),
        resolved_sha256=digest,
    )
    selected = context.Selection(rr, rm, ir, im, sources, q, full_cfg, tmp_path, tmp_path, "bm25")
    value = context.assemble(selected, cfg, tmp_path, None, fixture_counter=len)
    assert {h.paper_id for h in ranked.hits} == {"P001", "P002"}
    assert len(value.included_chunk_ids) == 4
    refs = context.publish(
        value, selected, cfg, tmp_path, None, tmp_path / "out", fixture_counter=len
    )
    assert (
        context.read(*refs, selected, cfg, tmp_path, None, tmp_path / "out", fixture_counter=len)
        == value
    )


def test_duplicate_reference_cannot_point_to_removed_duplicate(prepared):
    selected, cfg, root = prepared
    good = context.assemble(selected, cfg, root, None, fixture_counter=len)
    ds = [
        good.decisions[0],
        good.decisions[1].model_copy(update={"decision": "duplicate", "duplicate_of_rank": 1}),
        good.decisions[1].model_copy(
            update={"rank": 3, "decision": "duplicate", "duplicate_of_rank": 2}
        ),
    ]
    with pytest.raises(ValueError, match="retained inclusion"):
        ContextRecord.model_validate(
            good.model_copy(
                update={"decisions": ds, "included_chunk_ids": good.included_chunk_ids[:1]}
            ).model_dump()
        )


def test_retrieval_reference_substitution_renews_manifest_but_rejects(prepared):
    selected, cfg, root = prepared
    good = context.assemble(selected, cfg, root, None, fixture_counter=len)
    other = ArtifactRef(
        artifact_id="other-ranked-result",
        logical_ref=selected.retrieval.logical_ref,
        sha256=selected.retrieval.sha256,
    )
    bad = good.model_copy(update={"retrieval": other})
    refs = counterfeit(bad, root / "substitute")
    with pytest.raises(ValueError):
        context.read(*refs, selected, cfg, root, None, root / "substitute", fixture_counter=len)


def test_valid_other_paper_context_cannot_be_submitted_for_selected_paper(prepared):
    selected, cfg, root = prepared
    other_query = query(paper="P002")
    index = bm25.read_index(selected.index, selected.index_manifest, selected.sources, root)
    ranked = bm25.retrieve(
        index, selected.sources, other_query, selected.retrieval_configuration, root
    )
    rr, rm = bm25.publish_retrieval(
        ranked,
        selected.index,
        selected.index_manifest,
        selected.sources,
        other_query,
        selected.retrieval_configuration,
        root,
        root,
    )
    other = replace(selected, retrieval=rr, retrieval_manifest=rm, query=other_query)
    value = context.assemble(other, cfg, root, None, fixture_counter=len)
    refs = context.publish(value, other, cfg, root, None, root / "other-paper", fixture_counter=len)
    assert (
        context.read(*refs, other, cfg, root, None, root / "other-paper", fixture_counter=len)
        == value
    )
    with pytest.raises(ValueError, match="independently selected"):
        context.read(*refs, selected, cfg, root, None, root / "other-paper", fixture_counter=len)
