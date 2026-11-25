"""Native-backed classifications and fully rehashed uncertainty suppression."""

import json
from types import SimpleNamespace as Item

import pytest
from test_documents_table_fidelity import _native_data, _typed_native
from test_publication_authorization import CASES, attack, candidate
from test_publication_authorization import authority as authority  # noqa: PLC0414
from test_publication_authorization import pair as pair  # noqa: PLC0414

from biorag_repro.core.identity import ArtifactInput
from biorag_repro.core.manifest import ArtifactRef, BuildInput
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.documents.authorization import (
    candidate_ref,
    validate_candidate,
    validate_scientific_pair,
)
from biorag_repro.documents.coverage import validate_pdf_coverage
from biorag_repro.documents.ingestion import publish_validation, validate_build
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.documents.pdf import TRANSFORMATION_ID, canonicalize_native
from biorag_repro.documents.pdf_ingest import (
    PdfCanonicalBuild,
    _manifest,
    _ref,
    canonical_build_ref,
    publish_pdf_result,
)


def classification_fixture(label, layer="body", role="texts"):
    ref = f"#/{role}/0" if role not in {"body", "furniture"} else f"#/{role}"
    item = {
        "self_ref": ref,
        "label": label,
        "content_layer": layer,
        "children": [],
    }
    data = {role: item if role in {"body", "furniture"} else [item]}
    obj = Item(**item)
    native = Item(**{role: obj if role in {"body", "furniture"} else [obj]})
    native.iterate_items = lambda with_groups: iter([(obj, 0)])
    source = ArtifactRef(artifact_id="raw", logical_ref="source.pdf", sha256="a" * 64)
    saved = ArtifactRef(artifact_id="native", logical_ref="native.json", sha256="b" * 64)
    doc = canonicalize_native(native, "P001", source, saved, "c" * 64)
    return doc, data


CLASSIFICATIONS = [
    ("section_header", "body", "texts", "title", "exact_native"),
    ("text", "body", "texts", "paragraph", "exact_native"),
    ("list_item", "body", "texts", "list_item", "exact_native"),
    ("list", "body", "groups", "list", "exact_native"),
    ("picture", "body", "pictures", "figure", "exact_native"),
    ("formula", "body", "texts", "formula", "exact_native"),
    ("caption", "body", "texts", "caption", "exact_native"),
    ("page_header", "furniture", "texts", "page_furniture", "exact_native"),
    ("text", "furniture", "texts", "page_furniture", "exact_native"),
    ("unspecified", "furniture", "furniture", "source_construct", "ambiguous"),
    ("unspecified", "body", "body", "source_construct", "ambiguous"),
    ("key_value_area", "body", "groups", "source_construct", "ambiguous"),
    ("form_area", "body", "groups", "source_construct", "ambiguous"),
    ("code", "body", "texts", "source_construct", "ambiguous"),
    ("unrecognized", "body", "groups", "source_construct", "ambiguous"),
]


@pytest.mark.parametrize("label,layer,role,kind,state", CLASSIFICATIONS)
def test_native_classifications_match_literal_expectations(label, layer, role, kind, state):
    doc, data = classification_fixture(label, layer, role)
    assert (doc.nodes[0].kind, doc.nodes[0].interpretation_state) == (kind, state)
    assert validate_pdf_coverage(doc, data) == (1, int(role == "texts"), 0)


@pytest.mark.parametrize(
    "classification,state",
    [
        (case, state)
        for case in CLASSIFICATIONS
        for state in [
            None,
            "exact_native",
            "derived",
            "approximate",
            "unavailable",
            "parser_damaged",
            "failed",
            "ambiguous",
        ]
        if state != case[4]
    ],
)
def test_native_classification_rejects_other_states(classification, state):
    label, layer, role, _, _ = classification
    doc, data = classification_fixture(label, layer, role)
    doc.nodes[0].interpretation_state = state
    with pytest.raises(ValueError, match="native classification"):
        validate_pdf_coverage(doc, data)


@pytest.mark.parametrize(
    "kind,state",
    [
        ("source_construct", "exact_native"),
        ("paragraph", "exact_native"),
        ("source_construct", None),
    ],
)
def test_fallback_cannot_claim_certainty_or_another_kind(kind, state):
    doc, data = classification_fixture("key_value_area", role="groups")
    doc.nodes[0].kind = kind
    doc.nodes[0].interpretation_state = state
    with pytest.raises(ValueError, match="native classification.*#/groups/0"):
        validate_pdf_coverage(doc, data)


@pytest.mark.parametrize("kind", ["source_construct", "table_cell", "title"])
def test_native_kind_cannot_be_substituted(kind):
    doc, data = classification_fixture("text")
    doc.nodes[0].kind = kind
    with pytest.raises(ValueError, match="native classification.*#/texts/0"):
        validate_pdf_coverage(doc, data)


def test_missing_and_invented_interpretation_information():
    doc, data = classification_fixture("text")
    value = doc.model_dump()
    del value["nodes"][0]["interpretation_state"]
    with pytest.raises(ValueError, match="native classification"):
        validate_pdf_coverage(CanonicalDocument.model_validate(value), data)
    value["nodes"][0]["interpretation_state"] = "certain"
    with pytest.raises(ValueError, match="interpretation_state"):
        CanonicalDocument.model_validate(value)


@pytest.mark.parametrize("grid_only", [False, True])
def test_table_and_cell_classifications_are_native_exact(grid_only):
    data = _native_data()
    native = _typed_native(data)
    if grid_only:
        data["tables"][0]["data"]["table_cells"].pop()
        native.tables[0].data.table_cells.pop()
    source = ArtifactRef(artifact_id="raw", logical_ref="source.pdf", sha256="a" * 64)
    saved = ArtifactRef(artifact_id="native", logical_ref="native.json", sha256="b" * 64)
    doc = canonicalize_native(native, "P001", source, saved, "c" * 64, saved_native=data)
    assert [(n.kind, n.interpretation_state) for n in doc.nodes] == [
        ("source_construct", "ambiguous"),
        ("table", "exact_native"),
        *[("table_cell", "exact_native")] * 4,
    ]
    assert validate_pdf_coverage(doc, data) == (2, 0, 4)
    cell = doc.nodes[-1]
    assert ("/grid/" in cell.mapping.native_ref) == grid_only
    cell.interpretation_state = "ambiguous"
    with pytest.raises(ValueError, match="native classification.*#/tables/0/data/"):
        validate_pdf_coverage(doc, data)


def rehash_classification(pair, authority, *, change_kind=False):
    """Replace only derived evidence; a submitted PASS is not a validation oracle."""
    doc = CanonicalDocument.model_validate_json(
        (pair.output / pair.pdf.canonical_artifact.logical_ref).read_bytes()
    )
    for node in doc.nodes:
        if node.kind == "source_construct":
            node.interpretation_state = "exact_native"
            if change_kind:
                node.kind = "paragraph"
    content = serialize_project_json(doc, CanonicalDocument)
    settings = json.loads((pair.output / pair.pdf.build_manifest.logical_ref).read_bytes())[
        "transformation_settings_sha256"
    ]
    inputs = [
        BuildInput(role="raw_pdf", artifact=pair.pdf.source),
        BuildInput(role="docling_native", artifact=pair.pdf.native_artifact),
    ]
    ref = _ref(
        "canonical_pdf_docling",
        [ArtifactInput(i.role, i.artifact.artifact_id, i.artifact.sha256) for i in inputs],
        TRANSFORMATION_ID,
        settings,
        content,
        "4",
        "canonical/PDF_DOCLING",
    )
    (pair.output / ref.logical_ref).write_bytes(content)
    manifest = _manifest(
        inputs, TRANSFORMATION_ID, settings, ref, "canonical_pdf_docling", "4", pair.output
    )
    build = PdfCanonicalBuild(
        paper_id=pair.pdf.paper_id,
        source=pair.pdf.source,
        configuration_sha256=pair.pdf.configuration_sha256,
        environment_sha256=pair.pdf.environment_sha256,
        asset_inventory_sha256=pair.pdf.asset_inventory_sha256,
        original_attempt_artifact=authority.native_attempt,
        native_artifact=pair.pdf.native_artifact,
        native_build_manifest=pair.pdf.native_build_manifest,
        canonical_artifact=ref,
        build_manifest=manifest,
    )
    build_ref = canonical_build_ref(build)
    (pair.output / build_ref.logical_ref).write_bytes(
        serialize_project_json(build, PdfCanonicalBuild)
    )
    report = pair.pr.model_copy(
        update={
            "canonical_artifact": ref,
            "build_manifest": manifest,
            "canonical_build_artifact": build_ref,
            "warnings": [
                w
                for w in pair.pr.warnings
                if w
                not in (
                    {"native_classification_uncertain", "source_construct_preserved"}
                    if change_kind
                    else {"native_classification_uncertain"}
                )
            ],
        }
    )
    pair.index.pdf_validation = publish_validation(report, pair.output)
    submitted = candidate(pair)
    assert candidate_ref(submitted).artifact_id.startswith("drv_")
    return build, submitted


@pytest.mark.parametrize("boundary", ["build", "pair", "candidate"])
@pytest.mark.parametrize("change_kind", [False, True])
def test_rehashed_uncertainty_is_rejected_before_approval(pair, authority, boundary, change_kind):
    original = candidate(pair)
    validate_candidate(original, pair.output, pair.corpus, authority, "f" * 40)
    build, submitted = rehash_classification(pair, authority, change_kind=change_kind)
    with pytest.raises(ValueError, match="native classification.*#/body"):
        if boundary == "build":
            validate_build(build, pair.output, corpus_root=pair.corpus)
        elif boundary == "pair":
            validate_scientific_pair(pair.index, pair.output, pair.corpus, authority, "f" * 40)
        else:
            validate_candidate(submitted, pair.output, pair.corpus, authority, "f" * 40)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_classification_warning_matches_native_structure(pair, tmp_path, ambiguous):
    data = _native_data()
    native = _typed_native(data)
    if not ambiguous:
        data["body"]["label"] = "list"
        native.body.label = "list"

    class Native:
        def save_as_json(self, path, **kwargs):
            path.write_text(json.dumps(data))

    result = Item(status=Item(value="success"), errors=[], pages=[], document=Native())
    output = tmp_path / "warning"
    built = publish_pdf_result(
        result,
        pair.corpus / "corpus/raw" / pair.pdf.source.logical_ref,
        pair.pdf.source,
        "P001",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        output,
        load_native=lambda path: native,
    )
    report = validate_build(built, output, corpus_root=pair.corpus)
    assert ("native_classification_uncertain" in report.warnings) == ambiguous


@pytest.mark.parametrize("node_index", [1, 2])
@pytest.mark.parametrize("field,value", [("kind", "paragraph"), ("interpretation_state", None)])
def test_table_and_cell_classification_substitutions(node_index, field, value):
    data = _native_data()
    source = ArtifactRef(artifact_id="raw", logical_ref="source.pdf", sha256="a" * 64)
    saved = ArtifactRef(artifact_id="native", logical_ref="native.json", sha256="b" * 64)
    doc = canonicalize_native(_typed_native(data), "P001", source, saved, "c" * 64)
    setattr(doc.nodes[node_index], field, value)
    with pytest.raises(ValueError, match="native classification.*#/tables/0"):
        validate_pdf_coverage(doc, data)


def test_source_tag_cannot_authorize_another_classification():
    doc, data = classification_fixture("key_value_area", role="groups")
    doc.nodes[0].source_tag = "TextItem"
    doc.nodes[0].kind = "paragraph"
    doc.nodes[0].interpretation_state = "exact_native"
    with pytest.raises(ValueError, match="native classification.*#/groups/0"):
        validate_pdf_coverage(doc, data)


@pytest.mark.parametrize("field", ["source_artifact_id", "native_artifact"])
def test_classification_requires_original_source_identity(field):
    doc, data = classification_fixture("text")
    setattr(
        doc.nodes[0].mapping, field, "another_source" if field == "source_artifact_id" else None
    )
    with pytest.raises(ValueError, match="native classification source identity.*#/texts/0"):
        validate_pdf_coverage(doc, data)


@pytest.mark.parametrize("case", CASES)
def test_historical_attacks_rejected_by_ordinary_candidate(pair, authority, case):
    original = candidate(pair)
    validate_candidate(original, pair.output, pair.corpus, authority, "f" * 40)
    index = attack(pair, case)
    submitted = original.model_copy(
        update={
            "reproduction_git_sha": index.implementation_git_sha,
            "corpus_release_sha256": index.corpus_release_sha256,
            "corpus_version": index.corpus_version,
            "jats_validation": index.jats_validation,
            "pdf_validation": index.pdf_validation,
        }
    )
    with pytest.raises(ValueError) as error:
        validate_candidate(submitted, pair.output, pair.corpus, authority, "f" * 40)
    assert "human authorization" not in str(error.value)
