"""Frozen synthetic supplement fixture used by the R tests.

The R tests read the committed bytes under analysis/tests/testthat/fixtures/supplement and never
call Python. This module builds them with the real publication path, reads the supplement back
through the Python verifier, and fails if the committed files drift. Everything is synthetic.

To refresh after an intentional change:

    python tests/test_supplement_fixture.py --write
"""

import json
import sys
import tempfile
from pathlib import Path

from test_analysis_export import observation
from test_analysis_handoff import rehearsal
from test_analysis_handoff_pairing import (
    COMPONENTS,
    FIXTURE_REVIEWER,
    deliver,
    documents,
    pdf_chunk_with_evidence,
)
from test_stage_a import judgment, make_item, occurrence_ids
from test_stage_b import finish
from test_supplement import gen, raw

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.evaluation import analysis_handoff as ah
from biorag_repro.evaluation import supplement as sp
from biorag_repro.evaluation.analysis_table import serialize_csv
from biorag_repro.evaluation.models import ComponentLocus, Placement, SetLocus

FIXTURE_DIR = Path(__file__).resolve().parent.parent / (
    "analysis/tests/testthat/fixtures/supplement"
)

# item -> answers per repeat for JATS and PDF_DOCLING. A string is the answer given (gold is "1"),
# "abstain" is a completed abstention. Padding after the JSON changes the raw bytes only.
CASES = {
    "q_001": (("2", "2 ", "2"), ("1", "1", "1")),  # JATS stable wrong; PDF right
    "q_002": (("1", "2", "1"), ("1.0", "1", "1")),  # JATS right, wrong, right; PDF right
    "q_003": (("2", "3", "2"), ("2", "abstain", "2")),  # varying wrong; PDF abstains once
}


# item -> (JATS reference path, whether the PDF reference names a table). Valid indexed paths,
# unlike the shared fixture item, so that form and location are established by structure.
REFS = {
    "q_001": ("/article/body[1]/sec[1]/p[2]", False),
    "q_002": ("/article/body[1]/sec[2]/table-wrap[1]/table[1]/tbody[1]/tr[1]/td[2]", True),
    "q_003": ("/article/front[1]/article-meta[1]/abstract[1]/p[1]", False),
}


def worlds(item_id):
    """As test_analysis_handoff_pairing.worlds, with structurally valid original references."""
    jats, jats_chunks, pdf, pdf_chunks = documents()
    path, pdf_table = REFS[item_id]
    jats_ref = {
        "artifact_id": jats.source.artifact_id,
        "source_sha256": jats.source.sha256,
        "paper_id": jats.paper_id,
        "xml_path": path,
        "source_text": "source",
    }
    pdf_ref = {
        "artifact_id": pdf.source.artifact_id,
        "source_sha256": pdf.source.sha256,
        "paper_id": jats.paper_id,
        "page": 1,
        "source_text": "source",
    }
    if pdf_table:
        pdf_ref |= {"table_id": "T1", "row_label": "row", "column_label": "column"}
        jats_ref |= {"table_id": "T1", "row_label": "row", "column_label": "column"}
    item = make_item(jats, COMPONENTS, item_id=item_id, jats_refs=[jats_ref], pdf_refs=[pdf_ref])
    table_node = next(n.node_id for n in pdf.nodes if n.kind == "table")
    placement = Placement(
        container_node_id=table_node,
        components=[
            ComponentLocus(role=r, text=t, occurrence_ids=occurrence_ids(pdf, t))
            for r, t in COMPONENTS
        ],
    )
    reviewed = judgment(
        item,
        pdf,
        "preserved",
        sets=[SetLocus(set_index=0, placements=[placement])],
        reviewer=FIXTURE_REVIEWER,
    )
    return finish(jats, jats_chunks, item), finish(pdf, pdf_chunks, item, stage_a=reviewed)


def _result(item, condition, repeat, answer):
    if answer == "abstain":
        return gen(
            item,
            '{"answer": null, "abstained": true}',
            None,
            condition=condition,
            repeat=repeat,
            abstained=True,
        )
    text = answer.strip()
    return gen(
        item, raw(text, " " * (len(answer) - len(text))), text, condition=condition, repeat=repeat
    )


def observations():
    out = []
    for item_id, (jats_answers, pdf_answers) in CASES.items():
        jats_world, pdf_world = worlds(item_id)
        for repeat in (1, 2, 3):
            out.append(
                observation(
                    jats_world,
                    case=deliver(jats_world, [jats_world.chunk("ClusterONE")]),
                    result=_result(jats_world.item, "JATS", repeat, jats_answers[repeat - 1]),
                    repeat_index=repeat,
                )
            )
            out.append(
                observation(
                    pdf_world,
                    case=deliver(pdf_world, [pdf_chunk_with_evidence(pdf_world)]),
                    condition="PDF_DOCLING",
                    result=_result(pdf_world.item, "PDF_DOCLING", repeat, pdf_answers[repeat - 1]),
                    repeat_index=repeat,
                )
            )
    return out


def build_fixture(root: Path) -> dict[str, bytes]:
    root.mkdir(parents=True, exist_ok=True)
    _, _, handoff = rehearsal(root, observations())
    handoff_ref, handoff_build = ah.publish_handoff(handoff, root)
    verified = ah.read_handoff(handoff_build, root)
    supplement = sp.build_supplement(verified, observations())
    digest = sp.publish_supplement(supplement, root, "supplement")
    sp.read_supplement(root, "supplement", digest, verified=verified)
    files = {
        "analysis_table.csv": serialize_csv(verified.rows),
        "handoff.json": (root / handoff_ref.logical_ref).read_bytes(),
        **{f"supplement/{n}": b for n, b in supplement.tables.items()},
        f"supplement/{sp.RECORD_FILE}": sp.serialize_record(supplement.record),
    }
    meta = {
        "note": "synthetic supplement fixture; no scientific authority; not a result",
        "handoff_sha256": sha256_bytes(files["handoff.json"]),
        "csv_sha256": sha256_bytes(files["analysis_table.csv"]),
        "supplement_record_sha256": digest,
        "supplement_sha256": supplement.record.supplement_sha256,
        "python_check": "read_supplement passed on these bytes when they were generated",
    }
    files["fixture.json"] = (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode()
    return files


def test_committed_supplement_fixture_matches_a_fresh_publication():
    with tempfile.TemporaryDirectory() as tmp:
        fresh = build_fixture(Path(tmp))
    committed = {name: (FIXTURE_DIR / name).read_bytes() for name in fresh}
    assert committed == fresh


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: test_supplement_fixture.py --write")
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in build_fixture(Path(tmp)).items():
            target = FIXTURE_DIR / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
