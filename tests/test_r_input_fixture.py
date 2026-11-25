"""Frozen synthetic rehearsal table and handoff used by the R tests.

The R tests read the committed bytes under analysis/tests/testthat/fixtures/rehearsal and
never call Python. This module builds those bytes with the real publication path, runs the
Python preflight (`read_handoff`) on them, and fails if the committed files drift.

Everything is synthetic. No row is a result. To refresh the committed files after an
intentional change to the table or handoff contract:

    python tests/test_r_input_fixture.py --write
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from test_analysis_export import failed_for, observation, result_for, roster_of
from test_analysis_handoff_pairing import (
    deliver,
    pdf_chunk_with_evidence,
    pdf_chunk_without_evidence,
    worlds,
)

from biorag_repro.core.hashing import sha256_bytes, sha256_file
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_manifest
from biorag_repro.evaluation import analysis_handoff as ah
from biorag_repro.evaluation.analysis_models import PlannedRoster, RosterItem
from biorag_repro.evaluation.analysis_storage import publish_table
from biorag_repro.evaluation.analysis_table import Table, build_table
from biorag_repro.generation.environment import GitState

FIXTURE_DIR = Path(__file__).resolve().parent.parent / (
    "analysis/tests/testthat/fixtures/rehearsal"
)
CODE = GitState(status="ok", commit="a" * 40, dirty=False, dirty_paths=0)
ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
SEED = "20260601"
needs_r = pytest.mark.skipif(
    shutil.which("Rscript") is None or not (ANALYSIS_DIR / "renv" / "library").is_dir(),
    reason="R and the analysis renv library are needed to produce a real R result",
)

# item -> (paper, JATS cell, PDF_DOCLING cell). A cell is (evidence delivered, answer):
# delivered True/False, or None when no Stage A/B record exists (non_evaluable);
# answer "right", "wrong" or "failed" (operational failure, answer_correct = missing).
CASES = {
    "q_001": ("P001", (True, "right"), (False, "wrong")),
    "q_002": ("P002", (True, "right"), (True, "right")),
    "q_003": ("P002", (True, "wrong"), (True, "right")),
    "q_004": ("P003", (True, "right"), (True, "failed")),
    "q_005": ("P003", (None, "right"), (True, "right")),
    "q_006": ("P003", (False, "wrong"), (True, "right")),
    "q_007": ("P004", (True, "right"), (False, "right")),
    "q_008": ("P005", (True, "failed"), (True, "failed")),
    "q_009": ("P005", (True, "right"), (None, "wrong")),
}
NOT_INCLUDED = RosterItem(item_id="q_010", state="missing", reason="synthetic: no accepted item")


def _cell(world, condition, delivered, answer, pdf_world=None):
    item = world.item
    if answer == "failed":
        result = failed_for(item, representation_or_condition=condition)
    else:
        result = result_for(
            item,
            answer="1" if answer == "right" else "2",
            representation_or_condition=condition,
        )
    plan = {} if condition == "JATS" else {"condition": condition}
    if delivered is None:
        plan["evidence"] = None
    elif condition == "JATS":
        picks = [world.chunk("ClusterONE")] if delivered else []
        plan["case"] = deliver(world, picks)
    else:
        chunk = (pdf_chunk_with_evidence if delivered else pdf_chunk_without_evidence)(world)
        plan["case"] = deliver(world, [chunk])
    return observation(world, result=result, **plan)


def build_table_with_papers():
    observations = []
    papers = {}
    for item_id, (paper, jats, pdf) in CASES.items():
        jats_world, pdf_world = worlds(item_id)
        papers[item_id] = paper
        observations.append(_cell(jats_world, "JATS", *jats))
        observations.append(_cell(pdf_world, "PDF_DOCLING", *pdf))
    # The shared test documents all belong to one paper; give each item its fixture paper in
    # both the rows and the roster, which must agree.
    table = build_table(observations, roster_of(observations))
    rows = [r.model_copy(update={"paper_id": papers[r.item_id]}) for r in table.rows]
    items = [i.model_copy(update={"paper_id": papers[i.item_id]}) for i in table.roster.items] + [
        NOT_INCLUDED
    ]
    roster = PlannedRoster.model_validate(
        table.roster.model_copy(update={"items": items}).model_dump()
    )
    return Table(rows, roster, table.inputs)


def build_fixture(root: Path) -> dict[str, bytes]:
    """Publish the synthetic table and a rehearsal handoff; return the bytes R will read."""
    root.mkdir(parents=True, exist_ok=True)
    csv_ref, _inventory, build = publish_table(build_table_with_papers(), root)
    handoff = ah.build_handoff(build, root, code=CODE, rehearsal=True)
    handoff_ref, handoff_build = ah.publish_handoff(handoff, root)
    ah.read_handoff(handoff_build, root)  # Python preflight on the exact published files
    csv_bytes = (root / csv_ref.logical_ref).read_bytes()
    handoff_bytes = (root / handoff_ref.logical_ref).read_bytes()
    meta = {
        "note": "synthetic rehearsal fixture; no scientific authority; not a result",
        "input_origin": handoff.input_origin,
        "handoff_sha256": sha256_bytes(handoff_bytes),
        "csv_sha256": sha256_bytes(csv_bytes),
        "python_preflight": "read_handoff passed on these bytes when they were generated",
    }
    return {
        "analysis_table.csv": csv_bytes,
        "handoff.json": handoff_bytes,
        "fixture.json": (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode(),
    }


def publish_frozen_handoff(root: Path) -> ArtifactRef:
    _csv, _inventory, build = publish_table(build_table_with_papers(), root)
    _ref, handoff_build = ah.publish_handoff(
        ah.build_handoff(build, root, code=CODE, rehearsal=True), root
    )
    return handoff_build


def run_r(root: Path, handoff_build: ArtifactRef, command: str, out: str, *args: str) -> str:
    """Run `run_analysis.R <command>` on a verified handoff; return its result dir under root."""
    checked = ah.read_handoff(handoff_build, root)
    handoff = read_manifest(root / handoff_build.logical_ref, handoff_build.sha256).outputs[0]
    (root / out).mkdir(parents=True, exist_ok=True)
    done = subprocess.run(
        [
            "Rscript",
            "run_analysis.R",
            command,
            "--handoff",
            str(root / handoff.artifact.logical_ref),
            "--handoff-sha256",
            handoff.artifact.sha256,
            "--csv",
            str(root / checked.handoff.csv.logical_ref),
            "--out-root",
            str(root / out),
            *args,
        ],
        cwd=ANALYSIS_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    printed = re.search(r"^result directory: (.+)$", done.stdout, re.MULTILINE)
    assert printed
    return os.path.relpath(printed.group(1), root)


def run_primary(root: Path, handoff_build: ArtifactRef) -> str:
    lock = sha256_file(ANALYSIS_DIR / "renv.lock")
    return run_r(
        root, handoff_build, "run", "analysis/r_results",
        "--lock-sha256", lock, "--seed", SEED, "--seed-label", "synthetic_rehearsal",
    )  # fmt: skip


def derive(root: Path, handoff_build: ArtifactRef, primary: str) -> str:
    return run_r(
        root, handoff_build, "derive", "analysis/r_secondary", "--primary", str(root / primary)
    )


def test_committed_r_fixture_matches_a_fresh_publication():
    with tempfile.TemporaryDirectory() as tmp:
        fresh = build_fixture(Path(tmp))
    committed = {name: (FIXTURE_DIR / name).read_bytes() for name in fresh}
    assert committed == fresh


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: test_r_input_fixture.py --write")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in build_fixture(Path(tmp)).items():
            (FIXTURE_DIR / name).write_bytes(data)
