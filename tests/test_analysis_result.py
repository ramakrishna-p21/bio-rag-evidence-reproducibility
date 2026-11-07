"""Python verification of a published R result directory and the typed result record.

The R side is the real analysis runner on the frozen synthetic rehearsal table, so these tests
also catch drift between R's output schemas and the Python parser. Everything is synthetic; no
value is a scientific result. Tests that need R are skipped when R or its renv library is absent.
"""

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from test_r_input_fixture import CODE, build_table_with_papers

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_manifest
from biorag_repro.evaluation import analysis_handoff as ah
from biorag_repro.evaluation import analysis_result as ar
from biorag_repro.evaluation.analysis_preflight import publish_preflight
from biorag_repro.evaluation.analysis_storage import publish_table

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
GOLDEN = "17c44709e57fd9ac259f3d239498693d31e34258f8577d9e73d0e6cb86b01e36"
SEED = "20260601"
R_RESULTS = "analysis/r_results"

needs_r = pytest.mark.skipif(
    shutil.which("Rscript") is None or not (ANALYSIS_DIR / "renv" / "library").is_dir(),
    reason="R and the analysis renv library are needed to produce a real R result",
)


@dataclass(frozen=True)
class World:
    root: Path
    handoff: ArtifactRef
    csv: ArtifactRef
    preflight: ArtifactRef
    preflight_build: ArtifactRef
    result: str  # result directory, relative to root
    record_sha256: str


def make_store(root: Path):
    """The frozen rehearsal table, its handoff and a preflight, published with the real paths."""
    csv_ref, _inventory, build = publish_table(build_table_with_papers(), root)
    handoff = ah.build_handoff(build, root, code=CODE, rehearsal=True)
    handoff_ref, handoff_build = ah.publish_handoff(handoff, root)
    pre_ref, pre_build = publish_preflight(handoff_build, root, code=CODE)
    return csv_ref, handoff_ref, pre_ref, pre_build


def run_r(
    root: Path, handoff: ArtifactRef, csv: ArtifactRef, pre: ArtifactRef, seed: str = SEED
) -> tuple[str, str]:
    """Run the real R analysis and return (result directory relative to root, record sha256)."""
    out_root = root / R_RESULTS
    out_root.mkdir(parents=True, exist_ok=True)
    lock = sha256_file(ANALYSIS_DIR / "renv.lock")
    command = [
        "Rscript",
        "run_analysis.R",
        "run",
        "--handoff",
        str(root / handoff.logical_ref),
        "--handoff-sha256",
        handoff.sha256,
        "--csv",
        str(root / csv.logical_ref),
        "--lock-sha256",
        lock,
        "--seed",
        seed,
        "--seed-label",
        "synthetic_rehearsal",
        "--out-root",
        str(out_root),
        "--preflight",
        str(root / pre.logical_ref),
        "--preflight-sha256",
        pre.sha256,
    ]
    done = subprocess.run(
        command, cwd=ANALYSIS_DIR, capture_output=True, text=True, timeout=300, check=False
    )
    assert done.returncode == 0, done.stderr
    directory = re.search(r"^result directory: (.+)$", done.stdout, re.MULTILINE)
    record = re.search(r"^run_record.json sha256: ([0-9a-f]{64})$", done.stdout, re.MULTILINE)
    assert directory and record
    return str(Path(directory[1]).relative_to(root)), record[1]


@pytest.fixture(scope="session")
def base_world(tmp_path_factory) -> World:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    root = tmp_path_factory.mktemp("result_store")
    csv, handoff, pre, pre_build = make_store(root)
    result, record_sha256 = run_r(root, handoff, csv, pre)
    return World(root, handoff, csv, pre, pre_build, result, record_sha256)


def writable_copy(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    for path in [target, *target.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    return target


@pytest.fixture
def world(base_world, tmp_path) -> World:
    """A private writable copy of the store, so a test may alter anything in it."""
    copy = writable_copy(base_world.root, tmp_path / "store")
    return World(
        copy,
        base_world.handoff,
        base_world.csv,
        base_world.preflight,
        base_world.preflight_build,
        base_world.result,
        base_world.record_sha256,
    )


def verify(w: World, **changes):
    arguments = {
        "root": w.root,
        "result_dir": w.result,
        "preflight_build": w.preflight_build,
        "run_record_sha256": w.record_sha256,
    } | changes
    return ar.verify_r_result(**arguments)


def publish(w: World, **changes):
    arguments = {
        "root": w.root,
        "result_dir": w.result,
        "preflight_build": w.preflight_build,
        "run_record_sha256": w.record_sha256,
        "code": CODE,
    } | changes
    return ar.publish_result_record(**arguments)


# ---- the verifier on a real R result ---------------------------------------------------------


@needs_r
def test_a_real_r_result_verifies_and_reproduces_the_pinned_scientific_hash(world):
    verified = verify(world)
    assert verified.claims["scientific_result_sha256"] == GOLDEN
    assert verified.claims["run_record_sha256"] == world.record_sha256
    assert [f.name for f in verified.claims["r_files"]] == list(ar.RESULT_FILES)


@needs_r
def test_the_identity_is_the_documented_payload_over_the_file_hashes(world):
    # Built here from the files and the record's inputs, independently of the module's function.
    directory = world.root / world.result
    record = json.loads((directory / "run_record.json").read_bytes())
    files = [
        {"name": n, "sha256": sha256_file(directory / n)}
        for n in ("coprimary_results.csv", "paper_counts.csv", "state_counts.csv")
    ]
    payload = {
        "csv_sha256": record["inputs"]["csv_sha256"],
        "files": files,
        "handoff_sha256": record["inputs"]["handoff_sha256"],
        "seed": int(SEED),
    }
    assert sha256_bytes(canonical_json_bytes(payload)) == GOLDEN
    assert (
        ar.scientific_result_id(
            payload["handoff_sha256"],
            payload["csv_sha256"],
            int(SEED),
            {f["name"]: f["sha256"] for f in files},
        )
        == GOLDEN
    )


# ---- the CSV contract, without R --------------------------------------------------------------


def test_python_schemas_equal_the_r_schemas_in_the_r_source():
    source = (ANALYSIS_DIR / "R" / "results.R").read_text()
    for r_name, python_schema in (
        ("COPRIMARY_SCHEMA", ar.COPRIMARY_SCHEMA),
        ("PAPER_COUNTS_SCHEMA", ar.PAPER_COUNTS_SCHEMA),
        ("STATE_COUNTS_SCHEMA", ar.STATE_COUNTS_SCHEMA),
        ("INVENTORY_SCHEMA", ar.INVENTORY_SCHEMA),
    ):
        body = re.search(rf"{r_name} <- c\((.*?)\n?\)\n", source, re.DOTALL)
        assert body, r_name
        declared = re.findall(r"(\w+) = \"(chr|int|dbl)\"", body[1])
        assert declared == list(python_schema.items()), r_name


def table(schema, *rows):
    header = ",".join(schema)
    return ("\n".join([header, *rows]) + "\n").encode()


def test_the_literal_text_NA_is_the_missing_token_in_a_csv_cell():
    data = table(ar.PAPER_COUNTS_SCHEMA, "evidence_delivered,P001,1,1")
    assert ar.parse_csv("paper_counts.csv", data) == [
        {"outcome": "evidence_delivered", "paper_id": "P001", "eligible": 1, "complete": 1}
    ]
    cell = table(ar.INVENTORY_SCHEMA, "R6,2.6.1,NA,project,yes")
    assert ar.parse_csv("package_inventory.csv", cell)[0]["priority"] is None


@pytest.mark.parametrize(
    "data",
    [
        b"outcome,paper_id,eligible,complete\r\nevidence_delivered,P001,1,1\r\n",  # CRLF
        b"\xef\xbb\xbfoutcome,paper_id,eligible,complete\nevidence_delivered,P001,1,1\n",  # BOM
        b"outcome,paper_id,eligible,complete\nevidence_delivered,P001,1,1",  # no final newline
        b"outcome,paper_id,eligible,complete\nevidence_delivered,P001,01,1\n",  # leading zero
        b"outcome,paper_id,eligible,complete\nevidence_delivered,P001,1.0,1\n",  # not an integer
        b"outcome,paper_id,eligible,complete\nevidence_delivered,P001,1,1\n\n",  # blank line
        b"outcome,paper_id,eligible\nevidence_delivered,P001,1\n",  # missing column
        b"outcome,eligible,paper_id,complete\nevidence_delivered,1,P001,1\n",  # column order
        b'outcome,paper_id,eligible,complete\nevidence_delivered,"P001",1,1\n',  # needless quote
        b'outcome,paper_id,eligible,complete\nevidence_delivered,"NA",1,1\n',  # quoted NA
        b"outcome,paper_id,eligible,complete\nevidence_delivered,P001,1,99999999999\n",  # > int32
        b"outcome,paper_id,eligible,complete\nevidence_delivered,\xff,1,1\n",  # not UTF-8
    ],
)
def test_noncanonical_csv_bytes_are_rejected(data):
    with pytest.raises(ar.ResultError) as error:
        ar.parse_csv("paper_counts.csv", data)
    assert error.value.stage == "csv"


@pytest.mark.parametrize(
    "cell",
    [
        "0.1",
        "1e-05",
        "1.0",
        "+0.5",
        ".5",
        "0.50000000000000000",
        "-0",
        "nan",
        "inf",
        "1e400",
        "0x1p-1",
    ],
)
def test_double_cells_are_only_accepted_in_the_exact_form_r_writes(cell):
    row = ",".join(["evidence_delivered", "1", "1", "1", "ok", *[cell] * 4])
    schema = {
        "outcome": "chr",
        "n_eligible": "int",
        "n_complete": "int",
        "n_papers": "int",
        "summary_status": "chr",
        "a": "dbl",
        "b": "dbl",
        "c": "dbl",
        "d": "dbl",
    }
    with pytest.raises(ar.ResultError) as error:
        ar.parse_csv_with(schema, "x.csv", table(schema, row))
    assert error.value.stage == "csv"


def test_full_precision_doubles_round_trip_exactly():
    schema = {"x": "dbl"}
    values = [
        "0.10000000000000001",
        "-1.296461178681746e-16",
        "0.99999999999999989",
        "0",
        "NA",
        "1.0000000000000001e+300",
    ]
    parsed = ar.parse_csv_with(schema, "x.csv", table(schema, *values))
    assert [r["x"] for r in parsed] == [
        0.1,
        -1.296461178681746e-16,
        0.99999999999999989,
        0.0,
        None,
        1e300,
    ]


# ---- publication -------------------------------------------------------------------------------


@needs_r
def test_the_result_record_is_published_read_back_and_states_its_limits(world):
    published = publish(world)
    assert published.created
    checked = ar.read_result_record(published.build, world.root)
    record = checked.record
    assert (record.purpose, record.input_origin, record.scientific_authority) == (
        "synthetic_rehearsal",
        "synthetic_rehearsal",
        "none",
    )
    assert record.scientific_result_sha256 == GOLDEN
    assert record.run_record_sha256 == world.record_sha256
    assert record.applied_bootstrap_seed == int(SEED)
    statuses = {c.name: c.status for c in record.checks}
    assert statuses["complete_case_arithmetic"] == "verified_by_python"
    assert statuses["statistical_values"] == "reported_by_r"
    assert statuses["source_run_lineage"] == statuses["record_authentication"] == "not_established"
    assert record.rerun == "not_performed"


@needs_r
def test_identical_evidence_gives_identical_bytes_and_a_second_publication_is_a_noop(world):
    first = publish(world)
    again = publish(world)
    assert not again.created
    assert (again.ref, again.build) == (first.ref, first.build)
    assert (world.root / first.ref.logical_ref).read_bytes() == (
        world.root / again.ref.logical_ref
    ).read_bytes()


@needs_r
def test_a_different_r_execution_gives_a_distinguishable_record_with_the_same_science(world):
    first = publish(world)
    second_result, second_hash = run_r(world.root, world.handoff, world.csv, world.preflight)
    assert second_result != world.result
    second = publish(world, result_dir=second_result, run_record_sha256=second_hash)
    assert second.ref.artifact_id != first.ref.artifact_id
    a = ar.read_result_record(first.build, world.root).record
    b = ar.read_result_record(second.build, world.root).record
    assert a.scientific_result_sha256 == b.scientific_result_sha256 == GOLDEN
    assert a.run_record_sha256 != b.run_record_sha256


@needs_r
def test_the_published_manifest_lists_the_preflight_and_the_record(world):
    published = publish(world)
    manifest = read_manifest(world.root / published.build.logical_ref, published.build.sha256)
    assert manifest.outputs[0].artifact_kind == "analysis_result_record"
    assert [i.role for i in manifest.inputs] == ["analysis_preflight_build", "analysis_preflight"]
    assert manifest.outputs[0].artifact == published.ref
