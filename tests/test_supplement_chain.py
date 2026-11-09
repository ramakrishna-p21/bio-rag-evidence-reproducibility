"""Synthetic end to end: table, handoff, Python supplement, real R primary, secondary and supplement
summary, release index with the supplement, and the command-line readers.

Everything is synthetic; no value is a scientific result.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from test_analysis_export import roster_of
from test_analysis_rehearsal import rehearse
from test_analysis_result import ANALYSIS_DIR, needs_r
from test_release_index import derive
from test_supplement_fixture import observations

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.evaluation import analysis_handoff as ah
from biorag_repro.evaluation import release_index as ri
from biorag_repro.evaluation import supplement as sp
from biorag_repro.evaluation.analysis_cli import main
from biorag_repro.evaluation.analysis_preflight import read_preflight
from biorag_repro.evaluation.analysis_result import ResultError
from biorag_repro.evaluation.analysis_storage import publish_table
from biorag_repro.evaluation.analysis_table import build_table


def tree_hashes(directory: Path) -> dict[str, str]:
    return {p.name: sha256_bytes(p.read_bytes()) for p in sorted(directory.iterdir())}


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    root = tmp_path_factory.mktemp("supplement_chain")
    obs = observations()
    _csv, _inventory, table_build = publish_table(build_table(obs, roster_of(obs)), root)
    from test_r_input_fixture import CODE

    _ref, handoff_build = ah.publish_handoff(
        ah.build_handoff(table_build, root, code=CODE, rehearsal=True), root
    )
    verified = ah.read_handoff(handoff_build, root)
    supplement = sp.build_supplement(verified, obs)
    (root / "analysis").mkdir(exist_ok=True)
    digest = sp.publish_supplement(supplement, root, "analysis/supplement_py")
    first = rehearse(root, handoff_build)
    primary_before = tree_hashes(root / first.result_dir)
    secondary, inputs = derive(
        root, first.preflight_build, first.result_dir, "analysis/r_secondary"
    )
    secondary_before = tree_hashes(root / secondary)
    checked = read_preflight(first.preflight_build, root)
    handoff, csv = checked.preflight.handoff, checked.preflight.csv
    out = root / "analysis/r_supplement"
    out.mkdir()
    done = subprocess.run(
        [
            "Rscript",
            "run_analysis.R",
            "supplement",
            "--handoff",
            str(root / handoff.logical_ref),
            "--handoff-sha256",
            handoff.sha256,
            "--csv",
            str(root / csv.logical_ref),
            "--supplement",
            str(root / "analysis/supplement_py"),
            "--supplement-record-sha256",
            digest,
            "--out-root",
            str(out),
            "--primary",
            str(root / first.result_dir),
        ],
        cwd=ANALYSIS_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    printed = re.search(r"^result directory: (.+)$", done.stdout, re.MULTILINE)
    assert printed
    summary = os.path.relpath(printed.group(1), root)
    record_sha = re.search(r"record.json sha256: ([0-9a-f]{64})", done.stdout)
    assert record_sha
    return {
        "root": root,
        "verified": verified,
        "handoff_build": handoff_build,
        "supplement": supplement,
        "digest": digest,
        "first": first,
        "secondary": secondary,
        "inputs": inputs,
        "summary": summary,
        "summary_record_sha": record_sha.group(1),
        "stdout": done.stdout,
        "primary_before": primary_before,
        "secondary_before": secondary_before,
    }


def index_args(c, **changes):
    return {
        "primary_dir": c["first"].result_dir,
        "secondary_dir": c["secondary"],
        "inputs": c["inputs"],
        "supplement_dir": "analysis/supplement_py",
        "supplement_summary_dir": c["summary"],
    } | changes


@needs_r
def test_the_chain_is_synthetic_and_cites_the_python_supplement(chain):
    assert "SYNTHETIC REHEARSAL" in chain["stdout"]
    record = chain["supplement"].record
    text = (chain["root"] / chain["summary"] / "supplement_summary_record.json").read_text()
    assert record.supplement_sha256 in text and '"citation"' not in text


@needs_r
def test_index_with_a_supplement_lists_it_and_keeps_old_indexes_unchanged(chain):
    root = chain["root"]
    plain = ri.build_release_index(
        root,
        primary_dir=chain["first"].result_dir,
        secondary_dir=chain["secondary"],
        inputs=chain["inputs"],
    )
    full = ri.build_release_index(root, **index_args(chain))
    assert b"supplement_sha256" not in ri.serialize_index(plain)
    assert full.supplement_sha256 == chain["supplement"].record.supplement_sha256
    roles = [a.role for a in full.included]
    assert len(roles) == len(plain.included) + 6 + 7
    assert sum(r.startswith("supplement:") for r in roles) == 6
    assert sum(r.startswith("supplement_summary:") for r in roles) == 7
    assert full.primary_scientific_result_sha256 == plain.primary_scientific_result_sha256
    assert full.secondary_result_sha256 == plain.secondary_result_sha256
    digest = ri.publish_release_index(full, root, "analysis/r_secondary/index_s.json")
    back = ri.read_release_index(
        root, "analysis/r_secondary/index_s.json", digest, **index_args(chain)
    )
    assert back == full


@needs_r
def test_a_supplement_is_indexed_only_together_with_its_summary(chain):
    with pytest.raises(ResultError, match="given together"):
        ri.build_release_index(chain["root"], **index_args(chain, supplement_summary_dir=None))


@needs_r
def test_index_refuses_a_summary_that_cites_another_supplement(chain):
    root = chain["root"]
    other = Path("analysis/r_supplement_forged")
    shutil.copytree(root / chain["summary"], root / other)
    for path in (root / other).iterdir():
        path.chmod(0o644)
    (root / other).chmod(0o755)
    record = root / other / "supplement_summary_record.json"
    text = record.read_text()
    old = chain["supplement"].record.supplement_sha256
    record.write_text(text.replace(old, "0" * 64))
    with pytest.raises(ResultError):
        ri.build_release_index(root, **index_args(chain, supplement_summary_dir=str(other)))


@needs_r
def test_python_cli_reads_the_supplement_back(chain, capsys):
    root = chain["root"]
    build = chain["handoff_build"]
    code = main(
        [
            "verify-supplement",
            "--root", str(root),
            "--handoff-build", build.logical_ref,
            "--handoff-sha256", build.sha256,
            "--supplement", "analysis/supplement_py",
            "--sha256", chain["digest"],
        ]
    )  # fmt: skip
    assert code == 0
    assert "valid synthetic supplement" in capsys.readouterr().out
    code = main(
        [
            "verify-supplement",
            "--root", str(root),
            "--handoff-build", build.logical_ref,
            "--handoff-sha256", build.sha256,
            "--supplement", "analysis/supplement_py",
            "--sha256", "0" * 64,
        ]
    )  # fmt: skip
    assert code == 1


@needs_r
def test_r_verifies_its_summary_and_reruns_byte_identically(chain):
    root = chain["root"]
    checked = read_preflight(chain["first"].preflight_build, root)
    handoff, csv = checked.preflight.handoff, checked.preflight.csv
    done = subprocess.run(
        [
            "Rscript", "run_analysis.R", "verify-supplement",
            "--result", str(root / chain["summary"]),
            "--record-sha256", chain["summary_record_sha"],
            "--primary", str(root / chain["first"].result_dir),
            "--rerun",
            "--handoff", str(root / handoff.logical_ref),
            "--handoff-sha256", handoff.sha256,
            "--csv", str(root / csv.logical_ref),
            "--supplement", str(root / "analysis/supplement_py"),
        ],
        cwd=ANALYSIS_DIR, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert "byte_identical_tables" in done.stdout and "authenticated" in done.stdout


@needs_r
def test_the_primary_and_secondary_results_are_byte_for_byte_unchanged(chain):
    root = chain["root"]
    assert tree_hashes(root / chain["first"].result_dir) == chain["primary_before"]
    assert tree_hashes(root / chain["secondary"]) == chain["secondary_before"]
    record = (root / chain["summary"] / "supplement_summary_record.json").read_text()
    assert chain["first"].scientific_result_sha256 in record  # cited, not recomputed or replaced


@needs_r
def test_cli_writes_and_reads_back_an_index_with_the_supplement(chain, capsys):
    root = chain["root"]
    inputs = chain["inputs"]
    common = [
        "--root", str(root),
        "--primary", chain["first"].result_dir,
        "--secondary", chain["secondary"],
        "--handoff", inputs["handoff"],
        "--analysis-table", inputs["analysis_table"],
        "--preflight", inputs["preflight"],
        "--supplement", "analysis/supplement_py",
        "--supplement-summary", chain["summary"],
    ]  # fmt: skip
    assert main(["release-index", *common, "--out", "analysis/r_secondary/index_cli.json"]) == 0
    digest = re.search(r"index_cli.json ([0-9a-f]{64})", capsys.readouterr().out)
    assert digest
    args = ["verify-release-index", *common, "--index", "analysis/r_secondary/index_cli.json"]
    assert main([*args, "--sha256", digest.group(1)]) == 0
    assert "valid release index" in capsys.readouterr().out
    # the same index no longer verifies when the supplement options are left out
    plain = [a for a in common if a not in {"--supplement", "analysis/supplement_py"}]
    plain = plain[:-2]
    assert (
        main(
            [
                "verify-release-index",
                *plain,
                "--index",
                "analysis/r_secondary/index_cli.json",
                "--sha256",
                digest.group(1),
            ]
        )
        == 1
    )
