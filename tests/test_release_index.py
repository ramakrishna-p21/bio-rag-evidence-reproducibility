# ruff: noqa: F811
"""Release index of a synthetic rehearsal: references, separation of external resources, tampering.

The secondary R result is produced by the real R program from the real primary rehearsal result.
Everything is synthetic; no value is a scientific result.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from test_analysis_rehearsal import real_runs  # noqa: F401  (session fixture)
from test_analysis_result import ANALYSIS_DIR, needs_r

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import read_manifest
from biorag_repro.evaluation import release_index as ri
from biorag_repro.evaluation.analysis_preflight import read_preflight
from biorag_repro.evaluation.analysis_result import ResultError


def derive(root: Path, build, result_dir: str, out_relative: str) -> tuple[str, dict[str, str]]:
    checked = read_preflight(build, root)
    handoff, csv = checked.preflight.handoff, checked.preflight.csv
    pre_file = read_manifest(root / build.logical_ref, build.sha256).outputs[0].artifact
    (root / out_relative).mkdir(parents=True, exist_ok=True)
    done = subprocess.run(
        [
            "Rscript",
            "run_analysis.R",
            "derive",
            "--handoff",
            str(root / handoff.logical_ref),
            "--handoff-sha256",
            handoff.sha256,
            "--csv",
            str(root / csv.logical_ref),
            "--primary",
            str(root / result_dir),
            "--out-root",
            str(root / out_relative),
        ],
        cwd=ANALYSIS_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    printed = re.search(r"^result directory: (.+)$", done.stdout, re.MULTILINE)
    assert printed
    inputs = {
        "handoff": handoff.logical_ref,
        "analysis_table": csv.logical_ref,
        "preflight": pre_file.logical_ref,
    }
    return os.path.relpath(printed.group(1), root), inputs


@pytest.fixture(scope="module")
def chain(real_runs):
    root, build, first, second = real_runs
    secondary, inputs = derive(
        root, first.preflight_build, first.result_dir, "analysis/r_secondary"
    )
    return root, build, first, second, secondary, inputs


def make(chain, **changes):
    root, _build, first, _second, secondary, inputs = chain
    arguments = {
        "primary_dir": first.result_dir,
        "secondary_dir": secondary,
        "inputs": inputs,
    } | changes
    return ri.build_release_index(root, **arguments)


def writable_secondary(chain, name: str) -> str:
    root, *_rest, secondary, _inputs = chain
    target = Path(f"analysis/r_secondary/{name}")
    shutil.copytree(root / secondary, root / target)
    for path in (root / target).iterdir():
        path.chmod(0o644)
    (root / target).chmod(0o755)
    return str(target)


@needs_r
def test_index_lists_included_files_separately_from_external_resources(chain):
    index = make(chain)
    roles = [a.role for a in index.included]
    assert len(roles) == 3 + 5 + len(ri.SECONDARY_FILES)
    assert {r.split(":")[0] for r in roles} == {"input", "r_primary", "r_secondary"}
    assert [e.role for e in index.external] == list(ri.EXTERNAL_RESOURCES)
    assert not {a.role for a in index.included} & {e.role for e in index.external}
    assert index.final is False and index.scientific_authority == "none"
    assert index.banner == ri.BANNER
    assert index.primary_scientific_result_sha256 == chain[2].scientific_result_sha256


@needs_r
def test_every_cited_hash_is_the_hash_of_the_file(chain):
    root = chain[0]
    index = make(chain)
    for artifact in index.included:
        data = (root / artifact.path).read_bytes()
        assert (artifact.bytes, artifact.sha256) == (len(data), sha256_bytes(data))


@needs_r
def test_the_serialized_index_has_no_local_paths_or_protected_content(chain):
    root = chain[0]
    text = ri.serialize_index(make(chain)).decode()
    assert str(root) not in text and "/home/" not in text
    assert all(not a.path.startswith("/") for a in make(chain).included)
    assert json.loads(text)["final"] is False


@needs_r
def test_the_same_files_give_the_same_index_bytes(chain):
    assert ri.serialize_index(make(chain)) == ri.serialize_index(make(chain))


@needs_r
def test_publication_never_replaces_and_reads_back_by_trusted_hash(chain):
    root, *_rest, secondary, inputs = chain
    first = chain[2]
    index = make(chain)
    digest = ri.publish_release_index(index, root, "analysis/r_secondary/release_index_a.json")
    with pytest.raises(FileExistsError):
        ri.publish_release_index(index, root, "analysis/r_secondary/release_index_a.json")
    read = ri.read_release_index(
        root,
        "analysis/r_secondary/release_index_a.json",
        digest,
        primary_dir=first.result_dir,
        secondary_dir=secondary,
        inputs=inputs,
    )
    assert read == index
    with pytest.raises(ResultError, match="trusted"):
        ri.read_release_index(
            root,
            "analysis/r_secondary/release_index_a.json",
            "0" * 64,
            primary_dir=first.result_dir,
            secondary_dir=secondary,
            inputs=inputs,
        )


@needs_r
def test_a_changed_secondary_table_is_refused(chain):
    root = chain[0]
    target = writable_secondary(chain, "tampered_table")
    path = root / target / "secondary_stage_a.csv"
    data = bytearray(path.read_bytes())
    data[-2] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ResultError, match="differs from its record"):
        make(chain, secondary_dir=target)


@needs_r
def test_a_report_without_the_synthetic_statement_is_refused(chain):
    root = chain[0]
    target = writable_secondary(chain, "tampered_report")
    path = root / target / "report.md"
    forged = path.read_bytes().replace(ri.BANNER.encode(), b"RESULTS")
    path.write_bytes(forged)
    record_path = root / target / ri.SECONDARY_RECORD
    record = json.loads(record_path.read_text())
    record["presentation"]["report.md"].update(bytes=len(forged), sha256=sha256_bytes(forged))
    record_path.write_text(json.dumps(record))
    with pytest.raises(ResultError, match="synthetic"):
        make(chain, secondary_dir=target)


@needs_r
def test_a_secondary_result_cannot_be_indexed_with_another_primary_result(chain):
    second = chain[3]
    with pytest.raises(ResultError, match="disagree"):
        make(chain, primary_dir=second.result_dir)


@needs_r
def test_an_input_that_differs_from_the_cited_hash_is_refused(chain):
    root, _b, _f, _s, _sec, inputs = chain
    other = dict(inputs, handoff=inputs["analysis_table"])
    with pytest.raises(ResultError, match="input handoff"):
        make(chain, inputs=other)
    with pytest.raises(ResultError, match="inputs are"):
        make(chain, inputs={k: v for k, v in inputs.items() if k != "preflight"})
    with pytest.raises(ResultError, match="plain relative"):
        make(chain, inputs=dict(inputs, preflight="../outside.json"))
    assert root.exists()


@needs_r
def test_an_index_that_no_longer_matches_its_files_is_refused(chain):
    root, *_rest, secondary, inputs = chain
    first = chain[2]
    index = make(chain)
    forged = index.model_copy(update={"renv_lock_sha256": "1" * 64})
    digest = ri.publish_release_index(forged, root, "analysis/r_secondary/release_index_b.json")
    with pytest.raises(ResultError, match="no longer matches"):
        ri.read_release_index(
            root,
            "analysis/r_secondary/release_index_b.json",
            digest,
            primary_dir=first.result_dir,
            secondary_dir=secondary,
            inputs=inputs,
        )


def test_the_model_rejects_inconsistent_resource_lists(chain=None):
    base = {
        "handoff_sha256": "a" * 64,
        "csv_sha256": "b" * 64,
        "primary_scientific_result_sha256": "c" * 64,
        "secondary_result_sha256": "d" * 64,
        "r_version_required": "4.6.1",
        "renv_lock_sha256": "e" * 64,
        "primary_code": {"git_commit": "f" * 40, "dirty": False, "script_sha256": {}},
        "secondary_code": {"git_commit": "f" * 40, "dirty": False, "script_sha256": {}},
    }
    artifact = {"role": "input:handoff", "path": "a/b", "bytes": 1, "sha256": "a" * 64}
    external = [
        {"role": r, "status": "externally_acquired_or_restricted", "reason": "x"}
        for r in ri.EXTERNAL_RESOURCES
    ]
    ri.ReleaseIndex.model_validate(base | {"included": [artifact], "external": external})
    with pytest.raises(ValueError, match="external resources"):
        ri.ReleaseIndex.model_validate(base | {"included": [artifact], "external": external[:-1]})
    with pytest.raises(ValueError, match="unique"):
        ri.ReleaseIndex.model_validate(
            base | {"included": [artifact, artifact], "external": external}
        )
    with pytest.raises(ValueError, match="external resource cannot also be included"):
        leak = dict(artifact, role="generator_weights", path="x/y")
        ri.ReleaseIndex.model_validate(base | {"included": [leak], "external": external})
    with pytest.raises(ValueError):
        ri.ReleaseIndex.model_validate(
            base | {"included": [artifact], "external": external, "final": True}
        )


@needs_r
def test_the_command_line_writes_and_reads_back_the_index(chain, capsys):
    from biorag_repro.evaluation.analysis_cli import main

    root, _b, first, _s, secondary, inputs = chain
    common = [
        "--root", str(root), "--primary", first.result_dir, "--secondary", secondary,
        "--handoff", inputs["handoff"], "--analysis-table", inputs["analysis_table"],
        "--preflight", inputs["preflight"],
    ]  # fmt: skip
    assert (
        main(["release-index", *common, "--out", "analysis/r_secondary/release_index_cli.json"])
        == 0
    )
    out = capsys.readouterr().out
    digest = re.search(r"release_index_cli\.json ([0-9a-f]{64})", out)
    assert digest and "NOT SCIENTIFIC FINDINGS" in out
    index = [
        "verify-release-index",
        *common,
        "--index",
        "analysis/r_secondary/release_index_cli.json",
    ]
    assert main([*index, "--sha256", digest.group(1)]) == 0
    assert main([*index, "--sha256", "0" * 64]) == 1
    assert (
        main(["release-index", *common, "--out", "analysis/r_secondary/release_index_cli.json"])
        == 1
    )


def edit_secondary_record(chain, name: str, change) -> str:
    root = chain[0]
    target = writable_secondary(chain, name)
    path = root / target / ri.SECONDARY_RECORD
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record))
    return target


@needs_r
@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        (
            "cited_scientific",
            lambda r: r["primary_result"].update(scientific_result_sha256="2" * 64),
            "disagree",
        ),
        ("record_banner", lambda r: r.update(banner="RESULTS"), "disagree"),
        (
            "record_bytes",
            lambda r: r["scientific_files"]["secondary_scope.csv"].update(bytes=1),
            "differs from its record",
        ),
        ("record_authority", lambda r: r.update(scientific_authority="primary"), "disagree"),
    ],
)
def test_a_secondary_record_that_misstates_the_primary_result_or_its_files_is_refused(
    chain, name, change, message
):
    target = edit_secondary_record(chain, name, change)
    with pytest.raises(ResultError, match=message):
        make(chain, secondary_dir=target)


@needs_r
def test_an_index_that_is_not_in_canonical_form_is_refused(chain):
    root, *_rest, secondary, inputs = chain
    first = chain[2]
    index = make(chain)
    pretty = json.dumps(index.model_dump(mode="json"), indent=2).encode() + b"\n"
    relative = "analysis/r_secondary/release_index_pretty.json"
    from biorag_repro.core.storage import publish_bytes

    publish_bytes(root / relative, pretty, sha256_bytes(pretty))
    with pytest.raises(ResultError, match="canonical"):
        ri.read_release_index(
            root,
            relative,
            sha256_bytes(pretty),
            primary_dir=first.result_dir,
            secondary_dir=secondary,
            inputs=inputs,
        )
