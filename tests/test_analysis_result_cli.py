# ruff: noqa: F811
"""Command line for the R result verification, the result record and the optional R rerun."""

import subprocess

import pytest
from test_analysis_result import (  # noqa: F401  (fixtures)
    ANALYSIS_DIR,
    base_world,
    needs_r,
    publish,
    verify,
    world,
)
from test_analysis_result_adversarial import edit_text, forge

from biorag_repro.evaluation import analysis_cli
from biorag_repro.evaluation import analysis_result as ar
from biorag_repro.evaluation.analysis_cli import main

pytestmark = needs_r


def arguments(w, operation, *extra):
    return [
        operation,
        "--root",
        str(w.root),
        "--preflight-build",
        w.preflight_build.logical_ref,
        "--preflight-sha256",
        w.preflight_build.sha256,
        "--result",
        w.result,
        "--run-record-sha256",
        w.record_sha256,
        *extra,
    ]


def test_verify_r_result_accepts_a_real_result_and_states_the_limits(world, capsys):
    assert main(arguments(world, "verify-r-result")) == 0
    out = capsys.readouterr().out
    assert "valid R result" in out and "supplied by you" in out
    assert "not a credential" in out and "no scientific authority" in out


def test_verify_r_result_accepts_an_absolute_path_inside_the_root(world):
    args = arguments(world, "verify-r-result")
    args[args.index("--result") + 1] = str(world.root / world.result)
    assert main(args) == 0


def test_verify_r_result_refuses_a_wrong_hash_a_path_outside_the_root_and_a_tampered_file(
    world, capsys
):
    args = arguments(world, "verify-r-result")
    args[args.index("--run-record-sha256") + 1] = "0" * 64
    assert main(args) == 1 and "[record_hash]" in capsys.readouterr().err
    args = arguments(world, "verify-r-result")
    args[args.index("--result") + 1] = "../outside"
    assert main(args) == 1 and "[result_location]" in capsys.readouterr().err
    edit_text(
        world, "paper_counts.csv", "evidence_delivered,P001,1,1", "evidence_delivered,P001,1,0"
    )
    assert main(arguments(world, "verify-r-result")) == 1
    assert "[output_hashes]" in capsys.readouterr().err


def test_record_result_publishes_then_reports_an_identical_republication(world, capsys):
    assert main(arguments(world, "record-result", "--repository", str(ANALYSIS_DIR.parent))) == 0
    first = capsys.readouterr().out
    assert first.startswith("published") and "R rerun: not_performed" in first
    assert main(arguments(world, "record-result", "--repository", str(ANALYSIS_DIR.parent))) == 0
    assert capsys.readouterr().out.startswith("already published")


def test_record_result_publishes_nothing_when_verification_fails(world, capsys):
    edit_text(world, "state_counts.csv", "JATS,yes,7", "JATS,yes,8")
    assert main(arguments(world, "record-result", "--repository", str(ANALYSIS_DIR.parent))) == 1
    assert "result record not written" in capsys.readouterr().err
    assert not (world.root / "analysis" / "result").exists()


def test_verify_result_record_reads_back_and_rejects_tampering(world, capsys):
    published = publish(world)
    read_back = [
        "verify-result-record",
        "--root",
        str(world.root),
        "--result-build",
        published.build.logical_ref,
        "--result-sha256",
        published.build.sha256,
    ]
    assert main([*read_back, "--repository", str(ANALYSIS_DIR.parent), "--compare-current"]) == 0
    out = capsys.readouterr().out
    assert "valid result record" in out and "informational; not a test" in out
    edit_text(
        world, "paper_counts.csv", "evidence_delivered,P001,1,1", "evidence_delivered,P001,1,0"
    )
    assert main(read_back) == 1
    assert "result record is not valid" in capsys.readouterr().err


# ---- optional R rerun --------------------------------------------------------------------------


def completed(code=0, out="", err=""):
    return subprocess.CompletedProcess([], code, out, err)


@pytest.mark.parametrize(
    "result,expected",
    [
        (completed(0, "x\nre-execution: byte_identical\n"), "byte_identical"),
        (
            completed(0, "re-execution: numerically_equal_not_byte_identical\n"),
            "numerically_equal_not_byte_identical",
        ),
        (
            completed(
                1, "", "error: [verify] re-execution differs in coprimary_results.csv column rd"
            ),
            "failed_to_reproduce",
        ),
    ],
)
def test_the_rerun_verdicts_are_taken_from_r(world, result, expected):
    seen = {}

    def fake(command, **options):
        seen["command"], seen["cwd"] = command, options["cwd"]
        return result

    assert ar.rerun_in_r(verify(world), world.root, ANALYSIS_DIR, run=fake) == expected
    assert seen["command"][:4] == ["Rscript", "run_analysis.R", "verify", "--result"]
    assert "--rerun" in seen["command"] and seen["cwd"] == ANALYSIS_DIR


@pytest.mark.parametrize(
    "result",
    [
        completed(1, "", "error: [environment] renv is not synchronized"),
        completed(0, "no verdict line\n"),
        completed(127, "", ""),
    ],
)
def test_an_r_failure_without_a_verdict_is_an_error_not_a_result(world, result):
    with pytest.raises(ar.ResultError) as error:
        ar.rerun_in_r(verify(world), world.root, ANALYSIS_DIR, run=lambda *a, **k: result)
    assert error.value.stage == "rerun"


def test_r_that_cannot_be_started_is_an_error(world):
    def missing(*_a, **_k):
        raise FileNotFoundError("Rscript")

    with pytest.raises(ar.ResultError) as error:
        ar.rerun_in_r(verify(world), world.root, ANALYSIS_DIR, run=missing)
    assert error.value.stage == "rerun"


def test_a_real_rerun_reproduces_the_bytes_and_is_recorded(world, capsys):
    assert ar.rerun_in_r(verify(world), world.root, ANALYSIS_DIR) == "byte_identical"
    assert (
        main(arguments(world, "record-result", "--repository", str(ANALYSIS_DIR.parent), "--rerun"))
        == 0
    )
    assert "R rerun: byte_identical" in capsys.readouterr().out


def test_a_consistent_forgery_of_an_r_only_value_passes_python_but_not_the_rerun(world, capsys):
    """The bootstrap interval is R's value, so Python cannot refute a hash-consistent change of it.
    Only R's own recomputation can, which is why the rerun exists and what it is limited to."""
    edit_text(
        world,
        "coprimary_results.csv",
        "-0.25,0.80000000000000004",
        format(-0.3125, ".17g") + ",0.80000000000000004",
    )
    forged = forge(world)
    verified = verify(world, run_record_sha256=forged)
    assert verified.claims["inference"][0].boot_status == "ok"  # Python accepted it
    assert ar.rerun_in_r(verified, world.root, ANALYSIS_DIR) == "failed_to_reproduce"
    world_forged = world.__class__(**{**world.__dict__, "record_sha256": forged})
    assert (
        main(
            arguments(
                world_forged, "record-result", "--repository", str(ANALYSIS_DIR.parent), "--rerun"
            )
        )
        == 0
    )
    assert "R rerun: failed_to_reproduce" in capsys.readouterr().out


def test_a_cli_rerun_that_cannot_run_publishes_nothing(world, monkeypatch, capsys):
    def broken(*_a, **_k):
        raise ar.ResultError("rerun", "R could not be run")

    monkeypatch.setattr(analysis_cli, "rerun_in_r", broken)
    assert (
        main(arguments(world, "record-result", "--repository", str(ANALYSIS_DIR.parent), "--rerun"))
        == 1
    )
    assert not (world.root / "analysis" / "result").exists()
    assert "result record not written" in capsys.readouterr().err


def test_record_result_can_compare_the_current_checkout_without_it_deciding_anything(world, capsys):
    args = arguments(world, "record-result", "--repository", str(ANALYSIS_DIR.parent))
    assert main([*args, "--compare-current"]) == 0
    out = capsys.readouterr().out
    assert "current checkout vs recorded git_commit:" in out and "informational; not a test" in out
    assert main(args) == 0
    assert "current checkout" not in capsys.readouterr().out
