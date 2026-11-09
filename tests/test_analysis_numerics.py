"""The R analysis runs with a fixed OpenBLAS kernel and one thread, whatever the caller's environment.

OpenBLAS 0.3.20 chooses its kernel from the CPU. On the AMD EPYC 9V45 CI runners it falls back to
the generic Barcelona kernel, which changes the last bit of the CR2 standard error of the
answer_correct outcome (risk difference exactly 0) and so the primary result hash. The synthetic
rehearsal golden is the Haswell result.
"""

import os
import re
import subprocess

import pytest
from test_analysis_rehearsal import (
    CODE,
    LOCK,
    SEED,
    fixed_preflight_hooks,
    publish_frozen_handoff,
    rehearse,
)
from test_analysis_result import (  # noqa: F401  (fixtures)
    ANALYSIS_DIR,
    GOLDEN,
    base_world,
    needs_r,
    verify,
    world,
)

from biorag_repro.evaluation import analysis_result as ar
from biorag_repro.evaluation.analysis_preflight import publish_preflight, read_preflight

# The kernel the failing runners selected by default (OpenBLAS "Core: Barcelona"); it can be
# requested on any x86-64 CPU, so the failure is reproducible without that hardware.
FAILING_KERNEL = {"OPENBLAS_CORETYPE": "BARCELONA", "OPENBLAS_NUM_THREADS": "4"}


def test_the_r_process_environment_pins_the_kernel_and_keeps_the_rest(monkeypatch):
    monkeypatch.setenv("OPENBLAS_CORETYPE", "ZEN")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.setenv("BIORAG_UNRELATED", "kept")
    env = ar.r_process_env()
    assert env["OPENBLAS_CORETYPE"] == "HASWELL"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["BIORAG_UNRELATED"] == "kept"
    assert os.environ["OPENBLAS_CORETYPE"] == "ZEN"  # the caller's own environment is untouched


@needs_r
def test_the_golden_holds_when_the_caller_selects_the_kernel_of_the_failing_runners(
    tmp_path, monkeypatch
):
    for name, value in FAILING_KERNEL.items():
        monkeypatch.setenv(name, value)
    build = publish_frozen_handoff(tmp_path)
    assert rehearse(tmp_path, build).scientific_result_sha256 == GOLDEN


def test_the_rerun_hands_the_pinned_environment_to_r(world):  # noqa: F811
    seen = []

    def stub(command, **kwargs):
        seen.append(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, "re-execution: byte_identical\n", "")

    ar.rerun_in_r(verify(world), world.root, ANALYSIS_DIR, run=stub)
    assert seen[0]["OPENBLAS_CORETYPE"] == "HASWELL"
    assert seen[0]["OPENBLAS_NUM_THREADS"] == "1"


@needs_r
@pytest.mark.parametrize(
    "overrides",
    [{}, FAILING_KERNEL, {"OPENBLAS_CORETYPE": "HASWELL", "OPENBLAS_NUM_THREADS": "2"}],
    ids=["unset", "other_kernel", "other_threads"],
)
def test_r_refuses_to_start_a_run_without_the_pinned_kernel(tmp_path, overrides):
    build = publish_frozen_handoff(tmp_path)
    _, pre = publish_preflight(build, tmp_path, code=CODE, **fixed_preflight_hooks())
    checked = read_preflight(pre, tmp_path).preflight
    out = tmp_path / "unpinned_out"
    out.mkdir()
    command = [
        "Rscript",
        "run_analysis.R",
        "run",
        "--handoff",
        str(tmp_path / checked.handoff.logical_ref),
        "--handoff-sha256",
        checked.handoff.sha256,
        "--csv",
        str(tmp_path / checked.csv.logical_ref),
        "--lock-sha256",
        LOCK,
        "--seed",
        str(SEED),
        "--seed-label",
        "synthetic_rehearsal",
        "--out-root",
        str(out),
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENBLAS_")} | overrides
    done = subprocess.run(
        command, cwd=ANALYSIS_DIR, capture_output=True, text=True, env=env, check=False
    )
    assert done.returncode != 0
    assert "OPENBLAS_CORETYPE=HASWELL" in done.stderr
    assert not any(out.iterdir())


def test_the_python_and_r_pins_are_the_same():
    text = (ANALYSIS_DIR / "run_analysis.R").read_text(encoding="utf-8")
    block = re.search(r"BLAS_PIN <- c\((.*?)\)", text, re.DOTALL)
    assert block is not None
    in_r = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', block[1]))
    assert in_r == ar.R_BLAS_PIN
