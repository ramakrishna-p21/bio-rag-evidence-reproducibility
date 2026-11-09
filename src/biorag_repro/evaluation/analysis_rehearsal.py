"""Run the offline Python-to-R rehearsal in a fixed order.

preflight -> R run -> Python verification of the R result -> result record -> read-back.

Each step takes what the step before it verified, never what the caller supplied. This module
only sequences the existing verifiers and publishers; it does not repeat their checks and it
does not compute any statistic. A synthetic rehearsal has no scientific authority, and the
preflight citation is a content dependency and an operational order, not proof of when anything ran.
"""

import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from biorag_repro.core.identity import new_run_id
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_manifest
from biorag_repro.evaluation.analysis_handoff import OfficialProvenanceNotVerified
from biorag_repro.evaluation.analysis_preflight import publish_preflight, read_preflight
from biorag_repro.evaluation.analysis_result import (
    PublishedResult,
    Rerun,
    publish_result_record,
    r_process_env,
    read_result_record,
    rerun_in_r,
    verify_r_result,
)
from biorag_repro.generation.environment import GitState, utc_now

SEED_LABEL = "synthetic_rehearsal"
R_TIMEOUT_SECONDS = 1800


class RehearsalError(ValueError):
    """The rehearsal stopped; `stage` names the step that failed. Nothing later was done."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


@dataclass(frozen=True)
class Rehearsal:
    preflight_build: ArtifactRef
    result_dir: str  # relative to the artifact root
    run_record_sha256: str
    scientific_result_sha256: str
    rerun: Rerun
    record: PublishedResult


def _inside(root: Path, value: str | Path, stage: str, what: str) -> str:
    """`value` as a path relative to the root; anything outside the root is refused."""
    base = Path(os.path.abspath(root))
    full = Path(os.path.abspath(base / value))
    if full == base or base not in full.parents:
        raise RehearsalError(stage, f"{what} must lie inside the artifact root")
    return full.relative_to(base).as_posix()


def _run_r(
    command: list[str], analysis_dir: Path, run: Callable[..., subprocess.CompletedProcess[str]]
) -> subprocess.CompletedProcess[str]:
    try:
        done = run(
            command,
            cwd=analysis_dir,
            capture_output=True,
            text=True,
            timeout=R_TIMEOUT_SECONDS,
            check=False,
            env=r_process_env(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RehearsalError("r_run", f"R could not be run: {error}") from error
    if done.returncode != 0:
        raise RehearsalError(
            "r_run", f"R exited with status {done.returncode}: {done.stderr.strip()[-500:]}"
        )
    return done


def _printed(stdout: str, pattern: str, what: str) -> str:
    found = re.search(pattern, stdout, re.MULTILINE)
    if not found:
        raise RehearsalError("r_run", f"R did not print {what}")
    return found[1]


def rehearse(
    root: Path,
    handoff_build: ArtifactRef,
    *,
    analysis_dir: Path,
    code: GitState,
    seed: int,
    lock_sha256: str,
    out_root: str,
    rerun: bool = False,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    now: Callable[[], str] = utc_now,
    new_id: Callable[[], str] = new_run_id,
) -> Rehearsal:
    """Run the whole chain for a synthetic rehearsal handoff, or stop at the first failure.

    The seed, its label and the lock hash are passed to R exactly as given; there are no
    defaults. Every later step is fed from the verified result of the one before.
    """
    out_relative = _inside(root, out_root, "arguments", "the R output root")

    try:
        _, pre_build = publish_preflight(handoff_build, root, code=code, now=now, new_id=new_id)
        checked = read_preflight(pre_build, root)
        pre_file = read_manifest(root / pre_build.logical_ref, pre_build.sha256).outputs[0].artifact
    except OfficialProvenanceNotVerified:
        raise
    except (ValueError, OSError) as error:
        raise RehearsalError("preflight", str(error)) from error
    handoff, csv = checked.preflight.handoff, checked.preflight.csv

    (root / out_relative).mkdir(parents=True, exist_ok=True)
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
        lock_sha256,
        "--seed",
        str(seed),
        "--seed-label",
        SEED_LABEL,
        "--out-root",
        str(root / out_relative),
        "--preflight",
        str(root / pre_file.logical_ref),
        "--preflight-sha256",
        pre_file.sha256,
    ]
    done = _run_r(command, analysis_dir, run)
    printed_dir = _printed(done.stdout, r"^result directory: (.+)$", "its result directory")
    run_record_sha256 = _printed(
        done.stdout, r"^run_record\.json sha256: ([0-9a-f]{64})$", "the run record hash"
    )
    printed_result = _printed(
        done.stdout, r"^scientific_result_sha256: ([0-9a-f]{64})$", "the scientific result hash"
    )
    result_dir = _inside(root, printed_dir, "r_run", "the R result directory")
    if Path(out_relative) not in Path(result_dir).parents:
        raise RehearsalError("r_run", "the R result directory is not under the R output root")

    try:
        verified = verify_r_result(root, result_dir, pre_build, run_record_sha256)
    except (ValueError, OSError) as error:
        raise RehearsalError("postflight", str(error)) from error
    claims = verified.claims
    for name, expected in (
        ("scientific_result_sha256", printed_result),
        ("applied_bootstrap_seed", seed),
        ("r_dependency_lock_sha256", lock_sha256),
    ):
        if claims[name] != expected:
            raise RehearsalError(
                "postflight", f"{name} in the verified result is not the value that was requested"
            )

    try:
        verdict: Rerun = rerun_in_r(verified, root, analysis_dir) if rerun else "not_performed"
        published = publish_result_record(
            root, result_dir, pre_build, run_record_sha256, code=code, rerun=verdict
        )
    except (ValueError, OSError) as error:
        raise RehearsalError("publication", str(error)) from error

    try:
        back = read_result_record(published.build, root).record
    except (ValueError, OSError) as error:
        raise RehearsalError("read_back", str(error)) from error
    if (
        back.preflight_build != pre_build
        or back.preflight != pre_file
        or back.handoff_build != handoff_build
        or back.run_record_sha256 != run_record_sha256
        or back.scientific_result_sha256 != claims["scientific_result_sha256"]
        or back.applied_bootstrap_seed != seed
        or back.result_dir != result_dir
    ):
        raise RehearsalError("read_back", "the stored record does not cite what was just verified")
    return Rehearsal(
        pre_build,
        result_dir,
        run_record_sha256,
        back.scientific_result_sha256,
        back.rerun,
        published,
    )
