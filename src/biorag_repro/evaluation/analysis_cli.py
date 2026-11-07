"""Verify a published analysis table or a rehearsal analysis handoff, and write a preflight.

`verify-handoff` checks a rehearsal handoff only. It cannot establish official provenance:
that needs freshly verified official runs, which this command does not have. `preflight`
records what Python verified about a rehearsal handoff before R is run; `verify-preflight`
reads such a record back against its handoff. After R has run, `verify-r-result` checks the
published R result directory, `record-result` publishes the Python result record, and
`verify-result-record` reads that record back against the R files and the preflight.
`rehearse` runs the whole chain for a synthetic rehearsal handoff in that order and stops at the
first failure.
"""

import argparse
import os
import sys
from pathlib import Path

from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_manifest
from biorag_repro.evaluation.analysis_handoff import OfficialProvenanceNotVerified, read_handoff
from biorag_repro.evaluation.analysis_preflight import publish_preflight, read_preflight
from biorag_repro.evaluation.analysis_rehearsal import RehearsalError, rehearse
from biorag_repro.evaluation.analysis_result import (
    AnalysisResultRecord,
    Rerun,
    ResultError,
    compare_current_environment,
    publish_result_record,
    read_result_record,
    rerun_in_r,
    verify_r_result,
)
from biorag_repro.evaluation.analysis_storage import read_table
from biorag_repro.generation.environment import system_probes


def _ref(logical_ref: str, sha256: str) -> ArtifactRef:
    return ArtifactRef(artifact_id="build_" + sha256, logical_ref=logical_ref, sha256=sha256)


def verify_table(args: argparse.Namespace) -> int:
    build = _ref(args.build, args.sha256)
    try:
        rows = read_table(build, args.root)
    except (ValueError, OSError) as error:
        print(f"analysis table is not valid: {error}", file=sys.stderr)
        return 1
    manifest = read_manifest(args.root / args.build, args.sha256)
    table = next(o.artifact for o in manifest.outputs if o.artifact_kind == "analysis_table")
    print(f"valid analysis table: {len(rows)} rows, csv sha256 {table.sha256}")
    return 0


def verify_handoff(args: argparse.Namespace) -> int:
    build = _ref(args.handoff_build, args.handoff_sha256)
    try:
        checked = read_handoff(build, args.root)
    except OfficialProvenanceNotVerified:
        print(
            "this handoff claims an official origin; official provenance can only be checked "
            "against freshly verified official runs, which this command does not have. "
            "Nothing is accepted.",
            file=sys.stderr,
        )
        return 2
    except (ValueError, OSError) as error:
        print(f"analysis handoff is not valid: {error}", file=sys.stderr)
        return 1
    handoff = checked.handoff
    print(
        f"valid synthetic rehearsal handoff: {len(checked.rows)} rows, "
        f"csv sha256 {handoff.csv.sha256}"
    )
    for outcome in handoff.accounting:
        print(
            f"  {outcome.outcome}: {len(outcome.complete_item_ids)} of "
            f"{len(outcome.eligible_items)} eligible items are complete cases"
        )
    print(
        "This check cannot establish official provenance. A rehearsal handoff has "
        "no scientific authority."
    )
    return 0


def preflight(args: argparse.Namespace) -> int:
    build = _ref(args.handoff_build, args.handoff_sha256)
    try:
        file_ref, pre_build = publish_preflight(
            build, args.root, code=system_probes(args.repository).git()
        )
        checked = read_preflight(pre_build, args.root)
    except OfficialProvenanceNotVerified:
        print(
            "this handoff claims an official origin; a preflight is written for a synthetic "
            "rehearsal only. Nothing is written.",
            file=sys.stderr,
        )
        return 2
    except (ValueError, OSError) as error:
        print(f"preflight not written: {error}", file=sys.stderr)
        return 1
    handoff = checked.handoff.handoff
    print(f"preflight build: {pre_build.logical_ref} {pre_build.sha256}")
    print(f"preflight file: {args.root / file_ref.logical_ref} {file_ref.sha256}")
    print("for the R runner (verified by Python just now):")
    print(f"  --handoff {args.root / checked.preflight.handoff.logical_ref}")
    print(f"  --handoff-sha256 {checked.preflight.handoff.sha256}")
    print(f"  --csv {args.root / handoff.csv.logical_ref}")
    print(f"  --preflight {args.root / file_ref.logical_ref}")
    print(f"  --preflight-sha256 {file_ref.sha256}")
    print(
        "Established: the handoff and table bytes, the handoff contract and the complete-case "
        "accounting of a synthetic rehearsal. Not established: official source-run lineage, "
        "the production seed, authorship, or when this ran. Running this before R is "
        "operational sequencing; the record and its hash citation are not cryptographic proof "
        "of order. No scientific authority."
    )
    return 0


def verify_preflight(args: argparse.Namespace) -> int:
    build = _ref(args.preflight_build, args.preflight_sha256)
    try:
        checked = read_preflight(build, args.root)
    except (ValueError, OSError) as error:
        print(f"preflight is not valid: {error}", file=sys.stderr)
        return 1
    pre = checked.preflight
    print(f"valid preflight {pre.preflight_id} for handoff sha256 {pre.handoff.sha256}")
    print(
        "It records what Python verified for a synthetic rehearsal; it does not establish "
        "official lineage, authorship or the time of verification."
    )
    return 0


LIMITS = (
    "Verified by Python: the R result bytes, the run record structure, the scientific result "
    "identity, the CSV contract, the binding to the preflight and handoff, and recomputed counts, "
    "rates and the paired risk difference. Reported by R and not recomputed: the bootstrap, CR2, "
    "Satterthwaite and Holm values, the R runtime and the code identity. Not established: "
    "official source-run lineage, the production seed, authenticity (a hash you supply is not "
    "a credential). A synthetic rehearsal has no scientific authority."
)


def _result_location(root: Path, value: str) -> str:
    """The result directory as a path relative to the root; links are refused later, on opening."""
    full = Path(os.path.abspath(root / value))
    try:
        return full.relative_to(os.path.abspath(root)).as_posix()
    except ValueError:
        raise ResultError(
            "result_location", "the result directory must lie inside the root"
        ) from None


def _show_compared(record: AnalysisResultRecord, repository: Path | None) -> None:
    if repository is not None:
        for name, state in compare_current_environment(record, repository).items():
            print(f"current checkout vs recorded {name}: {state} (informational; not a test)")


def verify_r_result_command(args: argparse.Namespace) -> int:
    try:
        verified = verify_r_result(
            args.root,
            _result_location(args.root, args.result),
            _ref(args.preflight_build, args.preflight_sha256),
            args.run_record_sha256,
        )
    except (ResultError, ValueError, OSError) as error:
        print(f"R result is not valid: {error}", file=sys.stderr)
        return 1
    claims = verified.claims
    print(f"valid R result: scientific_result_sha256 {claims['scientific_result_sha256']}")
    print(f"run_record.json sha256 {claims['run_record_sha256']} (supplied by you)")
    print(LIMITS)
    return 0


def record_result(args: argparse.Namespace) -> int:
    try:
        result_dir = _result_location(args.root, args.result)
        preflight_build = _ref(args.preflight_build, args.preflight_sha256)
        rerun: Rerun = "not_performed"
        if args.rerun:
            verified = verify_r_result(
                args.root, result_dir, preflight_build, args.run_record_sha256
            )
            rerun = rerun_in_r(verified, args.root, args.repository / "analysis")
        published = publish_result_record(
            args.root,
            result_dir,
            preflight_build,
            args.run_record_sha256,
            code=system_probes(args.repository).git(),
            rerun=rerun,
        )
        checked = read_result_record(published.build, args.root)
    except (ResultError, ValueError, OSError) as error:
        print(f"result record not written: {error}", file=sys.stderr)
        return 1
    print(f"{'published' if published.created else 'already published (identical bytes verified)'}")
    print(f"result record build: {published.build.logical_ref} {published.build.sha256}")
    print(f"result record file: {args.root / published.ref.logical_ref} {published.ref.sha256}")
    print(f"R rerun: {checked.record.rerun}")
    _show_compared(checked.record, args.repository if args.compare_current else None)
    print(LIMITS)
    return 0


def verify_result_record(args: argparse.Namespace) -> int:
    try:
        checked = read_result_record(_ref(args.result_build, args.result_sha256), args.root)
    except (ResultError, ValueError, OSError) as error:
        print(f"result record is not valid: {error}", file=sys.stderr)
        return 1
    record = checked.record
    print(f"valid result record for scientific_result_sha256 {record.scientific_result_sha256}")
    print(f"R run record sha256 {record.run_record_sha256}; R rerun: {record.rerun}")
    _show_compared(record, args.repository if args.compare_current else None)
    print(LIMITS)
    return 0


def rehearse_command(args: argparse.Namespace) -> int:
    try:
        done = rehearse(
            args.root,
            _ref(args.handoff_build, args.handoff_sha256),
            analysis_dir=args.repository / "analysis",
            code=system_probes(args.repository).git(),
            seed=args.seed,
            lock_sha256=args.lock_sha256,
            out_root=args.out_root,
            rerun=args.rerun,
        )
    except OfficialProvenanceNotVerified:
        print(
            "this handoff claims an official origin; a rehearsal accepts a synthetic handoff "
            "only. Nothing is written.",
            file=sys.stderr,
        )
        return 2
    except RehearsalError as error:
        print(f"rehearsal stopped: {error}", file=sys.stderr)
        return 1
    print(f"rehearsal complete: scientific_result_sha256 {done.scientific_result_sha256}")
    print(f"preflight build: {done.preflight_build.logical_ref} {done.preflight_build.sha256}")
    print(f"R result directory: {args.root / done.result_dir}")
    print(f"run_record.json sha256: {done.run_record_sha256}; R rerun: {done.rerun}")
    print(f"result record build: {done.record.build.logical_ref} {done.record.build.sha256}")
    print(f"result record file: {args.root / done.record.ref.logical_ref} {done.record.ref.sha256}")
    print(f"The seed {args.seed} is a rehearsal seed, not the approved production seed. " + LIMITS)
    return 0


def _result_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preflight-build", required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--result", required=True, help="R result directory, inside --root")
    parser.add_argument(
        "--run-record-sha256", required=True, help="hash R printed for run_record.json"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)

    table = operations.add_parser("verify", help="verify a published analysis table")
    table.add_argument("--root", type=Path, required=True)
    table.add_argument("--build", required=True, help="logical_ref of the table build manifest")
    table.add_argument("--sha256", required=True, help="SHA-256 of the table build manifest")
    table.set_defaults(run=verify_table)

    handoff = operations.add_parser(
        "verify-handoff", help="verify a rehearsal analysis handoff (never an official one)"
    )
    handoff.add_argument("--root", type=Path, required=True)
    handoff.add_argument(
        "--handoff-build", required=True, help="logical_ref of the handoff build manifest"
    )
    handoff.add_argument(
        "--handoff-sha256", required=True, help="SHA-256 of the handoff build manifest"
    )
    handoff.set_defaults(run=verify_handoff)

    pre = operations.add_parser(
        "preflight", help="verify a rehearsal handoff and publish the preflight record for R"
    )
    pre.add_argument("--root", type=Path, required=True)
    pre.add_argument("--handoff-build", required=True)
    pre.add_argument("--handoff-sha256", required=True)
    pre.add_argument("--repository", type=Path, default=Path.cwd(), help="Git repository to record")
    pre.set_defaults(run=preflight)

    check = operations.add_parser("verify-preflight", help="read a preflight record back")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--preflight-build", required=True)
    check.add_argument("--preflight-sha256", required=True)
    check.set_defaults(run=verify_preflight)

    r_check = operations.add_parser("verify-r-result", help="verify a published R result directory")
    _result_arguments(r_check)
    r_check.set_defaults(run=verify_r_result_command)

    record = operations.add_parser(
        "record-result", help="verify an R result and publish the Python result record"
    )
    _result_arguments(record)
    record.add_argument(
        "--repository", type=Path, default=Path.cwd(), help="Git repository to record"
    )
    record.add_argument("--rerun", action="store_true", help="also ask R to recompute the result")
    record.add_argument("--compare-current", action="store_true")
    record.set_defaults(run=record_result)

    read_back = operations.add_parser("verify-result-record", help="read a result record back")
    read_back.add_argument("--root", type=Path, required=True)
    read_back.add_argument("--result-build", required=True)
    read_back.add_argument("--result-sha256", required=True)
    read_back.add_argument("--repository", type=Path, default=Path.cwd())
    read_back.add_argument("--compare-current", action="store_true")
    read_back.set_defaults(run=verify_result_record)

    rehearsal = operations.add_parser(
        "rehearse",
        help="preflight, run R, verify and record a synthetic rehearsal in one controlled sequence",
    )
    rehearsal.add_argument("--root", type=Path, required=True)
    rehearsal.add_argument("--handoff-build", required=True)
    rehearsal.add_argument("--handoff-sha256", required=True)
    rehearsal.add_argument("--seed", type=int, required=True, help="rehearsal seed, no default")
    rehearsal.add_argument("--lock-sha256", required=True, help="approved renv.lock SHA-256")
    rehearsal.add_argument("--out-root", required=True, help="R output root, inside --root")
    rehearsal.add_argument(
        "--repository", type=Path, default=Path.cwd(), help="Git repository to record"
    )
    rehearsal.add_argument("--rerun", action="store_true", help="also ask R to recompute")
    rehearsal.set_defaults(run=rehearse_command)

    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
