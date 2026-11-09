"""Evaluate or verify final-test readiness.

    python -m biorag_repro.readiness.cli evaluate --data-root DIR [--repo DIR]
    python -m biorag_repro.readiness.cli verify --data-root DIR --report-sha256 SHA [--repo DIR]

Exit status: 0 the verified decision is READY, 2 the verified decision is NOT_READY, 1 the
evaluator failed or the report is invalid. NOT_READY is a correct result, so it is not 1. The
three-way split is specific to this command; `biorag-environment` uses its own five codes.

The command is bound to the tracked approval policy of the checkout given by `--repo`
(default: the current directory). There is no option, environment variable or file that
selects another policy; `run` takes a policy only so tests can exercise decision logic with a
fixture. A report describes the moment it was made and is not
an authorization to run anything.
"""

import argparse
import sys
from pathlib import Path

from biorag_repro.readiness import publish
from biorag_repro.readiness.approvals import load_policy
from biorag_repro.readiness.evaluate import EvaluationError, evaluate, system_context
from biorag_repro.readiness.evidence import Context
from biorag_repro.readiness.policy import ApprovalPolicy

EXIT_READY, EXIT_ERROR, EXIT_NOT_READY = 0, 1, 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biorag-readiness", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("evaluate", help="evaluate readiness now and publish the report")
    make.add_argument("--data-root", type=Path, required=True)
    make.add_argument("--repo", type=Path, default=Path.cwd())
    check = commands.add_parser("verify", help="verify a stored report (historical)")
    check.add_argument("--data-root", type=Path, required=True)
    check.add_argument("--report-sha256", required=True)
    check.add_argument(
        "--repo",
        type=Path,
        help="checkout whose tracked policy is used and which is compared with the report "
        "(default: the current directory, without the comparison)",
    )
    return parser


def _summary(readback: publish.Readback) -> None:
    report = readback.report
    print(f"decision {readback.outcome.decision} (recomputed from the stored gate results)")
    for gate in report.gates:
        print(f"  {gate.gate_id}: {gate.status} ({gate.reason_code})")
    print(
        f"retained evidence verified: {readback.verified_evidence}; "
        f"not retained: {len(readback.not_retained)}; "
        f"checkout files never retained: {readback.not_historical}"
    )


def _exit(readback: publish.Readback) -> int:
    return EXIT_READY if readback.outcome.decision == "READY" else EXIT_NOT_READY


def run(
    argv: list[str] | None = None,
    *,
    _policy: ApprovalPolicy | None = None,
    _context: Context | None = None,
) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = _policy if _policy is not None else load_policy(args.repo or Path.cwd())
        if args.command == "evaluate":
            ctx = _context or system_context(args.repo, args.data_root)
            published = publish.publish_report(evaluate(ctx, policy), args.data_root)
            readback = publish.read_report(args.data_root, published.report_sha256, policy)
            print(f"report sha256 {published.report_sha256}")
            print(f"manifest sha256 {published.manifest_sha256}")
        else:
            readback = publish.read_report(args.data_root, args.report_sha256, policy)
            if args.repo is not None:
                for name, state in publish.compare_current(
                    readback.report, args.repo, args.data_root
                ).items():
                    print(f"current {name}: {state}")
    except (
        EvaluationError,
        publish.PublicationError,
        publish.ReadbackError,
        OSError,
        ValueError,
    ) as error:
        print(f"readiness failed: {error}", file=sys.stderr)
        return EXIT_ERROR
    _summary(readback)
    return _exit(readback)


def main(argv: list[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
