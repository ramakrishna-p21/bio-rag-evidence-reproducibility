"""Verify a published analysis table, or a rehearsal analysis handoff.

`verify-handoff` checks a rehearsal handoff only. It cannot establish official provenance:
that needs freshly verified official runs, which this command does not have.
"""

import argparse
import sys
from pathlib import Path

from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_manifest
from biorag_repro.evaluation.analysis_handoff import OfficialProvenanceNotVerified, read_handoff
from biorag_repro.evaluation.analysis_storage import read_table


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

    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
