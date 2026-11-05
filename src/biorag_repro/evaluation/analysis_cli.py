"""Verify a published analysis table (bytes, rows, inventory and cited sources)."""

import argparse
import sys
from pathlib import Path

from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_manifest
from biorag_repro.evaluation.analysis_storage import read_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["verify"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--build", required=True, help="logical_ref of the build manifest")
    parser.add_argument("--sha256", required=True, help="SHA-256 of the build manifest")
    args = parser.parse_args(argv)
    build = ArtifactRef(
        artifact_id="build_" + args.sha256, logical_ref=args.build, sha256=args.sha256
    )
    try:
        rows = read_table(build, args.root)
    except (ValueError, OSError) as error:
        print(f"analysis table is not valid: {error}", file=sys.stderr)
        return 1
    manifest = read_manifest(args.root / args.build, args.sha256)
    table = next(o.artifact for o in manifest.outputs if o.artifact_kind == "analysis_table")
    print(f"valid analysis table: {len(rows)} rows, csv sha256 {table.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
