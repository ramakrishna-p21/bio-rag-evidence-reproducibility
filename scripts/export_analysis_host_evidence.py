"""On the designated analysis host, retain its live G07 result for the final-test host."""

import argparse
from pathlib import Path

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.readiness import evidence
from biorag_repro.readiness.evaluate import system_context


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    ctx = system_context(args.repo, args.data_root)
    data = evidence.export_analysis_host_evidence(ctx, evidence.scientific_source(ctx))
    target = args.data_root / evidence.G07_HOST_RECORD
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:  # never replaces earlier evidence
        stream.write(data)
    print(target, sha256_bytes(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
