"""Initialize the approved final-test root and print its verified genesis identity."""

import argparse
import json
from pathlib import Path

from biorag_repro.finaltest.genesis import initialize_final_test_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--expected-genesis-head-sha256", required=True)
    args = parser.parse_args()
    result = initialize_final_test_root(
        args.repo,
        args.data_root,
        decision_sha256=args.decision_sha256,
        expected_genesis_head_sha256=args.expected_genesis_head_sha256,
    )
    print(
        json.dumps(
            {
                "anchor": str(result.anchor),
                "decision_sha256": result.decision_sha256,
                "genesis_head_sha256": result.genesis_head_sha256,
                "root": str(result.root),
                "root_id": result.root_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
