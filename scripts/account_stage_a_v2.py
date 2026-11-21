"""Build a development-only retrospective Stage A accounting dry run."""

import argparse
import json
from pathlib import Path

from biorag_repro.core.hashing import sha256_file
from biorag_repro.evaluation.stage_a_accounting_v2 import build_inventory, read_inventory

BASE = Path("artifacts/phase8/stage_a_153_recovery_v1")
QUEUE = Path("artifacts/phase8/stage_a_review_remaining_v1")
AMENDMENT = Path("docs/frozen/AMENDMENT_STAGE_A_RETROSPECTIVE_ACCOUNTING_V2.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("dry-run root already exists; preserve immutable output")
    sources = json.loads((BASE / "verified_103_sources_v2.json").read_bytes())
    originals = {
        handle: BASE / "fresh_full_preflight/drafts" / f"{handle}.json"
        for handle in ("R050", "R116", "R152")
    }
    authorization_paths = [BASE / f"focused_{count}_authorization.json" for count in (58, 13, 32)]
    approvals = [sha256_file(path) for path in authorization_paths]
    approved = {}
    for path in authorization_paths:
        approved.update(json.loads(path.read_bytes())["approved_resolution_sha256_by_handle"])
    if {handle: source["sha256"] for handle, source in sources.items()} != approved:
        raise ValueError("verified source map differs from the three researcher authorizations")
    inventory = build_inventory(
        decisions=BASE
        / "fresh_full_preflight/evaluation/stage_a/retrospective_sources/293f494a3f7a9cc14a68be1fbd9f6f800b8764450f9f983ef514e91b252e1b24.json",
        coordinates=QUEUE / "provenance/unmasked_evidence.json",
        session=QUEUE / "provenance/session_manifest.json",
        authorization=Path(
            "artifacts/phase8/retrospective_identity_pass_v1/decision_a_authorization.json"
        ),
        verified_sources=sources,
        original_drafts=originals,
        root=args.out,
        amendment_sha256=sha256_file(AMENDMENT),
        approval_sha256s=approvals,
    )
    manifest = args.out / "evaluation/stage_a/accounting_v2/inventory.json"
    digest = sha256_file(manifest)
    if read_inventory(manifest, digest) != inventory:
        raise ValueError("inventory readback differs")
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "sha256": digest,
                "by_representation": inventory["by_representation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
