"""Reproduce a frozen pair as a candidate; complete publication is unavailable."""

import argparse
import json
from pathlib import Path

from biorag_repro.core.config import resolve_config
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_project_json
from biorag_repro.documents.authorization import (
    PHASE2H_SHA256,
    QUALIFIED_ENVIRONMENT,
    ScientificAuthority,
)
from biorag_repro.documents.models import JatsConfiguration
from biorag_repro.documents.pdf import PdfConfiguration, verify_assets, verify_qualified_lock
from biorag_repro.documents.phase4 import Phase4GateIndex
from biorag_repro.documents.reproduction import reproduce_pair


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", choices=["candidate", "complete"], default="candidate")
    preliminary, _ = parser.parse_known_args()
    if preliminary.publication == "complete":
        parser.error("independent human authorization is unavailable in production")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--release-record", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reference-index", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--native-attempt-id", required=True)
    parser.add_argument("--native-attempt-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--jats-config", type=Path, default=Path("configs/jats.toml"))
    parser.add_argument("--pdf-config", type=Path, default=Path("configs/pdf_docling.toml"))
    args = parser.parse_args()
    import re

    if re.fullmatch("[0-9a-f]{32}", args.native_attempt_id) is None:
        parser.error("native attempt ID must be 32 lowercase hexadecimal characters")
    jats = resolve_config(args.jats_config.read_bytes(), JatsConfiguration)
    pdf = resolve_config(args.pdf_config.read_bytes(), PdfConfiguration)
    verify_qualified_lock(args.lock_path, pdf.settings)
    assets = verify_assets(args.assets_root, pdf.settings)
    attempt = ArtifactRef(
        artifact_id="attempt_" + args.native_attempt_id,
        logical_ref=f"corpus/derived/attempts/{args.native_attempt_id}.json",
        sha256=args.native_attempt_sha256,
    )
    authority = ScientificAuthority(
        args.release_record,
        PHASE2H_SHA256,
        attempt,
        jats.configuration_sha256,
        pdf.configuration_sha256,
        QUALIFIED_ENVIRONMENT,
        assets,
    )
    reference = read_project_json(args.reference_index, args.reference_sha256, Phase4GateIndex)
    ref = reproduce_pair(
        reference,
        args.reference_root,
        args.data_root,
        args.corpus_root,
        authority,
        Path(__file__).resolve().parents[1],
    )
    print(
        json.dumps(
            {
                "candidate_attestation": ref.model_dump(mode="json"),
                "status": "validated_candidate_pending_human_review",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
