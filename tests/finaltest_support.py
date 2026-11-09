"""Synthetic final-test storage fixtures. Nothing here touches a real root or a real policy."""

import hashlib
import json
from pathlib import Path

from biorag_repro.finaltest.inventory import initialize_root
from biorag_repro.finaltest.layout import StorageAuthorization

CORPUS = hashlib.sha256(b"corpus").hexdigest()
RELEASE = hashlib.sha256(b"release").hexdigest()


def authorization(base: Path, kind: str = "rehearsal", **changes: object) -> StorageAuthorization:
    values: dict[str, object] = {
        "schema_version": "1",
        "layout_version": "final_test_v1",
        "root_kind": kind,
        "authorized_root": str(base / "final_test_v1"),
        "storage_label": "synthetic storage",
        "corpus_content_sha256": CORPUS,
        "benchmark_release_sha256": RELEASE,
        "matrix_id": "experiment_plan_v1",
    }
    return StorageAuthorization.model_validate(values | changes)


def made(base: Path, kind: str = "rehearsal"):
    """An initialized synthetic root: (root, anchor_dir, authorization)."""
    auth = authorization(base, kind)
    root, anchors = Path(auth.authorized_root), base / "anchors"
    initialize_root(root, auth, anchors)
    return root, anchors, auth


def oracle_chain_ok(registry: Path) -> bool:
    """Independent check of the registry chain: plain hashlib and json, no project code."""
    files = sorted(registry.glob("[0-9]*.json"))
    previous_bytes = None
    for number, path in enumerate(files, start=1):
        data = path.read_bytes()
        event = json.loads(data)
        if event["sequence"] != number or path.name != f"{number:06d}.json":
            return False
        expected_previous = (
            None if previous_bytes is None else hashlib.sha256(previous_bytes).hexdigest()
        )
        if event["previous_event_sha256"] != expected_previous:
            return False
        identity = {
            key: event[key]
            for key in (
                "schema_version",
                "layout_version",
                "root_id",
                "sequence",
                "event_type",
                "previous_event_sha256",
                "evidence",
            )
        }
        text = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if hashlib.sha256(text.encode()).hexdigest() != event["identity_sha256"]:
            return False
        previous_bytes = data
    return bool(files)
