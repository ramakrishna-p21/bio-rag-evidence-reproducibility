"""Synthetic Drive head requests and off-runtime acknowledgment."""

import json

import pytest
from test_finaltest_history_v2 import batch, root_fixture

from biorag_repro.core.hashing import sha256_bytes, sha256_file
from biorag_repro.finaltest.amended_custody import (
    AmendedCustodyRefused,
    _publish_drive_record,
    publish_request,
    verify_chain,
)
from biorag_repro.finaltest.history_v2 import anchor_batch, verify_history
from biorag_repro.finaltest.layout import root_id_for


def test_incomplete_drive_record_fails_closed_without_replacement(tmp_path):
    target = tmp_path / "head.json"
    content = b'{"head":"complete"}'
    target.write_bytes(content[:8])

    with pytest.raises(AmendedCustodyRefused, match="incomplete"):
        _publish_drive_record(target, content, sha256_bytes(content), reuse_identical=True)

    assert target.read_bytes() == content[:8]


def test_two_synthetic_batches_require_exact_researcher_acknowledgment(tmp_path):
    root, auth, anchors, genesis = root_fixture(tmp_path)
    drive = tmp_path / "drive"
    retained = tmp_path / "retained"
    retained.mkdir()
    previous = genesis
    records = []
    for number in (1, 2):
        batch_id, path = batch(root, number)
        head = anchor_batch(root, auth, anchors, batch_id, previous)
        request = publish_request("rk_capstone_final_test_2026", path, drive, previous, head)
        record = verify_history(root, auth, anchors, head)[-1]
        record["anchor_sha256"] = sha256_file(anchors / f"{root_id_for(auth)}.{number:06d}.json")
        records.append(record)
        with pytest.raises(AmendedCustodyRefused, match="acknowledgment"):
            verify_chain("rk_capstone_final_test_2026", root, drive, records, genesis, head, None)
        acknowledgment = retained / f"simulated_{head}.json"
        acknowledgment.write_text(
            json.dumps(
                {
                    "schema_version": "researcher_head_ack_v1",
                    "researcher": "Rama Krishna Pudota",
                    "request_sha256": sha256_file(request),
                    "campaign_id": "rk_capstone_final_test_2026",
                    "batch_id": batch_id,
                    "new_head_sha256": head,
                }
            )
        )
        verify_chain(
            "rk_capstone_final_test_2026", root, drive, records, genesis, head, acknowledgment
        )
        with pytest.raises(AmendedCustodyRefused):
            verify_chain("wrong_campaign", root, drive, records, genesis, head, acknowledgment)
        previous = head
    with pytest.raises(AmendedCustodyRefused, match="stale|rolled back"):
        verify_chain(
            "rk_capstone_final_test_2026", root, drive, records, genesis, genesis, acknowledgment
        )
