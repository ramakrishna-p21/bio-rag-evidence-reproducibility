"""One-time initialization of the exact researcher-authorized final-test root."""

from dataclasses import dataclass
from pathlib import Path

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.finaltest import inventory
from biorag_repro.finaltest.authorization import campaign_preflight
from biorag_repro.finaltest.history_v2 import verify_history
from biorag_repro.finaltest.layout import root_id_for
from biorag_repro.readiness import evidence


class GenesisRefused(ValueError):
    """The production genesis act is not exactly authorized or independently bound."""


@dataclass(frozen=True)
class GenesisResult:
    root_id: str
    root: Path
    anchor: Path
    genesis_head_sha256: str
    decision_sha256: str


def initialize_final_test_root(
    repo: Path,
    data_root: Path,
    *,
    decision_sha256: str,
    expected_genesis_head_sha256: str,
) -> GenesisResult:
    """Publish or read back one approved genesis, then verify its exact expected head.

    The expected head is an explicit custody input. After the first publication, its anchor
    bytes still have to be copied and read back at the approved off-runtime custody location;
    this function cannot claim that an inaccessible external copy exists.
    """
    preflight = campaign_preflight(repo, data_root, custody_mode="amended_drive")
    observed_decision_sha256 = sha256_bytes(preflight.decision_bytes)
    if decision_sha256 != observed_decision_sha256:
        raise GenesisRefused("explicit decision SHA-256 differs from the tracked decision")
    if preflight.storage.root_kind != "final_test":
        raise GenesisRefused("the approved storage authorization is not for a final_test root")
    decision = preflight.decision
    if expected_genesis_head_sha256 != decision.genesis_anchor_sha256:
        raise GenesisRefused("expected custody head differs from the researcher decision")
    records = inventory.genesis_records(preflight.storage, decision.genesis_created_utc)
    if sha256_bytes(records.anchor) != expected_genesis_head_sha256:
        raise GenesisRefused("authorized genesis bytes do not produce the expected custody head")

    root = Path(preflight.storage.authorized_root)
    anchors = data_root / evidence.ROOT_ANCHORS
    try:
        verdict = inventory._initialize(
            root,
            preflight.storage,
            anchors,
            lambda: decision.genesis_created_utc,
        )
        verify_history(root, preflight.storage, anchors, expected_genesis_head_sha256)
    except (
        inventory.AuthorityNotEstablished,
        inventory.InitializationRefused,
        ValueError,
    ) as error:
        raise GenesisRefused(str(error)) from error
    if verdict.anchor_sha256 != expected_genesis_head_sha256:
        raise GenesisRefused("initialized root readback differs from the expected custody head")
    root_id = root_id_for(preflight.storage)
    return GenesisResult(
        root_id=root_id,
        root=root,
        anchor=anchors / f"{root_id}.anchor.json",
        genesis_head_sha256=expected_genesis_head_sha256,
        decision_sha256=decision_sha256,
    )
