"""The approved analysis seed, read from its exact registered bytes."""

from pathlib import Path
from typing import Literal

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.records import Record
from biorag_repro.generation.environment import check_utc
from biorag_repro.readiness.files import loads_strict, read_regular

REGISTRY_PATH = "configs/analysis_seed.json"
APPROVED_REGISTRY_SHA256 = "c46bd692e612628af19f305de8ee118a24a656d945768d629aefa6a8ff66a6c9"


class AnalysisSeed(Record):
    schema_version: Literal["analysis_seed_v1"]
    name: Literal["primary_bootstrap"]
    value: int
    resamples: Literal[10000]
    method: Literal["whole_paper_percentile"]
    rng_algorithm: Literal["Mersenne-Twister"]
    normal_kind: Literal["Inversion"]
    sample_kind: Literal["Rejection"]
    experiment: Literal["EXPERIMENT_PLAN sections 6-9"]
    authority: Literal["Rama Krishna Pudota"]
    protocol_version: Literal["researcher_authority_v1"]
    decision_record_sha256: str
    registered_at_utc: str


def read_analysis_seed(repo: Path) -> AnalysisSeed:
    data = read_regular(repo, REGISTRY_PATH)
    if sha256_bytes(data) != APPROVED_REGISTRY_SHA256:
        raise ValueError("analysis seed bytes differ from the approved registry")
    seed = AnalysisSeed.model_validate(loads_strict(data))
    check_utc(seed.registered_at_utc)
    if not 0 <= seed.value <= 2147483647:
        raise ValueError("analysis seed is outside R's integer range")
    decision = read_regular(repo, "approvals/researcher_decisions.json")
    if sha256_bytes(decision) != seed.decision_record_sha256:
        raise ValueError("analysis seed decision record differs")
    return seed
