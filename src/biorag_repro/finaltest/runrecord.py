"""Run record of a future final-test batch run.

Schema only. No `RunStore` constructor accepts this record and nothing here creates a run;
`RunRecord` and `RunPhase` keep their development and qualification meaning. The record exists
so that the G13 verifier can name what a stored final-test run must say about itself. Its
`authorization_citation_sha256` is a reference to a historical assessment, not evidence that
the run was permitted.

`purpose` has the values of `GeneratorConfiguration.purpose` and names the *role of the
configuration*, not the stage of the experiment. The frozen final-test candidate is the
configuration that development qualified, so it carries `development_candidate` and keeps the
resolved-configuration hash that qualification used; no `final_test` purpose exists (decision
of 2026-11-08). The stage is `phase`, and the kind of storage is `root_kind`: a rehearsal root
holds non-official runs only, a final-test root holds official runs only, so a rehearsal record
can never be `official`.
"""

from typing import Annotated, Literal, Self

from pydantic import StringConstraints, field_validator, model_validator

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.finaltest.batches import Nonblank, Sha256, batch_id_for
from biorag_repro.finaltest.inputs import retry_policy_sha256 as retry_digest
from biorag_repro.finaltest.layout import FrozenRecord, RootKind, serialize
from biorag_repro.generation.attempts import (
    NonOfficialBackend,
    RetryPolicy,
    RunRecord,
    _check_backend_purpose,
)
from biorag_repro.generation.environment import BackendKind, check_utc

OpaqueId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]


class FinalTestRunRecord(FrozenRecord):
    schema_version: Literal["1"] = "1"
    run_id: OpaqueId
    created_utc: str  # informational
    phase: Literal["final_test"] = "final_test"
    root_id: Sha256
    root_kind: RootKind
    campaign_plan_sha256: Sha256
    batch_id: Sha256
    benchmark_version: Nonblank
    corpus_content_sha256: Sha256
    benchmark_release_sha256: Sha256
    matrix_arm: Nonblank
    resolved_config_sha256: Sha256
    raw_config_sha256: Sha256
    purpose: Literal["synthetic_test", "development_candidate"]
    backend_kind: BackendKind
    retry: RetryPolicy
    retry_policy_sha256: Sha256
    inputs_set_sha256: Sha256
    authorization_citation_sha256: Sha256

    _utc = field_validator("created_utc")(check_utc)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.batch_id != batch_id_for(self.root_id, self.campaign_plan_sha256, self.matrix_arm):
            raise ValueError("batch_id does not match root, plan and arm")
        if self.retry_policy_sha256 != retry_digest(self.retry):
            raise ValueError("retry_policy_sha256 does not match the retry policy")
        try:
            _check_backend_purpose(self.backend_kind, self.purpose)
        except NonOfficialBackend as error:
            raise ValueError(str(error)) from error
        if self.root_kind == "final_test" and self.backend_kind != "official":
            raise ValueError("a final-test root holds official runs only")
        if self.root_kind == "rehearsal" and self.backend_kind == "official":
            raise ValueError("a rehearsal root never holds an official run")
        return self

    @property
    def official(self) -> bool:
        return self.root_kind == "final_test" and self.backend_kind == "official"


def run_record_sha256(record: FinalTestRunRecord) -> str:
    """Identity cited by every event of the run (`AttemptEvent.run_record_sha256`)."""
    return sha256_bytes(serialize(record))


def run_record_projection(record: FinalTestRunRecord) -> RunRecord:
    """The `RunRecord` the store's key computation reads: the same run with phase `final_test`.

    Every field is validated by constructing a development record first. Only `phase` is then
    replaced, which `model_copy` does not validate; `RunPhase` has no `final_test` value and is
    not widened. The result must only reach stores built for final-test runs and the read-only
    view; it is never serialized and must not be passed to `RunStore.create` or `RunStore.open`.
    """
    development = RunRecord(
        run_id=record.run_id,
        created_utc=record.created_utc,
        phase="development",
        benchmark_version=record.benchmark_version,
        resolved_config_sha256=record.resolved_config_sha256,
        raw_config_sha256=record.raw_config_sha256,
        purpose=record.purpose,
        matrix_arm=record.matrix_arm,
        backend_kind=record.backend_kind,
        retry=record.retry,
    )
    return development.model_copy(update={"phase": "final_test"})
