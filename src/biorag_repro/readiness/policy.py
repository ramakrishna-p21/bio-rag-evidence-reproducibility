"""Tracked approval policy: the only place a human decision can enter the readiness decision.

Each slot holds the SHA-256 of the canonical identity subject a person approved for one gate,
or `None` while nothing is approved. A slot is a statement made in this file, not in the
evidence: a report, fixture or configuration that says "approved" grants nothing. A gate can
pass only when its result names the slot it matched and quotes the slot's current value.

Trust limit: this module is code in a Git repository. Its history gives versioning and
auditability. It is not a signature, and anyone who can change the repository can change the
policy and the evaluator. Entries are added in their own reviewed commits, never by
evaluation, and never in the same commit as code that depends on them.

Every slot is unset. The identity subject each slot digests (for example the corpus
identities, the freeze record or the storage approval) is defined together with the verifier
that produces it, so a slot is not filled before its subject exists. The real policy
therefore cannot authorize final-test execution: no gate that needs a slot can pass.

The values live in the tracked data file `approvals/policy.json`; this module is the schema
and the hash, `approvals.load_policy` reads the file, and both stay inside the G09 source digest. The values file is the one
path excluded from that digest (`git_state.APPROVAL_FILE`), because it holds the G09 slot and
would otherwise be hashed into the value it stores. Its identity is `policy_sha256`, which
every report binds. G09 still requires the file to be tracked and unmodified.

Approval values are frozen with the scientific source before final-test execution. Any later
change gives a different policy hash, so an earlier report no longer verifies under the
current policy and nothing is authorized until a new campaign is reviewed.

There is no synthetic policy here. Tests build their own `ApprovalPolicy`; no command-line
option, environment variable or configuration file selects a policy.
"""

from typing import Annotated, Final

from pydantic import ConfigDict, StringConstraints

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.records import Record
from biorag_repro.readiness.models import MANDATORY_GATES, GateId

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ApprovalSlots(Record):
    """One field per approval. `None` is the explicit unset state."""

    model_config = ConfigDict(frozen=True)

    # G01: the frozen 50-paper corpus: release record, content hash and stored source bytes.
    corpus_identity: Sha256 | None = None
    # G02: the frozen benchmark release, its accepted item set and the 10/40 split.
    benchmark_identity: Sha256 | None = None
    # G03: the final-test freeze record binding every frozen scientific setting.
    final_test_freeze: Sha256 | None = None
    # G04: Phase 7 qualification evidence that is independently checkable. A descriptive
    # report and its hash are not enough: peak GPU memory and full-length-context handling
    # must be established by evidence the qualification interfaces do not record today.
    generator_qualification_acceptance: Sha256 | None = None
    # G05: approved checkpoint file manifest, tokenizer, BGE, Docling and JATS-adapter identities.
    model_parser_identities: Sha256 | None = None
    # G06: the approved Python/Conda environment lock, verified against a live inventory.
    python_environment_lock: Sha256 | None = None
    # G07: the R lock and the designated statistical-analysis host and workflow. This is the
    # environment that will produce official results, which need not be the generator host.
    analysis_environment: Sha256 | None = None
    # G08: approved seed registries, including the named analysis seed with its operation,
    # algorithm and binding to the frozen experiment. No integer is synthetic by value.
    seed_approvals: Sha256 | None = None
    # G09: the exact approved scientific source and configuration identity, or an explicitly
    # reviewed change set with documented compatibility. A clean tree at some descendant
    # commit of an approved revision is not equivalent to it.
    scientific_source_identity: Sha256 | None = None
    # G10: development restart and recovery evidence from a real run, accepted for this purpose.
    restart_evidence_acceptance: Sha256 | None = None
    # G11: approved writable project/model storage for the execution host.
    storage_authorization: Sha256 | None = None
    # G12: advisor or administrator guidance for the intended shared-GPU use.
    gpu_authorization: Sha256 | None = None


class ApprovalPolicy(Record):
    model_config = ConfigDict(frozen=True)

    policy_version: Annotated[str, StringConstraints(pattern=r"\S")]
    slots: ApprovalSlots


SLOT_NAMES: Final[tuple[str, ...]] = tuple(ApprovalSlots.model_fields)

# G13 has no slot: its verifier classifies final-test history (see models.HISTORY_PASS_CODES)
# against the authorized batches rather than against one approved digest.
GATE_SLOT: Final[dict[GateId, str | None]] = {
    "G01_corpus_sources": "corpus_identity",
    "G02_benchmark_freeze": "benchmark_identity",
    "G03_final_test_configuration_freeze": "final_test_freeze",
    "G04_generator_qualification": "generator_qualification_acceptance",
    "G05_model_parser_identities": "model_parser_identities",
    "G06_python_environment": "python_environment_lock",
    "G07_r_analysis_environment": "analysis_environment",
    "G08_seed_registry": "seed_approvals",
    "G09_scientific_source_revision": "scientific_source_identity",
    "G10_development_restart_recovery": "restart_evidence_acceptance",
    "G11_storage_authorization": "storage_authorization",
    "G12_shared_gpu_authorization": "gpu_authorization",
    "G13_final_test_history_integrity": None,
}
assert tuple(GATE_SLOT) == MANDATORY_GATES
assert {s for s in GATE_SLOT.values() if s} == set(SLOT_NAMES)


def policy_sha256(policy: ApprovalPolicy) -> str:
    return sha256_bytes(canonical_json_bytes(policy.model_dump(mode="json")))
