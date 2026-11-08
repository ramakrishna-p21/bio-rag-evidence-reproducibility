"""Run every gate adapter and assemble the readiness report.

Three outcomes stay apart. A report with decision NOT_READY is a successful evaluation. An
`EvaluationError` means no trustworthy report could be built (an adapter crashed, the report
failed its own validation); no report exists and nothing is published. READY is possible only
under a policy that approves every gate, which the real policy does not.
"""

import platform
import secrets
import socket
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.generation.environment import GpuDevice, system_probes, utc_now
from biorag_repro.generation.environment_lock import capture_inventory
from biorag_repro.readiness import evidence
from biorag_repro.readiness.decide import build_report
from biorag_repro.readiness.evidence import Context, Probes
from biorag_repro.readiness.files import FileProblem
from biorag_repro.readiness.gather import Gathered, to_result
from biorag_repro.readiness.models import (
    MANDATORY_GATES,
    Candidate,
    EvaluatorIdentity,
    GateId,
    GateResult,
    HostBinding,
    ReadinessReport,
)
from biorag_repro.readiness.policy import ApprovalPolicy, policy_sha256

EVALUATOR_VERSION = "p10b-1"


class EvaluationError(RuntimeError):
    """The evaluator could not produce a trustworthy report."""


@dataclass(frozen=True)
class Evaluation:
    report: ReadinessReport
    snapshots: dict[str, bytes]  # canonical JSON observations by SHA-256, including the policy


def evaluator_identity() -> EvaluatorIdentity:
    """Hash of the source files that decide readiness: the readiness package and the
    final-test root verifier that G13 depends on."""
    data: dict[str, str] = {}
    for name in ("readiness", "finaltest"):
        package = resources.files(f"biorag_repro.{name}")
        for path in sorted(package.iterdir(), key=lambda p: p.name):
            if path.name.endswith(".py"):
                data[f"{name}/{path.name}"] = sha256_bytes(path.read_bytes())
    return EvaluatorIdentity(
        version=EVALUATOR_VERSION, code_sha256=sha256_bytes(canonical_json_bytes(data))
    )


def system_context(repo: Path, data_root: Path) -> Context:
    probes = Probes(
        inventory=cache(lambda: capture_inventory(system_probes(repo))),
        r_runtime=evidence.run_r_runtime,
    )
    return Context(repo=repo, data_root=data_root, probes=probes)


def _host(ctx: Context) -> HostBinding:
    try:
        gpus: list[GpuDevice] = list(ctx.probes.inventory().gpus)
    except Exception:  # noqa: BLE001 - the host record is informational
        gpus = []
    return HostBinding(
        hostname=socket.gethostname(),
        operating_system=platform.system(),
        machine=platform.machine(),
        gpu_names=sorted({g.name for g in gpus}),
    )


def gather_all(ctx: Context, policy: ApprovalPolicy | None = None) -> dict[GateId, Gathered]:
    problem: str | None = None
    release = release_bytes = None
    try:
        release, release_bytes = evidence.read_benchmark_record(ctx)
    except (FileProblem, ValueError) as error:
        problem = str(error)[:300]
    return {
        "G01_corpus_sources": evidence.corpus_sources(ctx, release, problem),
        "G02_benchmark_freeze": evidence.benchmark_freeze(ctx, release, release_bytes, problem),
        "G03_final_test_configuration_freeze": evidence.final_test_freeze(),
        "G04_generator_qualification": evidence.generator_qualification(),
        "G05_model_parser_identities": evidence.model_parser_identities(),
        "G06_python_environment": evidence.python_environment(ctx),
        "G07_r_analysis_environment": evidence.r_environment(ctx),
        "G08_seed_registry": evidence.seed_registry(ctx, release),
        "G09_scientific_source_revision": evidence.scientific_source(ctx),
        "G10_development_restart_recovery": evidence.restart_recovery(),
        "G11_storage_authorization": evidence.storage_authorization(),
        "G12_shared_gpu_authorization": evidence.shared_gpu_authorization(),
        "G13_final_test_history_integrity": evidence.history_integrity(ctx, policy),
    }


def _candidate(found: dict[GateId, Gathered]) -> Candidate:
    facts: dict[str, Any] = {}
    for gathered in found.values():
        facts.update({k: v for k, v in gathered.facts.items() if v is not None})
    source = found["G09_scientific_source_revision"]
    identity = None
    if source.subject is not None:
        identity = sha256_bytes(canonical_json_bytes(source.subject))
    return Candidate(
        head_sha=facts.get("head_sha"),
        scientific_source_identity_sha256=identity,
        freeze_record_sha256=None,
        configuration_sha256_by_arm={},
        benchmark_release_sha256=facts.get("benchmark_release_sha256"),
        corpus_content_sha256=facts.get("corpus_content_sha256"),
        python_lock_sha256=facts.get("python_lock_sha256"),
        renv_lock_sha256=facts.get("renv_lock_sha256"),
        generation_seed_registry_sha256=None,
        benchmark_seed_registry_sha256=facts.get("benchmark_seed_registry_sha256"),
    )


def evaluate(
    ctx: Context,
    policy: ApprovalPolicy,
    *,
    now: Callable[[], str] = utc_now,
    nonce: Callable[[], str] = lambda: secrets.token_hex(16),
) -> Evaluation:
    try:
        found = gather_all(ctx, policy)
        results: list[GateResult] = [to_result(g, found[g], policy) for g in MANDATORY_GATES]
        report = build_report(
            policy=policy,
            results=results,
            candidate=_candidate(found),
            host=_host(ctx),
            evaluator=evaluator_identity(),
            warnings=ctx.warnings,
            created_at_utc=now(),
            nonce=nonce(),
        )
    except Exception as error:
        raise EvaluationError(f"{type(error).__name__}: {str(error)[:300]}") from error
    snapshots: dict[str, bytes] = {}
    for gathered in found.values():
        snapshots.update(gathered.snapshots)
    policy_bytes = canonical_json_bytes(policy.model_dump(mode="json"))
    snapshots[policy_sha256(policy)] = policy_bytes
    return Evaluation(report, snapshots)
