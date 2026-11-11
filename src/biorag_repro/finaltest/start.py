"""Guard-connected start of one final-test batch.

Two entry points, deliberately different in kind.

`start_batch(data_root, report_sha256, repo=..., matrix_arm=...)` is the production route. It
takes no policy, evaluator, authorization object, flag or path other than the data root and
the repository. It calls the readiness guard (`authorize_final_test`) and then refuses:
`PRODUCTION_START` is False and no production request builder exists, so even a guard result
of READY ends in `ProductionStartRefused` before any file is written, any lock taken or any
protected input read. Today the guard itself refuses first (no approval is set, G13 is
unavailable), and that refusal is what callers see.

`_start_with` is the orchestration itself, written once. It takes an `assess` callable and an
open `Session`, accepts rehearsal roots only (`allowed_root_kinds`), and is used by tests with
a fixture assessor built on the existing readiness test seam. The same refusal of root kinds is
made again by every publisher it calls (`registry`, `store`) and by the verifier, so
removing the check here does not open a final-test root. None of this is a security boundary
against code that can edit this package: it makes the intended route the only convenient
one, and the evidence it leaves (root kind, mock backend, fixture policy hash) tells a
rehearsal from a real record.

Order of one start, inside the session and the registry lock, nothing written before step 5:

    1. verify the whole root (only the unestablished-head note may remain)
    2. run the guard, then check the registry head did not move while it ran
    3. render every observation and build rosters, binding, citation and run record in
       memory; verify them against the plan, manifest, configuration and seeds
    4. reserve and materialize the batch (existing protocol, manifest decided in advance)
    5. `batch_inputs_bound`, then the roster files, then the binding file
    6. `batch_authorized`, then the citation file
    7. create the run directory with the record bytes the event already cites

Hash dependencies are acyclic: roster <- binding <- citation <- run record, and each registry
event cites the digest of the file it announces, decided before the event is written; the run
record cites the citation and the binding, never the reverse.

The guard result is read, copied into the citation and dropped; it is never a token. The guard
runs against the genesis head only in a real root, because the first appended event makes G13
unavailable for good. That is why production resume, a second batch and any later
authorization stay impossible, and why the citation records `head_authority`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn

from biorag_repro.core.config import ResolvedConfiguration
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import new_run_id
from biorag_repro.finaltest import layout, registry
from biorag_repro.finaltest.batches import (
    AuthorizationCitation,
    batch_id_for,
    citation_sha256,
    make_manifest,
)
from biorag_repro.finaltest.campaign import CampaignPlan, Population, plan_sha256
from biorag_repro.finaltest.continuity import BatchScope, scientific_identity_sha256
from biorag_repro.finaltest.execution import SessionGate, StartedBatch
from biorag_repro.finaltest.inputs import (
    binding_sha256,
    build_binding,
    retry_policy_sha256,
    verify_binding,
)
from biorag_repro.finaltest.inventory import HEAD_NOT_ESTABLISHED_TEXT, verify_root
from biorag_repro.finaltest.layout import EvidenceCitation, StorageAuthorization
from biorag_repro.finaltest.protected import (
    ProtectedInputError,
    ProtectedInputs,
    build_rosters,
    facts_by_key,
)
from biorag_repro.finaltest.runrecord import FinalTestRunRecord, run_record_sha256
from biorag_repro.finaltest.session import Session
from biorag_repro.finaltest.store import create_final_test_store
from biorag_repro.generation.attempts import RetryPolicy, retry_policy_of
from biorag_repro.generation.config import GeneratorConfiguration, resolve_generator_config
from biorag_repro.generation.environment import BackendKind, utc_now
from biorag_repro.generation.prompt import InputRenderer
from biorag_repro.generation.seeds import GenerationSeedRegistry, check_seeds_for_configuration
from biorag_repro.readiness import guard, publish
from biorag_repro.readiness.approvals import load_policy
from biorag_repro.readiness.guard import FinalTestAuthorization, authorize_final_test
from biorag_repro.readiness.models import ReadinessReport

PRODUCTION_START: Final = False


class StartRefused(Exception):
    """The start was refused. `code` says at which check; nothing before step 4 is written."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


class ProductionStartRefused(StartRefused):
    """The production route is closed in this version."""


@dataclass(frozen=True)
class Assessed:
    """What an assessor returns: the guard result and the stored report it was made from."""

    authorization: FinalTestAuthorization
    report: ReadinessReport


@dataclass(frozen=True)
class StartRequest:
    root: Path
    authorization: StorageAuthorization
    anchors: Path
    witnesses: Path
    plan: CampaignPlan
    population: Population
    matrix_arm: str
    config_toml: bytes
    renderer: InputRenderer
    seeds: GenerationSeedRegistry | None
    inputs: ProtectedInputs
    backend_kind: BackendKind = "mock"


def _production_assessed(data_root: Path, report_sha256: str, repo: Path) -> Assessed:
    authorization = authorize_final_test(data_root, report_sha256, repo=repo)
    stored = publish.read_report(data_root, report_sha256, load_policy(repo))
    return Assessed(authorization, stored.report)


def start_batch(data_root: Path, report_sha256: str, *, repo: Path, matrix_arm: str) -> NoReturn:
    """Production start. Always refuses in this version; see the module text."""
    _production_assessed(data_root, report_sha256, repo)  # AuthorizationRefused when not READY
    if not PRODUCTION_START:
        raise ProductionStartRefused("production_disabled", "final-test start is not enabled")
    raise ProductionStartRefused("no_production_request", "no production request builder exists")


def _checked(assessed: Assessed) -> tuple[FinalTestAuthorization, ReadinessReport]:
    authorization, report = assessed.authorization, assessed.report
    if authorization.scope != guard.SCOPE:
        raise StartRefused("guard_scope", authorization.scope)
    if (
        authorization.report_binding_sha256 != report.binding_sha256
        or authorization.policy_sha256 != report.policy_sha256
        or authorization.evaluator_code_sha256 != report.evaluator.code_sha256
    ):
        raise StartRefused("guard_report_mismatch", "the report is not the one the guard assessed")
    if report.candidate.scientific_source_identity_sha256 is None:
        raise StartRefused(
            "source_identity_missing", "the report has no scientific source identity"
        )
    return authorization, report


def _check_request(
    request: StartRequest, session: Session, allowed_root_kinds: tuple[str, ...]
) -> tuple[ResolvedConfiguration[GeneratorConfiguration], RetryPolicy]:
    """Everything that can be refused before the root is read. Shared by start and resume."""
    storage = request.authorization
    root, arm, plan = request.root, request.matrix_arm, request.plan
    if storage.root_kind not in allowed_root_kinds:
        raise StartRefused("root_kind", f"{storage.root_kind} roots are not started here")
    if Path(storage.authorized_root) != root or session.root != root:
        raise StartRefused("root_location", "the root is not the authorized location")
    batch_plan = next((b for b in plan.batches if b.matrix_arm == arm), None)
    if batch_plan is None:
        raise StartRefused("arm", f"the plan has no arm {arm}")
    try:
        resolved = resolve_generator_config(request.config_toml)
    except ValueError as error:
        raise StartRefused("configuration", str(error)[:200]) from error
    settings = resolved.settings
    if (
        resolved.configuration_sha256 != batch_plan.resolved_config_sha256
        or resolved.configuration_sha256 != request.renderer.config.configuration_sha256
        or settings.matrix_arm != arm
    ):
        raise StartRefused("configuration", "not the configuration the plan names for this arm")
    if (plan.benchmark_release_sha256, plan.corpus_content_sha256) != (
        storage.benchmark_release_sha256,
        storage.corpus_content_sha256,
    ):
        raise StartRefused("benchmark_identity", "the plan is for another benchmark than the root")
    population = request.population
    if (
        population.benchmark_release_sha256,
        population.corpus_content_sha256,
        population.split_sha256,
        population.benchmark_version,
    ) != (
        plan.benchmark_release_sha256,
        plan.corpus_content_sha256,
        plan.split_sha256,
        plan.benchmark_version,
    ):
        raise StartRefused("benchmark_identity", "the population is not the one the plan names")
    try:
        if settings.decoding.do_sample and request.seeds is None:
            raise ValueError("sampling decoding needs the seed registry")
        if request.seeds is not None:
            check_seeds_for_configuration(settings, request.seeds)
    except ValueError as error:
        raise StartRefused("seeds", str(error)[:200]) from error
    session.assert_held()
    return resolved, retry_policy_of(settings)


def _start_with(
    request: StartRequest,
    assess: Callable[[], Assessed],
    session: Session,
    *,
    allowed_root_kinds: tuple[str, ...] = ("rehearsal",),
    now: Callable[[], str] = utc_now,
) -> StartedBatch:
    # step 0: nothing is read or written
    storage = request.authorization
    root, arm, plan = request.root, request.matrix_arm, request.plan
    resolved, retry = _check_request(request, session, allowed_root_kinds)
    settings = resolved.settings

    with session.registry() as held:
        # step 1
        verdict = verify_root(root, storage, request.anchors, request.witnesses)
        left = [p for p in verdict.problems if p != HEAD_NOT_ESTABLISHED_TEXT]
        if left:
            raise StartRefused("root_unverified", f"{verdict.code}: {left[:2]}")
        root_id = layout.root_id_for(storage)
        if batch_id_for(root_id, plan_sha256(plan), arm) in verdict.batches:
            raise StartRefused(
                "arm_already_reserved", "an arm has one batch and is never restarted"
            )
        head_before = registry.registry_head(root)
        # step 2
        authorization, report = _checked(assess())
        if registry.registry_head(root) != head_before:
            raise StartRefused("head_changed", "the registry moved while the guard ran")

        # step 3
        manifest = make_manifest(root_id, plan, arm, now())
        try:
            prepared = build_rosters(
                inputs=request.inputs,
                population=request.population,
                matrix_arm=arm,
                batch_id=manifest.batch_id,
                renderer=request.renderer,
                seeds=request.seeds,
            )
            binding = build_binding(
                manifest=manifest,
                plan=plan,
                resolved=resolved,
                retry=retry,
                seeds=request.seeds,
                rosters=prepared.rosters,
            )
        except ValueError as error:  # includes ProtectedInputError and roster mismatches
            code = "inputs" if isinstance(error, ProtectedInputError) else "inputs_invalid"
            raise StartRefused(code, str(error)[:200]) from error
        findings = verify_binding(
            binding,
            plan=plan,
            manifest=manifest,
            item_papers=request.population.item_papers,
            rosters={r.condition: r for r in prepared.rosters},
            resolved=resolved,
            retry=retry,
            seeds=request.seeds,
        )
        if findings:
            raise StartRefused("inputs_invalid", ", ".join(sorted({f.code for f in findings})))
        binding_bytes = layout.serialize(binding)
        binding_sha = binding_sha256(binding)
        head_sequence, head_file_sha = head_before
        scope = BatchScope(
            campaign_plan_sha256=manifest.campaign_plan_sha256,
            batch_id=manifest.batch_id,
            matrix_arm=arm,
            resolved_config_sha256=manifest.resolved_config_sha256,
            ownership_sha256=manifest.ownership,
            key_set_sha256=manifest.key_set_sha256,
            inputs_set_sha256=binding_sha,
        )
        source, head_sha = (
            report.candidate.scientific_source_identity_sha256,
            report.candidate.head_sha,
        )
        assert source is not None and head_sha is not None
        citation = AuthorizationCitation(
            kind="initial",
            batch_id=manifest.batch_id,
            initial_report_sha256=authorization.report_sha256,
            readiness_report_sha256=authorization.report_sha256,
            policy_sha256=authorization.policy_sha256,
            evaluator_code_sha256=authorization.evaluator_code_sha256,
            report_binding_sha256=authorization.report_binding_sha256,
            head_sha=head_sha,
            scientific_source_identity_sha256=source,
            scientific_identity_sha256=scientific_identity_sha256(report, scope),
            registry_head_sequence=head_sequence,
            registry_head_event_sha256=head_file_sha,
            head_authority="genesis_anchored" if head_sequence == 1 else "not_established",
            inputs_set_sha256=binding_sha,
            previous_citation_sha256=None,
            continuity_assessment_sha256=None,
            created_utc=now(),
        )
        citation_bytes, citation_digest = layout.serialize(citation), citation_sha256(citation)
        run_record = FinalTestRunRecord(
            run_id=new_run_id(),
            created_utc=now(),
            root_id=root_id,
            root_kind=storage.root_kind,
            campaign_plan_sha256=manifest.campaign_plan_sha256,
            batch_id=manifest.batch_id,
            benchmark_version=manifest.benchmark_version,
            corpus_content_sha256=manifest.corpus_content_sha256,
            benchmark_release_sha256=manifest.benchmark_release_sha256,
            matrix_arm=arm,
            resolved_config_sha256=resolved.configuration_sha256,
            raw_config_sha256=resolved.raw_toml_sha256,
            purpose=settings.purpose,
            backend_kind=request.backend_kind,
            retry=retry,
            retry_policy_sha256=retry_policy_sha256(retry),
            inputs_set_sha256=binding_sha,
            authorization_citation_sha256=citation_digest,
        )
        run_digest = run_record_sha256(run_record)

        # steps 4 to 7: the first writes
        batch = manifest.batch_id
        args = (held, root, storage)
        registry.reserve_locked(
            *args, request.anchors, request.witnesses, plan, arm, now=now, manifest=manifest
        )
        registry.materialize_locked(*args, request.anchors, request.witnesses, batch, now=now)
        materialized = registry.registry_head(root)[1]
        registry.append_event_locked(
            *args,
            request.witnesses,
            "batch_inputs_bound",
            [
                EvidenceCitation(role="batch_id", sha256=batch),
                EvidenceCitation(role="materialization_event", sha256=materialized),
                EvidenceCitation(role="inputs_binding", sha256=binding_sha),
            ],
            now=now,
        )
        inputs_event = registry.registry_head(root)[1]
        base = f"{layout.BATCHES_DIR}/{batch}"
        roster_paths = {}
        for roster in prepared.rosters:
            data = layout.serialize(roster)
            path = f"{base}/{layout.INPUTS_DIR}/{layout.ROSTERS_DIR}/{sha256_bytes(data)}.json"
            registry.publish_batch_file(held, root, storage, path, data)
            roster_paths[path] = sha256_bytes(data)
        binding_path = f"{base}/{layout.INPUTS_DIR}/{layout.BINDING_NAME}"
        registry.publish_batch_file(held, root, storage, binding_path, binding_bytes)
        registry.append_event_locked(
            *args,
            request.witnesses,
            "batch_authorized",
            [
                EvidenceCitation(role="batch_id", sha256=batch),
                EvidenceCitation(role="inputs_bound_event", sha256=inputs_event),
                EvidenceCitation(role="authorization_citation", sha256=citation_digest),
                EvidenceCitation(role="run_record", sha256=run_digest),
                EvidenceCitation(role="readiness_report", sha256=authorization.report_sha256),
            ],
            now=now,
        )
        citation_path = f"{base}/{layout.CITATIONS_DIR}/{layout.CITATION_NAME}"
        registry.publish_batch_file(held, root, storage, citation_path, citation_bytes)

        # the gate watches exactly what the start published
        head = registry.registry_head(root)
        files = {
            f"{layout.REGISTRY_DIR}/{n:06d}.json": sha256_bytes(
                (root / layout.REGISTRY_DIR / f"{n:06d}.json").read_bytes()
            )
            for n in range(1, head[0] + 1)
        }
        files[f"{base}/{layout.MANIFEST_NAME}"] = sha256_bytes(layout.serialize(manifest))
        files[binding_path] = binding_sha
        files[citation_path] = citation_digest
        files.update(roster_paths)
        files[f"runs/{run_record.run_id}/run.json"] = run_digest
        rows = facts_by_key(prepared)
        session.head = head
        gate = SessionGate(session, files, head, rows)
        store = create_final_test_store(root, run_record, resolved, gate, now=now)
    return StartedBatch(
        session=session,
        store=store,
        batch_id=batch,
        run_id=run_record.run_id,
        binding_sha256=binding_sha,
        citation_sha256=citation_digest,
        run_record_sha256=run_digest,
        report_sha256=authorization.report_sha256,
        head=head,
        rows=rows,
        entries=prepared.entries,
        seeds=request.seeds,
        root=root,
    )
