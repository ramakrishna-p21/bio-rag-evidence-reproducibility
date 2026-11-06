"""The only path from a generation run to official development evidence.

A result qualifies only if it comes from a completed run whose records agree with a
retained completion anchor, whose run and every attempt declare the official backend,
whose configuration names the approved tokenizer, and whose roster cites the verified
development-only benchmark view. A bare `GenerationResult`, a backend label or a mock
wrapper proves nothing and is refused by `check_official_lineage`.

What this cannot prove yet: that the backend really loaded the Qwen3-8B weights. The
environment record names the model and tokenizer revisions the backend reports but
carries no hash of loaded weight files. `OFFICIAL_BACKENDS` is empty until the real
backend exists and records that evidence, so no run qualifies before then.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.core.config import ResolvedConfiguration
from biorag_repro.core.storage import read_project_json
from biorag_repro.evaluation.stage_a import item_digest
from biorag_repro.evaluation.stage_c import generation_digest
from biorag_repro.evaluation.stage_c_models import GenerationResult
from biorag_repro.generation.attempts import LifecycleError, NonOfficialBackend, RunRecord
from biorag_repro.generation.completion import verify_completed_run
from biorag_repro.generation.config import GeneratorConfiguration
from biorag_repro.generation.environment import EnvironmentRecord
from biorag_repro.generation.runner import decided_results
from biorag_repro.retrieval import context as phase5

# Names of backends accepted as the official generator. Deliberately empty: the Hugging Face
# backend does not exist yet, and nothing may qualify on a caller's say-so.
OFFICIAL_BACKENDS: frozenset[str] = frozenset()


class NotVerifiedOfficial(LifecycleError):
    """A result is not part of a verified official run."""


@dataclass(frozen=True)
class VerifiedOfficialRun:
    run_id: str
    manifest_sha256: str
    results: list[GenerationResult]
    digests: frozenset[str]


def verify_official_run(
    root: Path,
    run_id: str,
    manifest_sha256: str,
    *,
    config: ResolvedConfiguration[GeneratorConfiguration],
    view_directory: Path,
    view_record_sha256: str,
) -> VerifiedOfficialRun:
    run_path = root / "runs" / run_id / "run.json"
    try:
        record = RunRecord.model_validate_json(run_path.read_bytes())
    except (OSError, ValueError) as error:
        raise NonOfficialBackend(f"run record of {run_id} is unreadable") from error
    if not record.official:
        raise NonOfficialBackend(
            f"a {record.backend_kind} run with a {record.purpose} configuration is not official"
        )
    # The peeked record is not trusted by itself: its hash is bound by the completion anchor.
    completed = verify_completed_run(
        root, run_id, manifest_sha256, config=config, backend_kind=record.backend_kind
    )

    settings = config.settings
    if (
        settings.tokenizer.revision != phase5.REVISION
        or settings.tokenizer.file_sha256 != phase5.ASSETS
    ):
        raise NonOfficialBackend("the configuration does not name the approved tokenizer")
    for digest in completed.completion.environments:
        environment = read_project_json(
            root / "runs" / run_id / "environments" / f"{digest}.json", digest, EnvironmentRecord
        )
        backend = environment.backend
        if backend.kind != "official" or backend.name not in OFFICIAL_BACKENDS:
            raise NonOfficialBackend(
                f"backend {backend.name!r} is not a recognised official backend"
            )

    view, items = read_development_view(view_directory, view_record_sha256)
    roster = completed.roster
    if (roster.development_items, roster.benchmark_version, roster.corpus_version) != (
        view.output_items,
        view.benchmark_version,
        view.corpus_version,
    ):
        raise NonOfficialBackend("the roster does not cite this development view")
    by_id = {item.item_id: item for item in items}
    for row in roster.rows:
        item = by_id.get(row.item_id)
        if item is None:
            raise NonOfficialBackend("a rostered item is not in the development view")
        if (row.paper_id, row.item_sha256) != (item.paper_id, item_digest(item)):
            raise NonOfficialBackend("a rostered item differs from the view record")

    results = decided_results(completed.store)
    return VerifiedOfficialRun(
        run_id, manifest_sha256, results, frozenset(generation_digest(r) for r in results)
    )


def check_official_lineage(
    results: Iterable[GenerationResult], verified: VerifiedOfficialRun
) -> None:
    """Every result must be byte-identical to one produced by the verified run."""
    for result in results:
        if generation_digest(result) not in verified.digests:
            raise NotVerifiedOfficial(
                f"result for {result.observation_key} is not part of verified run {verified.run_id}"
            )
