"""Export persistent JSON schemas from their Pydantic models."""

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from biorag_repro.benchmark.audit import AuditDraw, AuditReport
from biorag_repro.benchmark.discovery import CandidateLead
from biorag_repro.benchmark.flags import ReviewFlag
from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.benchmark.pdf import PdfHit
from biorag_repro.benchmark.release import BenchmarkRelease
from biorag_repro.benchmark.review import DecisionEvent, FlagDisposition, WorksheetRow
from biorag_repro.benchmark.seeds import BenchmarkSeedRegistry
from biorag_repro.benchmark.split import PaperSplit, RelatednessRecord
from biorag_repro.benchmark.validation import ValidationReport as BenchmarkValidationReport
from biorag_repro.core.manifest import BuildManifest
from biorag_repro.corpus.manifest import ReleaseRecord, ValidationReport
from biorag_repro.documents.authorization import ReproductionCandidate
from biorag_repro.documents.models import CanonicalDocument, JatsConfiguration
from biorag_repro.documents.pdf import PdfConfiguration
from biorag_repro.documents.pdf_ingest import ConversionAttempt, PdfCanonicalBuild
from biorag_repro.documents.phase4 import JatsBuild, Phase4GateIndex, Phase4Validation
from biorag_repro.evaluation.analysis_models import AnalysisInventory, AnalysisRow, GeneratorInput
from biorag_repro.evaluation.models import StageACandidate, StageAJudgment
from biorag_repro.evaluation.stage_b_models import StageBCandidate, StageBJudgment
from biorag_repro.evaluation.stage_c_models import GenerationResult, StageCScore
from biorag_repro.generation.attempts import AttemptEvent, RunRecord
from biorag_repro.generation.checkpoint import CheckpointInventory, CheckpointManifest
from biorag_repro.generation.completion import RunCompletion
from biorag_repro.generation.config import GeneratorConfiguration
from biorag_repro.generation.driver import Anchor
from biorag_repro.generation.environment import EnvironmentRecord
from biorag_repro.generation.environment_lock import (
    EnvironmentInventory,
    EnvironmentSpec,
    EnvironmentVerification,
)
from biorag_repro.generation.inputs import GeneratorInputRecord
from biorag_repro.generation.prompt import PromptArtifact, RenderedInput
from biorag_repro.generation.receipt import ExecutionReceipt
from biorag_repro.generation.roster import RosterRecord
from biorag_repro.generation.seeds import GenerationSeedRegistry
from biorag_repro.generation.sources import ContextSourceSpec
from biorag_repro.retrieval.models import (
    Bm25Index,
    CanonicalInput,
    Chunk,
    ChunkSet,
    ContextRecord,
    DenseIndex,
    Phase5Configuration,
    RankedRetrieval,
)

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"
SCHEMAS: dict[str, type[BaseModel]] = {
    "stage_a_candidate.schema.json": StageACandidate,
    "stage_a_judgment.schema.json": StageAJudgment,
    "stage_b_candidate.schema.json": StageBCandidate,
    "stage_b_judgment.schema.json": StageBJudgment,
    "generation_result.schema.json": GenerationResult,
    "stage_c_score.schema.json": StageCScore,
    "analysis_row.schema.json": AnalysisRow,
    "analysis_inventory.schema.json": AnalysisInventory,
    "generator_input.schema.json": GeneratorInput,
    "generator_configuration.schema.json": GeneratorConfiguration,
    "generator_prompt.schema.json": PromptArtifact,
    "generator_rendered_input.schema.json": RenderedInput,
    "generation_seed_registry.schema.json": GenerationSeedRegistry,
    "checkpoint_inventory.schema.json": CheckpointInventory,
    "checkpoint_manifest.schema.json": CheckpointManifest,
    "generation_run.schema.json": RunRecord,
    "generation_attempt_event.schema.json": AttemptEvent,
    "generation_environment.schema.json": EnvironmentRecord,
    "environment_inventory.schema.json": EnvironmentInventory,
    "environment_spec.schema.json": EnvironmentSpec,
    "environment_verification.schema.json": EnvironmentVerification,
    "generation_roster.schema.json": RosterRecord,
    "generation_run_completion.schema.json": RunCompletion,
    "generation_input_record.schema.json": GeneratorInputRecord,
    "generation_execution_receipt.schema.json": ExecutionReceipt,
    "generation_run_anchor.schema.json": Anchor,
    "generation_context_sources.schema.json": ContextSourceSpec,
    "phase5_configuration.schema.json": Phase5Configuration,
    "canonical_input.schema.json": CanonicalInput,
    "chunk.schema.json": Chunk,
    "chunk_set.schema.json": ChunkSet,
    "bm25_index.schema.json": Bm25Index,
    "dense_index.schema.json": DenseIndex,
    "ranked_retrieval.schema.json": RankedRetrieval,
    "context_record.schema.json": ContextRecord,
    "phase4_reproduction_candidate.schema.json": ReproductionCandidate,
    "benchmark_item.schema.json": BenchmarkItem,
    "benchmark_split.schema.json": PaperSplit,
    "benchmark_decision.schema.json": DecisionEvent,
    "benchmark_flag_disposition.schema.json": FlagDisposition,
    "benchmark_worksheet.schema.json": WorksheetRow,
    "benchmark_audit_draw.schema.json": AuditDraw,
    "benchmark_audit_report.schema.json": AuditReport,
    "benchmark_validation_report.schema.json": BenchmarkValidationReport,
    "benchmark_release.schema.json": BenchmarkRelease,
    "benchmark_seed_registry.schema.json": BenchmarkSeedRegistry,
    "benchmark_relatedness.schema.json": RelatednessRecord,
    "benchmark_candidate_lead.schema.json": CandidateLead,
    "benchmark_review_flag.schema.json": ReviewFlag,
    "benchmark_pdf_hit.schema.json": PdfHit,
    "build_manifest.schema.json": BuildManifest,
    "corpus_validation_report.schema.json": ValidationReport,
    "corpus_release_record.schema.json": ReleaseRecord,
    "canonical_document.schema.json": CanonicalDocument,
    "jats_configuration.schema.json": JatsConfiguration,
    "pdf_configuration.schema.json": PdfConfiguration,
    "pdf_conversion_attempt.schema.json": ConversionAttempt,
    "pdf_canonical_build.schema.json": PdfCanonicalBuild,
    "phase4_jats_build.schema.json": JatsBuild,
    "phase4_validation.schema.json": Phase4Validation,
    "phase4_gate_index.schema.json": Phase4GateIndex,
}
SCHEMA_PATH = SCHEMA_DIR / "build_manifest.schema.json"


def schema_bytes(model: type[BaseModel] = BuildManifest) -> bytes:
    return (
        json.dumps(model.model_json_schema(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def check_schema(path: Path = SCHEMA_PATH) -> bool:
    return path.is_file() and path.read_bytes() == schema_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export persistent JSON Schemas")
    parser.add_argument("--check", action="store_true", help="fail if the generated schema differs")
    args = parser.parse_args()
    if args.check:
        stale = [
            str(SCHEMA_DIR / name)
            for name, model in SCHEMAS.items()
            if not (SCHEMA_DIR / name).is_file()
            or (SCHEMA_DIR / name).read_bytes() != schema_bytes(model)
        ]
        if stale:
            parser.exit(1, f"schema differs or is missing: {', '.join(stale)}\n")
        return 0
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMAS.items():
        (SCHEMA_DIR / name).write_bytes(schema_bytes(model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
