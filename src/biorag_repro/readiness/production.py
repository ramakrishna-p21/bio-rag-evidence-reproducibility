"""Read the retained production and qualification evidence without granting approval."""

import platform
import socket
from pathlib import Path

from biorag_repro.core.hashing import sha256_bytes, sha256_file
from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
from biorag_repro.generation import config as generator_config
from biorag_repro.generation.config import require_qualified_settings, resolve_generator_config
from biorag_repro.readiness import git_state
from biorag_repro.readiness.files import FileMissing, FileProblem, loads_strict, read_regular
from biorag_repro.readiness.gather import Gathered, file_ref
from biorag_repro.readiness.models import EvidenceRef

REPORT = "docs/reports/remaining_gate_subjects_2026_11_20/"
BASELINE = "20e6efdec5a3c4ed1d3700a0754b4aa8790cbcdb"
ARMS = tuple(FROZEN_MATRIX)
DEVELOPMENT_SHA = {
    "confirmatory": "11290e58973f3eec13ae2ce2712b9848f19fb57900bc42e7a73bd6f687949927",
    "dense_robustness": "38bf4944a798e2b42baf855d3dccda8adb2a51804fa0f8b8c8d868deabee5c9c",
    "chunk_robustness": "2406e990544267d41d86ab0a487ccd14b1e14d26dc80d824352fd03d728c51c0",
    "scope_robustness": "ac88bb4faeaa22907c7b6663c96299983ddf2f946e35c107721339b8c987e3cb",
    "closed_book": "c247b14982c8bdba5d8110816465e1e2dc07789c8f1d7d8debe0948cfe397888",
}
CANDIDATE_SHA = {
    "g04": "4f4b4f9d8805176445901e8f9fa7078a476138ecf1fe701470bba4c865c3b47c",
    "g05": "a1a105959a956081d1efd5a3ef7bed3b5cdd8be72eb4cbe0687bee0c0ec32c92",
}


def _verified_candidate(repo: Path, name: str) -> tuple[dict, bytes]:
    data = read_regular(
        repo,
        REPORT
        + f"{name}_"
        + ("qualification_candidate.json" if name == "g04" else "identity_candidate.json"),
    )
    if sha256_bytes(data) != CANDIDATE_SHA[name]:
        raise FileProblem(f"{name} retained candidate changed")
    value = loads_strict(data)
    if not isinstance(value, dict):
        raise FileProblem(f"{name} candidate is not an object")
    return value, data


def production_freeze(ctx, source: Gathered) -> Gathered:
    """Verify the provisional choices; emit a subject only for a committed G09 source."""
    out = Gathered("absent", "production_source_revision_pending", "production freeze")
    try:
        record_bytes = read_regular(ctx.repo, "docs/reports/g03_production_freeze_2026_11_20.json")
        record = loads_strict(record_bytes)
        if not isinstance(record, dict) or set(record) != {
            "schema_version",
            "baseline_commit",
            "execution_source_sha256",
            "prompt_sha256",
            "output_schema_sha256",
            "parser",
            "stage_c_rules",
            "chunking",
            "arm_sha256",
            "development_sha256",
            "retrieval_sha256",
            "corpus_subject_sha256",
            "benchmark_subject_sha256",
            "seed_subject_sha256",
            "generation_seeds_sha256",
            "analysis",
            "matrix",
            "planned_observations",
            "context",
            "model_manifest_sha256",
        }:
            raise FileProblem("production freeze fields differ")
        if (
            record["schema_version"],
            record["baseline_commit"],
            record["stage_c_rules"],
            record["chunking"],
            record["parser"],
            record["planned_observations"],
        ) != (
            "production_freeze_v1",
            BASELINE,
            "stage_c_rules_v2",
            "canonical_whole_units_v2",
            "short_answer_json_v1",
            4575,
        ):
            raise FileProblem("production freeze scientific choices differ")
        if (
            set(record["arm_sha256"]) != set(ARMS)
            or record["development_sha256"] != DEVELOPMENT_SHA
        ):
            raise FileProblem("production arm roster or development identities differ")
        if not git_state.is_ancestor(ctx.repo, BASELINE):
            raise FileProblem("approved baseline is not in source history")
        if (
            record["corpus_subject_sha256"],
            record["benchmark_subject_sha256"],
            record["seed_subject_sha256"],
            record["model_manifest_sha256"],
            record["output_schema_sha256"],
        ) != (
            "f51b52e3cd5e0700575d0dda3394de26e2fd5a5314e09051e27feac1422a3768",
            "a7d4c33e7a51a6339ce0891961e46035ede4f865f6584e7fdcb710e3f7d1dd49",
            "07c518097c6dd56142a10c44d1635f5986a7b0c51f17557379463ab9725a921a",
            "4a133fa0d16ea3fdd42e987d7aa5135f3f8562ce45c55bf478b2f777f53e6fad",
            "bcf82c6821edd4bfd16f520ff4a5e57a52866f2f5ff0cea76b1a2ace80c903de",
        ):
            raise FileProblem("approved source, model, seed or schema identity differs")
        if record["context"] != {
            "deduplicate_exact_within_representation": True,
            "evidence_cap_qwen_tokens": 2048,
            "on_first_eligible_overflow": "stop_without_truncation_or_searching_later_chunks",
            "preserve_rank": True,
            "ranked_chunk_limit": 20,
        }:
            raise FileProblem("context assembly rules differ")
        if record["analysis"] != {
            "bootstrap_resamples": 10000,
            "bootstrap_unit": "whole_paper",
            "confidence_interval": "percentile",
            "equal_paper_weighted": "sensitivity",
            "multiplicity": "Holm",
            "paired_difference": "JATS - PDF_DOCLING",
            "primary_estimand": "item_weighted_paired_absolute_risk_difference",
            "primary_repeat": 1,
            "test": "intercept_only_paired_difference_CR2_Satterthwaite",
        }:
            raise FileProblem("frozen analysis differs")
        witness = loads_strict(read_regular(ctx.repo, "approvals/benchmark_count_witness_v1.json"))
        if witness["approved"] is not True or witness["final_test_items"] != 305:
            raise FileProblem("approved final-test item count differs")
        expected_matrix = {
            a: {"calls_per_item": len(c) * len(r), "conditions": list(c), "repeats": list(r)}
            for a, (c, r) in FROZEN_MATRIX.items()
        }
        if record["matrix"] != expected_matrix:
            raise FileProblem("experiment matrix differs")
        prompt = read_regular(ctx.repo, "artifacts/phase8/colab_development_v5/inputs/prompt.json")
        seeds = read_regular(ctx.repo, "artifacts/phase8/colab_development_v5/inputs/seeds.json")
        if (sha256_bytes(prompt), sha256_bytes(seeds)) != (
            record["prompt_sha256"],
            record["generation_seeds_sha256"],
        ) or record[
            "prompt_sha256"
        ] != "06b0f611f0cc1f492baa40d38c2fffc306442837cb0f18044bc40433242688c2":
            raise FileProblem("prompt or generation seed identity differs")
        prompt_value = loads_strict(prompt)
        if prompt_value["prompt_version"] != "qualification-candidate-1":
            raise FileProblem("retained prompt provenance changed")
        if (
            record["generation_seeds_sha256"]
            != "54637ea3144ba165f13726d9f8934534e4e28d4b89e1f1a5f354499c069fd099"
        ):
            raise FileProblem("generation seed registry differs")
        out.refs.append(
            file_ref(
                "production_freeze",
                "repo/docs/reports/g03_production_freeze_2026_11_20.json",
                record_bytes,
            )
        )
        for arm in ARMS:
            old = resolve_generator_config(
                read_regular(
                    ctx.repo, f"artifacts/phase8/colab_development_v5/inputs/configs/{arm}.toml"
                )
            )
            new_data = read_regular(ctx.repo, f"configs/production/{arm}.toml")
            new = resolve_generator_config(new_data)
            if old.configuration_sha256 != DEVELOPMENT_SHA[arm]:
                raise FileProblem(f"{arm}: retained development identity differs")
            expected = old.settings.model_dump(mode="json") | {
                "scoring_version": "stage_c_rules_v2"
            }
            if (
                new.settings.model_dump(mode="json") != expected
                or new.configuration_sha256 != record["arm_sha256"][arm]
            ):
                raise FileProblem(f"{arm}: production controls differ beyond scoring")
            if new.settings.retrieval_config_sha256 != record["retrieval_sha256"][arm]:
                raise FileProblem(f"{arm}: retrieval identity differs")
            if new.settings.output_schema_sha256 != record["output_schema_sha256"]:
                raise FileProblem(f"{arm}: output schema differs")
            require_qualified_settings(new.settings)
            out.refs.append(
                file_ref("production_arm", f"repo/configs/production/{arm}.toml", new_data)
            )
        if sum(len(c) * len(r) for c, r in FROZEN_MATRIX.values()) != 15:
            raise FileProblem("frozen matrix differs")
        out.facts = {
            "configuration_sha256_by_arm": record["arm_sha256"],
            "freeze_record_sha256": sha256_bytes(record_bytes),
            "generation_seed_registry_sha256": record["generation_seeds_sha256"],
        }
        if record["execution_source_sha256"] is None:
            return out
        if source.state != "ok" or source.subject is None:
            raise FileProblem("execution source has no verified G09 subject")
        if (
            source.facts.get("execution_source_identity_sha256")
            != record["execution_source_sha256"]
        ):
            raise FileProblem("execution source differs from the frozen execution identity")
        out.state, out.reason, out.subject = "ok", "checks_passed", record
        return out
    except FileMissing as error:
        return Gathered("absent", "production_freeze_evidence_missing", out.scope, str(error))
    except (FileProblem, ValueError, KeyError, TypeError, git_state.GitUnavailable) as error:
        return Gathered("failed", "production_freeze_invalid", out.scope, str(error)[:300])


def qualification(ctx) -> Gathered:
    scope = "retained original S1/S2 L4 qualification"
    out = Gathered("ok", "checks_passed", scope)
    try:
        value, candidate = _verified_candidate(ctx.repo, "g04")
        for name, expected in value["original_archives_sha256"].items():
            session = "S1" if name.startswith(("S1", "candidate")) else "S2"
            relative = f"readiness/qualification/{session}/originals/{name}"
            data = read_regular(ctx.data_root, relative)
            if sha256_bytes(data) != expected:
                raise FileProblem(f"qualification archive missing or changed: {name}")
            out.refs.append(file_ref("qualification_archive", "data/" + relative, data))
        audits = {}
        for session in ("s1", "s2"):
            script_relative = f"readiness/qualification/audit_{session}.py"
            output_relative = f"readiness/qualification/audit_{session}_output.txt"
            script = read_regular(ctx.data_root, script_relative)
            output = read_regular(ctx.data_root, output_relative)
            if (
                sha256_bytes(script) != value["audit_scripts_sha256"][session]
                or sha256_bytes(output) != value["audit_outputs_sha256"][session]
            ):
                raise FileProblem(f"{session} audit identity differs")
            audits[session] = output.decode("utf-8")
            out.refs.append(
                file_ref(
                    "qualification_audit_script",
                    "data/" + script_relative,
                    script,
                )
            )
            out.refs.append(
                file_ref(
                    "qualification_audit",
                    "data/" + output_relative,
                    output,
                )
            )
        if not all(
            marker in audits["s1"]
            for marker in (
                "ok   Q3: 821 calls measured, 0 breaches",
                "ok   Q4: probe JATS 2048 + 128, length, complete",
                "ok   Q4: probe PDF_DOCLING 2048 + 128, length, complete",
                "ok   Q7: target attempt is exactly started then interrupted",
                "ok   Q7: open started event sealed as it stood in the post-kill anchor",
                "0 failures",
            )
        ) or not all(
            marker in audits["s2"]
            for marker in (
                "ok   Q1: S2 model file manifest",
                "ok   S2 ran on a new VM",
                "ok   Q6(ii): all 45 S2 observations equal their S1 canary counterparts",
                "ok   Q6(ii): all 45 S2 observations equal their S1 cohort counterparts",
                "0 failures",
            )
        ):
            raise FileProblem("authenticated qualification audit omits a required criterion")
        if not (
            value["q1_to_q10"] == "PASS"
            and value["qualified_host_class"] == "NVIDIA L4"
            and value["boundary"]
            == {
                "both_complete": True,
                "context_tokens": 2048,
                "output_tokens": 128,
                "probes": 2,
                "stop_reason": "length",
            }
            and value["restart"]["planned_kill"]
            and value["restart"]["seals_verified"]
            and value["restart"]["completed_observations_regenerated"] == 0
            and value["cross_session_replay"]["matching_observations"] == 45
            and value["cross_session_replay"]["output_token_ids_equal"] == 45
            and value["cross_session_replay"]["new_vm"]
            and value["cross_session_replay"]["later_utc_date"]
            and value["measured_fit"]["s1_calls"] == 821
        ):
            raise FileProblem("qualification criterion differs")
        out.refs.append(
            file_ref(
                "qualification_identity",
                "repo/" + REPORT + "g04_qualification_candidate.json",
                candidate,
            )
        )
        out.subject = {k: v for k, v in value.items() if k != "status"}
        return out
    except FileMissing as error:
        return Gathered("absent", "qualification_evidence_missing", scope, str(error))
    except (FileProblem, ValueError, KeyError, TypeError, OSError) as error:
        return Gathered("failed", "qualification_evidence_invalid", scope, str(error)[:300])


def model_parser(ctx) -> Gathered:
    scope = "Qwen, BGE, Docling and JATS retained identities"
    out = Gathered("ok", "checks_passed", scope)
    try:
        value, candidate = _verified_candidate(ctx.repo, "g05")
        qwen = value["qwen"]
        if (qwen["revision"], qwen["model_manifest_sha256"]) != (
            "b968826d9c46dd6066d109eabc6255188de91218",
            "4a133fa0d16ea3fdd42e987d7aa5135f3f8562ce45c55bf478b2f777f53e6fad",
        ):
            raise FileProblem("Qwen approval identity differs")
        if (
            generator_config.APPROVED_CHECKPOINTS.get((qwen["repository"], qwen["revision"]))
            != qwen["model_manifest_sha256"]
        ):
            raise FileProblem("Qwen checkpoint is not approved at this exact manifest")
        bases = (
            (
                qwen["tokenizer_files_sha256"],
                ctx.repo / "artifacts/phase8/colab_development_v5/tokenizer" / qwen["revision"],
            ),
            (
                value["bge"]["files_sha256"],
                ctx.repo / "models/bge-base-en-v1.5" / value["bge"]["revision"],
            ),
            (
                value["docling"]["asset_files_sha256"],
                ctx.repo / "artifacts/phase4_docling_qualification/assets",
            ),
        )
        for files, root in bases:
            for name, expected in files.items():
                path = root / name
                if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
                    raise FileProblem(f"model/parser asset missing or changed: {name}")
                out.refs.append(
                    EvidenceRef(
                        kind="model_parser_asset",
                        locator=f"repo/{path.relative_to(ctx.repo)}",
                        sha256=expected,
                        subject_kind="stored_file_bytes",
                    )
                )
        for relative, expected in (
            (
                "models/bge-base-en-v1.5/" + value["bge"]["revision"] + "/verified-assets.json",
                value["bge"]["manifest_sha256"],
            ),
            (
                "artifacts/phase4_docling_qualification/environment_lock.txt",
                value["docling"]["environment_lock_sha256"],
            ),
            (
                "artifacts/phase4_docling_qualification/qualification_configuration.json",
                value["docling"]["qualification_configuration_sha256"],
            ),
            (
                "artifacts/phase4_docling_qualification/asset_inventory.json",
                value["docling"]["asset_inventory_sha256"],
            ),
        ):
            data = read_regular(ctx.repo, relative)
            if sha256_bytes(data) != expected:
                raise FileProblem(f"model/parser manifest changed: {relative}")
            out.refs.append(file_ref("model_parser_manifest", "repo/" + relative, data))
        entries = {path: blob for _, blob, path in git_state.probe(ctx.repo).entries}
        jats = value["jats"]
        for key, path in {
            "adapter_git_blob": "src/biorag_repro/documents/jats.py",
            "configuration_git_blob": "configs/jats.toml",
            "models_git_blob": "src/biorag_repro/documents/models.py",
            "storage_git_blob": "src/biorag_repro/documents/storage.py",
        }.items():
            if entries.get(path) != jats[key]:
                raise FileProblem(f"JATS source identity differs: {path}")
        if sha256_file(ctx.repo / "configs/jats.toml") != jats["configuration_sha256"]:
            raise FileProblem("JATS configuration bytes differ")
        out.refs.append(
            file_ref(
                "model_parser_identity", "repo/" + REPORT + "g05_identity_candidate.json", candidate
            )
        )
        out.subject = {k: v for k, v in value.items() if k != "status"}
        return out
    except FileMissing as error:
        return Gathered("absent", "model_parser_evidence_missing", scope, str(error))
    except (
        FileProblem,
        ValueError,
        KeyError,
        TypeError,
        OSError,
        git_state.GitUnavailable,
    ) as error:
        return Gathered("failed", "model_parser_evidence_invalid", scope, str(error)[:300])


def designated_host() -> bool:
    return socket.gethostname().upper() == "YOGA-C940" and "microsoft" in platform.release().lower()


def host_designation(
    ctx, sources: dict[str, str], lock_sha256: str, *, evaluating_host: bool = True
) -> dict:
    """Validate the researcher's analysis-host designation.

    With `evaluating_host` the live host must be the designated WSL host. Retained evidence
    from that host is checked with `evaluating_host=False`: the designation, lock, sources and
    BLAS enforcement are still validated here, and the host is checked against the record.
    """
    data = read_regular(
        ctx.repo, "docs/reports/g07_researcher_designation_evidence_2026_11_25.json"
    )
    if sha256_bytes(data) != "7b50f78abbb7b629787c4193b766fe547730d97a4f70319f946f166565f867f5":
        raise FileProblem("researcher host designation changed")
    value = loads_strict(data)
    if (
        value["designation"]["host"],
        value["designation"]["platform"],
        value["designation"]["entry_point"],
        value["renv_lock_sha256"],
    ) != (
        "YOGA-C940",
        "WSL Ubuntu 22.04.5",
        "analysis/run_analysis.R",
        lock_sha256,
    ) or value["analysis_sources_sha256"] != sources:
        raise FileProblem("analysis designation or source identity differs")
    if evaluating_host:
        if not designated_host():
            raise FileProblem("evaluating host is not designated WSL host")
        os_release = Path("/etc/os-release").read_text()
        if 'VERSION_ID="22.04"' not in os_release and "VERSION_ID=22.04" not in os_release:
            raise FileProblem("evaluating Ubuntu release differs")
    entry = read_regular(ctx.repo, "analysis/run_analysis.R").decode()
    if not all(
        x in entry
        for x in (
            'OPENBLAS_CORETYPE = "HASWELL"',
            'OPENBLAS_NUM_THREADS = "1"',
            'if (a$command == "run" || (a$command == "verify" && isTRUE(a$rerun))) require_blas_pin()',
        )
    ):
        raise FileProblem("analysis BLAS enforcement source differs")
    return value
