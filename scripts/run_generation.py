"""Generate answers for one batch of planned observations: write-once, append-only, resumable.

Candidate runner for development and qualification batches only. It refuses `final_test`:
official execution stays behind the existing readiness mechanism until a replacement is
approved. Built from the Dev35 runner; rendering, parsing, seeds, configuration identity and
observation keys come from the package, so there is one implementation of each.

    python scripts/run_generation.py run --contexts F --contexts-sha256 H --configs DIR \\
        --prompt F --seeds F --tokenizer-root DIR --model-root DIR --device cuda:0 \\
        --phase development --out DIR [--anchor F --anchor-sha256 H] [--stop-after N]
    python scripts/run_generation.py seal --out DIR
    python scripts/run_generation.py verify --out DIR --anchor F --anchor-sha256 H
    python scripts/run_generation.py inventory --out FILE [--device cuda:0]
    python scripts/run_generation.py probe --contexts F --contexts-sha256 H --configs DIR \\
        --prompt F --seeds F --tokenizer-root DIR --model-root DIR --device cuda:0 \\
        --item-id Q --condition JATS --out FILE

`inventory` and `probe` are qualification aids. They write one standalone write-once file each,
never into a batch: an inventory is a read-only record of the GPUs, driver and packages; a
probe is one development-only call at the exact evidence budget with the full output
allowance forced. A probe has no observation key, is never parsed or scored and cannot enter
the five-arm plan.

Layout of --out: batch.json (identity, written once); attempts/NNNNNN.json (one write-once
file per attempt event: started, completed, failed, interrupted); results/<key>.json (one
write-once file per finished observation: a completed generation or a recorded operational
failure); anchors/NNNNNN.json (sealed lists of finished results and events).

Trust: files here can be rewritten by anyone with write access, so read-only modes and the
batch's own records are not evidence by themselves. A sealed anchor is evidence only through
its SHA-256 recorded outside the data root (in DECISIONS.md, a report, or a commit); `verify`
and `run --anchor` check every sealed result and event against it.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from biorag_repro.core.identity import observation_key
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
from biorag_repro.evaluation.stage_c_models import GenerationResult
from biorag_repro.generation.config import check_retrieval_reference, resolve_generator_config
from biorag_repro.generation.model_snapshot import inventory_snapshot
from biorag_repro.generation.parsing import parse_output
from biorag_repro.generation.prompt import InputRenderer, PromptArtifact, prompt_sha256
from biorag_repro.generation.receipt import token_ids_sha256
from biorag_repro.generation.seeds import (
    GenerationSeedRegistry,
    check_seeds_for_configuration,
    repeat_seed,
    seed_registry_sha256,
)

PHASES = ("development", "qualification")  # final_test is refused; see the module text


class Refused(SystemExit):
    def __init__(self, message: str):
        super().__init__(f"refused: {message}")


@dataclass(frozen=True)
class Planned:
    key: str
    record: dict
    repeat: int
    config: Any  # ResolvedConfiguration of the record's arm


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def write_once(path: Path, data: bytes) -> None:
    """Create `path` with exactly these bytes, or fail if it exists. Never partially visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)  # atomic and refuses an existing name
    finally:
        tmp.unlink()
    os.chmod(path, 0o444)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def dump(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True).encode("utf-8") + b"\n"


# ---- inputs and plan ---------------------------------------------------------------------


def load_inputs(args: argparse.Namespace) -> tuple[dict, list[Planned], Any, PromptArtifact]:
    data = args.contexts.read_bytes()
    if sha256_bytes(data) != args.contexts_sha256:
        raise Refused("the contexts file differs from the trusted SHA-256")
    package = json.loads(data)
    if args.phase not in PHASES:
        raise Refused(f"phase {args.phase!r}: this runner executes development or qualification")
    # The expected observations come from the declared eligible items and arms and the frozen
    # matrix, never from the records that happen to be present: a missing context is an
    # input-integrity failure that stops the batch, not a smaller plan.
    header = package.get("header", {})
    items, arms = header.get("item_ids"), header.get("arms")
    if not items or not arms or any(arm not in FROZEN_MATRIX for arm in arms):
        raise Refused("input integrity: the contexts header must declare its items and arms")
    expected = {(i, a, c) for i in items for a in arms for c in FROZEN_MATRIX[a][0]}
    present = [(r["item_id"], r["matrix_arm"], r["condition"]) for r in package["records"]]
    if len(set(present)) != len(present) or set(present) != expected:
        missing, extra = sorted(expected - set(present)), sorted(set(present) - expected)
        raise Refused(f"input integrity: missing {missing[:5]}, unexpected {extra[:5]}")
    if getattr(args, "published_context_root", None) is not None:
        check_published_contexts(package, args.published_context_root)
    prompt = PromptArtifact.model_validate_json(args.prompt.read_bytes())
    seeds = GenerationSeedRegistry.model_validate_json(args.seeds.read_bytes())
    configs = {}
    plan = []
    for record in package["records"]:
        arm, condition = record["matrix_arm"], record["condition"]
        if record["split"] != "development":
            raise Refused(f"{record['item_id']} is not a development item")
        conditions, repeats = FROZEN_MATRIX[arm]
        if condition not in conditions:
            raise Refused(f"{arm} has no condition {condition}")
        if (record["context"] is None) != (arm == "closed_book"):
            raise Refused("exactly the closed-book arm has no context")
        if (
            record["context"] is not None
            and sha256_bytes(record["context"].encode("utf-8")) != record["context_sha256"]
        ):
            raise Refused(f"context of {record['item_id']} {condition} does not match its hash")
        if arm not in configs:
            config = resolve_generator_config((args.configs / f"{arm}.toml").read_bytes())
            if config.settings.matrix_arm != arm:
                raise Refused(f"{arm}.toml configures {config.settings.matrix_arm}")
            check_seeds_for_configuration(config.settings, seeds)
            configs[arm] = config
        config = configs[arm]
        check_retrieval_reference(config.settings, record["retrieval_config_sha256"])
        for repeat in repeats:
            key = observation_key(
                phase=args.phase,
                benchmark_version=record["benchmark_version"],
                item_id=record["item_id"],
                matrix_arm=arm,
                representation_or_condition=condition,
                repeat_index=repeat,
                resolved_config_sha256=config.configuration_sha256,
            )
            plan.append(Planned(key, record, repeat, config))
    keys = [p.key for p in plan]
    if len(set(keys)) != len(keys):
        raise Refused("the contexts file plans an observation twice")
    plan.sort(key=lambda p: p.repeat)  # every repeat 1 before any repeat 2, as in Dev35
    return package, plan, seeds, prompt


def check_published_contexts(package: dict, root: Path) -> None:
    """Check stored context links; this alone never establishes official origin."""
    from biorag_repro.core.storage import read_manifest, read_project_json
    from biorag_repro.retrieval.models import ContextRecord, RankedRetrieval
    from biorag_repro.retrieval.storage import record_ref

    for record in package["records"]:
        if record["matrix_arm"] == "closed_book":
            if record.get("context_record") is not None:
                raise Refused("closed book cannot cite a ContextRecord")
            continue
        try:
            ref = ArtifactRef(**record["context_record"])
            manifest = ArtifactRef(**record["context_manifest"])
            value = read_project_json(root / ref.logical_ref, ref.sha256, ContextRecord)
            build = read_manifest(root / manifest.logical_ref, manifest.sha256)
            if build.status != "complete" or not any(o.artifact == ref for o in build.outputs):
                raise ValueError("context manifest does not publish the selected record")
            ranked = read_project_json(
                root / value.retrieval.logical_ref, value.retrieval.sha256, RankedRetrieval
            )
            if (
                record_ref(value) != ref
                or value.exact_context != record["context"]
                or value.generator_tokens != record["context_tokens"]
                or value.representation != record["condition"]
                or ranked.query_id != record["item_id"]
                or ranked.question != record["question"]
                or ranked.configuration.resolved_sha256 != record["retrieval_config_sha256"]
            ):
                raise ValueError("context or query lineage mismatch")
        except (KeyError, ValueError, OSError) as error:
            raise Refused(f"published ContextRecord verification failed: {error}") from error


def preflight(args: argparse.Namespace) -> dict:
    """Check the complete development roster and CPU renderings without loading a generator."""
    from biorag_repro.generation.config import require_qualified_settings

    package, plan, seeds, prompt = load_inputs(args)
    header = package["header"]
    if (
        len(plan) != 1290
        or len(package["records"]) != 774
        or len(header["item_ids"]) != 86
        or set(header["arms"]) != set(FROZEN_MATRIX)
        or set(header.get("pool_papers", [])) != {f"P{n:03d}" for n in range(1, 51)}
    ):
        raise Refused("preflight requires the complete five-arm frozen development matrix")
    if getattr(args, "published_context_root", None) is None:
        raise Refused("preflight requires published context lineage")
    configs = {p.record["matrix_arm"]: p.config for p in plan}
    renderers = {}
    for arm, config in configs.items():
        require_qualified_settings(config.settings)
        renderers[arm] = InputRenderer(args.tokenizer_root, config, prompt)
    rendered = {}
    for planned in plan:
        key = "/".join(
            (planned.record["item_id"], planned.record["matrix_arm"], planned.record["condition"])
        )
        if key not in rendered:
            value, ids = prepare(renderers[planned.record["matrix_arm"]], planned)
            rendered[key] = {
                "rendered_sha256": value.rendered_sha256,
                "input_token_ids_sha256": token_ids_sha256(ids),
                "input_tokens": len(ids),
            }
    return {
        "status": "development_inputs_checked",
        "scientific_execution": False,
        "contexts_sha256": args.contexts_sha256,
        "items": 86,
        "item_arm_cells": 430,
        "conditions": len(rendered),
        "planned_observations": len(plan),
        "planned_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(p.key for p in plan))),
        "generator_configurations": {a: c.configuration_sha256 for a, c in configs.items()},
        "prompt_sha256": prompt_sha256(prompt),
        "seed_registry_sha256": seed_registry_sha256(seeds),
        "rendered_inputs": rendered,
    }


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "head": git("rev-parse", "HEAD"),
        "tracked_changes_sha256": sha256_bytes(status.encode()),
    }


def environment(device: Any) -> dict:
    import tokenizers
    import torch
    import transformers

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "pip_freeze_sha256": sha256_bytes(freeze.encode()),
    }


def prospective_freeze_identity(
    raw_freeze: str,
    candidate_lock: bytes,
    git_head: str,
    *,
    packaging_root: Path | None = None,
    packaging_reference: bytes | None = None,
    project_root: Path | None = None,
) -> dict:
    """Propose a path-independent batch identity without changing the current runner.

    The candidate lock's complete lines are compared independent of locale order.
    The editable project locator stays outside the portable hash. A verified local
    build is represented by content identity; its exact raw line remains checked provenance.
    Nonlocal direct references remain exact. The retained local packaging build needs
    verified candidate content and an observed clean project revision. This produces
    a review proposal, never approval or a reconstruction of historical installations.
    """
    try:
        locked = candidate_lock.decode("utf-8")
    except UnicodeDecodeError:
        raise Refused("the candidate lock is not UTF-8") from None
    if not locked.endswith("\n"):
        raise Refused("the candidate lock is not valid UTF-8 text ending in a newline")
    if not re.fullmatch(r"[0-9a-f]{40}", git_head):
        raise Refused("the proposed identity requires the exact Git commit")
    lines = raw_freeze.splitlines(keepends=True)
    editable = [line for line in lines if line.startswith(("-e ", "--editable "))]
    if len(editable) != 1:
        raise Refused("the freeze must contain exactly one editable project install")
    project = editable[0]
    if not re.fullmatch(
        rf"(?:-e|--editable) git\+file://[^\s]+@{git_head}#egg=biorag_repro\n", project
    ):
        raise Refused("the editable project install does not identify this Git commit")
    dependencies = [line for line in lines if line != project]
    if sorted(dependencies) != sorted(locked.splitlines(keepends=True)):
        raise Refused("installed dependencies differ from the exact candidate lock")
    local_lines = [line for line in dependencies if re.search(r"\s@\s*file:", line, re.IGNORECASE)]
    verified_build = None
    if local_lines:
        if packaging_root is None or packaging_reference is None or project_root is None:
            raise Refused(
                "portable identity requires a verified candidate content identity and project "
                "checkout for non-project local build references; activation still requires approval"
            )
        from biorag_repro.generation.environment import system_probes
        from biorag_repro.generation.local_build import (
            CANDIDATE_LOCK_SHA256,
            verify_packaging_payload,
        )

        if sha256_bytes(candidate_lock) != CANDIDATE_LOCK_SHA256:
            raise Refused("local-build evidence requires the exact retained S1/S2 candidate lock")

        observed = system_probes(project_root).git()
        if observed.status != "ok" or observed.dirty or observed.commit != git_head:
            raise Refused("project checkout is not the exact clean reviewed revision")
        try:
            verified_build = verify_packaging_payload(packaging_root, packaging_reference)
        except ValueError as error:
            raise Refused(str(error)) from error
        if local_lines != [f"packaging @ {verified_build['source_url']}\n"]:
            raise Refused("local dependency reference differs from the verified packaging build")
    elif packaging_root is not None or packaging_reference is not None:
        raise Refused("packaging evidence is unrelated to this lock")
    requirements = [
        {"local_build": verified_build["identity"]}
        if line in local_lines
        else {"requirement": line.rstrip("\n")}
        for line in sorted(dependencies)
    ]
    portable = {
        "rule": "prospective_freeze_identity_v1",
        "dependency_identity_sha256": sha256_bytes(canonical_json_bytes(requirements)),
        "project_revision": git_head,
    }
    return {
        "schema_version": "prospective_freeze_identity_v1",
        "activation_status": "pending_researcher_patch_review",
        "portable_identity": portable,
        "portable_identity_sha256": sha256_bytes(canonical_json_bytes(portable)),
        "candidate_lock_sha256": sha256_bytes(candidate_lock),
        "raw_freeze_sha256": sha256_bytes(raw_freeze.encode()),
        "editable_install": project.rstrip("\n"),
        "git_head": git_head,
        "local_build_verification": verified_build,
        "approved_for_execution": False,
    }


def batch_identity(args, package_sha, plan, seeds, prompt, snapshot, env) -> dict:
    """Everything that must be identical for a resume to continue the same batch."""
    configs = {p.record["matrix_arm"]: p.config for p in plan}
    return {
        "phase": args.phase,
        "contexts_sha256": package_sha,
        "planned_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(p.key for p in plan))),
        "generator_configurations": {a: c.configuration_sha256 for a, c in sorted(configs.items())},
        "prompt_sha256": prompt_sha256(prompt),
        "seed_registry_sha256": seed_registry_sha256(seeds),
        "model_manifest_sha256": snapshot.manifest_sha256,
        "git": git_state(),
        "environment": env,
    }


# ---- state of an existing batch ------------------------------------------------------------


def read_events(out: Path) -> list[dict]:
    files = sorted((out / "attempts").glob("[0-9]*.json"))
    events = [json.loads(f.read_bytes()) for f in files]
    if [e["seq"] for e in events] != list(range(1, len(events) + 1)):
        raise Refused("the attempt history has a gap or a foreign file")
    return events


def append_event(out: Path, events: list[dict], **fields: Any) -> None:
    event = {"seq": len(events) + 1, "at_utc": utc_now(), **fields}
    write_once(out / "attempts" / f"{event['seq']:06d}.json", dump(event))
    events.append(event)


def check_result(
    stored: dict,
    planned: Planned,
    renderer: InputRenderer | None,
    run_id: str,
    events: list[dict],
    input_root: Path | None = None,
) -> None:
    """A stored result must be exactly what this batch would record for that observation."""
    try:
        result = GenerationResult.model_validate(stored["generation_result"])
    except (KeyError, ValueError) as error:
        raise Refused(f"result {planned.key} is not a valid generation result: {error}") from None
    record = planned.record
    key = observation_key(
        phase=result.phase,
        benchmark_version=result.benchmark_version,
        item_id=result.item_id,
        matrix_arm=result.matrix_arm,
        representation_or_condition=result.representation_or_condition,
        repeat_index=result.repeat_index,
        resolved_config_sha256=result.resolved_config_sha256,
    )
    if key != result.observation_key or result.paper_id != record["paper_id"]:
        raise Refused("result scientific coordinates differ from its observation identity")
    if (
        result.observation_key != planned.key
        or result.run_id != run_id
        or result.repeat_index != planned.repeat
        or result.resolved_config_sha256 != planned.config.configuration_sha256
        or stored["context_sha256"] != record["context_sha256"]
    ):
        raise Refused(f"result {planned.key} does not belong to this batch's plan")
    ended = {(e["attempt_id"], e["event"]) for e in events}
    expected_end = "completed" if result.status == "completed" else "failed"
    if (result.attempt_id, expected_end) not in ended:
        raise Refused(f"result {planned.key} has no matching attempt event")
    if result.status == "completed":
        parsed = parse_output(result.raw_output or "", planned.config.settings.parser_version)
        if (parsed.parse_status, parsed.answer, parsed.abstained) != (
            result.parse_status,
            result.answer,
            result.abstained,
        ):
            raise Refused(f"result {planned.key}: stored parse differs from re-parsing")
        if renderer is not None:
            rendered = renderer.render(
                record["question"], record["context"], record["context_tokens"]
            )
            if rendered.rendered_sha256 != stored["rendered_sha256"]:
                raise Refused(f"result {planned.key}: input differs from re-rendering")
            if "generator_input" in stored:
                from biorag_repro.core.hashing import sha256_file
                from biorag_repro.generation.inputs import check_tokenization, read_input_record

                input_ref = ArtifactRef(**stored["generator_input"])
                # The caller supplies the batch root; never follow a stored absolute path.
                if input_ref.logical_ref != f"inputs/{input_ref.sha256}.json":
                    raise Refused("receipt has an invalid stored-input path")
                if input_root is None:
                    raise Refused("receipt verification requires the batch input root")
                try:
                    saved_input = read_input_record(input_root, input_ref.sha256)
                    check_tokenization(saved_input, renderer.tokenizer)
                    if (
                        saved_input.observation_key != planned.key
                        or saved_input.rendered != rendered
                    ):
                        raise ValueError("stored input differs from this observation")
                    if saved_input.input_token_ids != stored["input_token_ids"]:
                        raise ValueError("receipt input tokens differ from the stored input")
                    if (
                        stored["input_token_ids_sha256"] != rendered.input_token_ids_sha256
                        or stored["input_tokens"] != rendered.input_tokens
                    ):
                        raise ValueError("receipt input token summary differs")
                    decoding = planned.config.settings.decoding
                    expected_arguments = {
                        "do_sample": decoding.do_sample,
                        "num_beams": 1,
                        "max_new_tokens": planned.config.settings.budgets.max_new_tokens,
                        "use_model_defaults": False,
                        "eos_token_id": [
                            renderer.tokenizer.convert_tokens_to_ids(t)
                            for t in ("<|im_end|>", "<|endoftext|>")
                        ],
                        "pad_token_id": renderer.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                    }
                    if decoding.do_sample:
                        expected_arguments.update(
                            temperature=decoding.temperature,
                            top_p=decoding.top_p,
                            top_k=decoding.top_k,
                        )
                    if stored["generation_arguments"] != expected_arguments:
                        raise ValueError("receipt generation arguments differ from configuration")
                    ids = stored["output_token_ids"]
                    if token_ids_sha256(ids) != stored["output_token_ids_sha256"]:
                        raise ValueError("receipt output token hash differs")
                    if len(ids) != stored["output_tokens"]:
                        raise ValueError("receipt output token count differs")
                    stop_ids = [
                        renderer.tokenizer.convert_tokens_to_ids(t)
                        for t in ("<|im_end|>", "<|endoftext|>")
                    ]
                    if not ids or any(i in stop_ids for i in ids[:-1]):
                        raise ValueError("receipt output stops before the last token")
                    if stored["stop_reason"] == "eos":
                        if ids[-1] not in stop_ids:
                            raise ValueError("receipt eos lacks a stop token")
                        body = ids[:-1]
                    else:
                        if len(ids) != planned.config.settings.budgets.max_new_tokens:
                            raise ValueError("receipt length stop is incomplete")
                        body = ids
                    raw = renderer.tokenizer.decode(
                        body, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )
                    if raw != result.raw_output:
                        raise ValueError("receipt tokens decode to another output")
                    from biorag_repro.core.storage import read_project_json
                    from biorag_repro.generation.environment import EnvironmentRecord
                    from biorag_repro.generation.receipt import ExecutionReceipt

                    ref = stored["execution_receipt"]
                    expected_path = f"receipts/{planned.key}/{result.attempt_id}.json"
                    if ref["logical_ref"] != expected_path:
                        raise ValueError("receipt path differs from this attempt")
                    receipt = read_project_json(
                        input_root / expected_path, ref["sha256"], ExecutionReceipt
                    )
                    if (
                        receipt.run_id != run_id.removeprefix("run_")
                        or receipt.attempt_id != result.attempt_id.removeprefix("att_")
                        or receipt.observation_key != planned.key
                        or receipt.input_record_sha256 != input_ref.sha256
                        or receipt.rendered_input_sha256 != rendered.rendered_sha256
                        or receipt.raw_output_sha256 != result.output.sha256
                        or receipt.facts.output_token_ids != ids
                        or receipt.facts.input_token_ids_sha256 != rendered.input_token_ids_sha256
                        or receipt.facts.input_token_count != rendered.input_tokens
                    ):
                        raise ValueError("typed receipt differs from input/output provenance")
                    facts = receipt.facts
                    generation = facts.generation
                    if (
                        generation.max_new_tokens != expected_arguments["max_new_tokens"]
                        or generation.do_sample != decoding.do_sample
                        or generation.temperature != decoding.temperature
                        or generation.top_p != decoding.top_p
                        or generation.top_k != decoding.top_k
                        or generation.eos_token_ids != stop_ids
                        or generation.pad_token_id != stop_ids[-1]
                        or generation.seed != stored["seed"]
                        or (facts.stop_reason == "eos") != (stored["stop_reason"] == "eos")
                        or facts.measurements is None
                        or facts.measurements.generation_ns != stored["generation_ns"]
                    ):
                        raise ValueError("typed receipt settings differ from execution")
                    for name in (
                        "max_new_tokens",
                        "do_sample",
                        "num_beams",
                        "eos_token_id",
                        "pad_token_id",
                    ):
                        if generation.effective[name] != expected_arguments[name]:
                            raise ValueError("effective decoding differs from the explicit call")
                    env_sha = receipt.environment_sha256
                    env = read_project_json(
                        input_root / "environments" / f"{env_sha}.json", env_sha, EnvironmentRecord
                    )
                    if (
                        env.execution is None
                        or env.model_revision != planned.config.settings.model.revision
                    ):
                        raise ValueError("receipt environment lacks the selected model")
                    raw_path = f"raw/{planned.key}/{result.attempt_id}.txt"
                    if (
                        result.output.logical_ref != raw_path
                        or sha256_file(input_root / raw_path) != result.output.sha256
                    ):
                        raise ValueError("receipt raw output is missing or changed")
                    if sha256_file(input_root / input_ref.logical_ref) != input_ref.sha256:
                        raise ValueError("receipt stored input hash differs")
                except (KeyError, ValueError, OSError) as error:
                    raise Refused(f"invalid execution receipt: {error}") from error


def check_anchor(out: Path, anchor: Path, anchor_sha256: str) -> None:
    data = anchor.read_bytes()
    if sha256_bytes(data) != anchor_sha256:
        raise Refused("the anchor differs from its trusted SHA-256")
    sealed = json.loads(data)
    if not isinstance(sealed, dict) or not isinstance(sealed.get("files"), dict):
        raise Refused("malformed seal file inventory")
    for relative, digest in sealed["files"].items():
        if not isinstance(relative, str):
            raise Refused("malformed sealed path")
        relative_path = Path(relative)
        path = out / relative_path
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not path.resolve().is_relative_to(out.resolve())
        ):
            raise Refused("sealed path is outside the batch")
        if not path.is_file() or sha256_file(path) != digest:
            raise Refused(f"{relative} is missing or differs from the sealed anchor")


# ---- generation --------------------------------------------------------------------------


def load_model(model_root: Path, device: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_root), dtype=torch.bfloat16, local_files_only=True, use_safetensors=True
    )
    return model.to(device).eval()


def prepare(renderer: InputRenderer, planned: Planned) -> tuple[Any, list[int]]:
    """The rendered input and the token ids the model will receive, checked before any attempt.

    The runner tokenizes the rendered text itself; those ids must equal the renderer's count
    and hash. A mismatch is a hard stop, not an operational failure: no attempt is started.
    """
    record = planned.record
    rendered = renderer.render(record["question"], record["context"], record["context_tokens"])
    ids = renderer.tokenizer(rendered.rendered_text, add_special_tokens=False)["input_ids"]
    if (
        len(ids) != rendered.input_tokens
        or token_ids_sha256(ids) != rendered.input_token_ids_sha256
    ):
        raise Refused(
            f"{planned.key}: generation input has {len(ids)} tokens, the rendered input "
            f"{rendered.input_tokens}, or their token-id hashes differ"
        )
    return rendered, ids


def measured_call(torch: Any, device: Any, call: Any) -> tuple[Any, int, dict]:
    """Run `call`; return its result, elapsed monotonic nanoseconds and a CUDA memory reading.

    On CUDA the device is synchronized before the clock starts and before it is read, so queued
    work from earlier calls is excluded and this call's work is included. Peak counters are
    reset just before the call, so the peaks are this process's maximum during the call and
    include the resident weights.
    """
    cuda = device.type == "cuda"
    if cuda:
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.synchronize(index)
        before = torch.cuda.memory_allocated(index)
        torch.cuda.reset_peak_memory_stats(index)
    start = time.perf_counter_ns()
    result = call()
    if cuda:
        torch.cuda.synchronize(index)
    elapsed = time.perf_counter_ns() - start
    if not cuda:
        return result, elapsed, {"status": "unavailable", "reason": f"device {device} is not CUDA"}
    free, total = torch.cuda.mem_get_info(index)
    return (
        result,
        elapsed,
        {
            "status": "measured",
            "scope": "this process, peak since reset immediately before model.generate",
            "device_index": index,
            "device_name": torch.cuda.get_device_name(index),
            "allocated_before_bytes": before,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
            "device_total_bytes": total,
            "device_used_after_bytes": total - free,  # includes other processes on the device
        },
    )


def generate(model: Any, tokenizer: Any, rendered: Any, ids: list[int], planned: Planned,
             seeds, force_full_output: bool = False, retain_token_ids: bool = False) -> dict:  # fmt: skip
    """One call, as in the Dev35 runner: exact input, explicit decoding, stop on end of turn."""
    import torch
    from transformers import set_seed

    settings = planned.config.settings
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    decoding = settings.decoding
    options: dict[str, Any] = {"do_sample": decoding.do_sample, "num_beams": 1}
    if force_full_output:  # boundary probe only: end-of-turn is suppressed until the limit
        options["min_new_tokens"] = settings.budgets.max_new_tokens
    seed = None
    if decoding.do_sample:
        seed = repeat_seed(seeds, planned.repeat)  # same named seed for JATS and PDF_DOCLING
        set_seed(seed)
        options |= {
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "top_k": decoding.top_k,
        }
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], device=device)

    def call() -> Any:
        with torch.inference_mode():
            return model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=settings.budgets.max_new_tokens,
                use_model_defaults=False,
                eos_token_id=stop_ids,
                pad_token_id=stop_ids[-1],
                **options,
            )

    out, elapsed_ns, memory = measured_call(torch, device, call)
    if retain_token_ids and out[0, : len(ids)].tolist() != ids:
        raise Refused("generation changed the supplied input prefix")
    output_ids = out[0, len(ids) :].tolist()
    stop = "eos" if output_ids and output_ids[-1] in stop_ids else "length"
    body = output_ids[:-1] if stop == "eos" else output_ids
    raw = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    facts = None
    if retain_token_ids:
        from biorag_repro.generation.receipt import ExecutionFacts, GenerationSettings

        effective, _ = model._prepare_generation_config(
            None,
            use_model_defaults=False,
            max_new_tokens=settings.budgets.max_new_tokens,
            eos_token_id=stop_ids,
            pad_token_id=stop_ids[-1],
            **options,
        )
        memory_fields = {k: v for k, v in memory.items() if k not in {"scope", "device_name"}}
        facts = ExecutionFacts(
            input_token_ids_sha256=rendered.input_token_ids_sha256,
            input_token_count=len(ids),
            output_token_ids=output_ids,
            stop_reason="eos" if stop == "eos" else "max_new_tokens",
            generation=GenerationSettings(
                max_new_tokens=settings.budgets.max_new_tokens,
                do_sample=decoding.do_sample,
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                top_k=decoding.top_k,
                eos_token_ids=stop_ids,
                pad_token_id=stop_ids[-1],
                seed=seed,
                effective=effective.to_dict(),
            ),
            measurements={
                "clock": "perf_counter_ns",
                "generation_ns": elapsed_ns,
                "cuda_memory": memory_fields,
            },
        )
    return {
        **(
            {
                "execution_facts": facts,
                "output_token_ids": output_ids,
                "input_token_ids": ids,
                "generation_arguments": {
                    **options,
                    "max_new_tokens": settings.budgets.max_new_tokens,
                    "use_model_defaults": False,
                    "eos_token_id": stop_ids,
                    "pad_token_id": stop_ids[-1],
                },
            }
            if retain_token_ids
            else {}
        ),
        "raw": raw,
        "rendered_sha256": rendered.rendered_sha256,
        "input_tokens": rendered.input_tokens,
        "input_token_ids_sha256": rendered.input_token_ids_sha256,
        "output_tokens": len(output_ids),
        "output_token_ids_sha256": token_ids_sha256(output_ids),  # includes the stop token
        "stop_reason": stop,
        "seed": seed,
        "generation_ns": elapsed_ns,  # model.generate only; see measured_call
        "cuda_memory": memory,
    }


def failure_kind(error: BaseException) -> str:
    try:
        import torch

        if isinstance(error, torch.cuda.OutOfMemoryError):
            return "out_of_memory"
    except ImportError:
        pass
    return "unclassified_error"


def result_bytes(planned, phase, run_id, attempt_id, raw=None, failure=None, call=None) -> bytes:
    record = planned.record
    common = {
        "phase": phase,
        "benchmark_version": record["benchmark_version"],
        "item_id": record["item_id"],
        "paper_id": record["paper_id"],
        "matrix_arm": record["matrix_arm"],
        "representation_or_condition": record["condition"],
        "repeat_index": planned.repeat,
        "resolved_config_sha256": planned.config.configuration_sha256,
        "observation_key": planned.key,
        "run_id": run_id,
        "attempt_id": attempt_id,
    }
    if failure is not None:
        result = GenerationResult(**common, status="operational_failure", failure=failure)
    else:
        parsed = parse_output(raw, planned.config.settings.parser_version)
        output = (
            ArtifactRef(**call["raw_output"])
            if call and "raw_output" in call
            else ArtifactRef(
                artifact_id="generator_output_" + sha256_bytes(raw.encode("utf-8")),
                logical_ref=f"results/{planned.key}.json",
                sha256=sha256_bytes(raw.encode("utf-8")),
            )
        )
        result = GenerationResult(
            **common, status="completed", output=output, raw_output=raw,
            parse_status=parsed.parse_status, answer=parsed.answer, abstained=parsed.abstained,
        )  # fmt: skip
    lineage = {k: record.get(k) for k in ("context_sha256", "context_tokens", "context_stop",
                                          "included_chunk_ids", "candidate_chunk_ids",
                                          "retrieval_config_sha256")}  # fmt: skip
    return dump({"generation_result": result.model_dump(mode="json"), **lineage, **(call or {})})


def run(args: argparse.Namespace, model: Any = None) -> int:
    """Generate every planned observation that has no result yet; return how many finished."""
    import torch

    if getattr(args, "official", False) and model is not None:
        raise Refused("official execution cannot use an injected model")
    if getattr(args, "official", False):
        preflight(args)
    package, plan, seeds, prompt = load_inputs(args)
    configs = {p.record["matrix_arm"]: p.config for p in plan}
    renderers = {a: InputRenderer(args.tokenizer_root, c, prompt) for a, c in configs.items()}
    revisions = {c.settings.model.revision for c in configs.values()}
    if len(revisions) != 1:
        raise Refused("all arms of a batch use one model revision")
    snapshot = inventory_snapshot(args.model_root, revisions.pop())
    official_inventory = None
    session_gpus = None
    if getattr(args, "official", False):
        from biorag_repro.generation import config as policy
        from biorag_repro.generation.environment import system_probes
        from biorag_repro.generation.environment_lock import capture_inventory, verify_environment

        if (
            args.phase != "development"
            or len(plan) != 1290
            or getattr(args, "published_context_root", None) is None
            or getattr(args, "selection", None) is None
        ):
            raise Refused(
                "official development requires full inputs and independent source selection"
            )
        if any(c.settings.purpose != "development_candidate" for c in configs.values()):
            raise Refused("synthetic configurations cannot create official execution")
        for config in configs.values():
            policy.require_qualified_settings(config.settings)
            pin = config.settings.model
            if (
                policy.APPROVED_CHECKPOINTS.get((pin.repository, pin.revision))
                != snapshot.manifest_sha256
            ):
                raise Refused("checkpoint is not independently approved")
        if getattr(args, "environment_lock", None) is None:
            raise Refused("official execution requires an approved environment lock")
        lock_bytes = args.environment_lock.read_bytes()
        lock_sha = sha256_bytes(lock_bytes)
        if lock_sha not in policy.APPROVED_ENVIRONMENT_LOCKS:
            raise Refused("environment lock is not independently approved")
        official_inventory = capture_inventory(system_probes(Path.cwd()))
        checked = verify_environment(
            lock_bytes, official_inventory, expected_spec_sha256=lock_sha, inventory_source="live"
        )
        if checked.outcome != "matches_approved_lock":
            raise Refused("live environment does not match the approved lock")
        if not re.fullmatch(r"cuda:\d+", args.device):
            raise Refused("official execution requires one explicit CUDA device")
        if torch.cuda.get_device_name(args.device) != policy.APPROVED_CHECKPOINT_GPU:
            raise Refused("checkpoint approval is limited to the qualified NVIDIA L4")
        session_gpus = gpu_inventory()
        if session_gpus.get("status") != "measured":
            raise Refused("official execution requires measured GPU UUID inventory")
    device = torch.device(args.device)
    identity = batch_identity(args, args.contexts_sha256, plan, seeds, prompt, snapshot,
                              environment(device))  # fmt: skip
    if official_inventory is not None:
        identity.update(
            official_execution=True,
            environment_lock_sha256=lock_sha,
            session_hostname=platform.node(),
            session_boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            session_gpu_inventory=session_gpus["gpus"],
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], text=True
        ).strip()
        if status:
            raise Refused("official execution requires reviewed clean code")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "batch.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise Refused(f"{lock} exists: another session is running or one crashed") from None
    try:
        batch_file = out / "batch.json"
        if batch_file.exists():
            batch = json.loads(batch_file.read_bytes())
            if batch["identity"] != identity:
                changed = sorted(k for k in identity if identity[k] != batch["identity"].get(k))
                raise Refused(f"resume would change the batch identity: {changed}")
        else:
            batch = {"run_id": "run_" + uuid.uuid4().hex, "created_utc": utc_now(),
                     "researcher": "Rama Krishna Pudota", "identity": identity,
                     "model_snapshot": snapshot.model_dump(mode="json"),
                     "contexts_header": package.get("header", {})}  # fmt: skip
            write_once(batch_file, dump(batch))
        run_id = batch["run_id"]
        if official_inventory is not None:
            invocation = len(list((out / "gpu_inventories").glob("*_start.json"))) + 1
            write_once(out / "gpu_inventories" / f"{invocation:06d}_start.json", dump(session_gpus))
            data = dump(official_inventory.model_dump(mode="json"))
            start = out / "environment_start.json"
            if start.exists():
                if start.read_bytes() != data:
                    raise Refused("same-session resume environment differs from its start")
            else:
                write_once(start, data)
        if args.anchor is not None:
            check_anchor(out, args.anchor, args.anchor_sha256)
        events = read_events(out)
        by_key = {p.key: p for p in plan}
        stored = {}
        for path in sorted((out / "results").glob("obs_*.json")):
            if path.stem not in by_key:
                raise Refused(f"result {path.stem} is not in this batch's plan")
            stored[path.stem] = json.loads(path.read_bytes())
        # Crash recovery. A result written just before a crash lacks its end event: the result
        # proves how the attempt ended, so the event is added and marked recovered. A final
        # failure recorded just before a crash lacks its result: the result is written now,
        # without a further attempt. Any other open attempt was interrupted.
        closed = {e["attempt_id"] for e in events if e["event"] != "started"}
        for key, value in stored.items():
            result = value["generation_result"]
            if result["attempt_id"] not in closed:
                end = "completed" if result["status"] == "completed" else "failed"
                if (
                    getattr(args, "published_context_root", None) is not None
                    and result["status"] == "completed"
                    and "generator_input" not in value
                ):
                    raise Refused("published recovery is missing its input receipt")
                # Validate the candidate completion before changing the append-only history.
                # The prospective end event exists only in this validation call.
                check_result(
                    value,
                    by_key[key],
                    renderers[by_key[key].record["matrix_arm"]],
                    run_id,
                    [*events, {"attempt_id": result["attempt_id"], "event": end}],
                    out,
                )
                append_event(out, events, run_id=run_id, attempt_id=result["attempt_id"],
                             observation_key=key, event=end, recovered=True, final=True)  # fmt: skip
                closed.add(result["attempt_id"])
        for event in [e for e in events if e["event"] == "failed" and e.get("final")]:
            key = event["observation_key"]
            if key not in stored:
                data = result_bytes(by_key[key], args.phase, run_id, event["attempt_id"],
                                    failure=f"{event['failure_kind']}: {event['failure_detail'][:300]}")  # fmt: skip
                write_once(out / "results" / f"{key}.json", data)
                stored[key] = json.loads(data)
        for event in [e for e in events if e["event"] == "started"]:
            if event["attempt_id"] not in closed:
                append_event(out, events, run_id=run_id, attempt_id=event["attempt_id"],
                             observation_key=event["observation_key"], event="interrupted")  # fmt: skip
        for key, value in stored.items():
            planned = by_key[key]
            if (
                getattr(args, "published_context_root", None) is not None
                and value["generation_result"]["status"] == "completed"
                and "generator_input" not in value
            ):
                raise Refused("published execution is missing its input receipt")
            check_result(
                value, planned, renderers[planned.record["matrix_arm"]], run_id, events, out
            )
        finished = set(stored)
        print(f"run {run_id}: {len(finished)} of {len(plan)} observations finished")
        if model is None:
            model = load_model(args.model_root, args.device)
        execution_environment = None
        if getattr(args, "published_context_root", None) is not None:
            from importlib.metadata import version

            from biorag_repro.core.storage import serialize_project_json
            from biorag_repro.generation.environment import (
                BackendIdentity,
                EnvironmentRecord,
                ExecutionRecord,
                capture_environment,
                system_probes,
            )
            from biorag_repro.generation.hf_backend import PACKAGES

            settings = next(iter(configs.values())).settings
            parameter = next(model.parameters())
            if official_inventory is not None:
                if any(p.dtype != torch.bfloat16 or p.device != device for p in model.parameters()):
                    raise Refused("loaded model is not wholly bfloat16 on the selected device")
                if getattr(model, "hf_quantizer", None) is not None:
                    raise Refused("loaded model is quantized")
                if inventory_snapshot(args.model_root, settings.model.revision) != snapshot:
                    raise Refused("model snapshot changed during loading")
            captured = capture_environment(
                backend=BackendIdentity(
                    kind="official" if official_inventory else "smoke",
                    name="scripts.run_generation",
                    version=version("transformers"),
                ),
                model=(settings.model.repository, settings.model.revision),
                tokenizer=(settings.tokenizer.repository, settings.tokenizer.revision),
                packages=PACKAGES,
                probes=system_probes(Path.cwd()),
                execution=ExecutionRecord(
                    snapshot=snapshot,
                    tokenizer_file_sha256=dict(settings.tokenizer.file_sha256),
                    device=str(parameter.device),
                    dtype=str(parameter.dtype).removeprefix("torch."),
                    device_name=torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None,
                    max_position_embeddings=model.config.max_position_embeddings,
                ),
            )
            data = serialize_project_json(captured, EnvironmentRecord)
            execution_environment = sha256_bytes(data)
            write_once(out / "environments" / f"{execution_environment}.json", data)
        done = 0
        for planned in plan:
            if planned.key in finished:
                continue
            if args.stop_after is not None and done >= args.stop_after:
                break
            settings = planned.config.settings
            renderer = renderers[settings.matrix_arm]
            rendered, ids = prepare(renderer, planned)
            input_ref = None
            if getattr(args, "published_context_root", None) is not None:
                from biorag_repro.generation.inputs import build_input_record, publish_input_record

                context_ref = planned.record.get("context_record")
                input_record = build_input_record(
                    renderer,
                    rendered,
                    observation_key=planned.key,
                    context=ArtifactRef(**context_ref) if context_ref else None,
                )
                input_ref = publish_input_record(out, input_record)
            while True:
                failures = sum(e["event"] == "failed" and e["observation_key"] == planned.key
                               for e in events)  # fmt: skip
                attempt_id = "att_" + uuid.uuid4().hex
                append_event(out, events, run_id=run_id, attempt_id=attempt_id,
                             observation_key=planned.key, event="started")  # fmt: skip
                try:
                    call = generate(
                        model,
                        renderer.tokenizer,
                        rendered,
                        ids,
                        planned,
                        seeds,
                        retain_token_ids=input_ref is not None,
                    )
                    if input_ref is not None:
                        call["generator_input"] = input_ref.model_dump(mode="json")
                except Exception as error:  # noqa: BLE001 - an operational failure, never a scientific score
                    kind = failure_kind(error)
                    retryable = kind in settings.retryable_failures
                    final = not retryable or failures + 1 >= settings.retry_max_attempts
                    append_event(out, events, run_id=run_id, attempt_id=attempt_id,
                                 observation_key=planned.key, event="failed", failure_kind=kind,
                                 failure_detail=str(error)[:500], retryable=retryable,
                                 final=final)  # fmt: skip
                    if not final:
                        continue
                    write_once(out / "results" / f"{planned.key}.json",
                               result_bytes(planned, args.phase, run_id, attempt_id,
                                            failure=f"{kind}: {str(error)[:300]}"))  # fmt: skip
                    break
                raw = call.pop("raw")
                if input_ref is not None:
                    raw_bytes = raw.encode("utf-8")
                    relative = f"raw/{planned.key}/{attempt_id}.txt"
                    write_once(out / relative, raw_bytes)
                    call["raw_output"] = ArtifactRef(
                        artifact_id="generator_output_" + sha256_bytes(raw_bytes),
                        logical_ref=relative,
                        sha256=sha256_bytes(raw_bytes),
                    ).model_dump(mode="json")
                    from biorag_repro.core.storage import serialize_project_json
                    from biorag_repro.generation.receipt import ExecutionReceipt

                    receipt = ExecutionReceipt(
                        run_id=run_id.removeprefix("run_"),
                        observation_key=planned.key,
                        attempt_id=attempt_id.removeprefix("att_"),
                        environment_sha256=execution_environment,
                        input_record_sha256=input_ref.sha256,
                        rendered_input_sha256=rendered.rendered_sha256,
                        raw_output_sha256=sha256_bytes(raw_bytes),
                        output_token_ids_sha256=call["output_token_ids_sha256"],
                        facts=call.pop("execution_facts"),
                    )
                    receipt_bytes = serialize_project_json(receipt, ExecutionReceipt)
                    relative = f"receipts/{planned.key}/{attempt_id}.json"
                    write_once(out / relative, receipt_bytes)
                    call["execution_receipt"] = {
                        "logical_ref": relative,
                        "sha256": sha256_bytes(receipt_bytes),
                    }
                write_once(out / "results" / f"{planned.key}.json",
                           result_bytes(planned, args.phase, run_id, attempt_id, raw=raw,
                                        call=call))  # fmt: skip
                append_event(out, events, run_id=run_id, attempt_id=attempt_id,
                             observation_key=planned.key, event="completed")  # fmt: skip
                break
            done += 1
        return done
    finally:
        try:
            if official_inventory is not None:
                gpu_end = gpu_inventory()
                write_once(out / "gpu_inventories" / f"{invocation:06d}_end.json", dump(gpu_end))
                if gpu_end.get("gpus") != session_gpus["gpus"]:
                    raise Refused("GPU UUID inventory changed; retain failure evidence")
                end = capture_inventory(system_probes(Path.cwd()))
                data = dump(end.model_dump(mode="json"))
                target = out / "environment_end.json"
                if target.exists():
                    if target.read_bytes() != data:
                        raise Refused("end environment changed during this session")
                else:
                    write_once(target, data)
                if end != official_inventory:
                    raise Refused("environment changed during execution; retain failure evidence")
        finally:
            os.close(lock_fd)
            lock.unlink()


# ---- qualification aids: GPU inventory and boundary probe --------------------------------

GPU_FIELDS = ("index", "name", "uuid", "memory.total", "driver_version", "compute_cap",
              "pci.bus_id")  # fmt: skip


def verify_official_batch(args: argparse.Namespace) -> list[GenerationResult]:
    """Verify approved development execution against an independently retained seal.

    Local selection, parser success and a seal alone never establish official origin.
    Qualification and synthetic batches are refused before any source reconstruction.
    """
    from tempfile import TemporaryDirectory

    from biorag_repro.core.storage import read_project_json
    from biorag_repro.generation import config as policy
    from biorag_repro.generation.environment_lock import EnvironmentInventory, verify_environment

    if __package__:
        from .build_contexts import build_published
    else:
        from build_contexts import build_published

    batch = json.loads((args.out / "batch.json").read_bytes())
    identity = batch["identity"]
    if identity.get("phase") != "development" or not identity.get("official_execution", False):
        raise Refused("batch was not created as approved official development execution")
    if not policy.APPROVED_CHECKPOINTS or not policy.APPROVED_ENVIRONMENT_LOCKS:
        raise Refused("official verification has no independently approved checkpoint/environment")
    if identity["git"]["tracked_changes_sha256"] != sha256_bytes(b""):
        raise Refused("official batch was executed with modified tracked code")
    if getattr(args, "published_context_root", None) is None:
        raise Refused("official verification requires published context lineage")
    package, plan, seeds, prompt = load_inputs(args)
    if len(plan) != 1290 or set(package["header"]["arms"]) != set(FROZEN_MATRIX):
        raise Refused("official development requires the complete five-arm roster")
    lock = args.environment_lock.read_bytes()
    lock_sha = sha256_bytes(lock)
    if lock_sha not in policy.APPROVED_ENVIRONMENT_LOCKS:
        raise Refused("environment lock is not independently approved")
    check_anchor(args.out, args.anchor, args.anchor_sha256)
    anchor = json.loads(args.anchor.read_bytes())
    sealed = anchor["files"]
    inventories = []
    for name in ("environment_start.json", "environment_end.json"):
        if name not in sealed:
            raise Refused("official batch lacks sealed start/end environment inventories")
        inventory = read_project_json(args.out / name, sealed[name], EnvironmentInventory)
        inventories.append(inventory)
        checked = verify_environment(
            lock, inventory, expected_spec_sha256=lock_sha, inventory_source="stored"
        )
        if checked.outcome != "matches_approved_lock":
            raise Refused("sealed environment differs from the approved lock")
    if inventories[0] != inventories[1]:
        raise Refused("start/end inventories differ; single-session execution is not established")
    gpu_starts = sorted((args.out / "gpu_inventories").glob("*_start.json"))
    if not identity.get("session_boot_id") or not gpu_starts:
        raise Refused("official single-session GPU provenance is missing")
    for start in gpu_starts:
        end = start.with_name(start.name.replace("_start", "_end"))
        for path in (start, end):
            if sealed.get(path.relative_to(args.out).as_posix()) != sha256_file(path):
                raise Refused("GPU session inventory is absent from the external seal")
            value = json.loads(path.read_bytes())
            if (
                value.get("status") != "measured"
                or value.get("gpus") != identity["session_gpu_inventory"]
            ):
                raise Refused("GPU session identity differs between invocations")
    configurations = {p.record["matrix_arm"]: p.config for p in plan}
    snapshot = batch["model_snapshot"]
    for config in configurations.values():
        policy.require_qualified_settings(config.settings)
        settings = config.settings
        if (
            settings.purpose != "development_candidate"
            or policy.APPROVED_CHECKPOINTS.get((settings.model.repository, settings.model.revision))
            != snapshot["manifest_sha256"]
        ):
            raise Refused("checkpoint or development configuration is not approved")
    if identity["contexts_sha256"] != args.contexts_sha256:
        raise Refused("batch contexts differ from independently selected inputs")
    if identity.get("environment_lock_sha256") != lock_sha:
        raise Refused("batch names another environment lock")
    if identity["model_manifest_sha256"] != snapshot["manifest_sha256"]:
        raise Refused("batch model identity differs from its approved snapshot")
    from biorag_repro.generation.environment import ModelSnapshot

    ModelSnapshot.model_validate(snapshot)
    expected_keys = sorted(p.key for p in plan)
    if identity["planned_keys_sha256"] != sha256_bytes(canonical_json_bytes(expected_keys)):
        raise Refused("batch observation plan differs")
    if identity["generator_configurations"] != {
        arm: config.configuration_sha256 for arm, config in configurations.items()
    }:
        raise Refused("batch generator configurations differ")
    if identity["prompt_sha256"] != prompt_sha256(prompt) or identity[
        "seed_registry_sha256"
    ] != seed_registry_sha256(seeds):
        raise Refused("batch prompt or seed registry differs")
    # Reproduce published lineage from independently supplied source selections. This
    # does not accept a submitted context's own claimed upstream selection as authority.
    selection = json.loads(args.selection.read_bytes())
    with TemporaryDirectory(prefix="biorag-official-contexts-") as temporary:
        root = Path(temporary)
        rebuilt = root / "contexts.json"
        build_published(selection, rebuilt, root / "published")
        if json.loads(rebuilt.read_bytes()) != package:
            raise Refused("contexts differ from independently reconstructed scientific lineage")
    events = read_events(args.out)
    expected_set = set(expected_keys)
    if {p.stem for p in (args.out / "results").glob("obs_*.json")} != expected_set:
        raise Refused("stored results differ from the complete observation plan")
    starts, ends = {}, set()
    for event in events:
        if (
            event.get("run_id") != batch["run_id"]
            or event.get("observation_key") not in expected_set
        ):
            raise Refused("attempt history contains foreign provenance")
        attempt = event.get("attempt_id")
        if event["event"] == "started":
            if attempt in starts:
                raise Refused("attempt history starts the same attempt twice")
            starts[attempt] = event["observation_key"]
        elif event["event"] in {"completed", "failed", "interrupted"}:
            if (
                attempt not in starts
                or attempt in ends
                or starts[attempt] != event["observation_key"]
            ):
                raise Refused("attempt history closes an absent, duplicate or foreign attempt")
            ends.add(attempt)
        else:
            raise Refused("attempt history contains an unsupported event")
    if set(starts) != ends:
        raise Refused("official completion contains unfinished attempts")
    for path in (args.out / "attempts").glob("*.json"):
        if path.relative_to(args.out).as_posix() not in sealed:
            raise Refused("attempt history extends beyond the independently supplied seal")
    renderers = {
        arm: InputRenderer(args.tokenizer_root, c, prompt) for arm, c in configurations.items()
    }
    results = []
    for planned in plan:
        relative = f"results/{planned.key}.json"
        if relative not in sealed:
            raise Refused("official observation is absent from the external seal")
        value = json.loads((args.out / relative).read_bytes())
        result = GenerationResult.model_validate(value["generation_result"])
        if result.phase != "development":
            raise Refused("observation belongs to another execution phase")
        if result.status == "completed":
            if "generator_input" not in value:
                raise Refused("official completion has no stored input receipt")
            from biorag_repro.generation.environment import EnvironmentRecord
            from biorag_repro.generation.receipt import ExecutionReceipt

            receipt_ref = value.get("execution_receipt")
            if not receipt_ref or sealed.get(receipt_ref["logical_ref"]) != receipt_ref["sha256"]:
                raise Refused("official completion lacks a sealed typed execution receipt")
            receipt = read_project_json(
                args.out / receipt_ref["logical_ref"], receipt_ref["sha256"], ExecutionReceipt
            )
            env_path = f"environments/{receipt.environment_sha256}.json"
            if sealed.get(env_path) != receipt.environment_sha256:
                raise Refused("official receipt environment is outside the external seal")
            env = read_project_json(
                args.out / env_path, receipt.environment_sha256, EnvironmentRecord
            )
            execution = env.execution
            settings = planned.config.settings
            if (
                env.backend.kind != "official"
                or execution is None
                or execution.snapshot.model_dump(mode="json") != snapshot
                or execution.dtype != "bfloat16"
                or execution.device_name != policy.APPROVED_CHECKPOINT_GPU
                or not re.fullmatch(r"cuda:\d+", execution.device)
                or env.model_repository != settings.model.repository
                or env.model_revision != settings.model.revision
                or env.tokenizer_repository != settings.tokenizer.repository
                or env.tokenizer_revision != settings.tokenizer.revision
                or execution.tokenizer_file_sha256 != settings.tokenizer.file_sha256
                or execution.max_position_embeddings < settings.budgets.context_window_tokens
                or env.git.status != "ok"
                or env.git.dirty
                or env.git.commit != identity["git"]["head"]
                or env.hostname != identity["session_hostname"]
            ):
                raise Refused("official receipt model/environment provenance differs")
            for ref in (value["generator_input"], result.output.model_dump()):
                if sealed.get(ref["logical_ref"]) != ref["sha256"]:
                    raise Refused("official input/output is not covered by the external seal")
        check_result(
            value,
            planned,
            renderers[planned.record["matrix_arm"]],
            batch["run_id"],
            events,
            args.out,
        )
        results.append(result)
    return results


def gpu_inventory() -> dict:
    """What nvidia-smi reports about the visible GPUs and driver. Queries only; changes nothing."""
    try:
        listing = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout  # fmt: skip
        full = subprocess.run(
            ["nvidia-smi", "-q"], capture_output=True, text=True, check=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "reason": f"nvidia-smi: {error}"[:300]}
    gpus = [
        dict(zip(GPU_FIELDS, (part.strip() for part in line.split(","))))
        for line in listing.splitlines()
        if line.strip()
    ]
    cuda = re.search(r"^CUDA Version\s*:\s*(\S+)", full, re.MULTILINE)
    return {
        "status": "measured",
        "gpus": gpus,
        "driver_cuda_version": cuda.group(1) if cuda else None,
        "nvidia_smi_q_sha256": sha256_bytes(full.encode()),
        "nvidia_smi_q": full,
    }


def inventory(device: str | None) -> dict:
    """A session record: host, packages, nvidia-smi and, for a named CUDA device, torch's view."""
    import tokenizers
    import torch
    import transformers

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    record = {
        "captured_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "pip_freeze": freeze,
        "pip_freeze_sha256": sha256_bytes(freeze.encode()),
        "nvidia": gpu_inventory(),
        "torch_device": None,
    }
    if device is not None:
        target = torch.device(device)
        if target.type != "cuda" or target.index is None:
            raise Refused("inventory --device names one explicit CUDA device, for example cuda:0")
        free, total = torch.cuda.mem_get_info(target.index)
        record["torch_device"] = {
            "device": device,
            "name": torch.cuda.get_device_name(target.index),
            "capability": list(torch.cuda.get_device_capability(target.index)),
            "total_bytes": total,
            "free_bytes": free,
            "cudnn": torch.backends.cudnn.version(),
        }
    return record


def model_positions(model_root: Path) -> dict:
    """The checkpoint's own context-length settings, read from its config.json."""
    config = json.loads((model_root / "config.json").read_bytes())
    return {k: config.get(k) for k in ("max_position_embeddings", "rope_scaling")}


BOUNDARY_ORIGIN = "development contexts joined and cut to the evidence budget; not retrieval output"


def boundary_context(renderer: InputRenderer, texts: list[str], size: int) -> tuple[str, int]:
    """Development text cut to exactly `size` tokens by the pinned counter, and how many of
    `texts` it draws on.

    Used only by the probe, never as supplied evidence: the scientific path never cuts a
    chunk. Decoding a token prefix can merge tokens at the cut, so nearby cut points are
    tried and the first one that recounts to exactly `size` is kept.
    """
    tokenizer = renderer.tokenizer
    joined = "\n\n".join(texts)
    ids = tokenizer(joined, add_special_tokens=False)["input_ids"]
    if len(ids) < size:
        raise Refused(f"the development contexts hold {len(ids)} tokens, fewer than {size}")
    for n in sorted(range(size - 8, size + 9), key=lambda n: abs(n - size)):
        text = tokenizer.decode(ids[:n], skip_special_tokens=False)
        if n <= len(ids) and renderer.counter(text) == size:
            starts = [len("\n\n".join(texts[:i])) + (2 if i else 0) for i in range(len(texts))]
            return text, sum(start < len(text) for start in starts)
    raise Refused(f"no cut of the development contexts recounts to exactly {size} tokens")


def boundary_input(args: argparse.Namespace) -> dict:
    """The probe's input, built and checked without a model: everything except the GPU call."""
    out = args.out
    if (out.parent / "batch.json").exists() or out.parent.name in (
        "results",
        "attempts",
        "anchors",
    ):
        raise Refused("a probe record is never written into a batch directory")
    package, plan, seeds, prompt = load_inputs(args)
    chosen = [p for p in plan if p.record["item_id"] == args.item_id
              and p.record["matrix_arm"] == "confirmatory"
              and p.record["condition"] == args.condition and p.repeat == 1]  # fmt: skip
    if len(chosen) != 1:
        raise Refused(f"no confirmatory {args.condition} record for {args.item_id}")
    planned = chosen[0]
    budgets = planned.config.settings.budgets
    positions = model_positions(args.model_root)
    if positions["rope_scaling"] is not None or (
        budgets.context_window_tokens > positions["max_position_embeddings"]
    ):
        raise Refused(f"configured window {budgets.context_window_tokens} exceeds the "
                      f"checkpoint's positions {positions}")  # fmt: skip
    renderer = InputRenderer(args.tokenizer_root, planned.config, prompt)
    # the chosen record's own context first, then the same condition's in package order
    sources = [planned.record] + [
        r for r in package["records"]
        if r["condition"] == args.condition and r["context"] is not None
        and r is not planned.record
    ]  # fmt: skip
    context, used = boundary_context(renderer, [r["context"] for r in sources],
                                     budgets.evidence_tokens)  # fmt: skip
    record = dict(planned.record, context=context, context_tokens=budgets.evidence_tokens,
                  context_sha256=sha256_bytes(context.encode("utf-8")))  # fmt: skip
    boundary = Planned("probe", record, 1, planned.config)
    rendered, ids = prepare(renderer, boundary)
    if rendered.input_tokens + budgets.max_new_tokens > budgets.context_window_tokens:
        raise Refused("input plus output allowance exceeds the configured window")
    return {
        "planned": planned, "boundary": boundary, "renderer": renderer, "rendered": rendered,
        "ids": ids, "seeds": seeds, "positions": positions,
        "sources": [(r["item_id"], r["matrix_arm"], r["condition"]) for r in sources[:used]],
    }  # fmt: skip


def probe(args: argparse.Namespace, model: Any = None) -> dict:
    """One development-only call at the exact evidence budget with the full output allowance.

    Writes one standalone record; no batch, observation key, parse or score is created.
    """
    import torch

    built = boundary_input(args)
    planned, rendered = built["planned"], built["rendered"]
    settings = planned.config.settings
    budgets = settings.budgets
    snapshot = inventory_snapshot(args.model_root, settings.model.revision)
    if model is None:
        model = load_model(args.model_root, args.device)
    call = generate(model, built["renderer"].tokenizer, rendered, built["ids"], built["boundary"],
                    built["seeds"], force_full_output=True)  # fmt: skip
    raw = call.pop("raw")
    result = {
        "kind": "boundary_probe",
        "not_an_observation": True,
        "researcher": "Rama Krishna Pudota",
        "probed_utc": utc_now(),
        "contexts_sha256": args.contexts_sha256,
        "item_id": args.item_id,
        "condition": args.condition,
        "resolved_config_sha256": planned.config.configuration_sha256,
        "model_manifest_sha256": snapshot.manifest_sha256,
        "checkpoint_positions": built["positions"],
        "budgets": budgets.model_dump(mode="json"),
        "context_origin": BOUNDARY_ORIGIN,
        "context_sources": [list(source) for source in built["sources"]],
        "boundary_context_sha256": built["boundary"].record["context_sha256"],
        "context_tokens": budgets.evidence_tokens,
        "output_sha256": sha256_bytes(raw.encode("utf-8")),
        "device": str(torch.device(args.device)),
        **call,
    }
    if result["output_tokens"] != budgets.max_new_tokens:
        result["probe_status"] = "output_allowance_not_reached"
    else:
        result["probe_status"] = "complete"
    write_once(args.out, dump(result))
    return result


def seal(out: Path) -> str:
    """Write the next anchor over every finished result and event; return its SHA-256."""
    files = {
        p.relative_to(out).as_posix(): sha256_file(p)
        for folder, pattern in (
            ("results", "obs_*.json"),
            ("attempts", "[0-9]*.json"),
            ("inputs", "*.json"),
            ("raw", "**/*.txt"),
            ("receipts", "**/*.json"),
            ("environments", "*.json"),
            ("gpu_inventories", "*.json"),
        )
        for p in sorted((out / folder).glob(pattern))
    }
    for name in ("environment_start.json", "environment_end.json"):
        if (out / name).exists():
            files[name] = sha256_file(out / name)
    files["batch.json"] = sha256_file(out / "batch.json")
    previous = sorted((out / "anchors").glob("[0-9]*.json"))
    anchor = {"sealed_utc": utc_now(), "files": files,
              "previous_anchor_sha256": sha256_file(previous[-1]) if previous else None}  # fmt: skip
    data = dump(anchor)
    write_once(out / "anchors" / f"{len(previous) + 1:06d}.json", data)
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    go = commands.add_parser("run")
    for name in ("contexts", "configs", "prompt", "seeds", "tokenizer-root", "model-root", "out"):
        go.add_argument(f"--{name}", type=Path, required=True)
    go.add_argument("--published-context-root", type=Path)
    go.add_argument("--official", action="store_true")
    go.add_argument("--environment-lock", type=Path)
    go.add_argument("--selection", type=Path)
    go.add_argument("--contexts-sha256", required=True)
    go.add_argument("--phase", required=True)
    go.add_argument("--device", required=True, help="explicit device, for example cuda:0")
    go.add_argument("--anchor", type=Path)
    go.add_argument("--anchor-sha256")
    go.add_argument("--stop-after", type=int)
    check_inputs = commands.add_parser("preflight")
    for name in (
        "contexts",
        "configs",
        "prompt",
        "seeds",
        "tokenizer-root",
        "published-context-root",
    ):
        check_inputs.add_argument(f"--{name}", type=Path, required=True)
    check_inputs.add_argument("--contexts-sha256", required=True)
    check_inputs.set_defaults(phase="development")
    official_check = commands.add_parser("verify-official")
    for name in (
        "contexts",
        "configs",
        "prompt",
        "seeds",
        "tokenizer-root",
        "out",
        "published-context-root",
        "environment-lock",
        "selection",
        "anchor",
    ):
        official_check.add_argument(f"--{name}", type=Path, required=True)
    official_check.add_argument("--contexts-sha256", required=True)
    official_check.add_argument("--anchor-sha256", required=True)
    official_check.set_defaults(phase="development")
    test = commands.add_parser("probe")
    for name in ("contexts", "configs", "prompt", "seeds", "tokenizer-root", "model-root", "out"):
        test.add_argument(f"--{name}", type=Path, required=True)
    test.add_argument("--contexts-sha256", required=True)
    test.add_argument("--device", required=True)
    test.add_argument("--item-id", required=True)
    test.add_argument("--condition", required=True, choices=("JATS", "PDF_DOCLING"))
    test.set_defaults(phase="qualification")
    look = commands.add_parser("inventory")
    look.add_argument("--out", type=Path, required=True)
    look.add_argument("--device")
    proposed = commands.add_parser("propose-environment-identity")
    proposed.add_argument("--freeze", type=Path, required=True)
    proposed.add_argument("--candidate-lock", type=Path, required=True)
    proposed.add_argument("--git-head", required=True)
    proposed.add_argument("--packaging-root", type=Path)
    proposed.add_argument("--packaging-reference", type=Path)
    proposed.add_argument("--project-root", type=Path)
    proposed.add_argument("--out", type=Path, required=True)
    for name in ("seal", "verify"):
        sub = commands.add_parser(name)
        sub.add_argument("--out", type=Path, required=True)
        if name == "verify":
            sub.add_argument("--anchor", type=Path, required=True)
            sub.add_argument("--anchor-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        if (args.anchor is None) != (args.anchor_sha256 is None):
            parser.error("--anchor and --anchor-sha256 go together")
        print(f"finished {run(args)} observations in this session")
    elif args.command == "preflight":
        print(json.dumps(preflight(args), sort_keys=True))
    elif args.command == "verify-official":
        results = verify_official_batch(args)
        print(
            f"verified {len(results)} approved development observations; final test remains closed"
        )
    elif args.command == "probe":
        result = probe(args)
        print(f"probe {result['probe_status']}: {result['output_tokens']} output tokens")
    elif args.command == "inventory":
        write_once(args.out, dump(inventory(args.device)))
        print(f"inventory sha256 {sha256_file(args.out)}; record it outside the data root")
    elif args.command == "propose-environment-identity":
        record = prospective_freeze_identity(
            args.freeze.read_text(encoding="utf-8"),
            args.candidate_lock.read_bytes(),
            args.git_head,
            packaging_root=args.packaging_root,
            packaging_reference=args.packaging_reference.read_bytes()
            if args.packaging_reference
            else None,
            project_root=args.project_root,
        )
        write_once(args.out, dump(record))
        print(f"candidate identity sha256 {sha256_file(args.out)}; not approved for execution")
    elif args.command == "seal":
        print(f"anchor sha256 {seal(args.out)}; record it outside the data root")
    else:
        check_anchor(args.out, args.anchor, args.anchor_sha256)
        print("every sealed file matches the anchor")


if __name__ == "__main__":
    main()
