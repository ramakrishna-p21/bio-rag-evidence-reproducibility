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
    stored: dict, planned: Planned, renderer: InputRenderer | None, run_id: str, events: list[dict]
) -> None:
    """A stored result must be exactly what this batch would record for that observation."""
    try:
        result = GenerationResult.model_validate(stored["generation_result"])
    except (KeyError, ValueError) as error:
        raise Refused(f"result {planned.key} is not a valid generation result: {error}") from None
    record = planned.record
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


def check_anchor(out: Path, anchor: Path, anchor_sha256: str) -> None:
    data = anchor.read_bytes()
    if sha256_bytes(data) != anchor_sha256:
        raise Refused("the anchor differs from its trusted SHA-256")
    sealed = json.loads(data)
    for relative, digest in sealed["files"].items():
        path = out / relative
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
             seeds, force_full_output: bool = False) -> dict:  # fmt: skip
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
    output_ids = out[0, len(ids) :].tolist()
    stop = "eos" if output_ids and output_ids[-1] in stop_ids else "length"
    body = output_ids[:-1] if stop == "eos" else output_ids
    raw = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {
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
        output = ArtifactRef(
            artifact_id="generator_output_" + sha256_bytes(raw.encode("utf-8")),
            logical_ref=f"results/{planned.key}.json",
            sha256=sha256_bytes(raw.encode("utf-8")),
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

    package, plan, seeds, prompt = load_inputs(args)
    configs = {p.record["matrix_arm"]: p.config for p in plan}
    renderers = {a: InputRenderer(args.tokenizer_root, c, prompt) for a, c in configs.items()}
    revisions = {c.settings.model.revision for c in configs.values()}
    if len(revisions) != 1:
        raise Refused("all arms of a batch use one model revision")
    snapshot = inventory_snapshot(args.model_root, revisions.pop())
    device = torch.device(args.device)
    identity = batch_identity(args, args.contexts_sha256, plan, seeds, prompt, snapshot,
                              environment(device))  # fmt: skip
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
            check_result(value, planned, renderers[planned.record["matrix_arm"]], run_id, events)
        finished = set(stored)
        print(f"run {run_id}: {len(finished)} of {len(plan)} observations finished")
        if model is None:
            model = load_model(args.model_root, args.device)
        done = 0
        for planned in plan:
            if planned.key in finished:
                continue
            if args.stop_after is not None and done >= args.stop_after:
                break
            settings = planned.config.settings
            renderer = renderers[settings.matrix_arm]
            rendered, ids = prepare(renderer, planned)
            while True:
                failures = sum(e["event"] == "failed" and e["observation_key"] == planned.key
                               for e in events)  # fmt: skip
                attempt_id = "att_" + uuid.uuid4().hex
                append_event(out, events, run_id=run_id, attempt_id=attempt_id,
                             observation_key=planned.key, event="started")  # fmt: skip
                try:
                    call = generate(model, renderer.tokenizer, rendered, ids, planned, seeds)
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
                write_once(out / "results" / f"{planned.key}.json",
                           result_bytes(planned, args.phase, run_id, attempt_id, raw=raw,
                                        call=call))  # fmt: skip
                append_event(out, events, run_id=run_id, attempt_id=attempt_id,
                             observation_key=planned.key, event="completed")  # fmt: skip
                break
            done += 1
        return done
    finally:
        os.close(lock_fd)
        lock.unlink()


# ---- qualification aids: GPU inventory and boundary probe --------------------------------

GPU_FIELDS = ("index", "name", "uuid", "memory.total", "driver_version", "compute_cap",
              "pci.bus_id")  # fmt: skip


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
        for folder, pattern in (("results", "obs_*.json"), ("attempts", "[0-9]*.json"))
        for p in sorted((out / folder).glob(pattern))
    }
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
    go.add_argument("--contexts-sha256", required=True)
    go.add_argument("--phase", required=True)
    go.add_argument("--device", required=True, help="explicit device, for example cuda:0")
    go.add_argument("--anchor", type=Path)
    go.add_argument("--anchor-sha256")
    go.add_argument("--stop-after", type=int)
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
    elif args.command == "probe":
        result = probe(args)
        print(f"probe {result['probe_status']}: {result['output_tokens']} output tokens")
    elif args.command == "inventory":
        write_once(args.out, dump(inventory(args.device)))
        print(f"inventory sha256 {sha256_file(args.out)}; record it outside the data root")
    elif args.command == "seal":
        print(f"anchor sha256 {seal(args.out)}; record it outside the data root")
    else:
        check_anchor(args.out, args.anchor, args.anchor_sha256)
        print("every sealed file matches the anchor")


if __name__ == "__main__":
    main()
