"""Hugging Face Transformers backend: one local checkpoint, offline, explicit settings.

Nothing is downloaded and nothing is chosen for the caller. The model directory must be
named by the configured revision, its files are inventoried before and after loading, the
loaded weights must have the configured precision on the one requested device, and the
decoding settings come from the generator configuration alone: the checkpoint's own
sampling defaults are ignored (its stop tokens are used, and reported).

Whether a backend is official is derived here from what was verified, never accepted from
the caller. Even then the inventory shows which files were present, not which bytes the
process executed; see model_snapshot.
"""

import json
import re
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from biorag_repro.core.config import ResolvedConfiguration
from biorag_repro.generation import config as generator_config
from biorag_repro.generation.config import GeneratorConfiguration
from biorag_repro.generation.environment import (
    BackendIdentity,
    BackendKind,
    EnvironmentProbes,
    EnvironmentRecord,
    ExecutionRecord,
    capture_environment,
)
from biorag_repro.generation.model_snapshot import SnapshotError, inventory_snapshot
from biorag_repro.generation.prompt import RenderedInput
from biorag_repro.generation.receipt import ExecutionFacts, GenerationSettings, token_ids_sha256
from biorag_repro.generation.runner import BackendFailure, GenerationOutput
from biorag_repro.retrieval import context as phase5
from biorag_repro.retrieval.models import Snapshot

HF_BACKEND_NAME = "biorag_repro.generation.hf_backend.HuggingFaceBackend"
PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "numpy",
    "huggingface-hub",
    "accelerate",
)


def is_official_device(device: str) -> bool:
    """The official condition is one explicitly named CUDA device."""
    return re.fullmatch(r"cuda:\d+", device) is not None


def load_local_model(directory: Path, device: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(directory),
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        device_map=None,
    )
    return model.to(device)  # type: ignore[arg-type]


def _stop_tokens(directory: Path) -> tuple[list[int], int | None]:
    """Stop and padding ids from the checkpoint's generation_config.json; nothing is guessed."""
    try:
        stored = json.loads((directory / "generation_config.json").read_text(encoding="utf-8"))
        eos = stored["eos_token_id"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SnapshotError("generation_config.json does not name the stop tokens") from error
    ids = [eos] if isinstance(eos, int) else eos
    pad = stored.get("pad_token_id")
    if not ids or not all(type(i) is int for i in ids) or not (pad is None or type(pad) is int):
        raise SnapshotError("generation_config.json has invalid stop token ids")
    return list(ids), pad


class HuggingFaceBackend:
    def __init__(
        self,
        config: ResolvedConfiguration[GeneratorConfiguration],
        *,
        model_dir: Path,
        tokenizer_root: Path,
        device: str,
        load_model: Callable[[Path, str], Any] = load_local_model,
    ) -> None:
        import torch

        self.config = config
        settings = config.settings
        self.device = device
        self.snapshot = inventory_snapshot(model_dir, settings.model.revision)
        self.stop_ids, self.pad_id = _stop_tokens(model_dir)
        stored = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        positions = stored.get("max_position_embeddings")
        if type(positions) is not int or positions <= 0:
            raise SnapshotError("config.json does not state max_position_embeddings")
        if settings.budgets.context_window_tokens > positions:
            raise ValueError(
                f"configured context window {settings.budgets.context_window_tokens} exceeds "
                f"the model's {positions} positions"
            )
        self.max_positions = positions

        pin = settings.tokenizer
        self.tokenizer: Any = phase5.QwenCounter(
            tokenizer_root,
            Snapshot(
                repository=pin.repository,
                state="pinned",
                revision=pin.revision,
                file_hashes=dict(pin.file_sha256),
            ),
        ).tokenizer
        if self.tokenizer.eos_token_id not in self.stop_ids:
            raise SnapshotError("the checkpoint's stop tokens do not include the chat end token")

        self.model: Any = load_model(model_dir, device)
        self.model.eval()
        parameters = list(self.model.parameters())
        if not parameters or any(p.dtype != torch.bfloat16 for p in parameters):
            raise ValueError("loaded weights do not have the configured dtype bfloat16")
        if any(p.device != torch.device(device) for p in parameters):
            raise ValueError(f"loaded weights are not all on the requested device {device}")
        if getattr(self.model, "hf_quantizer", None) is not None:
            raise ValueError("a quantized model was loaded")
        if inventory_snapshot(model_dir, settings.model.revision) != self.snapshot:
            raise SnapshotError("model files changed while loading")

        official = (
            settings.purpose == "development_candidate"
            and generator_config.APPROVED_CHECKPOINTS.get(
                (settings.model.repository, settings.model.revision)
            )
            == self.snapshot.manifest_sha256
            and is_official_device(device)
        )
        kind: BackendKind = "official" if official else "smoke"
        cls = type(self)
        self.identity = BackendIdentity(
            kind=kind,
            name=f"{cls.__module__}.{cls.__qualname__}",
            version=metadata.version("transformers"),
        )

    def environment(self, probes: EnvironmentProbes) -> EnvironmentRecord:
        import torch

        settings = self.config.settings
        name = torch.cuda.get_device_name(self.device) if self.device.startswith("cuda") else None
        return capture_environment(
            backend=self.identity,
            model=(settings.model.repository, settings.model.revision),
            tokenizer=(settings.tokenizer.repository, settings.tokenizer.revision),
            packages=PACKAGES,
            probes=probes,
            execution=ExecutionRecord(
                snapshot=self.snapshot,
                tokenizer_file_sha256=dict(settings.tokenizer.file_sha256),
                device=self.device,
                device_name=name,
                dtype="bfloat16",
                max_position_embeddings=self.max_positions,
            ),
        )

    def generate(
        self, rendered: RenderedInput, *, seed: int | None, max_new_tokens: int
    ) -> GenerationOutput:
        import torch
        from transformers import GenerationConfig

        settings = self.config.settings
        decoding, budgets = settings.decoding, settings.budgets
        if (
            rendered.resolved_config_sha256 != self.config.configuration_sha256
            or max_new_tokens != budgets.max_new_tokens
        ):
            raise BackendFailure("configuration_mismatch", "call differs from the configuration")
        if decoding.do_sample and seed is None:
            raise BackendFailure("missing_seed", "sampling decoding needs the planned seed")
        ids = self.tokenizer(
            rendered.rendered_text, add_special_tokens=False, truncation=False, padding=False
        )["input_ids"]
        if token_ids_sha256(ids) != rendered.input_token_ids_sha256:
            raise BackendFailure(
                "input_token_mismatch", "rendered text does not tokenize to the recorded ids"
            )
        if len(ids) > budgets.max_input_tokens or len(ids) + max_new_tokens > self.max_positions:
            raise BackendFailure("input_too_long", f"{len(ids)} input tokens exceed a limit")

        sampling = (
            {"temperature": decoding.temperature, "top_p": decoding.top_p, "top_k": decoding.top_k}
            if decoding.do_sample
            else {}
        )
        generation_config = GenerationConfig(  # type: ignore[no-untyped-call]
            max_new_tokens=max_new_tokens,
            do_sample=decoding.do_sample,
            eos_token_id=self.stop_ids,
            pad_token_id=self.pad_id,
            **sampling,
        )
        if seed is not None:
            torch.manual_seed(seed)
        input_ids = torch.tensor([ids], device=self.device)
        try:
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    generation_config=generation_config,
                    use_model_defaults=False,  # the checkpoint's sampling defaults must not leak in
                )
        except torch.cuda.OutOfMemoryError as error:
            raise BackendFailure("out_of_memory", str(error)) from error

        row = output[0].tolist()
        new = row[len(ids) :]
        if row[: len(ids)] != ids:
            raise BackendFailure("prompt_altered", "generation returned a different input prefix")
        if any(token in self.stop_ids for token in new[:-1]):
            raise BackendFailure("early_stop_token", "a stop token occurs before the last token")
        stop: Literal["eos", "max_new_tokens"]
        if new and new[-1] in self.stop_ids:
            stop, body = "eos", new[:-1]
        elif len(new) == max_new_tokens:
            stop, body = "max_new_tokens", new
        else:
            raise BackendFailure(
                "unexpected_stop", f"stopped after {len(new)} tokens without cause"
            )
        text = self.tokenizer.decode(
            body, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        facts = ExecutionFacts(
            input_token_ids_sha256=token_ids_sha256(ids),
            input_token_count=len(ids),
            generation=GenerationSettings(
                max_new_tokens=max_new_tokens,
                do_sample=decoding.do_sample,
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                top_k=decoding.top_k,
                eos_token_ids=self.stop_ids,
                pad_token_id=self.pad_id,
                seed=seed,
                effective=generation_config.to_dict(),
            ),
            output_token_ids=new,
            stop_reason=stop,
        )
        return GenerationOutput(text, facts)
