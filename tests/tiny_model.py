"""Tiny random-weight Qwen3 model for the opt-in engineering smoke tests (never scientific)."""

import json
from pathlib import Path

EOS, PAD = 151645, 151643


def build_tiny(directory: Path, *, generation_config: dict | None = None) -> Path:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=8192,
        bos_token_id=None,
        eos_token_id=EOS,
        pad_token_id=PAD,
    )
    Qwen3ForCausalLM(config).save_pretrained(directory, safe_serialization=True)
    if generation_config is not None:
        (directory / "generation_config.json").write_text(json.dumps(generation_config))
    return directory


_LOADED: dict[tuple[str, str], object] = {}


def load_once(directory: Path, device: str):
    """Load the real tiny model at most once per directory in a smoke session."""
    from biorag_repro.generation.hf_backend import load_local_model

    key = (str(directory), device)
    if key not in _LOADED:
        _LOADED[key] = load_local_model(directory, device)
    return _LOADED[key]
