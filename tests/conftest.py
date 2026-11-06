"""Opt-in engineering smoke tests that run a real (tiny) model.

Default runs never load model weights. Set BIORAG_MODEL_SMOKE=1 to run tests marked
`model_smoke`; BIORAG_SMOKE_TIMEOUT_S (default 120) bounds each one. The model is built
once per session. Smoke results are engineering checks, never scientific evidence.
"""

import os
import signal

import pytest

ENABLED = os.environ.get("BIORAG_MODEL_SMOKE") == "1"
TIMEOUT_S = int(os.environ.get("BIORAG_SMOKE_TIMEOUT_S", "120"))


def pytest_collection_modifyitems(config, items):
    if ENABLED:
        return
    skip = pytest.mark.skip(reason="model smoke test; set BIORAG_MODEL_SMOKE=1 to run")
    for item in items:
        if "model_smoke" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _smoke_timeout(request):
    if not ENABLED or "model_smoke" not in request.keywords:
        yield
        return

    def expired(signum, frame):
        pytest.fail(f"smoke test exceeded {TIMEOUT_S}s; record the limitation, do not retry")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(TIMEOUT_S)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture(scope="session")
def real_tiny(tmp_path_factory):
    from tiny_model import build_tiny

    directory = tmp_path_factory.mktemp("real_tiny") / ("1" * 40)
    directory.mkdir()
    return build_tiny(directory)


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    """A snapshot directory of small fake files: enough to inventory, never loaded as weights."""
    import json

    from tiny_model import EOS, PAD

    directory = tmp_path_factory.mktemp("fake") / ("1" * 40)
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"max_position_embeddings": 8192}))
    (directory / "generation_config.json").write_text(
        json.dumps({"eos_token_id": EOS, "pad_token_id": PAD})
    )
    (directory / "model.safetensors").write_bytes(b"not real weights")
    return directory
