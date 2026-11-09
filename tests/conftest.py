"""Test tiers.

fast        every test without a marker: synthetic fixtures, no weights, no local data.
real_data   reads the authoritative local development artifacts; opt in with
            BIORAG_REAL_DATA=<integration manifest> (see tests/test_real_data.py).
model_smoke runs a real (tiny) model; opt in with BIORAG_MODEL_SMOKE=1. BIORAG_SMOKE_TIMEOUT_S
            (default 120) bounds each one, and the model is built once per session.

Unselected tiers are skipped with a reason, so `pytest -rs` always lists them and a tier can
never disappear silently. Smoke results are engineering checks, never scientific evidence.
"""

import os
import shutil
import signal
from pathlib import Path

import pytest

ENABLED = os.environ.get("BIORAG_MODEL_SMOKE") == "1"
TIMEOUT_S = int(os.environ.get("BIORAG_SMOKE_TIMEOUT_S", "120"))


def pytest_configure(config):
    """Tests that start Rscript themselves get the same pinned BLAS kernel as the product code.

    The pin is read when R starts, so it has to be in the environment of this process; see
    biorag_repro.evaluation.analysis_result.R_BLAS_PIN.
    """
    from biorag_repro.evaluation.analysis_result import R_BLAS_PIN

    os.environ.update(R_BLAS_PIN)


def pytest_sessionstart(session):
    """BIORAG_REQUIRE_R=1 makes a missing R toolchain a failure instead of silent skips.

    The Python/R handoff, rehearsal, result and supplement tests skip when Rscript or the
    analysis renv library is absent. CI sets this so those tests cannot disappear unnoticed.
    """
    if os.environ.get("BIORAG_REQUIRE_R") != "1":
        return
    library = Path(__file__).resolve().parent.parent / "analysis" / "renv" / "library"
    if shutil.which("Rscript") is None or not library.is_dir():
        pytest.exit("BIORAG_REQUIRE_R=1 but Rscript or analysis/renv/library is missing", 2)


def pytest_collection_modifyitems(config, items):
    skips = {}
    if not ENABLED:
        skips["model_smoke"] = "model smoke test; set BIORAG_MODEL_SMOKE=1 to run"
    if not os.environ.get("BIORAG_REAL_DATA"):
        skips["real_data"] = "real development artifacts; set BIORAG_REAL_DATA=<manifest> to run"
    for item in items:
        for marker, reason in skips.items():
            if marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))


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
