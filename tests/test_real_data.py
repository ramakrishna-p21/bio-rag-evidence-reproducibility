"""Integration tier: authoritative local development artifacts, tokenizer only, no model.

Skipped unless BIORAG_REAL_DATA names a local JSON manifest, so the machine-specific paths and
pinned hashes stay out of the repository. Run with:

    BIORAG_REAL_DATA=/path/to/p027_integration.json pytest -m real_data -rs

Manifest keys (relative paths are relative to the manifest): context_sources,
context_sources_sha256, view_directory, view_record_sha256, tokenizer_root, scope,
retrieval_config_sha256, and optionally expected_roster_sha256 and expected_observations.
"""

import json
import os
import shutil
from pathlib import Path

import pytest
from test_generation_config import synthetic_settings, to_toml
from test_generation_prompt import artifact

from biorag_repro.generation import driver
from biorag_repro.generation.prompt import output_schema_sha256, prompt_sha256
from biorag_repro.generation.roster import roster_sha256
from biorag_repro.generation.sources import load_context_sources
from biorag_repro.retrieval import context

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def manifest():
    path = Path(os.environ["BIORAG_REAL_DATA"]).resolve()
    value = json.loads(path.read_text())
    for key in ("context_sources", "view_directory", "tokenizer_root", "scope"):
        value[key] = (path.parent / value[key]).resolve()
    return value


@pytest.fixture(scope="module")
def sources(manifest):
    return load_context_sources(
        manifest["context_sources"],
        manifest["context_sources_sha256"],
        manifest["view_directory"],
        manifest["view_record_sha256"],
    )


def test_every_published_context_reads_back_from_its_inputs(sources, manifest):
    """Each context is recomputed from its chunk sets and ranking and must equal its pinned file."""
    assert sources
    for source in sources:
        value = context.read(
            source.context,
            source.manifest,
            source.selection,
            source.configuration,
            source.configuration_root,
            manifest["tokenizer_root"],
            source.root,
        )
        assert value.exact_context and value.generator_tokens <= 2048


def test_roster_reconstruction_has_the_recorded_identity(sources, manifest, tmp_path):
    prompt = artifact()
    settings = synthetic_settings()
    settings.update(
        prompt_sha256=prompt_sha256(prompt),
        output_schema_sha256=output_schema_sha256(prompt),
        retrieval_config_sha256=manifest["retrieval_config_sha256"],
    )
    (tmp_path / "generator.toml").write_text(to_toml(settings))
    (tmp_path / "prompt.json").write_text(prompt.model_dump_json())
    shutil.copy(manifest["scope"], tmp_path / "scope.json")
    plan = driver.DriverPlan(
        run_root=tmp_path / "runs",
        anchor_dir=tmp_path / "anchors",
        config=tmp_path / "generator.toml",
        prompt=tmp_path / "prompt.json",
        seeds=None,
        tokenizer_root=manifest["tokenizer_root"],
        view_directory=manifest["view_directory"],
        view_record_sha256=manifest["view_record_sha256"],
        scope=tmp_path / "scope.json",
        repository=Path.cwd(),
        context_sources=manifest["context_sources"],
        context_sources_sha256=manifest["context_sources_sha256"],
    )
    record = driver.prepare(plan, sources).built.record
    if "expected_observations" in manifest:
        assert len(record.rows) == manifest["expected_observations"]
    if "expected_roster_sha256" in manifest:
        assert roster_sha256(record) == manifest["expected_roster_sha256"]
