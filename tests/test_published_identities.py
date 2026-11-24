"""Published identities of the synthetic rehearsal chain, pinned byte for byte.

The values were recorded before the storage and verification code was simplified. They fail
on any change of the handoff artifact id or build manifest, or of the bytes of the primary
and secondary R tables. Left out: records that embed a run id, a timestamp or the execution
host (`run_record.json`, `secondary_record.json`, `report.md`), and files that depend on the
local installation rather than on the data (`package_inventory.csv`, the rendered figure).
Everything is synthetic; no value is a scientific result.
"""

from pathlib import Path

from test_r_input_fixture import derive, needs_r, publish_frozen_handoff, run_primary

from biorag_repro.core.hashing import sha256_file

HANDOFF_BUILD = (
    "build_0af0f50bc463b61c97a275ed27a457fb1d455fd8d00a6666d48d7a01f2b2232e",
    "0af0f50bc463b61c97a275ed27a457fb1d455fd8d00a6666d48d7a01f2b2232e",
    "analysis/handoff/b568c4d2cd9c3b7d2bc66eaaeb62d1a88aec0bccb49568c5f9431605dc2cb960.build.json",
)
PRIMARY = {
    "coprimary_results.csv": "2f27d0f51c50279825ce29bf3749da12af051b4da0f8c104f214a14fb36e1239",
    "paper_counts.csv": "a10733bc370026725562ae2f7dd706eed314b7c10227115469e75b81a659cb33",
    "state_counts.csv": "d182b7397d699bf1d9ed233fda506ca01567d1b278513dac45876dc220b0e95a",
}
SECONDARY = {
    "secondary_answer_class.csv": "52e435c788786be782551df85ccc18a5bb2cf92c1aba1301a938c4dd06ac3a0d",
    "secondary_completion.csv": "2c2033425df99c1c32cfd658630b3e79eba969a815a39b3dfab430d6978baccf",
    "secondary_correct_without_evidence.csv": "47e9714d4f344011e941570c8c03da015304e33e4c300f72c9d9e3723607f2c4",
    "secondary_discordance.csv": "3dab22ae3500ad67f5d6d19dd1113848f22a8f58af37047380300f54ebf1d3e5",
    "secondary_populations.csv": "77dc53f2226eb2539547e2fe18177507638e7c24e7159b21de7ec5d28b96f25a",
    "secondary_repeat_outcomes.csv": "b6cf564cadf641b33198858febd753cb71623840fab1810955819d07575ff12c",
    "secondary_repeat_variability.csv": "e33e9d1ca3784056951748a771737a775ded9c42d9ef8e17706ed5d4ceb32bbd",
    "secondary_robustness.csv": "a0c0ad7d832b065c553a70c6f51fe70dbbafc9f89bc1d6e2de1f827489cd88cd",
    "secondary_scope.csv": "f526a4bf2f9b453f58acc0fa132e76039bdc2a5977c7dae783506b9d85b98ba5",
    "secondary_stage_a.csv": "f142a1ad58de6b6d1442a84e4ff9624e87e426b48f8aac513bbf441d88c03ab6",
    "secondary_stage_b_failure.csv": "52fd47f8394b14c2340fa8ec95c25bddf327ffd7b541bf1abfddc8d8d124942e",
    "secondary_unavailable.csv": "5aa9eb287e0d31a43324a33cffb5b1b101899cc0fe118dd579fa7d8372698b02",
}


def hashes(directory: Path, names: dict[str, str]) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in names}


@needs_r
def test_handoff_and_r_tables_keep_their_published_bytes(tmp_path):
    build = publish_frozen_handoff(tmp_path)
    assert (build.artifact_id, build.sha256, build.logical_ref) == HANDOFF_BUILD
    primary = run_primary(tmp_path, build)
    assert hashes(tmp_path / primary, PRIMARY) == PRIMARY
    assert hashes(tmp_path / derive(tmp_path, build, primary), SECONDARY) == SECONDARY
