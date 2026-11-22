import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from synthetic_environment import LOCK, inventory

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.generation.environment import (
    BackendIdentity,
    EnvironmentRecord,
    ExecutionRecord,
    GitState,
    ModelFile,
    ModelSnapshot,
    manifest_sha256,
)
from biorag_repro.generation.environment_lock import EnvironmentInventory
from biorag_repro.generation.receipt import (
    CudaMemoryMeasured,
    ExecutionFacts,
    ExecutionMeasurements,
    ExecutionReceipt,
    GenerationSettings,
    token_ids_sha256,
)
from biorag_repro.readiness.gather import to_result
from biorag_repro.readiness.policy import ApprovalPolicy, ApprovalSlots
from biorag_repro.readiness.restart import (
    EVIDENCE_RECORD,
    PRODUCTION_CONTRACT,
    RestartContract,
    restart_recovery,
)


def dump(value) -> bytes:
    return canonical_json_bytes(value)


def write(root: Path, relative: str, data: bytes) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def ref(relative: str, digest: str) -> dict[str, str]:
    return {"logical_ref": relative, "sha256": digest}


class RecoveryFixture:
    def __init__(self, root: Path):
        self.root = root
        self.package = "g10/package"
        self.batch = "g10/exercise/batch"
        self.run_id = "run_" + "1" * 32
        self.first_key = "obs_" + "2" * 64
        self.recovered_key = "obs_" + "3" * 64
        self.first_attempt = "att_" + "4" * 32
        self.interrupted_attempt = "att_" + "5" * 32
        self.recovery_attempt = "att_" + "6" * 32
        self.execution_revision = "7" * 40
        self.historical_revision = "8" * 40
        self.model_revision = "9" * 40
        self.model_manifest = "a" * 64
        self.contexts_sha = "b" * 64
        self.plan_sha = sha256_bytes(dump(sorted([self.first_key, self.recovered_key])))
        self.configurations = {
            "confirmatory": "c" * 64,
            "dense_robustness": "d" * 64,
            "chunk_robustness": "e" * 64,
            "scope_robustness": "f" * 64,
            "closed_book": "0" * 64,
        }
        self.prompt_sha = "1" * 64
        self.seed_sha = "2" * 64
        self.tokenizer_files = {"tokenizer.json": "3" * 64}
        self.historical_run = "run_" + "a" * 32
        self.historical_anchor = "4" * 64
        self._build_package()
        self._build_batch()
        self._build_record_and_custody()

    def _build_package(self):
        v4 = write(self.root, "g10/v4.tar", b"historical-v4")
        update = write(self.root, "g10/v5-update.tar", b"historical-v5-update")
        files = {
            "code_revision.txt": (self.historical_revision + "\n").encode(),
            "environment/approved_environment_lock_v1.toml": LOCK,
            "inputs/contexts.json": b"{}",
            "inputs/prompt.json": b"{}",
            "inputs/seeds.json": b"{}",
            "inputs/configs/confirmatory.toml": b"arm='confirmatory'\n",
            "inputs/configs/dense_robustness.toml": b"arm='dense_robustness'\n",
            "inputs/configs/chunk_robustness.toml": b"arm='chunk_robustness'\n",
            "inputs/configs/scope_robustness.toml": b"arm='scope_robustness'\n",
            "inputs/configs/closed_book.toml": b"arm='closed_book'\n",
        }
        preflight = {
            "status": "development_inputs_checked",
            "scientific_execution": False,
            "contexts_sha256": self.contexts_sha,
            "items": 2,
            "item_arm_cells": 10,
            "conditions": 2,
            "planned_observations": 2,
            "planned_keys_sha256": self.plan_sha,
            "generator_configurations": self.configurations,
            "prompt_sha256": self.prompt_sha,
            "seed_registry_sha256": self.seed_sha,
            "rendered_inputs": {},
        }
        planned_inputs = {}
        for key in (self.first_key, self.recovered_key):
            input_data = dump({"observation_key": key})
            input_sha = sha256_bytes(input_data)
            logical_ref = f"inputs/{input_sha}.json"
            files["planned/" + logical_ref] = input_data
            planned_inputs[key] = {
                "artifact_id": f"input-{key}",
                "logical_ref": logical_ref,
                "sha256": input_sha,
            }
        planned = {
            "status": "planned_generator_inputs_not_execution_evidence",
            "inputs": planned_inputs,
        }
        files["preflight.json"] = dump(preflight)
        files["planned_inputs.json"] = dump(planned)
        manifest_files = {}
        for name, data in files.items():
            digest = write(self.root, f"{self.package}/{name}", data)
            manifest_files[name] = {"sha256": digest, "bytes": len(data)}
        manifest = dump({"schema_version": "1", "files": manifest_files})
        manifest_sha = write(self.root, f"{self.package}/manifest.json", manifest)
        self.contract = RestartContract(
            v4_archive_sha256=v4,
            v5_update_sha256=update,
            v5_manifest_sha256=manifest_sha,
            historical_code_revision=self.historical_revision,
            historical_run_id=self.historical_run,
            historical_anchor_sha256=self.historical_anchor,
            planned_observations=2,
            contexts_sha256=self.contexts_sha,
            planned_keys_sha256=self.plan_sha,
            generator_configurations=self.configurations,
            prompt_sha256=self.prompt_sha,
            seed_registry_sha256=self.seed_sha,
            environment_lock_sha256=sha256_bytes(LOCK),
            model_repository="Qwen/Qwen3-8B",
            model_revision=self.model_revision,
            model_manifest_sha256=self.model_manifest,
            tokenizer_repository="Qwen/Qwen3-8B",
            tokenizer_revision=self.model_revision,
            tokenizer_file_sha256=self.tokenizer_files,
            approved_gpu_name="Synthetic GPU",
            minimum_model_positions=32768,
        )

    def _environment_record(self) -> tuple[str, bytes]:
        model_files = [ModelFile(path="config.json", size=2, sha256=sha256_bytes(b"{}"))]
        snapshot = ModelSnapshot(
            directory="model", files=model_files, manifest_sha256=manifest_sha256(model_files)
        )
        # The contract pins the manifest used by the batch. Tests replace the snapshot digest
        # with its real typed value so the model identity is independently recomputed.
        self.model_manifest = snapshot.manifest_sha256
        self.contract = replace(self.contract, model_manifest_sha256=self.model_manifest)
        record = EnvironmentRecord(
            captured_utc="2026-11-21T10:00:00Z",
            backend=BackendIdentity(kind="official", name="scripts.run_generation", version="1"),
            model_repository=self.contract.model_repository,
            model_revision=self.model_revision,
            tokenizer_repository=self.contract.tokenizer_repository,
            tokenizer_revision=self.model_revision,
            python_version="3.11.9",
            python_implementation="CPython",
            packages={"torch": "2.15.0+cu999"},
            operating_system="Linux",
            os_release="test",
            machine="x86_64",
            hostname="colab-l4",
            cpu_count=2,
            ram_total_mib=1024,
            gpu_status="ok",
            gpus=inventory().gpus,
            cuda_runtime="99.9",
            git=GitState(status="ok", commit=self.execution_revision, dirty=False, dirty_paths=0),
            execution=ExecutionRecord(
                snapshot=snapshot,
                tokenizer_file_sha256=self.tokenizer_files,
                device="cuda:0",
                device_name=self.contract.approved_gpu_name,
                dtype="bfloat16",
                max_position_embeddings=40960,
            ),
        )
        data = serialize_project_json(record, EnvironmentRecord)
        return sha256_bytes(data), data

    def _result_files(self, key: str, attempt: str, answer: str, env_sha: str) -> dict[str, bytes]:
        raw = answer.encode()
        raw_sha = sha256_bytes(raw)
        input_data = dump({"observation_key": key})
        input_sha = sha256_bytes(input_data)
        facts = ExecutionFacts(
            input_token_ids_sha256="7" * 64,
            input_token_count=1,
            generation=GenerationSettings(
                max_new_tokens=1,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                eos_token_ids=[99],
                pad_token_id=99,
                seed=42,
                effective={
                    "max_new_tokens": 1,
                    "do_sample": False,
                    "num_beams": 1,
                    "eos_token_id": [99],
                    "pad_token_id": 99,
                },
            ),
            output_token_ids=[7],
            stop_reason="max_new_tokens",
            measurements=ExecutionMeasurements(
                clock="perf_counter_ns",
                generation_ns=10,
                cuda_memory=CudaMemoryMeasured(
                    device_index=0,
                    allocated_before_bytes=1,
                    peak_allocated_bytes=2,
                    peak_reserved_bytes=3,
                    device_total_bytes=10,
                    device_used_after_bytes=4,
                ),
            ),
        )
        receipt = ExecutionReceipt(
            run_id=self.run_id.removeprefix("run_"),
            observation_key=key,
            attempt_id=attempt.removeprefix("att_"),
            environment_sha256=env_sha,
            input_record_sha256=input_sha,
            rendered_input_sha256="8" * 64,
            raw_output_sha256=raw_sha,
            output_token_ids_sha256=token_ids_sha256([7]),
            facts=facts,
        )
        receipt_data = serialize_project_json(receipt, ExecutionReceipt)
        receipt_path = f"receipts/{key}/{attempt}.json"
        raw_path = f"raw/{key}/{attempt}.txt"
        input_path = f"inputs/{input_sha}.json"
        config_sha = self.configurations["confirmatory"]
        result = {
            "generation_result": {
                "schema_version": "1",
                "phase": "development",
                "benchmark_version": "benchmark_v1",
                "item_id": "q_1" if key == self.first_key else "q_2",
                "paper_id": "P001",
                "matrix_arm": "confirmatory",
                "representation_or_condition": "JATS",
                "repeat_index": 1,
                "resolved_config_sha256": config_sha,
                "observation_key": key,
                "run_id": self.run_id,
                "attempt_id": attempt,
                "status": "completed",
                "output": {
                    "artifact_id": f"output-{key}",
                    "logical_ref": raw_path,
                    "sha256": raw_sha,
                },
                "raw_output": answer,
                "parse_status": "parsed",
                "answer": answer,
                "abstained": False,
            },
            "context_sha256": "9" * 64,
            "rendered_sha256": "8" * 64,
            "generator_input": {
                "artifact_id": f"input-{key}",
                "logical_ref": input_path,
                "sha256": input_sha,
            },
            "execution_receipt": {
                "logical_ref": receipt_path,
                "sha256": sha256_bytes(receipt_data),
            },
            "input_token_ids_sha256": receipt.facts.input_token_ids_sha256,
            "input_tokens": receipt.facts.input_token_count,
        }
        return {
            f"results/{key}.json": dump(result),
            raw_path: raw,
            input_path: input_data,
            receipt_path: receipt_data,
        }

    def _build_batch(self):
        env_sha, env_data = self._environment_record()
        self.env_sha = env_sha
        batch = {
            "run_id": self.run_id,
            "created_utc": "2026-11-21T10:00:00Z",
            "researcher": "Rama Krishna Pudota",
            "identity": {
                "phase": "development",
                "contexts_sha256": self.contexts_sha,
                "planned_keys_sha256": self.plan_sha,
                "generator_configurations": self.configurations,
                "prompt_sha256": self.prompt_sha,
                "seed_registry_sha256": self.seed_sha,
                "model_manifest_sha256": self.model_manifest,
                "git": {
                    "head": self.execution_revision,
                    "tracked_changes_sha256": sha256_bytes(b""),
                },
                "official_execution": True,
                "environment_lock_sha256": sha256_bytes(LOCK),
                "session_hostname": "colab-l4",
                "session_boot_id": "boot-1",
                "session_gpu_inventory": [
                    {"name": self.contract.approved_gpu_name, "uuid": "GPU-1"}
                ],
            },
            "model_snapshot": EnvironmentRecord.model_validate_json(
                env_data
            ).execution.snapshot.model_dump(mode="json"),
            "contexts_header": {"arms": list(self.configurations)},
        }
        files = {"batch.json": dump(batch), f"environments/{env_sha}.json": env_data}
        inv_data = serialize_project_json(inventory(), EnvironmentInventory)
        files["environment_start.json"] = inv_data
        files["environment_end.json"] = inv_data
        gpu = dump({"status": "measured", "gpus": batch["identity"]["session_gpu_inventory"]})
        for invocation in (1, 2):
            files[f"gpu_inventories/{invocation:06d}_start.json"] = gpu
            files[f"gpu_inventories/{invocation:06d}_end.json"] = gpu
        events = [
            (self.first_attempt, self.first_key, "started", "2026-11-21T10:00:01Z"),
            (self.first_attempt, self.first_key, "completed", "2026-11-21T10:00:02Z"),
            (self.interrupted_attempt, self.recovered_key, "started", "2026-11-21T10:00:03Z"),
            (self.interrupted_attempt, self.recovered_key, "interrupted", "2026-11-21T10:00:05Z"),
            (self.recovery_attempt, self.recovered_key, "started", "2026-11-21T10:00:06Z"),
            (self.recovery_attempt, self.recovered_key, "completed", "2026-11-21T10:00:07Z"),
        ]
        for seq, (attempt, key, event, at) in enumerate(events, 1):
            files[f"attempts/{seq:06d}.json"] = dump(
                {
                    "seq": seq,
                    "at_utc": at,
                    "run_id": self.run_id,
                    "attempt_id": attempt,
                    "observation_key": key,
                    "event": event,
                }
            )
        files.update(self._result_files(self.first_key, self.first_attempt, "first", env_sha))
        files.update(
            self._result_files(self.recovered_key, self.recovery_attempt, "second", env_sha)
        )
        for name, data in files.items():
            write(self.root, f"{self.batch}/{name}", data)
        first_names = {
            "batch.json",
            f"environments/{env_sha}.json",
            "environment_start.json",
            "environment_end.json",
            "gpu_inventories/000001_start.json",
            "gpu_inventories/000001_end.json",
            "attempts/000001.json",
            "attempts/000002.json",
            "attempts/000003.json",
            *self._result_files(self.first_key, self.first_attempt, "first", env_sha).keys(),
        }
        first_inventory = {name: sha256_bytes(files[name]) for name in sorted(first_names)}
        interruption = dump(
            {
                "sealed_utc": "2026-11-21T10:00:04.750000Z",
                "files": first_inventory,
                "previous_anchor_sha256": None,
            }
        )
        self.interruption_ref = f"{self.batch}/anchors/000001.json"
        self.interruption_sha = write(self.root, self.interruption_ref, interruption)
        recovery_inventory = {name: sha256_bytes(data) for name, data in sorted(files.items())}
        recovery = dump(
            {
                "sealed_utc": "2026-11-21T10:00:08Z",
                "files": recovery_inventory,
                "previous_anchor_sha256": self.interruption_sha,
            }
        )
        self.recovery_ref = f"{self.batch}/anchors/000002.json"
        self.recovery_sha = write(self.root, self.recovery_ref, recovery)
        self.recovery_files = recovery_inventory

    def _build_record_and_custody(self):
        command_sha = write(self.root, "g10/initial_command.txt", b"python run_generation.py\n")
        process = {
            "schema_version": "g10_interruption_process_v1",
            "runner": "scripts/run_generation.py",
            "command_sha256": command_sha,
            "pid": 123,
            "signal": "SIGINT",
            "exit_code": 130,
            "run_id": self.run_id,
            "observation_key": self.recovered_key,
            "attempt_id": self.interrupted_attempt,
            "session_boot_id": "boot-1",
            "started_at_utc": "2026-11-21T10:00:00Z",
            "signal_at_utc": "2026-11-21T10:00:04Z",
            "exited_at_utc": "2026-11-21T10:00:04.500000Z",
            "synthetic": False,
        }
        process_sha = write(self.root, "g10/process.json", dump(process))
        record = {
            "schema_version": "g10_restart_recovery_evidence_v1",
            "exercise_id": "g10-development-recovery-2026-11-21",
            "evidence_class": "live_development_gpu_interruption",
            "researcher": "Rama Krishna Pudota",
            "package": {
                "package_root": self.package,
                "v4_archive": ref("g10/v4.tar", self.contract.v4_archive_sha256),
                "v5_update": ref("g10/v5-update.tar", self.contract.v5_update_sha256),
                "manifest": ref(f"{self.package}/manifest.json", self.contract.v5_manifest_sha256),
            },
            "execution": {
                "batch_root": self.batch,
                "execution_source_revision": self.execution_revision,
                "interruption_anchor": ref(self.interruption_ref, self.interruption_sha),
                "recovery_anchor": ref(self.recovery_ref, self.recovery_sha),
                "process_record": ref("g10/process.json", process_sha),
                "command_record": ref("g10/initial_command.txt", command_sha),
            },
            "historical": {
                "package_code_revision": self.historical_revision,
                "published_run_id": self.historical_run,
                "published_anchor_sha256": self.historical_anchor,
            },
            "custody": {
                "drive_manifest": {"logical_ref": "g10/drive_manifest.json", "sha256": "0" * 64},
                "wsl_manifest": {"logical_ref": "g10/wsl_manifest.json", "sha256": "0" * 64},
            },
        }
        published = {
            **{f"{self.batch}/{name}": digest for name, digest in self.recovery_files.items()},
            self.interruption_ref: self.interruption_sha,
            self.recovery_ref: self.recovery_sha,
            "g10/process.json": process_sha,
            "g10/initial_command.txt": command_sha,
            "g10/v4.tar": self.contract.v4_archive_sha256,
            "g10/v5-update.tar": self.contract.v5_update_sha256,
            f"{self.package}/manifest.json": self.contract.v5_manifest_sha256,
        }
        drive = {
            "schema_version": "g10_drive_publication_v1",
            "publication_id": "drive-g10-1",
            "published_at_utc": "2026-11-21T10:01:00Z",
            "files": dict(sorted(published.items())),
        }
        drive_data = dump(drive)
        drive_sha = write(self.root, "g10/drive_manifest.json", drive_data)
        wsl = {
            "schema_version": "g10_wsl_custody_v1",
            "publication_id": drive["publication_id"],
            "drive_manifest_sha256": drive_sha,
            "received_at_utc": "2026-11-21T10:02:00Z",
            "custody_root_kind": "wsl_outside_drive_sync",
            "files": drive["files"],
        }
        wsl_sha = write(self.root, "g10/wsl_manifest.json", dump(wsl))
        record["custody"]["drive_manifest"]["sha256"] = drive_sha
        record["custody"]["wsl_manifest"]["sha256"] = wsl_sha
        write(self.root, EVIDENCE_RECORD, dump(record))

    def json(self, relative: str):
        return json.loads((self.root / relative).read_bytes())

    def replace_json(self, relative: str, mutate):
        value = self.json(relative)
        mutate(value)
        write(self.root, relative, dump(value))

    def reseal_recovery(self):
        anchor = self.json(self.recovery_ref)
        for name in list(anchor["files"]):
            anchor["files"][name] = sha256_bytes((self.root / self.batch / name).read_bytes())
        anchor["previous_anchor_sha256"] = sha256_bytes(
            (self.root / self.interruption_ref).read_bytes()
        )
        self.recovery_sha = write(self.root, self.recovery_ref, dump(anchor))
        record = self.json(EVIDENCE_RECORD)
        record["execution"]["recovery_anchor"]["sha256"] = self.recovery_sha
        write(self.root, EVIDENCE_RECORD, dump(record))
        self.refresh_custody()

    def reseal_both(self):
        first = self.json(self.interruption_ref)
        first["files"] = {
            name: sha256_bytes((self.root / self.batch / name).read_bytes())
            for name in first["files"]
        }
        self.interruption_sha = write(self.root, self.interruption_ref, dump(first))
        batch_root = self.root / self.batch
        files = {
            path.relative_to(batch_root).as_posix(): sha256_bytes(path.read_bytes())
            for path in batch_root.rglob("*")
            if path.is_file() and path.parent.name != "anchors"
        }
        recovery = self.json(self.recovery_ref)
        recovery["files"] = dict(sorted(files.items()))
        recovery["previous_anchor_sha256"] = self.interruption_sha
        self.recovery_sha = write(self.root, self.recovery_ref, dump(recovery))
        record = self.json(EVIDENCE_RECORD)
        record["execution"]["interruption_anchor"]["sha256"] = self.interruption_sha
        record["execution"]["recovery_anchor"]["sha256"] = self.recovery_sha
        write(self.root, EVIDENCE_RECORD, dump(record))
        drive = self.json("g10/drive_manifest.json")
        external = {
            name: digest
            for name, digest in drive["files"].items()
            if not name.startswith(self.batch + "/")
        }
        drive["files"] = {
            **external,
            **{f"{self.batch}/{name}": digest for name, digest in files.items()},
            self.interruption_ref: self.interruption_sha,
            self.recovery_ref: self.recovery_sha,
        }
        write(self.root, "g10/drive_manifest.json", dump(drive))
        self.refresh_custody()

    def refresh_custody(self):
        drive = self.json("g10/drive_manifest.json")
        drive["files"] = {
            relative: sha256_bytes((self.root / relative).read_bytes())
            for relative in drive["files"]
        }
        drive_sha = write(self.root, "g10/drive_manifest.json", dump(drive))
        wsl = self.json("g10/wsl_manifest.json")
        wsl["files"] = drive["files"]
        wsl["drive_manifest_sha256"] = drive_sha
        wsl_sha = write(self.root, "g10/wsl_manifest.json", dump(wsl))
        record = self.json(EVIDENCE_RECORD)
        record["custody"]["drive_manifest"]["sha256"] = drive_sha
        record["custody"]["wsl_manifest"]["sha256"] = wsl_sha
        write(self.root, EVIDENCE_RECORD, dump(record))

    def refresh_execution_ref(self, field: str, relative: str):
        record = self.json(EVIDENCE_RECORD)
        record["execution"][field]["sha256"] = sha256_bytes((self.root / relative).read_bytes())
        write(self.root, EVIDENCE_RECORD, dump(record))
        self.refresh_custody()

    def change_run_id(self, run_id: str):
        self.replace_json(f"{self.batch}/batch.json", lambda value: value.update(run_id=run_id))
        for seq in range(1, 7):
            self.replace_json(
                f"{self.batch}/attempts/{seq:06d}.json",
                lambda value: value.update(run_id=run_id),
            )
        for key in (self.first_key, self.recovered_key):
            result_path = f"{self.batch}/results/{key}.json"
            value = self.json(result_path)
            value["generation_result"]["run_id"] = run_id
            receipt_ref = value["execution_receipt"]
            receipt_path = f"{self.batch}/{receipt_ref['logical_ref']}"
            receipt = ExecutionReceipt.model_validate_json((self.root / receipt_path).read_bytes())
            receipt = receipt.model_copy(update={"run_id": run_id.removeprefix("run_")})
            receipt_data = serialize_project_json(receipt, ExecutionReceipt)
            write(self.root, receipt_path, receipt_data)
            receipt_ref["sha256"] = sha256_bytes(receipt_data)
            write(self.root, result_path, dump(value))
        self.replace_json("g10/process.json", lambda value: value.update(run_id=run_id))
        self.refresh_execution_ref("process_record", "g10/process.json")
        self.reseal_both()

    def change_execution_revision(self, revision: str):
        batch_path = f"{self.batch}/batch.json"
        batch = self.json(batch_path)
        batch["identity"]["git"]["head"] = revision
        write(self.root, batch_path, dump(batch))
        old_env_path = self.root / self.batch / f"environments/{self.env_sha}.json"
        environment = EnvironmentRecord.model_validate_json(old_env_path.read_bytes())
        environment = environment.model_copy(
            update={
                "git": environment.git.model_copy(update={"commit": revision}),
            }
        )
        environment_data = serialize_project_json(environment, EnvironmentRecord)
        new_env_sha = sha256_bytes(environment_data)
        new_env_path = self.root / self.batch / f"environments/{new_env_sha}.json"
        new_env_path.write_bytes(environment_data)
        old_env_path.unlink()
        first_anchor = self.json(self.interruption_ref)
        first_anchor["files"].pop(f"environments/{self.env_sha}.json")
        first_anchor["files"][f"environments/{new_env_sha}.json"] = new_env_sha
        write(self.root, self.interruption_ref, dump(first_anchor))
        for key in (self.first_key, self.recovered_key):
            result_path = f"{self.batch}/results/{key}.json"
            value = self.json(result_path)
            receipt_ref = value["execution_receipt"]
            receipt_path = self.root / self.batch / receipt_ref["logical_ref"]
            receipt = ExecutionReceipt.model_validate_json(receipt_path.read_bytes())
            receipt = receipt.model_copy(update={"environment_sha256": new_env_sha})
            receipt_data = serialize_project_json(receipt, ExecutionReceipt)
            receipt_path.write_bytes(receipt_data)
            receipt_ref["sha256"] = sha256_bytes(receipt_data)
            write(self.root, result_path, dump(value))
        record = self.json(EVIDENCE_RECORD)
        record["execution"]["execution_source_revision"] = revision
        write(self.root, EVIDENCE_RECORD, dump(record))
        self.env_sha = new_env_sha
        self.reseal_both()


@pytest.fixture
def recovery(tmp_path):
    return RecoveryFixture(tmp_path)


def verify(recovery):
    return restart_recovery(recovery.root, recovery.contract)


def test_valid_bounded_interruption_and_recovery_has_canonical_subject(recovery):
    gathered = verify(recovery)
    assert gathered.state == "ok"
    assert gathered.subject["kind"] == "development_restart_recovery_v1"
    assert gathered.subject["run_id"] == recovery.run_id
    assert gathered.subject["completed_observations"] == 2
    assert gathered.subject["planned_observations"] == 2
    assert gathered.subject["campaign_completion_claim"] is False
    assert gathered.subject["interrupted_attempt_id"] != gathered.subject["recovery_attempt_id"]
    assert sha256_bytes(dump(gathered.subject)) == sha256_bytes(dump(deepcopy(gathered.subject)))


def test_canonical_subject_uses_the_existing_policy_comparison(recovery):
    gathered = verify(recovery)
    digest = sha256_bytes(dump(gathered.subject))
    policy = ApprovalPolicy(
        policy_version="test",
        slots=ApprovalSlots(restart_evidence_acceptance=digest),
    )
    result = to_result("G10_development_restart_recovery", gathered, policy)
    assert (result.status, result.reason_code, result.expected_sha256) == (
        "passed",
        "verified_against_policy",
        digest,
    )


def test_production_contract_pins_the_exact_historical_v5_package():
    assert PRODUCTION_CONTRACT.v4_archive_sha256 == (
        "c8ddc4500329776b9ead10fdf5e3851cd9b4c8624e5e4cde396f0aa4633c0875"
    )
    assert PRODUCTION_CONTRACT.v5_update_sha256 == (
        "b067777b4612fafb89e16f6fd732fa4f032ba6736273e1372e4bc8e78ec5348c"
    )
    assert PRODUCTION_CONTRACT.v5_manifest_sha256 == (
        "0853ec27e1d7a6b5747c5d4762f1c7c64fc2909b02951aa24e1c3bea61472359"
    )
    assert PRODUCTION_CONTRACT.planned_observations == 1290


@pytest.mark.parametrize("event_index", [2, 3])
def test_missing_interruption_sequence_fails_closed(recovery, event_index):
    path = f"{recovery.batch}/attempts/{event_index + 1:06d}.json"
    (recovery.root / path).unlink()
    assert verify(recovery).state == "failed"


def test_interrupted_attempt_without_subsequent_completion_fails_closed(recovery):
    (recovery.root / recovery.batch / "attempts/000006.json").unlink()
    assert verify(recovery).state == "failed"


def test_reused_attempt_id_fails_closed(recovery):
    recovery.replace_json(
        f"{recovery.batch}/attempts/000005.json",
        lambda value: value.update(attempt_id=recovery.interrupted_attempt),
    )
    recovery.replace_json(
        f"{recovery.batch}/attempts/000006.json",
        lambda value: value.update(attempt_id=recovery.interrupted_attempt),
    )
    value = recovery.json(f"{recovery.batch}/results/{recovery.recovered_key}.json")
    old_raw = value["generation_result"]["output"]["logical_ref"]
    new_raw = old_raw.replace(recovery.recovery_attempt, recovery.interrupted_attempt)
    (recovery.root / recovery.batch / new_raw).parent.mkdir(parents=True, exist_ok=True)
    (recovery.root / recovery.batch / old_raw).rename(recovery.root / recovery.batch / new_raw)
    value["generation_result"]["attempt_id"] = recovery.interrupted_attempt
    value["generation_result"]["output"]["logical_ref"] = new_raw
    old_receipt = value["execution_receipt"]["logical_ref"]
    receipt = ExecutionReceipt.model_validate_json(
        (recovery.root / recovery.batch / old_receipt).read_bytes()
    )
    receipt = receipt.model_copy(
        update={"attempt_id": recovery.interrupted_attempt.removeprefix("att_")}
    )
    new_receipt = old_receipt.replace(recovery.recovery_attempt, recovery.interrupted_attempt)
    receipt_data = serialize_project_json(receipt, ExecutionReceipt)
    write(recovery.root, f"{recovery.batch}/{new_receipt}", receipt_data)
    (recovery.root / recovery.batch / old_receipt).unlink()
    value["execution_receipt"] = {
        "logical_ref": new_receipt,
        "sha256": sha256_bytes(receipt_data),
    }
    write(
        recovery.root,
        f"{recovery.batch}/results/{recovery.recovered_key}.json",
        dump(value),
    )
    recovery.reseal_both()
    assert verify(recovery).state == "failed"


def test_duplicate_completion_fails_closed(recovery):
    event = recovery.json(f"{recovery.batch}/attempts/000006.json")
    event["seq"] = 7
    write(recovery.root, f"{recovery.batch}/attempts/000007.json", dump(event))
    recovery.reseal_recovery()
    assert verify(recovery).state == "failed"


def test_modified_previously_sealed_result_fails_closed(recovery):
    path = f"{recovery.batch}/results/{recovery.first_key}.json"
    recovery.replace_json(path, lambda value: value.update(context_sha256="0" * 64))
    recovery.reseal_recovery()
    assert verify(recovery).state == "failed"


def test_corrupted_attempt_hash_fails_closed(recovery):
    path = recovery.root / recovery.batch / "attempts/000004.json"
    path.write_bytes(path.read_bytes() + b" ")
    assert verify(recovery).state == "failed"


def test_corrupted_package_manifest_hash_fails_closed(recovery):
    path = recovery.root / recovery.package / "manifest.json"
    path.write_bytes(path.read_bytes() + b" ")
    assert verify(recovery).state == "failed"


def test_wrong_anchor_predecessor_fails_closed(recovery):
    anchor = recovery.json(recovery.recovery_ref)
    anchor["previous_anchor_sha256"] = "0" * 64
    recovery.recovery_sha = write(recovery.root, recovery.recovery_ref, dump(anchor))
    record = recovery.json(EVIDENCE_RECORD)
    record["execution"]["recovery_anchor"]["sha256"] = recovery.recovery_sha
    write(recovery.root, EVIDENCE_RECORD, dump(record))
    assert verify(recovery).state == "failed"


@pytest.mark.parametrize("change", ["run", "boot", "gpu"])
def test_changed_run_boot_or_gpu_fails_closed(recovery, change):
    path = f"{recovery.batch}/batch.json"

    def mutate(value):
        if change == "run":
            value["run_id"] = "run_" + "f" * 32
        elif change == "boot":
            value["identity"]["session_boot_id"] = "other"
        else:
            value["identity"]["session_gpu_inventory"] = [{"name": "Other GPU", "uuid": "GPU-1"}]

    recovery.replace_json(path, mutate)
    recovery.reseal_recovery()
    assert verify(recovery).state == "failed"


@pytest.mark.parametrize("change", ["model", "tokenizer", "environment"])
def test_changed_model_tokenizer_or_environment_fails_closed(recovery, change):
    path = f"{recovery.batch}/environments/{recovery.env_sha}.json"

    def mutate(value):
        if change == "model":
            value["model_revision"] = "0" * 40
        elif change == "tokenizer":
            value["tokenizer_revision"] = "0" * 40
        else:
            value["hostname"] = "other-host"

    recovery.replace_json(path, mutate)
    recovery.reseal_recovery()
    assert verify(recovery).state == "failed"


def test_historical_publication_reuse_fails_closed(recovery):
    recovery.change_run_id(recovery.historical_run)
    assert verify(recovery).state == "failed"


def test_historical_source_revision_cannot_be_reused_for_execution(recovery):
    recovery.change_execution_revision(recovery.historical_revision)
    assert verify(recovery).state == "failed"


def test_nonofficial_batch_cannot_reuse_official_receipts(recovery):
    recovery.replace_json(
        f"{recovery.batch}/batch.json",
        lambda value: value["identity"].update(official_execution=False),
    )
    recovery.reseal_both()
    assert verify(recovery).state == "failed"


def test_executed_input_must_equal_the_authenticated_planned_input(recovery):
    key = recovery.first_key
    result_path = f"{recovery.batch}/results/{key}.json"
    value = recovery.json(result_path)
    old_input = value["generator_input"]["logical_ref"]
    new_input_data = dump({"observation_key": key, "forged": True})
    new_input_sha = sha256_bytes(new_input_data)
    new_input = f"inputs/{new_input_sha}.json"
    write(recovery.root, f"{recovery.batch}/{new_input}", new_input_data)
    (recovery.root / recovery.batch / old_input).unlink()
    first_anchor = recovery.json(recovery.interruption_ref)
    first_anchor["files"].pop(old_input)
    first_anchor["files"][new_input] = new_input_sha
    write(recovery.root, recovery.interruption_ref, dump(first_anchor))
    value["generator_input"] = {
        "artifact_id": "forged-input",
        "logical_ref": new_input,
        "sha256": new_input_sha,
    }
    receipt_ref = value["execution_receipt"]
    receipt_path = recovery.root / recovery.batch / receipt_ref["logical_ref"]
    receipt = ExecutionReceipt.model_validate_json(receipt_path.read_bytes())
    receipt = receipt.model_copy(update={"input_record_sha256": new_input_sha})
    receipt_data = serialize_project_json(receipt, ExecutionReceipt)
    receipt_path.write_bytes(receipt_data)
    receipt_ref["sha256"] = sha256_bytes(receipt_data)
    write(recovery.root, result_path, dump(value))
    recovery.reseal_both()
    assert verify(recovery).state == "failed"


@pytest.mark.parametrize("manifest", ["drive_manifest", "wsl_manifest"])
def test_incomplete_or_corrupt_custody_fails_closed(recovery, manifest):
    record = recovery.json(EVIDENCE_RECORD)
    path = record["custody"][manifest]["logical_ref"]
    recovery.replace_json(path, lambda value: value["files"].pop(recovery.recovery_ref))
    assert verify(recovery).state == "failed"


def test_unsealed_file_fails_closed(recovery):
    write(recovery.root, f"{recovery.batch}/results/obs_{'f' * 64}.json", b"{}")
    assert verify(recovery).state == "failed"


def test_unsafe_reference_fails_closed(recovery):
    record = recovery.json(EVIDENCE_RECORD)
    record["execution"]["batch_root"] = "../outside"
    write(recovery.root, EVIDENCE_RECORD, dump(record))
    assert verify(recovery).state == "failed"


def test_synthetic_fixture_cannot_claim_live_evidence(recovery):
    recovery.replace_json("g10/process.json", lambda value: value.update(synthetic=True))
    recovery.refresh_execution_ref("process_record", "g10/process.json")
    assert verify(recovery).state == "failed"


@pytest.mark.parametrize(
    ("field", "value"),
    [("signal", "SIGTERM"), ("exit_code", 0), ("runner", "synthetic.py")],
)
def test_forged_or_incomplete_process_evidence_fails_closed(recovery, field, value):
    recovery.replace_json("g10/process.json", lambda process: process.update({field: value}))
    recovery.refresh_execution_ref("process_record", "g10/process.json")
    assert verify(recovery).state == "failed"


def test_process_command_hash_must_reference_the_retained_command_bytes(recovery):
    (recovery.root / "g10/initial_command.txt").write_bytes(b"different command\n")
    recovery.refresh_execution_ref("command_record", "g10/initial_command.txt")
    assert verify(recovery).state == "failed"


def test_self_consistent_custody_manifest_omission_fails_closed(recovery):
    wsl = recovery.json("g10/wsl_manifest.json")
    wsl["files"].pop(recovery.recovery_ref)
    wsl_sha = write(recovery.root, "g10/wsl_manifest.json", dump(wsl))
    record = recovery.json(EVIDENCE_RECORD)
    record["custody"]["wsl_manifest"]["sha256"] = wsl_sha
    write(recovery.root, EVIDENCE_RECORD, dump(record))
    assert verify(recovery).state == "failed"


def test_symlinked_evidence_is_refused(recovery):
    path = recovery.root / "g10/process.json"
    copy = recovery.root / "g10/process-copy.json"
    copy.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(copy.name)
    assert verify(recovery).state == "failed"


def test_partial_evidence_never_claims_complete_campaign(recovery):
    gathered = verify(recovery)
    assert gathered.subject["campaign_completion_claim"] is False
    assert (
        gathered.subject["completed_observations"] < 1290
        or recovery.contract.planned_observations == 2
    )


def test_final_test_and_qualification_are_refused(recovery):
    for phase in ("final_test", "qualification"):
        clone = deepcopy(recovery.json(f"{recovery.batch}/batch.json"))
        clone["identity"]["phase"] = phase
        write(recovery.root, f"{recovery.batch}/batch.json", dump(clone))
        recovery.reseal_both()
        assert verify(recovery).state == "failed"


def test_absent_evidence_preserves_existing_pending_state(tmp_path):
    gathered = restart_recovery(tmp_path)
    assert (gathered.state, gathered.reason) == ("absent", "official_restart_evidence_absent")
