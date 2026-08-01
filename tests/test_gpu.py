from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tracebench.gpu import (
    EVIDENCE_FILES,
    GPU_EXECUTION_SCHEMA,
    GPU_MANIFEST_SCHEMA,
    GPU_PLAN_SCHEMA,
    analyze_gpu_evidence,
    build_gpu_schedules,
    model_snapshot_sha256,
    output_token_sha256,
    plan_sha256,
    request_input_sha256,
    sha256_file,
    validate_gpu_plan,
    verify_gpu_evidence,
    verify_model_snapshot,
    write_gpu_analysis,
)
from tracebench.model import TraceBenchError, canonical_json_bytes


TRACEBENCH_COMMIT = "c" * 40
PACKAGE_INVENTORY_TEXT = "fixture==1.0\n"
PACKAGE_INVENTORY_SHA256 = hashlib.sha256(
    PACKAGE_INVENTORY_TEXT.encode("utf-8")
).hexdigest()
MODEL_FILES = {
    "config.json": "1" * 64,
    "generation_config.json": "2" * 64,
    "merges.txt": "3" * 64,
    "model.safetensors": "4" * 64,
    "tokenizer.json": "5" * 64,
    "tokenizer_config.json": "6" * 64,
    "vocab.json": "7" * 64,
}


def gpu_plan() -> dict:
    return {
        "schema": GPU_PLAN_SCHEMA,
        "experiment_id": "fixture-gpu-experiment",
        "preregistration": {
            "registered_at": "2026-08-01T04:56:54Z",
            "primary_metric": "exact token divergence",
            "sample_size_rule": "fixed fixture",
            "reporting_rule": "retain every comparison",
            "hypothesis": "fixture hypothesis",
        },
        "model": {
            "repository": "fixture/model",
            "revision": "a" * 40,
            "files_sha256": dict(MODEL_FILES),
        },
        "request_count": 2,
        "requests": [
            {"request_id": "request-a", "prompt": "Alpha"},
            {"request_id": "request-b", "prompt": "Beta"},
        ],
        "decoding_modes": [{"name": "greedy", "do_sample": False, "max_new_tokens": 4}],
        "schedule": {
            "global_seed": 17,
            "variant_count": 1,
            "pinned_repetitions": 1,
            "batch_sizes": [2],
            "variant_seeds": [19],
        },
        "runtime": {
            "device": "cuda:0",
            "dtype": "float16",
            "padding_side": "left",
            "local_files_only": True,
            "deterministic_algorithms": False,
            "cudnn_benchmark": False,
            "allow_tf32": False,
            "chat_template": "tokenizer.apply_chat_template",
            "add_generation_prompt": True,
            "strip_trailing_pad_tokens": True,
            "seed_reset": "before_each_schedule",
        },
        "software_scope": {
            "python_version": "3.12.10",
            "torch_version": "2.6.0+cu124",
            "transformers_version": "5.5.0",
            "cuda_runtime": "12.4",
            "package_inventory_sha256": PACKAGE_INVENTORY_SHA256,
        },
        "hardware_scope": {
            "gpu_name": "NVIDIA GeForce RTX 2060",
            "vram_mib": 6144,
            "compute_capability": [7, 5],
            "native_os": "Windows",
            "engine": "Hugging Face Transformers generate",
        },
    }


def environment(plan: dict) -> dict:
    value = {
        "schema": "tracebench/gpu-environment/v1",
        "experiment_id": plan["experiment_id"],
        "plan_sha256": plan_sha256(plan),
        "python_version": plan["software_scope"]["python_version"],
        "python_build": "fixture build",
        "platform_system": plan["hardware_scope"]["native_os"],
        "platform": "fixture platform",
        "torch_version": plan["software_scope"]["torch_version"],
        "transformers_version": plan["software_scope"]["transformers_version"],
        "cuda_runtime": plan["software_scope"]["cuda_runtime"],
        "package_inventory_sha256": plan["software_scope"]["package_inventory_sha256"],
        "cuda_available": True,
        "gpu_name": plan["hardware_scope"]["gpu_name"],
        "gpu_capability": plan["hardware_scope"]["compute_capability"],
        "gpu_memory_mib": plan["hardware_scope"]["vram_mib"],
        "runtime_settings": dict(plan["runtime"]),
        "model_repository": plan["model"]["repository"],
        "model_revision": plan["model"]["revision"],
        "model_files_sha256": dict(plan["model"]["files_sha256"]),
        "model_snapshot_sha256": model_snapshot_sha256(plan["model"]["files_sha256"]),
        "tracebench_repository": "https://github.com/David-Wu1119/tracebench",
        "tracebench_commit": TRACEBENCH_COMMIT,
    }
    value["runtime_digest"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value


def execution(
    *,
    plan: dict,
    runtime: dict,
    cell: str,
    schedule_id: str,
    repetition: int,
    request_id: str,
    batch_index: int,
    batch_members: list[str],
    batch_position: int,
    tokens: list[int],
) -> dict:
    prompt = next(
        request["prompt"]
        for request in plan["requests"]
        if request["request_id"] == request_id
    )
    batch_document = {
        "mode": "greedy",
        "schedule_id": schedule_id,
        "seed": plan["schedule"]["global_seed"],
        "batch_members": batch_members,
    }
    return {
        "schema": GPU_EXECUTION_SCHEMA,
        "experiment_id": plan["experiment_id"],
        "tracebench_commit": TRACEBENCH_COMMIT,
        "mode": "greedy",
        "cell": cell,
        "schedule_id": schedule_id,
        "repetition": repetition,
        "request_id": request_id,
        "input_sha256": request_input_sha256(prompt),
        "batch_id": f"{schedule_id}-batch-{batch_index:03d}",
        "batch_members": batch_members,
        "batch_position": batch_position,
        "batch_fingerprint": hashlib.sha256(
            canonical_json_bytes(batch_document)
        ).hexdigest(),
        "global_seed": plan["schedule"]["global_seed"],
        "runtime_digest": runtime["runtime_digest"],
        "status": "completed",
        "output_token_ids": tokens,
        "output_sha256": output_token_sha256(tokens),
        "batch_duration_ms": 1.25,
        "error": None,
    }


class GpuContractTests(unittest.TestCase):
    def test_schedule_is_deterministic_and_covers_every_request_once(self) -> None:
        plan = gpu_plan()

        first = build_gpu_schedules(plan)
        second = build_gpu_schedules(plan)

        self.assertEqual(first, second)
        self.assertEqual(first["baseline"], (("request-a",), ("request-b",)))
        self.assertEqual(first["variant-00"], (("request-b", "request-a"),))
        for batches in first.values():
            flattened = [request_id for batch in batches for request_id in batch]
            self.assertEqual(sorted(flattened), ["request-a", "request-b"])

    def test_plan_validation_rejects_unbound_model_and_runtime_scope(self) -> None:
        plan = gpu_plan()
        del plan["model"]["files_sha256"]["tokenizer.json"]
        with self.assertRaisesRegex(TraceBenchError, "model snapshot"):
            validate_gpu_plan(plan)

        plan = gpu_plan()
        plan["software_scope"]["torch_version"] = ""
        with self.assertRaisesRegex(TraceBenchError, "torch_version"):
            validate_gpu_plan(plan)

        plan = gpu_plan()
        plan["hardware_scope"]["compute_capability"] = [7]
        with self.assertRaisesRegex(TraceBenchError, "compute_capability"):
            validate_gpu_plan(plan)

    def test_model_snapshot_requires_the_exact_regular_file_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory)
            files: dict[str, str] = {}
            for name in MODEL_FILES:
                (model_path / name).write_text(f"fixture {name}\n", encoding="utf-8")
                files[name] = sha256_file(model_path / name)
            model = {"files_sha256": files}

            digest = verify_model_snapshot(model_path, model)
            self.assertEqual(digest, model_snapshot_sha256(files))

            (model_path / "unbound.txt").write_text("unbound\n", encoding="utf-8")
            with self.assertRaisesRegex(TraceBenchError, "exact registered file set"):
                verify_model_snapshot(model_path, model)

    def write_evidence(self, root: Path) -> None:
        plan = gpu_plan()
        runtime = environment(plan)
        (root / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / "environment.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / "requests.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "schema": "tracebench/gpu-request/v1",
                        "request_id": request["request_id"],
                        "prompt": request["prompt"],
                        "input_sha256": request_input_sha256(request["prompt"]),
                    },
                    sort_keys=True,
                )
                + "\n"
                for request in plan["requests"]
            ),
            encoding="utf-8",
        )
        records = [
            execution(
                plan=plan,
                runtime=runtime,
                cell="baseline",
                schedule_id="baseline",
                repetition=0,
                request_id="request-a",
                batch_index=0,
                batch_members=["request-a"],
                batch_position=0,
                tokens=[1, 2],
            ),
            execution(
                plan=plan,
                runtime=runtime,
                cell="baseline",
                schedule_id="baseline",
                repetition=0,
                request_id="request-b",
                batch_index=1,
                batch_members=["request-b"],
                batch_position=0,
                tokens=[3],
            ),
            execution(
                plan=plan,
                runtime=runtime,
                cell="unpinned",
                schedule_id="variant-00",
                repetition=0,
                request_id="request-b",
                batch_index=0,
                batch_members=["request-b", "request-a"],
                batch_position=0,
                tokens=[3],
            ),
            execution(
                plan=plan,
                runtime=runtime,
                cell="unpinned",
                schedule_id="variant-00",
                repetition=0,
                request_id="request-a",
                batch_index=0,
                batch_members=["request-b", "request-a"],
                batch_position=1,
                tokens=[9, 2],
            ),
            execution(
                plan=plan,
                runtime=runtime,
                cell="pinned_replay",
                schedule_id="variant-00",
                repetition=0,
                request_id="request-b",
                batch_index=0,
                batch_members=["request-b", "request-a"],
                batch_position=0,
                tokens=[3],
            ),
            execution(
                plan=plan,
                runtime=runtime,
                cell="pinned_replay",
                schedule_id="variant-00",
                repetition=0,
                request_id="request-a",
                batch_index=0,
                batch_members=["request-b", "request-a"],
                batch_position=1,
                tokens=[9, 2],
            ),
        ]
        (root / "executions.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        (root / "packages.txt").write_text(PACKAGE_INVENTORY_TEXT, encoding="utf-8")
        for name in ("nvidia-smi.txt", "runner.log"):
            (root / name).write_text("fixture\n", encoding="utf-8")
        manifest = {
            "schema": GPU_MANIFEST_SCHEMA,
            "experiment_id": plan["experiment_id"],
            "plan_sha256": plan_sha256(plan),
            "started_at": "2026-08-01T05:00:00Z",
            "completed_at": "2026-08-01T05:01:00Z",
            "execution_records": len(records),
            "completed_records": len(records),
            "failed_records": 0,
            "executions_sha256": sha256_file(root / "executions.jsonl"),
            "environment_sha256": sha256_file(root / "environment.json"),
            "requests_sha256": sha256_file(root / "requests.jsonl"),
            "model_snapshot_sha256": model_snapshot_sha256(
                plan["model"]["files_sha256"]
            ),
            "tracebench_commit": TRACEBENCH_COMMIT,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.assertEqual({path.name for path in root.iterdir()}, EVIDENCE_FILES)
        self.rewrite_checksums(root)

    def rewrite_checksums(self, root: Path) -> None:
        (root / "checksums.sha256").write_text(
            "".join(
                f"{sha256_file(root / name)}  {name}\n"
                for name in sorted(EVIDENCE_FILES)
            ),
            encoding="ascii",
        )

    def rewrite_manifest_hashes(self, root: Path) -> None:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        manifest["executions_sha256"] = sha256_file(root / "executions.jsonl")
        manifest["environment_sha256"] = sha256_file(root / "environment.json")
        manifest["requests_sha256"] = sha256_file(root / "requests.jsonl")
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.rewrite_checksums(root)

    def test_evidence_verification_and_analysis_keep_invalids_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evidence = Path(temporary_directory) / "evidence"
            evidence.mkdir()
            self.write_evidence(evidence)

            verify_gpu_evidence(evidence)
            analysis = analyze_gpu_evidence(evidence)

            unpinned, pinned = analysis["rows"]
            self.assertEqual(unpinned["valid_comparisons"], 2)
            self.assertEqual(unpinned["divergences"], 1)
            self.assertEqual(unpinned["sequence_divergence_rate_pct"], 50.0)
            self.assertEqual(pinned["valid_comparisons"], 2)
            self.assertEqual(pinned["divergences"], 0)
            self.assertEqual(pinned["sequence_divergence_rate_pct"], 0.0)

            output = Path(temporary_directory) / "analysis"
            write_gpu_analysis(output, analysis)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"gpu-results.csv", "gpu-results.json", "gpu-results.md"},
            )

    def test_outer_checksum_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evidence = Path(temporary_directory)
            self.write_evidence(evidence)
            (evidence / "runner.log").write_text("tampered\n", encoding="utf-8")

            with self.assertRaisesRegex(TraceBenchError, "checksum mismatch"):
                verify_gpu_evidence(evidence)

    def test_manifest_internal_hash_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evidence = Path(temporary_directory)
            self.write_evidence(evidence)
            manifest = json.loads(
                (evidence / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["executions_sha256"] = "0" * 64
            (evidence / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.rewrite_checksums(evidence)

            with self.assertRaisesRegex(TraceBenchError, "manifest.*executions"):
                verify_gpu_evidence(evidence)

    def test_evidence_cannot_predate_preregistration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evidence = Path(temporary_directory)
            self.write_evidence(evidence)
            manifest = json.loads(
                (evidence / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["started_at"] = "2026-08-01T04:00:00Z"
            (evidence / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.rewrite_checksums(evidence)

            with self.assertRaisesRegex(TraceBenchError, "predates preregistration"):
                verify_gpu_evidence(evidence)

    def test_self_consistent_execution_set_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evidence = Path(temporary_directory)
            self.write_evidence(evidence)
            records = [
                json.loads(line)
                for line in (evidence / "executions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            records[0]["request_id"] = "unregistered-request"
            (evidence / "executions.jsonl").write_text(
                "".join(
                    json.dumps(record, sort_keys=True) + "\n" for record in records
                ),
                encoding="utf-8",
            )
            self.rewrite_manifest_hashes(evidence)

            with self.assertRaisesRegex(TraceBenchError, "execution record set"):
                verify_gpu_evidence(evidence)

    def test_self_consistent_batch_metadata_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evidence = Path(temporary_directory)
            self.write_evidence(evidence)
            records = [
                json.loads(line)
                for line in (evidence / "executions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            records[2]["batch_members"] = ["request-b"]
            (evidence / "executions.jsonl").write_text(
                "".join(
                    json.dumps(record, sort_keys=True) + "\n" for record in records
                ),
                encoding="utf-8",
            )
            self.rewrite_manifest_hashes(evidence)

            with self.assertRaisesRegex(TraceBenchError, "batch_members"):
                verify_gpu_evidence(evidence)

    def test_self_consistent_runtime_scope_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evidence = Path(temporary_directory)
            self.write_evidence(evidence)
            runtime = json.loads(
                (evidence / "environment.json").read_text(encoding="utf-8")
            )
            runtime["torch_version"] = "9.9.9"
            runtime_without_digest = dict(runtime)
            del runtime_without_digest["runtime_digest"]
            runtime["runtime_digest"] = hashlib.sha256(
                canonical_json_bytes(runtime_without_digest)
            ).hexdigest()
            (evidence / "environment.json").write_text(
                json.dumps(runtime, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.rewrite_manifest_hashes(evidence)

            with self.assertRaisesRegex(TraceBenchError, "torch_version"):
                verify_gpu_evidence(evidence)

    def test_self_consistent_request_file_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evidence = Path(temporary_directory)
            self.write_evidence(evidence)
            requests = [
                json.loads(line)
                for line in (evidence / "requests.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            requests[0]["prompt"] = "Changed"
            requests[0]["input_sha256"] = request_input_sha256("Changed")
            (evidence / "requests.jsonl").write_text(
                "".join(
                    json.dumps(request, sort_keys=True) + "\n" for request in requests
                ),
                encoding="utf-8",
            )
            self.rewrite_manifest_hashes(evidence)

            with self.assertRaisesRegex(TraceBenchError, "requests.jsonl"):
                verify_gpu_evidence(evidence)


if __name__ == "__main__":
    unittest.main()
