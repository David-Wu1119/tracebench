"""Optional PyTorch/Transformers runner for the registered GPU experiment."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tracebench.gpu import (
    EVIDENCE_FILES,
    GPU_ENVIRONMENT_SCHEMA,
    GPU_EXECUTION_SCHEMA,
    GPU_MANIFEST_SCHEMA,
    GPU_REQUEST_SCHEMA,
    build_gpu_schedules,
    load_gpu_plan,
    model_snapshot_sha256,
    output_token_sha256,
    plan_sha256,
    request_input_sha256,
    sha256_file,
    validate_gpu_environment,
    verify_model_snapshot,
)
from tracebench.model import TraceBenchError, canonical_json_bytes


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as destination:
        destination.write(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )


def _run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def _trim_generated(token_ids: list[int], pad_token_id: int) -> list[int]:
    result = list(token_ids)
    while result and result[-1] == pad_token_id:
        result.pop()
    return result


def _normalized_package_inventory() -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if not line.lower().startswith("pip==")
    ]
    if not lines or any(not line for line in lines):
        raise TraceBenchError("pip returned an empty or malformed package inventory")
    return "\n".join(sorted(lines)) + "\n"


def run_gpu_experiment(
    *,
    plan_path: Path,
    model_path: Path,
    output_dir: Path,
    implementation_commit: str,
) -> Path:
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise TraceBenchError(
            "gpu-run requires CUDA-enabled PyTorch and Transformers"
        ) from exc

    plan = load_gpu_plan(plan_path)
    if len(implementation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in implementation_commit
    ):
        raise TraceBenchError(
            "implementation_commit must be a lowercase 40-character Git commit"
        )
    if not torch.cuda.is_available():
        raise TraceBenchError("CUDA is not available")
    if output_dir.exists():
        raise TraceBenchError(f"GPU evidence output already exists: {output_dir}")
    staging = output_dir.with_name(output_dir.name + ".partial")
    if staging.exists():
        raise TraceBenchError(f"partial GPU evidence already exists: {staging}")
    snapshot_digest = verify_model_snapshot(model_path, plan["model"])
    package_inventory = _normalized_package_inventory()
    package_inventory_sha256 = hashlib.sha256(
        package_inventory.encode("utf-8")
    ).hexdigest()
    runtime = plan["runtime"]
    torch.use_deterministic_algorithms(runtime["deterministic_algorithms"])
    torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = runtime["allow_tf32"]
    environment = {
        "schema": GPU_ENVIRONMENT_SCHEMA,
        "experiment_id": plan["experiment_id"],
        "plan_sha256": plan_sha256(plan),
        "python_version": platform.python_version(),
        "python_build": sys.version,
        "platform_system": platform.system(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "cuda_runtime": torch.version.cuda,
        "package_inventory_sha256": package_inventory_sha256,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_memory_mib": round(
            torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        ),
        "runtime_settings": dict(runtime),
        "model_repository": plan["model"]["repository"],
        "model_revision": plan["model"]["revision"],
        "model_files_sha256": dict(plan["model"]["files_sha256"]),
        "model_snapshot_sha256": snapshot_digest,
        "tracebench_repository": "https://github.com/David-Wu1119/tracebench",
        "tracebench_commit": implementation_commit,
    }
    environment["runtime_digest"] = hashlib.sha256(
        canonical_json_bytes(environment)
    ).hexdigest()
    validate_gpu_environment(environment, plan)

    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    log_path = staging / "runner.log"

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8", newline="\n") as destination:
            destination.write(f"{_utc_now()} {message}\n")

    started_at = _utc_now()
    log(f"experiment_start experiment_id={plan['experiment_id']}")
    (staging / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    requests_by_id = {request["request_id"]: request for request in plan["requests"]}
    requests_path = staging / "requests.jsonl"
    for request in plan["requests"]:
        _append_jsonl(
            requests_path,
            {
                "schema": GPU_REQUEST_SCHEMA,
                "request_id": request["request_id"],
                "prompt": request["prompt"],
                "input_sha256": request_input_sha256(request["prompt"]),
            },
        )

    _write_json(staging / "environment.json", environment)
    (staging / "nvidia-smi.txt").write_text(
        _run_text(["nvidia-smi", "-q"]), encoding="utf-8"
    )
    (staging / "packages.txt").write_bytes(package_inventory.encode("utf-8"))

    log("model_load_start")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = runtime["padding_side"]
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float16,
        local_files_only=True,
    )
    model.to(runtime["device"])
    model.eval()
    log("model_load_complete")

    schedules = build_gpu_schedules(plan)
    executions_path = staging / "executions.jsonl"
    completed_records = 0
    failed_records = 0

    def execute_schedule(
        *,
        mode: Mapping[str, Any],
        cell: str,
        schedule_id: str,
        repetition: int,
    ) -> None:
        nonlocal completed_records, failed_records
        seed = plan["schedule"]["global_seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generation = {
            "do_sample": mode["do_sample"],
            "max_new_tokens": mode["max_new_tokens"],
            "pad_token_id": tokenizer.pad_token_id,
        }
        if mode["do_sample"]:
            generation.update(
                {
                    "temperature": mode["temperature"],
                    "top_p": mode["top_p"],
                }
            )
        for batch_index, batch_members in enumerate(schedules[schedule_id]):
            texts = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": requests_by_id[request_id]["prompt"],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=runtime["add_generation_prompt"],
                )
                for request_id in batch_members
            ]
            batch_document = {
                "mode": mode["name"],
                "schedule_id": schedule_id,
                "seed": seed,
                "batch_members": list(batch_members),
            }
            batch_fingerprint = hashlib.sha256(
                canonical_json_bytes(batch_document)
            ).hexdigest()
            started = time.perf_counter()
            try:
                encoded = tokenizer(texts, return_tensors="pt", padding=True)
                encoded = {
                    name: tensor.to(runtime["device"])
                    for name, tensor in encoded.items()
                }
                input_width = encoded["input_ids"].shape[1]
                with torch.inference_mode():
                    generated = model.generate(**encoded, **generation)
                torch.cuda.synchronize()
                duration_ms = round((time.perf_counter() - started) * 1000, 3)
                for position, request_id in enumerate(batch_members):
                    token_ids = (
                        _trim_generated(
                            generated[position, input_width:].detach().cpu().tolist(),
                            tokenizer.pad_token_id,
                        )
                        if runtime["strip_trailing_pad_tokens"]
                        else generated[position, input_width:].detach().cpu().tolist()
                    )
                    _append_jsonl(
                        executions_path,
                        {
                            "schema": GPU_EXECUTION_SCHEMA,
                            "experiment_id": plan["experiment_id"],
                            "tracebench_commit": implementation_commit,
                            "mode": mode["name"],
                            "cell": cell,
                            "schedule_id": schedule_id,
                            "repetition": repetition,
                            "request_id": request_id,
                            "input_sha256": request_input_sha256(
                                requests_by_id[request_id]["prompt"]
                            ),
                            "batch_id": f"{schedule_id}-batch-{batch_index:03d}",
                            "batch_members": list(batch_members),
                            "batch_position": position,
                            "batch_fingerprint": batch_fingerprint,
                            "global_seed": seed,
                            "runtime_digest": environment["runtime_digest"],
                            "status": "completed",
                            "output_token_ids": token_ids,
                            "output_sha256": output_token_sha256(token_ids),
                            "batch_duration_ms": duration_ms,
                            "error": None,
                        },
                    )
                    completed_records += 1
            except Exception as exc:
                duration_ms = round((time.perf_counter() - started) * 1000, 3)
                torch.cuda.empty_cache()
                error = f"{type(exc).__name__}: {exc}"[:500]
                log(
                    f"batch_failed mode={mode['name']} cell={cell} "
                    f"schedule={schedule_id} batch={batch_index} error={error}"
                )
                for position, request_id in enumerate(batch_members):
                    _append_jsonl(
                        executions_path,
                        {
                            "schema": GPU_EXECUTION_SCHEMA,
                            "experiment_id": plan["experiment_id"],
                            "tracebench_commit": implementation_commit,
                            "mode": mode["name"],
                            "cell": cell,
                            "schedule_id": schedule_id,
                            "repetition": repetition,
                            "request_id": request_id,
                            "input_sha256": request_input_sha256(
                                requests_by_id[request_id]["prompt"]
                            ),
                            "batch_id": f"{schedule_id}-batch-{batch_index:03d}",
                            "batch_members": list(batch_members),
                            "batch_position": position,
                            "batch_fingerprint": batch_fingerprint,
                            "global_seed": seed,
                            "runtime_digest": environment["runtime_digest"],
                            "status": "failed",
                            "output_token_ids": [],
                            "output_sha256": None,
                            "batch_duration_ms": duration_ms,
                            "error": error,
                        },
                    )
                    failed_records += 1

    for mode in plan["decoding_modes"]:
        log(f"mode_start mode={mode['name']}")
        execute_schedule(
            mode=mode,
            cell="baseline",
            schedule_id="baseline",
            repetition=0,
        )
        for variant in range(plan["schedule"]["variant_count"]):
            execute_schedule(
                mode=mode,
                cell="unpinned",
                schedule_id=f"variant-{variant:02d}",
                repetition=variant,
            )
        for repetition in range(plan["schedule"]["pinned_repetitions"]):
            execute_schedule(
                mode=mode,
                cell="pinned_replay",
                schedule_id="variant-00",
                repetition=repetition,
            )
        log(f"mode_complete mode={mode['name']}")

    expected_records = (
        len(plan["decoding_modes"])
        * plan["request_count"]
        * (
            1
            + plan["schedule"]["variant_count"]
            + plan["schedule"]["pinned_repetitions"]
        )
    )
    if completed_records + failed_records != expected_records:
        raise TraceBenchError("GPU runner did not emit the registered record count")
    log(
        f"experiment_complete completed_records={completed_records} "
        f"failed_records={failed_records}"
    )
    manifest = {
        "schema": GPU_MANIFEST_SCHEMA,
        "experiment_id": plan["experiment_id"],
        "plan_sha256": plan_sha256(plan),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "execution_records": expected_records,
        "completed_records": completed_records,
        "failed_records": failed_records,
        "executions_sha256": sha256_file(executions_path),
        "environment_sha256": sha256_file(staging / "environment.json"),
        "requests_sha256": sha256_file(requests_path),
        "model_snapshot_sha256": model_snapshot_sha256(plan["model"]["files_sha256"]),
        "tracebench_commit": implementation_commit,
    }
    _write_json(staging / "manifest.json", manifest)

    missing = EVIDENCE_FILES - {path.name for path in staging.iterdir()}
    if missing:
        raise TraceBenchError(f"GPU evidence is missing files: {sorted(missing)}")
    checksum_lines = [
        f"{sha256_file(staging / name)}  {name}" for name in sorted(EVIDENCE_FILES)
    ]
    (staging / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="ascii",
    )
    os.replace(staging, output_dir)
    return output_dir
