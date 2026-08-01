"""Pure-stdlib GPU experiment contract, schedule, verification, and analysis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from tracebench.model import TraceBenchError, assert_sha256, canonical_json_bytes


GPU_PLAN_SCHEMA = "tracebench/gpu-plan/v1"
GPU_EXECUTION_SCHEMA = "tracebench/gpu-execution/v1"
GPU_MANIFEST_SCHEMA = "tracebench/gpu-evidence-manifest/v1"
GPU_ANALYSIS_SCHEMA = "tracebench/gpu-analysis/v1"
GPU_ENVIRONMENT_SCHEMA = "tracebench/gpu-environment/v1"
GPU_REQUEST_SCHEMA = "tracebench/gpu-request/v1"
MODEL_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)
EVIDENCE_FILES = frozenset(
    {
        "environment.json",
        "executions.jsonl",
        "manifest.json",
        "nvidia-smi.txt",
        "packages.txt",
        "plan.json",
        "requests.jsonl",
        "runner.log",
    }
)
EVIDENCE_DIRECTORY_FILES = EVIDENCE_FILES | {"checksums.sha256"}

PLAN_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "preregistration",
        "model",
        "request_count",
        "requests",
        "decoding_modes",
        "schedule",
        "runtime",
        "software_scope",
        "hardware_scope",
    }
)
EXECUTION_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "tracebench_commit",
        "mode",
        "cell",
        "schedule_id",
        "repetition",
        "request_id",
        "input_sha256",
        "batch_id",
        "batch_members",
        "batch_position",
        "batch_fingerprint",
        "global_seed",
        "runtime_digest",
        "status",
        "output_token_ids",
        "output_sha256",
        "batch_duration_ms",
        "error",
    }
)
ENVIRONMENT_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "plan_sha256",
        "python_version",
        "python_build",
        "platform_system",
        "platform",
        "torch_version",
        "transformers_version",
        "cuda_runtime",
        "package_inventory_sha256",
        "cuda_available",
        "gpu_name",
        "gpu_capability",
        "gpu_memory_mib",
        "runtime_settings",
        "model_repository",
        "model_revision",
        "model_files_sha256",
        "model_snapshot_sha256",
        "tracebench_repository",
        "tracebench_commit",
        "runtime_digest",
    }
)
MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "plan_sha256",
        "started_at",
        "completed_at",
        "execution_records",
        "completed_records",
        "failed_records",
        "executions_sha256",
        "environment_sha256",
        "requests_sha256",
        "model_snapshot_sha256",
        "tracebench_commit",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def plan_sha256(plan: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(plan)).hexdigest()


def output_token_sha256(token_ids: list[int]) -> str:
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in token_ids
    ):
        raise TraceBenchError("output token IDs must be non-negative integers")
    return hashlib.sha256(
        canonical_json_bytes({"output_token_ids": token_ids})
    ).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceBenchError(f"{name} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise TraceBenchError(
            f"{name} fields do not match schema; missing={missing}, extra={extra}"
        )


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TraceBenchError(f"{name} is required")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TraceBenchError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TraceBenchError(f"{name} must be a non-negative integer")
    return value


def _require_commit(value: Any, name: str) -> str:
    value = _require_nonempty_string(value, name)
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise TraceBenchError(f"{name} must be a lowercase 40-character Git commit")
    return value


def _require_timestamp(value: Any, name: str) -> datetime:
    value = _require_nonempty_string(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TraceBenchError(f"{name} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise TraceBenchError(f"{name} must include a timezone")
    return parsed


def model_snapshot_sha256(files_sha256: Mapping[str, Any]) -> str:
    files = _require_mapping(files_sha256, "model.files_sha256")
    if set(files) != MODEL_FILES:
        raise TraceBenchError("model snapshot must bind the exact registered file set")
    normalized: dict[str, str] = {}
    for name in sorted(MODEL_FILES):
        digest = files.get(name)
        if not isinstance(digest, str):
            raise TraceBenchError(f"model.files_sha256.{name} is required")
        assert_sha256(digest, field_name=f"model.files_sha256.{name}")
        normalized[name] = digest
    return hashlib.sha256(
        canonical_json_bytes({"files_sha256": normalized})
    ).hexdigest()


def verify_model_snapshot(model_path: Path, model: Mapping[str, Any]) -> str:
    if not model_path.is_dir() or model_path.is_symlink():
        raise TraceBenchError("model path must be a local regular directory")
    expected = _require_mapping(model.get("files_sha256"), "model.files_sha256")
    model_snapshot_sha256(expected)
    try:
        entries = list(model_path.iterdir())
    except OSError as exc:
        raise TraceBenchError(f"cannot inventory model snapshot: {exc}") from exc
    if {entry.name for entry in entries} != MODEL_FILES:
        raise TraceBenchError(
            "model snapshot must contain the exact registered file set"
        )
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise TraceBenchError(
                f"model snapshot file is missing or unsafe: {entry.name}"
            )
        if sha256_file(entry) != expected[entry.name]:
            raise TraceBenchError(f"model snapshot checksum mismatch: {entry.name}")
    return model_snapshot_sha256(expected)


def validate_gpu_plan(plan: Mapping[str, Any]) -> None:
    canonical_json_bytes(plan)
    _require_exact_keys(plan, PLAN_FIELDS, "GPU plan")
    if plan.get("schema") != GPU_PLAN_SCHEMA:
        raise TraceBenchError(f"GPU plan schema must be {GPU_PLAN_SCHEMA}")
    _require_nonempty_string(plan.get("experiment_id"), "experiment_id")

    preregistration = _require_mapping(plan.get("preregistration"), "preregistration")
    _require_exact_keys(
        preregistration,
        frozenset(
            {
                "registered_at",
                "primary_metric",
                "sample_size_rule",
                "reporting_rule",
                "hypothesis",
            }
        ),
        "preregistration",
    )
    _require_timestamp(
        preregistration.get("registered_at"), "preregistration.registered_at"
    )
    for field in (
        "primary_metric",
        "sample_size_rule",
        "reporting_rule",
        "hypothesis",
    ):
        _require_nonempty_string(preregistration.get(field), f"preregistration.{field}")

    model = _require_mapping(plan.get("model"), "model")
    _require_exact_keys(
        model,
        frozenset({"repository", "revision", "files_sha256"}),
        "model",
    )
    for field in ("repository", "revision"):
        _require_nonempty_string(model.get(field), f"model.{field}")
    _require_commit(model["revision"], "model.revision")
    model_snapshot_sha256(model.get("files_sha256"))

    requests = plan.get("requests")
    if not isinstance(requests, list) or not requests:
        raise TraceBenchError("requests must be a nonempty array")
    request_count = _require_positive_int(plan.get("request_count"), "request_count")
    if len(requests) != request_count:
        raise TraceBenchError("request_count does not match requests")
    request_ids: list[str] = []
    for index, raw_request in enumerate(requests):
        request = _require_mapping(raw_request, f"requests[{index}]")
        _require_exact_keys(
            request,
            frozenset({"request_id", "prompt"}),
            f"requests[{index}]",
        )
        request_id = request.get("request_id")
        prompt = request.get("prompt")
        if not isinstance(request_id, str) or not request_id:
            raise TraceBenchError(f"requests[{index}].request_id is required")
        if not isinstance(prompt, str) or not prompt.strip():
            raise TraceBenchError(f"requests[{index}].prompt is required")
        request_ids.append(request_id)
    if len(set(request_ids)) != len(request_ids):
        raise TraceBenchError("request IDs must be unique")

    modes = plan.get("decoding_modes")
    if not isinstance(modes, list) or not modes:
        raise TraceBenchError("decoding_modes must be a nonempty array")
    mode_names: list[str] = []
    for index, raw_mode in enumerate(modes):
        mode = _require_mapping(raw_mode, f"decoding_modes[{index}]")
        name = mode.get("name")
        if not isinstance(name, str) or not name:
            raise TraceBenchError(f"decoding_modes[{index}].name is required")
        if not isinstance(mode.get("do_sample"), bool):
            raise TraceBenchError(f"decoding_modes[{index}].do_sample must be boolean")
        expected_mode_fields = {"name", "do_sample", "max_new_tokens"}
        if mode["do_sample"]:
            expected_mode_fields.update({"temperature", "top_p"})
        _require_exact_keys(
            mode,
            frozenset(expected_mode_fields),
            f"decoding_modes[{index}]",
        )
        _require_positive_int(
            mode.get("max_new_tokens"),
            f"decoding_modes[{index}].max_new_tokens",
        )
        if mode["do_sample"]:
            for field in ("temperature", "top_p"):
                value = mode.get(field)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TraceBenchError(
                        f"decoding_modes[{index}].{field} is required"
                    )
                if not math.isfinite(float(value)) or float(value) <= 0:
                    raise TraceBenchError(
                        f"decoding_modes[{index}].{field} must be finite and positive"
                    )
            if float(mode["top_p"]) > 1:
                raise TraceBenchError(
                    f"decoding_modes[{index}].top_p must not exceed 1"
                )
        mode_names.append(name)
    if len(set(mode_names)) != len(mode_names):
        raise TraceBenchError("decoding mode names must be unique")

    schedule = _require_mapping(plan.get("schedule"), "schedule")
    _require_exact_keys(
        schedule,
        frozenset(
            {
                "global_seed",
                "variant_count",
                "pinned_repetitions",
                "batch_sizes",
                "variant_seeds",
            }
        ),
        "schedule",
    )
    _require_positive_int(schedule.get("global_seed"), "schedule.global_seed")
    variants = _require_positive_int(
        schedule.get("variant_count"), "schedule.variant_count"
    )
    _require_positive_int(
        schedule.get("pinned_repetitions"), "schedule.pinned_repetitions"
    )
    batch_sizes = schedule.get("batch_sizes")
    if not isinstance(batch_sizes, list) or not batch_sizes:
        raise TraceBenchError("schedule.batch_sizes must be a nonempty array")
    for index, value in enumerate(batch_sizes):
        _require_positive_int(value, f"schedule.batch_sizes[{index}]")
        if value > request_count:
            raise TraceBenchError("schedule batch sizes must not exceed request_count")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise TraceBenchError("schedule.batch_sizes must be unique")
    seeds = schedule.get("variant_seeds")
    if not isinstance(seeds, list) or len(seeds) != variants:
        raise TraceBenchError("schedule.variant_seeds must match variant_count")
    for index, value in enumerate(seeds):
        _require_positive_int(value, f"schedule.variant_seeds[{index}]")
    if len(set(seeds)) != len(seeds):
        raise TraceBenchError("schedule.variant_seeds must be unique")

    runtime = _require_mapping(plan.get("runtime"), "runtime")
    _require_exact_keys(
        runtime,
        frozenset(
            {
                "device",
                "dtype",
                "padding_side",
                "local_files_only",
                "deterministic_algorithms",
                "cudnn_benchmark",
                "allow_tf32",
                "chat_template",
                "add_generation_prompt",
                "strip_trailing_pad_tokens",
                "seed_reset",
            }
        ),
        "runtime",
    )
    if runtime.get("device") != "cuda:0":
        raise TraceBenchError("runtime.device must be cuda:0")
    if runtime.get("dtype") != "float16":
        raise TraceBenchError("runtime.dtype must be float16")
    if runtime.get("padding_side") != "left":
        raise TraceBenchError("runtime.padding_side must be left")
    if runtime.get("local_files_only") is not True:
        raise TraceBenchError("runtime.local_files_only must be true")
    for field in (
        "deterministic_algorithms",
        "cudnn_benchmark",
        "allow_tf32",
        "add_generation_prompt",
        "strip_trailing_pad_tokens",
    ):
        if not isinstance(runtime.get(field), bool):
            raise TraceBenchError(f"runtime.{field} must be boolean")
    if runtime.get("chat_template") != "tokenizer.apply_chat_template":
        raise TraceBenchError(
            "runtime.chat_template must be tokenizer.apply_chat_template"
        )
    if runtime.get("seed_reset") != "before_each_schedule":
        raise TraceBenchError("runtime.seed_reset must be before_each_schedule")

    software = _require_mapping(plan.get("software_scope"), "software_scope")
    _require_exact_keys(
        software,
        frozenset(
            {
                "python_version",
                "torch_version",
                "transformers_version",
                "cuda_runtime",
                "package_inventory_sha256",
            }
        ),
        "software_scope",
    )
    for field in (
        "python_version",
        "torch_version",
        "transformers_version",
        "cuda_runtime",
    ):
        _require_nonempty_string(software.get(field), f"software_scope.{field}")
    package_inventory_sha256 = software.get("package_inventory_sha256")
    if not isinstance(package_inventory_sha256, str):
        raise TraceBenchError("software_scope.package_inventory_sha256 is required")
    assert_sha256(
        package_inventory_sha256,
        field_name="software_scope.package_inventory_sha256",
    )

    hardware = _require_mapping(plan.get("hardware_scope"), "hardware_scope")
    _require_exact_keys(
        hardware,
        frozenset(
            {
                "gpu_name",
                "vram_mib",
                "compute_capability",
                "native_os",
                "engine",
            }
        ),
        "hardware_scope",
    )
    for field in ("gpu_name", "native_os", "engine"):
        _require_nonempty_string(hardware.get(field), f"hardware_scope.{field}")
    _require_positive_int(hardware.get("vram_mib"), "hardware_scope.vram_mib")
    capability = hardware.get("compute_capability")
    if not isinstance(capability, list) or len(capability) != 2:
        raise TraceBenchError(
            "hardware_scope.compute_capability must have two integers"
        )
    for index, value in enumerate(capability):
        _require_nonnegative_int(value, f"hardware_scope.compute_capability[{index}]")


def load_gpu_plan(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceBenchError(f"cannot load GPU plan {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TraceBenchError("GPU plan must be a JSON object")
    validate_gpu_plan(payload)
    return payload


def build_gpu_schedules(
    plan: Mapping[str, Any],
) -> dict[str, tuple[tuple[str, ...], ...]]:
    validate_gpu_plan(plan)
    request_ids = [request["request_id"] for request in plan["requests"]]
    schedule = plan["schedule"]
    result: dict[str, tuple[tuple[str, ...], ...]] = {
        "baseline": tuple((request_id,) for request_id in request_ids)
    }
    for index, seed in enumerate(schedule["variant_seeds"]):
        random_source = random.Random(seed)
        shuffled = list(request_ids)
        random_source.shuffle(shuffled)
        batches: list[tuple[str, ...]] = []
        cursor = 0
        while cursor < len(shuffled):
            batch_size = random_source.choice(schedule["batch_sizes"])
            batches.append(tuple(shuffled[cursor : cursor + batch_size]))
            cursor += batch_size
        result[f"variant-{index:02d}"] = tuple(batches)
    return result


def request_input_sha256(prompt: str) -> str:
    document = {"messages": [{"role": "user", "content": prompt}]}
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _parse_checksum_sidecar(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    if not path.is_file() or path.is_symlink():
        raise TraceBenchError("checksum sidecar is missing or unsafe")
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise TraceBenchError(f"cannot read checksum sidecar: {exc}") from exc
    for line in lines:
        if "  " not in line:
            raise TraceBenchError("malformed checksum sidecar line")
        digest, name = line.split("  ", 1)
        assert_sha256(digest, field_name="checksum digest")
        if not name or Path(name).name != name or name in entries:
            raise TraceBenchError("unsafe or duplicate checksum path")
        entries[name] = digest
    if set(entries) != EVIDENCE_FILES:
        raise TraceBenchError(
            "checksum sidecar does not bind the exact evidence file set"
        )
    return entries


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceBenchError(f"cannot load {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise TraceBenchError(f"{name} must be a JSON object")
    return value


def _load_jsonl_objects(path: Path, name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        source = path.open(encoding="utf-8")
    except OSError as exc:
        raise TraceBenchError(f"cannot read {name}: {exc}") from exc
    with source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise TraceBenchError(f"{name}:{line_number} is blank")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceBenchError(f"{name}:{line_number} is invalid JSON") from exc
            if not isinstance(record, dict):
                raise TraceBenchError(f"{name}:{line_number} must be a JSON object")
            records.append(record)
    return records


def _verify_requests(path: Path, plan: Mapping[str, Any]) -> None:
    actual = _load_jsonl_objects(path, "requests.jsonl")
    expected = [
        {
            "schema": GPU_REQUEST_SCHEMA,
            "request_id": request["request_id"],
            "prompt": request["prompt"],
            "input_sha256": request_input_sha256(request["prompt"]),
        }
        for request in plan["requests"]
    ]
    if actual != expected:
        raise TraceBenchError("requests.jsonl does not exactly match the GPU plan")


def validate_gpu_environment(
    environment: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    _require_exact_keys(environment, ENVIRONMENT_FIELDS, "GPU environment")
    if environment.get("schema") != GPU_ENVIRONMENT_SCHEMA:
        raise TraceBenchError("GPU environment schema is invalid")
    expected_values = {
        "experiment_id": plan["experiment_id"],
        "plan_sha256": plan_sha256(plan),
        "python_version": plan["software_scope"]["python_version"],
        "platform_system": plan["hardware_scope"]["native_os"],
        "torch_version": plan["software_scope"]["torch_version"],
        "transformers_version": plan["software_scope"]["transformers_version"],
        "cuda_runtime": plan["software_scope"]["cuda_runtime"],
        "package_inventory_sha256": plan["software_scope"]["package_inventory_sha256"],
        "cuda_available": True,
        "gpu_name": plan["hardware_scope"]["gpu_name"],
        "gpu_capability": plan["hardware_scope"]["compute_capability"],
        "gpu_memory_mib": plan["hardware_scope"]["vram_mib"],
        "runtime_settings": plan["runtime"],
        "model_repository": plan["model"]["repository"],
        "model_revision": plan["model"]["revision"],
        "model_files_sha256": plan["model"]["files_sha256"],
        "model_snapshot_sha256": model_snapshot_sha256(plan["model"]["files_sha256"]),
        "tracebench_repository": "https://github.com/David-Wu1119/tracebench",
    }
    for field, expected in expected_values.items():
        if environment.get(field) != expected:
            raise TraceBenchError(f"GPU environment {field} does not match the plan")
    _require_nonempty_string(
        environment.get("python_build"), "environment.python_build"
    )
    _require_nonempty_string(environment.get("platform"), "environment.platform")
    _require_commit(
        environment.get("tracebench_commit"), "environment.tracebench_commit"
    )
    runtime_digest = environment.get("runtime_digest")
    if not isinstance(runtime_digest, str):
        raise TraceBenchError("environment.runtime_digest is required")
    assert_sha256(runtime_digest, field_name="environment.runtime_digest")
    digest_input = dict(environment)
    del digest_input["runtime_digest"]
    expected_digest = hashlib.sha256(canonical_json_bytes(digest_input)).hexdigest()
    if runtime_digest != expected_digest:
        raise TraceBenchError("GPU environment runtime_digest is invalid")


def _expected_execution_metadata(
    plan: Mapping[str, Any], environment: Mapping[str, Any]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    schedules = build_gpu_schedules(plan)
    prompts = {request["request_id"]: request["prompt"] for request in plan["requests"]}
    contexts: list[tuple[str, str, int]] = [("baseline", "baseline", 0)]
    contexts.extend(
        ("unpinned", f"variant-{variant:02d}", variant)
        for variant in range(plan["schedule"]["variant_count"])
    )
    contexts.extend(
        ("pinned_replay", "variant-00", repetition)
        for repetition in range(plan["schedule"]["pinned_repetitions"])
    )
    expected: dict[tuple[Any, ...], dict[str, Any]] = {}
    seed = plan["schedule"]["global_seed"]
    for mode in plan["decoding_modes"]:
        for cell, schedule_id, repetition in contexts:
            for batch_index, batch_members_tuple in enumerate(schedules[schedule_id]):
                batch_members = list(batch_members_tuple)
                batch_document = {
                    "mode": mode["name"],
                    "schedule_id": schedule_id,
                    "seed": seed,
                    "batch_members": batch_members,
                }
                fingerprint = hashlib.sha256(
                    canonical_json_bytes(batch_document)
                ).hexdigest()
                for position, request_id in enumerate(batch_members):
                    key = (
                        mode["name"],
                        cell,
                        schedule_id,
                        repetition,
                        request_id,
                    )
                    expected[key] = {
                        "schema": GPU_EXECUTION_SCHEMA,
                        "experiment_id": plan["experiment_id"],
                        "tracebench_commit": environment["tracebench_commit"],
                        "mode": mode["name"],
                        "cell": cell,
                        "schedule_id": schedule_id,
                        "repetition": repetition,
                        "request_id": request_id,
                        "input_sha256": request_input_sha256(prompts[request_id]),
                        "batch_id": f"{schedule_id}-batch-{batch_index:03d}",
                        "batch_members": batch_members,
                        "batch_position": position,
                        "batch_fingerprint": fingerprint,
                        "global_seed": seed,
                        "runtime_digest": environment["runtime_digest"],
                    }
    return expected


def _load_execution_records(
    path: Path,
    plan: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = _load_jsonl_objects(path, "executions.jsonl")
    seen: set[tuple[Any, ...]] = set()
    expected = _expected_execution_metadata(plan, environment)
    for line_number, record in enumerate(records, start=1):
        _require_exact_keys(
            record,
            EXECUTION_FIELDS,
            f"executions.jsonl:{line_number}",
        )
        key = (
            record.get("mode"),
            record.get("cell"),
            record.get("schedule_id"),
            record.get("repetition"),
            record.get("request_id"),
        )
        if key in seen:
            raise TraceBenchError(f"duplicate GPU execution record: {key}")
        seen.add(key)
    if seen != set(expected):
        raise TraceBenchError(
            "GPU execution record set does not exactly match the plan"
        )

    batch_states: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for record in records:
        key = (
            record["mode"],
            record["cell"],
            record["schedule_id"],
            record["repetition"],
            record["request_id"],
        )
        for field, expected_value in expected[key].items():
            if record.get(field) != expected_value:
                raise TraceBenchError(f"GPU execution {field} does not match the plan")
        status = record.get("status")
        if status not in {"completed", "failed"}:
            raise TraceBenchError("GPU execution status must be completed or failed")
        duration = record.get("batch_duration_ms")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration < 0
        ):
            raise TraceBenchError("GPU execution batch_duration_ms is invalid")
        if status == "completed":
            token_ids = record.get("output_token_ids")
            if not isinstance(token_ids, list):
                raise TraceBenchError("completed execution is missing output_token_ids")
            expected_output = output_token_sha256(token_ids)
            if record.get("output_sha256") != expected_output:
                raise TraceBenchError("GPU execution output digest is invalid")
            if record.get("error") is not None:
                raise TraceBenchError("completed GPU execution must not carry an error")
        else:
            if (
                record.get("output_token_ids") != []
                or record.get("output_sha256") is not None
            ):
                raise TraceBenchError(
                    "failed GPU execution must not carry output tokens"
                )
            if not isinstance(record.get("error"), str) or not record["error"]:
                raise TraceBenchError("failed GPU execution must carry an error")
        batch_key = (
            record["mode"],
            record["cell"],
            record["schedule_id"],
            record["repetition"],
            record["batch_id"],
        )
        batch_state = (status, duration, record.get("error"))
        if batch_key in batch_states and batch_states[batch_key] != batch_state:
            raise TraceBenchError(
                "GPU execution batch outcome metadata is inconsistent"
            )
        batch_states[batch_key] = batch_state
    return records


def verify_gpu_evidence(evidence_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not evidence_dir.is_dir() or evidence_dir.is_symlink():
        raise TraceBenchError(
            f"GPU evidence directory is missing or unsafe: {evidence_dir}"
        )
    try:
        directory_entries = list(evidence_dir.iterdir())
    except OSError as exc:
        raise TraceBenchError(f"cannot inventory GPU evidence: {exc}") from exc
    if {entry.name for entry in directory_entries} != EVIDENCE_DIRECTORY_FILES:
        raise TraceBenchError("GPU evidence directory has an unbound file set")
    entries = _parse_checksum_sidecar(evidence_dir / "checksums.sha256")
    for name, expected in entries.items():
        path = evidence_dir / name
        if not path.is_file() or path.is_symlink():
            raise TraceBenchError(f"evidence file is missing or unsafe: {name}")
        if sha256_file(path) != expected:
            raise TraceBenchError(f"evidence checksum mismatch: {name}")

    plan = load_gpu_plan(evidence_dir / "plan.json")
    environment = _read_json_object(
        evidence_dir / "environment.json", "GPU environment"
    )
    validate_gpu_environment(environment, plan)
    if (
        sha256_file(evidence_dir / "packages.txt")
        != environment["package_inventory_sha256"]
    ):
        raise TraceBenchError("packages.txt does not match the registered environment")
    _verify_requests(evidence_dir / "requests.jsonl", plan)
    records = _load_execution_records(
        evidence_dir / "executions.jsonl", plan, environment
    )
    manifest = _read_json_object(
        evidence_dir / "manifest.json", "GPU evidence manifest"
    )
    _require_exact_keys(manifest, MANIFEST_FIELDS, "GPU evidence manifest")
    if manifest.get("schema") != GPU_MANIFEST_SCHEMA:
        raise TraceBenchError("GPU evidence manifest schema is invalid")
    if manifest.get("experiment_id") != plan["experiment_id"]:
        raise TraceBenchError("manifest experiment_id does not match plan")
    if manifest.get("plan_sha256") != plan_sha256(plan):
        raise TraceBenchError("manifest does not bind the GPU plan")
    for field, filename in (
        ("executions_sha256", "executions.jsonl"),
        ("environment_sha256", "environment.json"),
        ("requests_sha256", "requests.jsonl"),
    ):
        digest = manifest.get(field)
        if not isinstance(digest, str):
            raise TraceBenchError(f"manifest {field} is required")
        assert_sha256(digest, field_name=f"manifest.{field}")
        if digest != sha256_file(evidence_dir / filename):
            raise TraceBenchError(f"manifest {field} does not match {filename}")
    expected_snapshot = model_snapshot_sha256(plan["model"]["files_sha256"])
    if manifest.get("model_snapshot_sha256") != expected_snapshot:
        raise TraceBenchError("manifest model snapshot does not match the plan")
    _require_commit(manifest.get("tracebench_commit"), "manifest.tracebench_commit")
    if manifest["tracebench_commit"] != environment["tracebench_commit"]:
        raise TraceBenchError("manifest TraceBench commit does not match environment")
    started = _require_timestamp(manifest.get("started_at"), "manifest.started_at")
    completed = _require_timestamp(
        manifest.get("completed_at"), "manifest.completed_at"
    )
    registered = _require_timestamp(
        plan["preregistration"]["registered_at"],
        "preregistration.registered_at",
    )
    if started < registered:
        raise TraceBenchError("GPU evidence predates preregistration")
    if completed < started:
        raise TraceBenchError("manifest completion precedes start")
    expected_records = len(_expected_execution_metadata(plan, environment))
    counts = {
        "execution_records": len(records),
        "completed_records": sum(record["status"] == "completed" for record in records),
        "failed_records": sum(record["status"] == "failed" for record in records),
    }
    for field, expected_count in counts.items():
        _require_nonnegative_int(manifest.get(field), f"manifest.{field}")
        if manifest[field] != expected_count:
            raise TraceBenchError(f"manifest {field} count is wrong")
    if manifest["execution_records"] != expected_records:
        raise TraceBenchError("manifest execution count does not match the GPU plan")
    if manifest["completed_records"] + manifest["failed_records"] != expected_records:
        raise TraceBenchError("manifest outcome counts do not sum to execution records")
    return plan, manifest


def _wilson_interval(successes: int, trials: int) -> tuple[float | None, float | None]:
    if trials == 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + (z**2 / trials)
    center = (proportion + (z**2 / (2 * trials))) / denominator
    margin = (
        z
        * math.sqrt((proportion * (1 - proportion) / trials) + (z**2 / (4 * trials**2)))
        / denominator
    )
    return round(max(0.0, center - margin) * 100, 3), round(
        min(1.0, center + margin) * 100,
        3,
    )


def _first_difference(left: list[int], right: list[int]) -> float:
    index = 0
    while index < min(len(left), len(right)) and left[index] == right[index]:
        index += 1
    return index / max(len(left), len(right), 1)


def analyze_gpu_evidence(evidence_dir: Path) -> dict[str, Any]:
    plan, manifest = verify_gpu_evidence(evidence_dir)
    environment = _read_json_object(
        evidence_dir / "environment.json", "GPU environment"
    )
    records = _load_execution_records(
        evidence_dir / "executions.jsonl", plan, environment
    )
    request_ids = [request["request_id"] for request in plan["requests"]]
    variants = plan["schedule"]["variant_count"]
    pinned_repetitions = plan["schedule"]["pinned_repetitions"]

    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            record["mode"],
            record["cell"],
            record["schedule_id"],
            record["repetition"],
            record["request_id"],
        )
        index[key] = record

    def compare(
        reference: dict[str, Any] | None,
        replay: dict[str, Any] | None,
    ) -> tuple[bool, bool, float | None]:
        if (
            reference is None
            or replay is None
            or reference.get("status") != "completed"
            or replay.get("status") != "completed"
        ):
            return False, False, None
        diverged = reference["output_sha256"] != replay["output_sha256"]
        difference = (
            _first_difference(reference["output_token_ids"], replay["output_token_ids"])
            if diverged
            else None
        )
        return True, diverged, difference

    rows: list[dict[str, Any]] = []
    for mode in (item["name"] for item in plan["decoding_modes"]):
        for condition in ("uncontrolled_batch_variation", "capsule_pinned_replay"):
            valid = 0
            divergences = 0
            first_differences: list[float] = []
            planned = (
                variants * len(request_ids)
                if condition == "uncontrolled_batch_variation"
                else pinned_repetitions * len(request_ids)
            )
            if condition == "uncontrolled_batch_variation":
                comparisons = (
                    (
                        index.get((mode, "baseline", "baseline", 0, request_id)),
                        index.get(
                            (
                                mode,
                                "unpinned",
                                f"variant-{variant:02d}",
                                variant,
                                request_id,
                            )
                        ),
                    )
                    for variant in range(variants)
                    for request_id in request_ids
                )
            else:
                comparisons = (
                    (
                        index.get((mode, "unpinned", "variant-00", 0, request_id)),
                        index.get(
                            (
                                mode,
                                "pinned_replay",
                                "variant-00",
                                repetition,
                                request_id,
                            )
                        ),
                    )
                    for repetition in range(pinned_repetitions)
                    for request_id in request_ids
                )
            for reference, replay in comparisons:
                is_valid, diverged, first_difference = compare(reference, replay)
                if not is_valid:
                    continue
                valid += 1
                divergences += int(diverged)
                if first_difference is not None:
                    first_differences.append(first_difference)
            lower, upper = _wilson_interval(divergences, valid)
            rows.append(
                {
                    "mode": mode,
                    "condition": condition,
                    "planned_comparisons": planned,
                    "valid_comparisons": valid,
                    "invalid_comparisons": planned - valid,
                    "divergences": divergences,
                    "sequence_divergence_rate_pct": (
                        round((divergences / valid) * 100, 3) if valid else None
                    ),
                    "wilson_95_lower_pct": lower,
                    "wilson_95_upper_pct": upper,
                    "mean_normalized_first_difference": (
                        round(sum(first_differences) / len(first_differences), 6)
                        if first_differences
                        else None
                    ),
                }
            )
    return {
        "schema": GPU_ANALYSIS_SCHEMA,
        "experiment_id": plan["experiment_id"],
        "plan_sha256": plan_sha256(plan),
        "evidence_manifest_sha256": sha256_file(evidence_dir / "manifest.json"),
        "rows": rows,
        "interpretation": (
            "Exact output-token sequence divergence on one declared hardware/runtime scope."
        ),
    }


def write_gpu_analysis(output_dir: Path, analysis: Mapping[str, Any]) -> None:
    if output_dir.exists():
        raise TraceBenchError(f"analysis output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    rows = analysis["rows"]
    fieldnames = (
        "mode",
        "condition",
        "planned_comparisons",
        "valid_comparisons",
        "invalid_comparisons",
        "divergences",
        "sequence_divergence_rate_pct",
        "wilson_95_lower_pct",
        "wilson_95_upper_pct",
        "mean_normalized_first_difference",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (output_dir / "gpu-results.csv").write_text(buffer.getvalue(), encoding="utf-8")
    (output_dir / "gpu-results.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# TraceBench GPU Divergence Results",
        "",
        "Exact token-sequence divergence for the declared experiment scope.",
        "",
        "| Mode | Condition | Valid / planned | Divergences | Rate | Wilson 95% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        rate = (
            "n/a"
            if row["sequence_divergence_rate_pct"] is None
            else f"{row['sequence_divergence_rate_pct']:.3f}%"
        )
        interval = (
            "n/a"
            if row["wilson_95_lower_pct"] is None
            else f"[{row['wilson_95_lower_pct']:.3f}%, {row['wilson_95_upper_pct']:.3f}%]"
        )
        lines.append(
            f"| {row['mode']} | {row['condition']} | "
            f"{row['valid_comparisons']} / {row['planned_comparisons']} | "
            f"{row['divergences']} | {rate} | {interval} |"
        )
    lines.extend(
        [
            "",
            "Invalid comparisons are excluded from the rate and reported separately.",
            "",
        ]
    )
    (output_dir / "gpu-results.md").write_text("\n".join(lines), encoding="utf-8")
