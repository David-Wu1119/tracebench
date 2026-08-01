"""Command-line entry points for CPU benchmarks and optional GPU evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tracebench.analysis import run_benchmark, write_results
from tracebench.calibration import derive_calibration, write_calibration
from tracebench.gpu import (
    analyze_gpu_evidence,
    compare_gpu_replications,
    write_gpu_analysis,
    write_gpu_replication_comparison,
)
from tracebench.workload import DriftSchedule, WorkloadConfig, generate_workload


def _config_from_mapping(payload: Mapping[str, Any]) -> WorkloadConfig:
    allowed = {
        "seed",
        "requests_per_regime",
        "base_rate_rps",
        "burst_multiplier",
        "mean_quiet_seconds",
        "mean_burst_seconds",
        "batch_window_ms",
        "median_input_tokens",
        "median_output_tokens",
        "token_sigma",
        "snapshot_retention_days",
        "drift",
    }
    unexpected = set(payload) - allowed
    if unexpected:
        raise ValueError(f"unexpected configuration fields: {sorted(unexpected)}")
    values = dict(payload)
    drift_payload = values.pop("drift", {})
    if not isinstance(drift_payload, dict):
        raise ValueError("drift must be a JSON object")
    values["drift"] = DriftSchedule(**drift_payload)
    return WorkloadConfig(**values)


def _load_config(path: Path | None) -> WorkloadConfig:
    if path is None:
        return WorkloadConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration must be a JSON object")
    return _config_from_mapping(payload)


def _request_document(request: Any, regime: str) -> dict[str, Any]:
    return {
        "schema": "tracebench/workload-request/v1",
        "workload": regime,
        "request_id": request.request_id,
        "arrival_s": round(request.arrival_s, 9),
        "input_tokens": request.input_tokens,
        "output_tokens": request.output_tokens,
        "input_sha256": request.input_sha256,
        "output_sha256": request.output_sha256,
        "artifacts": [artifact.as_dict() for artifact in request.artifacts],
        "controls": request.controls_dict,
    }


def _write_workload(
    output: Path,
    *,
    config: WorkloadConfig,
    regime: str,
    force: bool,
) -> None:
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    requests = generate_workload(config, regime)
    text = "".join(
        json.dumps(
            _request_document(request, regime),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for request in requests
    )
    output.write_text(text, encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tracebench",
        description="Measure structural replay sufficiency under evidence policies.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo", help="Run the deterministic CPU demonstration."
    )
    demo.add_argument("--output", type=Path, default=Path("results/demo"))
    demo.add_argument("--config", type=Path, default=Path("configs/demo.json"))
    demo.add_argument("--horizons", type=int, nargs="+", default=[30, 90, 365])

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run the structural benchmark with an explicit configuration.",
    )
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--horizons", type=int, nargs="+", default=[30, 90, 365])

    generate = subparsers.add_parser(
        "generate",
        help="Generate deterministic prompt-free workload JSONL.",
    )
    generate.add_argument("--config", type=Path, default=Path("configs/demo.json"))
    generate.add_argument("--regime", choices=("poisson", "burst"), required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--force", action="store_true")

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Derive demo parameters from downloaded public traces.",
    )
    calibrate.add_argument("--burstgpt", type=Path, required=True)
    calibrate.add_argument("--azure-code", type=Path, required=True)
    calibrate.add_argument("--azure-conversation", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)

    gpu_run = subparsers.add_parser(
        "gpu-run",
        help="Run the registered optional GPU batching experiment.",
    )
    gpu_run.add_argument("--plan", type=Path, required=True)
    gpu_run.add_argument("--model-path", type=Path, required=True)
    gpu_run.add_argument("--output", type=Path, required=True)
    gpu_run.add_argument("--implementation-commit", required=True)

    gpu_analyze = subparsers.add_parser(
        "gpu-analyze",
        help="Verify and analyze an exported GPU evidence directory.",
    )
    gpu_analyze.add_argument("--evidence", type=Path, required=True)
    gpu_analyze.add_argument("--output", type=Path, required=True)

    gpu_compare = subparsers.add_parser(
        "gpu-compare",
        help="Verify and compare two complete GPU evidence directories.",
    )
    gpu_compare.add_argument("--evidence", type=Path, nargs=2, required=True)
    gpu_compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "calibrate":
        calibration = derive_calibration(
            burstgpt_path=args.burstgpt,
            azure_code_path=args.azure_code,
            azure_conversation_path=args.azure_conversation,
        )
        write_calibration(args.output, calibration)
        print(f"wrote public-trace calibration to {args.output}")
        return
    if args.command == "gpu-run":
        from tracebench.gpu_runner import run_gpu_experiment

        output = run_gpu_experiment(
            plan_path=args.plan,
            model_path=args.model_path,
            output_dir=args.output,
            implementation_commit=args.implementation_commit,
        )
        print(f"wrote GPU evidence to {output}")
        return
    if args.command == "gpu-analyze":
        analysis = analyze_gpu_evidence(args.evidence)
        write_gpu_analysis(args.output, analysis)
        print(f"wrote GPU analysis to {args.output}")
        return
    if args.command == "gpu-compare":
        comparison = compare_gpu_replications(*args.evidence)
        write_gpu_replication_comparison(args.output, comparison)
        print(f"wrote GPU replication comparison to {args.output}")
        return
    config = _load_config(args.config)
    if args.command == "generate":
        _write_workload(
            args.output,
            config=config,
            regime=args.regime,
            force=args.force,
        )
        print(f"wrote {args.regime} workload to {args.output}")
        return
    rows = run_benchmark(config, horizons=args.horizons)
    manifest = write_results(
        args.output,
        rows=rows,
        config=config,
        horizons=args.horizons,
    )
    print(f"wrote {manifest['result_rows']} result rows to {args.output}")


if __name__ == "__main__":
    main()
