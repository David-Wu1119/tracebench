"""Benchmark orchestration and deterministic result rendering."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable

from tracebench.model import ARTIFACT_KINDS, CONTROL_FIELDS
from tracebench.policies import POLICIES
from tracebench.replay import evaluate_replay
from tracebench.workload import WorkloadConfig, generate_workload, snapshot_for


RESULT_FIELDS = (
    "workload",
    "policy",
    "horizon_days",
    "requests",
    "artifact_coverage_pct",
    "artifact_complete_pct",
    "control_coverage_pct",
    "control_complete_pct",
    "payload_coverage_pct",
    "payload_complete_pct",
    "replay_sufficient_pct",
    "evidence_bytes",
    "bytes_per_request",
    "unique_blob_bytes",
    "metadata_bytes",
    "payload_bytes",
)


def _percentage(matches: int, total: int) -> float:
    return round((matches / total) * 100.0, 3)


def _normalize_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    values = tuple(horizons)
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("horizons must contain positive integers")
    return tuple(sorted(set(values)))


def _workload_summary(config: WorkloadConfig, workload: str) -> dict[str, Any]:
    requests = generate_workload(config, workload)  # type: ignore[arg-type]
    arrivals = [request.arrival_s for request in requests]
    intervals = [right - left for left, right in zip(arrivals, arrivals[1:])]
    ordered_intervals = sorted(intervals)
    p95_index = max(0, math.ceil(0.95 * len(ordered_intervals)) - 1)

    left = 0
    peak_one_second = 0
    for right, arrival in enumerate(arrivals):
        while arrival - arrivals[left] >= 1.0:
            left += 1
        peak_one_second = max(peak_one_second, right - left + 1)

    return {
        "requests": len(requests),
        "elapsed_s": round(arrivals[-1], 6),
        "mean_interarrival_s": (
            round(sum(intervals) / len(intervals), 6) if intervals else None
        ),
        "p95_interarrival_s": (
            round(ordered_intervals[p95_index], 6) if ordered_intervals else None
        ),
        "subsecond_intervals": sum(interval < 1.0 for interval in intervals),
        "max_requests_per_1s_window": peak_one_second,
    }


def run_benchmark(
    config: WorkloadConfig,
    *,
    horizons: Iterable[int] = (30, 90, 365),
) -> list[dict[str, Any]]:
    normalized_horizons = _normalize_horizons(horizons)

    rows: list[dict[str, Any]] = []
    for workload_name in ("poisson", "burst"):
        requests = generate_workload(config, workload_name)
        for policy in POLICIES:
            evidence = policy.capture(requests)
            for horizon_days in normalized_horizons:
                outcomes = [
                    evaluate_replay(
                        request,
                        evidence=evidence,
                        snapshot=snapshot_for(
                            request,
                            horizon_days=horizon_days,
                            drift=config.drift,
                            retention_days=config.snapshot_retention_days,
                        ),
                    )
                    for request in requests
                ]
                total = len(outcomes)
                rows.append(
                    {
                        "workload": workload_name,
                        "policy": policy.name,
                        "horizon_days": horizon_days,
                        "requests": total,
                        "artifact_coverage_pct": _percentage(
                            sum(
                                len(ARTIFACT_KINDS)
                                - len(outcome.missing_artifact_hashes)
                                for outcome in outcomes
                            ),
                            total * len(ARTIFACT_KINDS),
                        ),
                        "artifact_complete_pct": _percentage(
                            sum(outcome.artifact_complete for outcome in outcomes),
                            total,
                        ),
                        "control_coverage_pct": _percentage(
                            sum(
                                len(CONTROL_FIELDS) - len(outcome.missing_controls)
                                for outcome in outcomes
                            ),
                            total * len(CONTROL_FIELDS),
                        ),
                        "control_complete_pct": _percentage(
                            sum(outcome.control_complete for outcome in outcomes),
                            total,
                        ),
                        "payload_coverage_pct": _percentage(
                            sum(
                                2 - len(outcome.missing_payloads)
                                for outcome in outcomes
                            ),
                            total * 2,
                        ),
                        "payload_complete_pct": _percentage(
                            sum(outcome.payload_complete for outcome in outcomes),
                            total,
                        ),
                        "replay_sufficient_pct": _percentage(
                            sum(outcome.replay_sufficient for outcome in outcomes),
                            total,
                        ),
                        "evidence_bytes": evidence.total_bytes,
                        "bytes_per_request": round(evidence.total_bytes / total, 3),
                        "unique_blob_bytes": evidence.blob_bytes,
                        "metadata_bytes": evidence.metadata_bytes,
                        "payload_bytes": evidence.payload_bytes,
                    }
                )
    return rows


def _csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=RESULT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _format_bytes(value: int | float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _markdown_text(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# TraceBench CPU Demo Results",
        "",
        (
            "These are deterministic structural sufficiency results over virtual artifact "
            "sizes. They are not GPU decision-divergence measurements."
        ),
        "",
        "| Workload | Policy | Horizon | Artifact coverage | Control coverage | Payload coverage | Replay sufficient | Evidence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {workload} | {policy} | {horizon_days}d | "
            "{artifact_coverage_pct:.3f}% | {control_coverage_pct:.3f}% | "
            "{payload_coverage_pct:.3f}% | {replay_sufficient_pct:.3f}% | "
            "{evidence} |".format(
                **row,
                evidence=_format_bytes(row["evidence_bytes"]),
            )
        )
    lines.extend(
        [
            "",
            "See `results.csv` for exact byte accounting and `manifest.json` for the configuration.",
            "",
        ]
    )
    return "\n".join(lines)


def write_results(
    output_dir: Path,
    *,
    rows: list[dict[str, Any]],
    config: WorkloadConfig,
    horizons: Iterable[int],
) -> dict[str, Any]:
    normalized_horizons = _normalize_horizons(horizons)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_text = _csv_text(rows)
    markdown_text = _markdown_text(rows)
    csv_path = output_dir / "results.csv"
    markdown_path = output_dir / "results.md"
    csv_path.write_text(csv_text, encoding="utf-8")
    markdown_path.write_text(markdown_text, encoding="utf-8")
    manifest = {
        "schema": "tracebench/demo-manifest/v1",
        "benchmark_version": "0.1.0a0",
        "configuration": config.as_dict(),
        "horizons_days": list(normalized_horizons),
        "policies": [policy.name for policy in POLICIES],
        "workload_summaries": {
            workload: _workload_summary(config, workload)
            for workload in ("poisson", "burst")
        },
        "result_rows": len(rows),
        "results_csv_sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
        "results_markdown_sha256": hashlib.sha256(
            markdown_text.encode("utf-8")
        ).hexdigest(),
        "interpretation": (
            "Structural replay eligibility over virtual artifact sizes; no model was executed."
        ),
        "measurement_scope": {
            "artifact_sizes": "virtual",
            "decision_divergence": "not_measured",
            "model_execution": "none",
            "replay_result": "structural_eligibility",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
