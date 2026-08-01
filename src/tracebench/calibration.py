"""Reproducible calibration from public BurstGPT and Azure LLM traces."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tracebench.model import TraceBenchError


BURSTGPT_URL = (
    "https://raw.githubusercontent.com/HPMLL/BurstGPT/"
    "8345c824bf744e21692186af2835521ba75e5f6d/data/BurstGPT_1.csv"
)
AZURE_CODE_URL = (
    "https://raw.githubusercontent.com/Azure/AzurePublicDataset/"
    "790921015d50dd6aae7f7e47f39ba0e235ad6b08/data/AzureLLMInferenceTrace_code.csv"
)
AZURE_CONVERSATION_URL = (
    "https://raw.githubusercontent.com/Azure/AzurePublicDataset/"
    "790921015d50dd6aae7f7e47f39ba0e235ad6b08/data/AzureLLMInferenceTrace_conv.csv"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _nearest_rank(values: list[float] | list[int], quantile: float) -> float | int:
    if not values:
        raise TraceBenchError("cannot calculate a percentile of an empty series")
    if quantile <= 0 or quantile > 1:
        raise TraceBenchError("quantile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _token_stats(values: list[int]) -> dict[str, Any]:
    if not values or any(value < 0 for value in values):
        raise TraceBenchError("token series must be nonempty and non-negative")
    positive = [value for value in values if value > 0]
    return {
        "p50": int(_nearest_rank(values, 0.50)),
        "p75": int(_nearest_rank(values, 0.75)),
        "p90": int(_nearest_rank(values, 0.90)),
        "p95": int(_nearest_rank(values, 0.95)),
        "positive_p50": int(_nearest_rank(positive, 0.50)),
        "zero_count": len(values) - len(positive),
        "positive_log_sigma": round(
            statistics.pstdev(math.log(value) for value in positive),
            6,
        ),
    }


def _bin_stats(
    timestamps: list[float], width_s: int
) -> tuple[dict[str, Any], list[int]]:
    start = min(timestamps)
    end = max(timestamps)
    counts = Counter(int((timestamp - start) // width_s) for timestamp in timestamps)
    number_of_bins = int((end - start) // width_s) + 1
    all_counts = [counts.get(index, 0) for index in range(number_of_bins)]
    mean_count = len(timestamps) / number_of_bins
    return (
        {
            "width_s": width_s,
            "bins": number_of_bins,
            "mean_count": round(mean_count, 6),
            "p50_count": int(_nearest_rank(all_counts, 0.50)),
            "p95_count": int(_nearest_rank(all_counts, 0.95)),
            "p99_count": int(_nearest_rank(all_counts, 0.99)),
            "max_count": max(all_counts),
        },
        all_counts,
    )


def _run_durations(
    counts: list[int],
    *,
    threshold: int,
    width_s: int,
) -> dict[str, int]:
    runs: list[tuple[bool, int]] = []
    current_state: bool | None = None
    current_bins = 0
    for count in counts:
        state = count >= threshold
        if state == current_state:
            current_bins += 1
            continue
        if current_state is not None:
            runs.append((current_state, current_bins * width_s))
        current_state = state
        current_bins = 1
    if current_state is not None:
        runs.append((current_state, current_bins * width_s))

    high = [duration for state, duration in runs if state]
    quiet = [duration for state, duration in runs if not state]
    if not high or not quiet:
        raise TraceBenchError(
            "calibration trace must contain both high and quiet states"
        )
    return {
        "threshold_count": threshold,
        "high_runs": len(high),
        "high_p50_s": int(_nearest_rank(high, 0.50)),
        "high_p75_s": int(_nearest_rank(high, 0.75)),
        "quiet_runs": len(quiet),
        "quiet_p50_s": int(_nearest_rank(quiet, 0.50)),
        "quiet_p75_s": int(_nearest_rank(quiet, 0.75)),
    }


def _read_trace(
    path: Path,
    *,
    timestamp_field: str,
    input_field: str,
    output_field: str,
    parse_timestamp: Callable[[str], float],
) -> tuple[list[float], list[int], list[int]]:
    timestamps: list[float] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            required = {timestamp_field, input_field, output_field}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise TraceBenchError(
                    f"{path} is missing required columns {sorted(required)}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    timestamps.append(parse_timestamp(row[timestamp_field]))
                    input_tokens.append(int(row[input_field]))
                    output_tokens.append(int(row[output_field]))
                except (TypeError, ValueError) as exc:
                    raise TraceBenchError(
                        f"{path}:{line_number} contains an invalid trace value"
                    ) from exc
    except OSError as exc:
        raise TraceBenchError(f"cannot read calibration trace {path}: {exc}") from exc
    if len(timestamps) < 2:
        raise TraceBenchError(f"{path} must contain at least two requests")
    if any(not math.isfinite(timestamp) for timestamp in timestamps):
        raise TraceBenchError(f"{path} contains a non-finite timestamp")
    if min(timestamps) == max(timestamps):
        raise TraceBenchError(f"{path} must span a nonzero duration")
    return timestamps, input_tokens, output_tokens


def _source_summary(
    path: Path,
    *,
    name: str,
    url: str,
    timestamp_field: str,
    input_field: str,
    output_field: str,
    parse_timestamp: Callable[[str], float],
) -> tuple[dict[str, Any], list[int]]:
    source_sha256 = _sha256_file(path)
    timestamps, input_tokens, output_tokens = _read_trace(
        path,
        timestamp_field=timestamp_field,
        input_field=input_field,
        output_field=output_field,
        parse_timestamp=parse_timestamp,
    )
    if _sha256_file(path) != source_sha256:
        raise TraceBenchError(f"calibration trace changed while being read: {path}")
    duration_s = max(timestamps) - min(timestamps)
    bins, counts = _bin_stats(timestamps, 10)
    return (
        {
            "name": name,
            "url": url,
            "sha256": source_sha256,
            "rows": len(timestamps),
            "duration_s": round(duration_s, 6),
            "mean_rps": round(len(timestamps) / duration_s, 6),
            "input_tokens": _token_stats(input_tokens),
            "output_tokens": _token_stats(output_tokens),
            "ten_second_bins": bins,
        },
        counts,
    )


def _nearest_power_of_two(value: float) -> int:
    return 2 ** round(math.log2(value))


def _parse_azure_timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.timestamp()


def derive_calibration(
    *,
    burstgpt_path: Path,
    azure_code_path: Path,
    azure_conversation_path: Path,
) -> dict[str, Any]:
    burst, burst_counts = _source_summary(
        burstgpt_path,
        name="BurstGPT_1",
        url=BURSTGPT_URL,
        timestamp_field="Timestamp",
        input_field="Request tokens",
        output_field="Response tokens",
        parse_timestamp=float,
    )
    azure_code, _ = _source_summary(
        azure_code_path,
        name="AzureLLMInferenceTrace_code",
        url=AZURE_CODE_URL,
        timestamp_field="TIMESTAMP",
        input_field="ContextTokens",
        output_field="GeneratedTokens",
        parse_timestamp=_parse_azure_timestamp,
    )
    azure_conversation, _ = _source_summary(
        azure_conversation_path,
        name="AzureLLMInferenceTrace_conv",
        url=AZURE_CONVERSATION_URL,
        timestamp_field="TIMESTAMP",
        input_field="ContextTokens",
        output_field="GeneratedTokens",
        parse_timestamp=_parse_azure_timestamp,
    )

    ten_second = burst["ten_second_bins"]
    state_runs = _run_durations(
        burst_counts,
        threshold=ten_second["p95_count"],
        width_s=ten_second["width_s"],
    )
    lower_multiplier = ten_second["p95_count"] / ten_second["mean_count"]
    upper_multiplier = ten_second["p99_count"] / ten_second["mean_count"]
    burst_multiplier = round(math.sqrt(lower_multiplier * upper_multiplier))

    sources = [burst, azure_code, azure_conversation]
    input_center = math.prod(
        source["input_tokens"]["positive_p50"] for source in sources
    ) ** (1 / len(sources))
    output_center = math.prod(
        source["output_tokens"]["positive_p50"] for source in sources
    ) ** (1 / len(sources))
    input_sigma = statistics.median(
        source["input_tokens"]["positive_log_sigma"] for source in sources
    )
    output_sigma = statistics.median(
        source["output_tokens"]["positive_log_sigma"] for source in sources
    )
    shared_sigma = round(((input_sigma + output_sigma) / 2) * 4) / 4

    return {
        "schema": "tracebench/public-trace-calibration/v1",
        "sources": sources,
        "burst_state": {
            "definition": "BurstGPT 10-second bin count at or above its p95 count",
            **state_runs,
            "p95_to_mean_rate_ratio": round(lower_multiplier, 6),
            "p99_to_mean_rate_ratio": round(upper_multiplier, 6),
        },
        "derived_demo_parameters": {
            "base_rate_rps": max(0.1, math.floor(burst["mean_rps"] * 10) / 10),
            "burst_multiplier": float(burst_multiplier),
            "mean_quiet_seconds": float(state_runs["quiet_p75_s"]),
            "mean_burst_seconds": float(state_runs["high_p50_s"]),
            "median_input_tokens": _nearest_power_of_two(input_center),
            "median_output_tokens": _nearest_power_of_two(output_center),
            "token_sigma": shared_sigma,
        },
        "derivation": {
            "base_rate_rps": "BurstGPT mean RPS rounded down to one decimal; BurstGPT permits RPS scaling.",
            "burst_multiplier": "Rounded geometric center of BurstGPT 10-second p95/mean and p99/mean ratios.",
            "state_durations": "Burst duration is high-state p50; quiet duration is quiet-state p75.",
            "token_medians": "Nearest powers of two to geometric centers of the three source medians.",
            "token_sigma": "Nearest 0.25 to the mean of median input and output log-space sigmas.",
        },
        "not_calibrated": [
            "artifact drift cadences",
            "artifact sizes",
            "batch window",
            "audit horizons",
        ],
    }


def write_calibration(path: Path, calibration: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
