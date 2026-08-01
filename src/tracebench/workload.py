"""Deterministic Poisson and burst-regime workload generation."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Literal

from tracebench.model import (
    ARTIFACT_KINDS,
    ArtifactVersion,
    Snapshot,
    TraceBenchError,
    WorkloadRequest,
    sha256_text,
)


SECONDS_PER_DAY = 86_400.0
Regime = Literal["poisson", "burst"]


@dataclass(frozen=True, slots=True)
class DriftSchedule:
    model_days: float = 90.0
    prompt_days: float = 7.0
    config_days: float = 30.0
    index_days: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0 or not math.isfinite(value):
                raise TraceBenchError(f"{name} must be finite and positive")

    def cadence_seconds(self, kind: str) -> float:
        if kind not in ARTIFACT_KINDS:
            raise TraceBenchError(f"unsupported artifact kind: {kind}")
        return float(getattr(self, f"{kind}_days")) * SECONDS_PER_DAY


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    seed: int = 20260801
    requests_per_regime: int = 96
    base_rate_rps: float = 0.20
    burst_multiplier: float = 8.0
    mean_quiet_seconds: float = 80.0
    mean_burst_seconds: float = 10.0
    batch_window_ms: int = 100
    median_input_tokens: int = 1024
    median_output_tokens: int = 32
    token_sigma: float = 1.0
    snapshot_retention_days: float = 0.0
    drift: DriftSchedule = DriftSchedule()

    def __post_init__(self) -> None:
        if self.requests_per_regime <= 0:
            raise TraceBenchError("requests_per_regime must be positive")
        for name in (
            "base_rate_rps",
            "burst_multiplier",
            "mean_quiet_seconds",
            "mean_burst_seconds",
            "token_sigma",
        ):
            value = float(getattr(self, name))
            if value <= 0 or not math.isfinite(value):
                raise TraceBenchError(f"{name} must be finite and positive")
        if self.batch_window_ms <= 0:
            raise TraceBenchError("batch_window_ms must be positive")
        if self.median_input_tokens <= 0 or self.median_output_tokens <= 0:
            raise TraceBenchError("median token counts must be positive")
        if self.snapshot_retention_days < 0 or not math.isfinite(
            self.snapshot_retention_days
        ):
            raise TraceBenchError(
                "snapshot_retention_days must be finite and non-negative"
            )

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["drift"] = asdict(self.drift)
        return payload


ARTIFACT_SIZES = {
    "model": 1_250_000_000,
    "prompt": 16_384,
    "config": 8_192,
    "index": 250_000_000,
}


def _sample_tokens(random_source: random.Random, median: int, sigma: float) -> int:
    value = random_source.lognormvariate(math.log(median), sigma)
    return max(1, min(int(round(value)), 32_768))


def _poisson_arrivals(
    random_source: random.Random,
    *,
    count: int,
    rate_rps: float,
) -> list[float]:
    now = 0.0
    arrivals: list[float] = []
    for _ in range(count):
        now += random_source.expovariate(rate_rps)
        arrivals.append(now)
    return arrivals


def _burst_arrivals(
    random_source: random.Random,
    *,
    count: int,
    base_rate_rps: float,
    burst_multiplier: float,
    mean_quiet_seconds: float,
    mean_burst_seconds: float,
) -> list[float]:
    """Sample a two-state Markov-modulated Poisson process."""

    now = 0.0
    in_burst = False
    state_ends = random_source.expovariate(1.0 / mean_quiet_seconds)
    arrivals: list[float] = []
    while len(arrivals) < count:
        rate = base_rate_rps * (burst_multiplier if in_burst else 1.0)
        candidate = now + random_source.expovariate(rate)
        if candidate < state_ends:
            now = candidate
            arrivals.append(now)
            continue
        now = state_ends
        in_burst = not in_burst
        mean_duration = mean_burst_seconds if in_burst else mean_quiet_seconds
        state_ends = now + random_source.expovariate(1.0 / mean_duration)
    return arrivals


def _artifact_for(
    *,
    kind: str,
    arrival_s: float,
    drift: DriftSchedule,
    seed: int,
) -> ArtifactVersion:
    cadence = drift.cadence_seconds(kind)
    version = int(arrival_s // cadence)
    digest = sha256_text(f"tracebench:artifact:{seed}:{kind}:v{version}")
    return ArtifactVersion(
        kind=kind,
        version=version,
        sha256=digest,
        size_bytes=ARTIFACT_SIZES[kind],
        mutable_uri=f"artifact://{kind}/current",
    )


def generate_workload(
    config: WorkloadConfig, regime: Regime
) -> tuple[WorkloadRequest, ...]:
    if regime not in {"poisson", "burst"}:
        raise TraceBenchError(f"unsupported arrival regime: {regime}")
    regime_offset = 0 if regime == "poisson" else 1_000_003
    random_source = random.Random(config.seed + regime_offset)
    if regime == "poisson":
        arrivals = _poisson_arrivals(
            random_source,
            count=config.requests_per_regime,
            rate_rps=config.base_rate_rps,
        )
    else:
        arrivals = _burst_arrivals(
            random_source,
            count=config.requests_per_regime,
            base_rate_rps=config.base_rate_rps,
            burst_multiplier=config.burst_multiplier,
            mean_quiet_seconds=config.mean_quiet_seconds,
            mean_burst_seconds=config.mean_burst_seconds,
        )

    runtime_digest = sha256_text("tracebench:runtime:cpu-surrogate:v1")
    requests: list[WorkloadRequest] = []
    for index, arrival_s in enumerate(arrivals):
        request_id = f"{regime}-{index:06d}"
        input_tokens = _sample_tokens(
            random_source,
            config.median_input_tokens,
            config.token_sigma,
        )
        output_tokens = _sample_tokens(
            random_source,
            config.median_output_tokens,
            config.token_sigma,
        )
        artifacts = tuple(
            _artifact_for(
                kind=kind,
                arrival_s=arrival_s,
                drift=config.drift,
                seed=config.seed,
            )
            for kind in ARTIFACT_KINDS
        )
        batch_id = int((arrival_s * 1000.0) // config.batch_window_ms)
        controls = tuple(
            sorted(
                {
                    "random_seed": str(random_source.randrange(0, 2**31)),
                    "batch_fingerprint": sha256_text(
                        f"tracebench:batch:{regime}:{batch_id}"
                    ),
                    "runtime_digest": runtime_digest,
                }.items()
            )
        )
        artifact_fingerprint = ":".join(artifact.sha256 for artifact in artifacts)
        control_fingerprint = ":".join(value for _, value in controls)
        input_sha256 = sha256_text(
            f"tracebench:input:{config.seed}:{regime}:{index}:{input_tokens}"
        )
        output_sha256 = sha256_text(
            f"tracebench:output:{input_sha256}:{artifact_fingerprint}:"
            f"{control_fingerprint}:{output_tokens}"
        )
        requests.append(
            WorkloadRequest(
                request_id=request_id,
                arrival_s=arrival_s,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_sha256=input_sha256,
                output_sha256=output_sha256,
                artifacts=artifacts,
                controls=controls,
            )
        )
    return tuple(requests)


def snapshot_for(
    request: WorkloadRequest,
    *,
    horizon_days: int,
    drift: DriftSchedule,
    retention_days: float,
) -> Snapshot:
    if horizon_days <= 0:
        raise TraceBenchError("horizon_days must be positive")
    if retention_days < 0 or not math.isfinite(retention_days):
        raise TraceBenchError("retention_days must be finite and non-negative")
    audit_s = request.arrival_s + (horizon_days * SECONDS_PER_DAY)
    retention_s = retention_days * SECONDS_PER_DAY
    available: set[str] = set()
    for artifact in request.artifacts:
        cadence = drift.cadence_seconds(artifact.kind)
        current_version = int(audit_s // cadence)
        if current_version == artifact.version:
            available.add(artifact.sha256)
            continue
        version_expired_s = (artifact.version + 1) * cadence
        # Retention is a half-open window: a zero-day policy preserves nothing,
        # and an N-day policy stops serving the old version exactly N days later.
        if audit_s - version_expired_s < retention_s:
            available.add(artifact.sha256)
    return Snapshot(
        request_id=request.request_id,
        horizon_days=horizon_days,
        available_hashes=frozenset(available),
    )
