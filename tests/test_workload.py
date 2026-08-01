from __future__ import annotations

import math
import unittest

from tracebench.model import TraceBenchError
from tracebench.workload import (
    DriftSchedule,
    WorkloadConfig,
    generate_workload,
    snapshot_for,
)
from tests.support import make_request


class WorkloadTests(unittest.TestCase):
    def test_generation_is_deterministic_and_seed_sensitive(self) -> None:
        config = WorkloadConfig(seed=41, requests_per_regime=16)

        first = generate_workload(config, "poisson")
        second = generate_workload(config, "poisson")
        different_seed = generate_workload(
            WorkloadConfig(seed=42, requests_per_regime=16),
            "poisson",
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, different_seed)
        self.assertEqual(len(first), 16)
        self.assertTrue(
            all(
                left.arrival_s < right.arrival_s
                for left, right in zip(first, first[1:])
            )
        )

    def test_burst_regime_is_measurably_more_concentrated(self) -> None:
        config = WorkloadConfig(seed=7, requests_per_regime=512)

        poisson = generate_workload(config, "poisson")
        burst = generate_workload(config, "burst")

        self.assertLess(burst[-1].arrival_s, poisson[-1].arrival_s * 0.85)
        poisson_subsecond = sum(
            right.arrival_s - left.arrival_s < 1.0
            for left, right in zip(poisson, poisson[1:])
        )
        burst_subsecond = sum(
            right.arrival_s - left.arrival_s < 1.0
            for left, right in zip(burst, burst[1:])
        )
        self.assertGreater(burst_subsecond, poisson_subsecond)

    def test_zero_retention_excludes_old_version_at_drift_boundary(self) -> None:
        request = make_request(arrival_s=0.0)
        hashes = {artifact.kind: artifact.sha256 for artifact in request.artifacts}

        snapshot = snapshot_for(
            request,
            horizon_days=30,
            drift=DriftSchedule(),
            retention_days=0.0,
        )

        self.assertIn(hashes["model"], snapshot.available_hashes)
        self.assertNotIn(hashes["prompt"], snapshot.available_hashes)
        self.assertNotIn(hashes["config"], snapshot.available_hashes)
        self.assertNotIn(hashes["index"], snapshot.available_hashes)

    def test_retention_window_is_half_open(self) -> None:
        request = make_request(arrival_s=0.0)
        config_hash = next(
            artifact.sha256
            for artifact in request.artifacts
            if artifact.kind == "config"
        )

        during_retention = snapshot_for(
            request,
            horizon_days=30,
            drift=DriftSchedule(),
            retention_days=1.0,
        )
        after_retention = snapshot_for(
            request,
            horizon_days=31,
            drift=DriftSchedule(),
            retention_days=1.0,
        )

        self.assertIn(config_hash, during_retention.available_hashes)
        self.assertNotIn(config_hash, after_retention.available_hashes)

    def test_invalid_regime_and_nonfinite_retention_fail_closed(self) -> None:
        config = WorkloadConfig(requests_per_regime=1)
        with self.assertRaises(TraceBenchError):
            generate_workload(config, "steady")  # type: ignore[arg-type]
        with self.assertRaises(TraceBenchError):
            snapshot_for(
                make_request(),
                horizon_days=30,
                drift=DriftSchedule(),
                retention_days=math.nan,
            )

    def test_invalid_workload_parameters_are_rejected(self) -> None:
        for values in (
            {"requests_per_regime": 0},
            {"base_rate_rps": math.inf},
            {"mean_burst_seconds": math.nan},
            {"snapshot_retention_days": -1.0},
        ):
            with self.subTest(values=values), self.assertRaises(TraceBenchError):
                WorkloadConfig(**values)


if __name__ == "__main__":
    unittest.main()
