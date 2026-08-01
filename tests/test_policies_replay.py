from __future__ import annotations

import unittest

from tracebench.model import ARTIFACT_KINDS, CONTROL_FIELDS, TraceBenchError
from tracebench.policies import POLICIES, policy_by_name
from tracebench.replay import evaluate_replay
from tracebench.workload import (
    ARTIFACT_SIZES,
    WorkloadConfig,
    generate_workload,
    snapshot_for,
)


class PolicyReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = WorkloadConfig(seed=9, requests_per_regime=4)
        self.requests = generate_workload(self.config, "poisson")

    def outcome(
        self, policy_name: str, request_index: int = 0, horizon_days: int = 365
    ):
        policy = policy_by_name(policy_name)
        request = self.requests[request_index]
        evidence = policy.capture(self.requests)
        snapshot = snapshot_for(
            request,
            horizon_days=horizon_days,
            drift=self.config.drift,
            retention_days=self.config.snapshot_retention_days,
        )
        return evaluate_replay(request, evidence=evidence, snapshot=snapshot)

    def test_benchmark_declares_exactly_five_distinct_profiles(self) -> None:
        self.assertEqual(
            [policy.name for policy in POLICIES],
            [
                "input-output-only",
                "mlflow-reference",
                "wandb-reference",
                "full-artifact-dedup",
                "capsule",
            ],
        )

    def test_capsule_is_structurally_sufficient_at_every_horizon(self) -> None:
        for horizon in (30, 90, 365):
            with self.subTest(horizon=horizon):
                outcome = self.outcome("capsule", horizon_days=horizon)
                self.assertTrue(outcome.replay_sufficient)
                self.assertEqual(outcome.missing_artifact_hashes, ())
                self.assertEqual(outcome.missing_controls, ())
                self.assertEqual(outcome.missing_payloads, ())

    def test_full_artifact_profile_still_fails_without_runtime_controls(self) -> None:
        outcome = self.outcome("full-artifact-dedup")

        self.assertTrue(outcome.artifact_complete)
        self.assertTrue(outcome.payload_complete)
        self.assertFalse(outcome.control_complete)
        self.assertEqual(
            outcome.missing_controls,
            ("batch_fingerprint", "runtime_digest"),
        )
        self.assertFalse(outcome.replay_sufficient)

    def test_input_output_only_has_comparison_payload_but_no_dependencies(self) -> None:
        outcome = self.outcome("input-output-only")

        self.assertFalse(outcome.artifact_complete)
        self.assertFalse(outcome.control_complete)
        self.assertTrue(outcome.payload_complete)
        self.assertEqual(len(outcome.missing_artifact_hashes), len(ARTIFACT_KINDS))
        self.assertEqual(outcome.missing_controls, CONTROL_FIELDS)

    def test_mutable_references_fail_after_artifact_drift(self) -> None:
        mlflow = self.outcome("mlflow-reference", horizon_days=365)
        wandb = self.outcome("wandb-reference", horizon_days=365)

        self.assertFalse(mlflow.artifact_complete)
        self.assertFalse(wandb.artifact_complete)
        self.assertFalse(mlflow.payload_complete)
        self.assertFalse(wandb.payload_complete)

    def test_content_addressed_blobs_are_deduplicated_across_requests(self) -> None:
        evidence = policy_by_name("capsule").capture(self.requests)

        self.assertEqual(evidence.blob_bytes, sum(ARTIFACT_SIZES.values()))
        self.assertEqual(len(evidence.blobs), len(ARTIFACT_KINDS))
        self.assertLess(
            evidence.blob_bytes, sum(ARTIFACT_SIZES.values()) * len(self.requests)
        )

    def test_unknown_policy_is_rejected(self) -> None:
        with self.assertRaises(TraceBenchError):
            policy_by_name("magic-logger")


if __name__ == "__main__":
    unittest.main()
