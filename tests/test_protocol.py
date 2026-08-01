from __future__ import annotations

import math
import unittest

from tracebench.model import TraceBenchError, sha256_text
from tracebench.protocol import RunnerProtocol, TerminalReport


def make_report(
    lease, *, report_id: str = "report-1", status: str = "completed"
) -> TerminalReport:
    return TerminalReport(
        report_id=report_id,
        run_id=lease.run_id,
        owner=lease.owner,
        lease_generation=lease.generation,
        lease_token=lease.token,
        status=status,
        evidence_sha256=sha256_text("fixture:evidence"),
        payload={"adapter_sha256": sha256_text("fixture:adapter")},
    )


class RunnerProtocolTests(unittest.TestCase):
    def test_live_lease_blocks_second_claim(self) -> None:
        protocol = RunnerProtocol()
        first = protocol.claim("run-1", "runner-a", now_s=0.0, ttl_s=10.0)

        with self.assertRaisesRegex(TraceBenchError, "live lease"):
            protocol.claim("run-1", "runner-b", now_s=9.999, ttl_s=10.0)
        self.assertEqual(first.generation, 1)

    def test_expiry_reclaim_increments_generation_and_fences_old_holder(self) -> None:
        protocol = RunnerProtocol()
        stale = protocol.claim("run-1", "runner-a", now_s=0.0, ttl_s=10.0)
        current = protocol.claim("run-1", "runner-b", now_s=10.0, ttl_s=10.0)

        self.assertEqual(current.generation, 2)
        with self.assertRaisesRegex(TraceBenchError, "stale or foreign"):
            protocol.heartbeat(stale, now_s=10.0, ttl_s=10.0)
        with self.assertRaisesRegex(TraceBenchError, "stale or foreign"):
            protocol.submit(make_report(stale), now_s=10.0)

    def test_heartbeat_renews_same_generation_and_token(self) -> None:
        protocol = RunnerProtocol()
        lease = protocol.claim("run-1", "runner-a", now_s=0.0, ttl_s=10.0)

        renewed = protocol.heartbeat(lease, now_s=5.0, ttl_s=20.0)

        self.assertEqual(renewed.generation, lease.generation)
        self.assertEqual(renewed.token, lease.token)
        self.assertEqual(renewed.expires_at_s, 25.0)

    def test_terminal_report_is_idempotent_only_for_identical_bytes(self) -> None:
        protocol = RunnerProtocol()
        lease = protocol.claim("run-1", "runner-a", now_s=0.0, ttl_s=10.0)
        report = make_report(lease)

        self.assertEqual(protocol.submit(report, now_s=1.0), "accepted")
        self.assertEqual(protocol.submit(report, now_s=9.0), "idempotent")
        changed = TerminalReport(
            report_id=report.report_id,
            run_id=report.run_id,
            owner=report.owner,
            lease_generation=report.lease_generation,
            lease_token=report.lease_token,
            status="failed",
            evidence_sha256=report.evidence_sha256,
            payload=report.payload,
        )
        with self.assertRaisesRegex(TraceBenchError, "reused with different bytes"):
            protocol.submit(changed, now_s=2.0)

    def test_expired_or_malformed_terminal_report_is_rejected(self) -> None:
        protocol = RunnerProtocol()
        lease = protocol.claim("run-1", "runner-a", now_s=0.0, ttl_s=10.0)

        with self.assertRaisesRegex(TraceBenchError, "after lease expiry"):
            protocol.submit(make_report(lease), now_s=10.0)

        protocol = RunnerProtocol()
        lease = protocol.claim("run-2", "runner-a", now_s=0.0, ttl_s=10.0)
        malformed = TerminalReport(
            report_id="report-bad",
            run_id=lease.run_id,
            owner=lease.owner,
            lease_generation=lease.generation,
            lease_token=lease.token,
            status="completed",
            evidence_sha256="not-a-digest",
            payload={},
        )
        with self.assertRaisesRegex(TraceBenchError, "evidence_sha256"):
            protocol.submit(malformed, now_s=1.0)

    def test_nonfinite_clock_and_ttl_values_are_rejected(self) -> None:
        for now_s, ttl_s in (
            (math.nan, 10.0),
            (0.0, math.inf),
            (math.inf, 10.0),
        ):
            with self.subTest(now_s=now_s, ttl_s=ttl_s):
                with self.assertRaises(TraceBenchError):
                    RunnerProtocol().claim(
                        "run-1", "runner-a", now_s=now_s, ttl_s=ttl_s
                    )


if __name__ == "__main__":
    unittest.main()
