"""Small generation-fenced runner protocol extracted from validated semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from tracebench.model import (
    TraceBenchError,
    assert_sha256,
    canonical_json_bytes,
    sha256_text,
)


TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


@dataclass(frozen=True, slots=True)
class Lease:
    run_id: str
    owner: str
    generation: int
    token: str
    expires_at_s: float


@dataclass(frozen=True, slots=True)
class TerminalReport:
    report_id: str
    run_id: str
    owner: str
    lease_generation: int
    lease_token: str
    status: str
    evidence_sha256: str
    payload: Mapping[str, Any]

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "schema": "tracebench/terminal-report/v1",
                "report_id": self.report_id,
                "run_id": self.run_id,
                "owner": self.owner,
                "lease_generation": self.lease_generation,
                "lease_token": self.lease_token,
                "status": self.status,
                "evidence_sha256": self.evidence_sha256,
                "payload": dict(self.payload),
            }
        )


class RunnerProtocol:
    """In-memory reference semantics for claim, reclaim, and terminal replay."""

    def __init__(self) -> None:
        self._leases: dict[str, Lease] = {}
        self._next_generation: dict[str, int] = {}
        self._reports: dict[str, bytes] = {}
        self._terminal_report_by_run: dict[str, str] = {}

    def claim(self, run_id: str, owner: str, *, now_s: float, ttl_s: float) -> Lease:
        if not run_id or not owner:
            raise TraceBenchError("run_id and owner are required")
        self._validate_clock(now_s=now_s, ttl_s=ttl_s)
        current = self._leases.get(run_id)
        if current is not None and current.expires_at_s > now_s:
            raise TraceBenchError("run already has a live lease")
        if run_id in self._terminal_report_by_run:
            raise TraceBenchError("terminal run cannot be claimed again")
        generation = self._next_generation.get(run_id, 0) + 1
        self._next_generation[run_id] = generation
        token = sha256_text(
            f"tracebench:lease:{run_id}:{owner}:{generation}:{now_s:.9f}"
        )
        lease = Lease(
            run_id=run_id,
            owner=owner,
            generation=generation,
            token=token,
            expires_at_s=now_s + ttl_s,
        )
        self._leases[run_id] = lease
        return lease

    def heartbeat(self, lease: Lease, *, now_s: float, ttl_s: float) -> Lease:
        self._validate_clock(now_s=now_s, ttl_s=ttl_s)
        self._require_current(lease, now_s=now_s)
        renewed = Lease(
            run_id=lease.run_id,
            owner=lease.owner,
            generation=lease.generation,
            token=lease.token,
            expires_at_s=now_s + ttl_s,
        )
        self._leases[lease.run_id] = renewed
        return renewed

    def submit(self, report: TerminalReport, *, now_s: float) -> str:
        if not math.isfinite(now_s):
            raise TraceBenchError("now_s must be finite")
        if report.status not in TERMINAL_STATUSES:
            raise TraceBenchError(f"non-terminal report status: {report.status}")
        if not report.report_id or not report.run_id or not report.owner:
            raise TraceBenchError("report_id, run_id, and owner are required")
        if report.lease_generation <= 0 or not report.lease_token:
            raise TraceBenchError("lease generation and token are required")
        assert_sha256(report.evidence_sha256, field_name="evidence_sha256")
        canonical = report.canonical_bytes()
        prior = self._reports.get(report.report_id)
        if prior is not None:
            if prior != canonical:
                raise TraceBenchError("report_id was reused with different bytes")
            return "idempotent"

        lease = Lease(
            run_id=report.run_id,
            owner=report.owner,
            generation=report.lease_generation,
            token=report.lease_token,
            expires_at_s=float("inf"),
        )
        self._require_current(lease, now_s=now_s, compare_expiry=False)
        current = self._leases[report.run_id]
        if current.expires_at_s <= now_s:
            raise TraceBenchError("terminal report arrived after lease expiry")
        if report.run_id in self._terminal_report_by_run:
            raise TraceBenchError("run already has a different terminal report")
        self._reports[report.report_id] = canonical
        self._terminal_report_by_run[report.run_id] = report.report_id
        return "accepted"

    @staticmethod
    def _validate_clock(*, now_s: float, ttl_s: float) -> None:
        if not math.isfinite(now_s):
            raise TraceBenchError("now_s must be finite")
        if ttl_s <= 0 or not math.isfinite(ttl_s):
            raise TraceBenchError("ttl_s must be finite and positive")

    def _require_current(
        self,
        lease: Lease,
        *,
        now_s: float,
        compare_expiry: bool = True,
    ) -> None:
        current = self._leases.get(lease.run_id)
        if current is None:
            raise TraceBenchError("run has no lease")
        identity = (lease.owner, lease.generation, lease.token)
        current_identity = (current.owner, current.generation, current.token)
        if identity != current_identity:
            raise TraceBenchError("stale or foreign lease holder")
        if compare_expiry and current.expires_at_s <= now_s:
            raise TraceBenchError("lease expired")
