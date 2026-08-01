"""Structural replay-sufficiency evaluation."""

from __future__ import annotations

from tracebench.model import (
    CONTROL_FIELDS,
    PolicyEvidence,
    ReplayOutcome,
    Snapshot,
    TraceBenchError,
    WorkloadRequest,
)


def evaluate_replay(
    request: WorkloadRequest,
    *,
    evidence: PolicyEvidence,
    snapshot: Snapshot,
) -> ReplayOutcome:
    """Evaluate whether evidence can reconstruct and compare one decision.

    This evaluator is deliberately structural. It does not claim that a GPU
    re-execution produced the same output; the GPU divergence experiment is a
    separate benchmark stage.
    """

    if snapshot.request_id != request.request_id:
        raise TraceBenchError("snapshot request does not match replay request")
    try:
        record = evidence.records[request.request_id]
    except KeyError as exc:
        raise TraceBenchError(
            f"policy {evidence.policy} has no evidence for {request.request_id}"
        ) from exc

    retrievable_references = record.referenced_hashes & snapshot.available_hashes
    available_artifacts = record.preserved_hashes | retrievable_references
    missing_artifacts = tuple(sorted(request.artifact_hashes - available_artifacts))

    recorded_controls = record.controls_dict
    expected_controls = request.controls_dict
    missing_controls = tuple(
        name
        for name in CONTROL_FIELDS
        if recorded_controls.get(name) != expected_controls[name]
    )
    missing_payloads: list[str] = []
    if not record.input_preserved:
        missing_payloads.append("request_input")
    if not record.output_preserved:
        missing_payloads.append("reference_output")

    artifact_complete = not missing_artifacts
    control_complete = not missing_controls
    payload_complete = not missing_payloads
    return ReplayOutcome(
        policy=evidence.policy,
        request_id=request.request_id,
        horizon_days=snapshot.horizon_days,
        missing_artifact_hashes=missing_artifacts,
        missing_controls=missing_controls,
        missing_payloads=tuple(missing_payloads),
        artifact_complete=artifact_complete,
        control_complete=control_complete,
        payload_complete=payload_complete,
        replay_sufficient=(artifact_complete and control_complete and payload_complete),
    )
