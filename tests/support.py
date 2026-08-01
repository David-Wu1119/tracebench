"""Deterministic test fixtures built through TraceBench's public records."""

from __future__ import annotations

from tracebench.model import (
    ARTIFACT_KINDS,
    ArtifactVersion,
    WorkloadRequest,
    sha256_text,
)


def make_request(
    *, request_id: str = "request-001", arrival_s: float = 0.0
) -> WorkloadRequest:
    artifacts = tuple(
        ArtifactVersion(
            kind=kind,
            version=0,
            sha256=sha256_text(f"fixture:{kind}:v0"),
            size_bytes=(index + 1) * 1_000,
            mutable_uri=f"artifact://{kind}/current",
        )
        for index, kind in enumerate(ARTIFACT_KINDS)
    )
    controls = tuple(
        sorted(
            {
                "random_seed": "17",
                "batch_fingerprint": sha256_text("fixture:batch:0"),
                "runtime_digest": sha256_text("fixture:runtime:v1"),
            }.items()
        )
    )
    input_sha256 = sha256_text(f"fixture:input:{request_id}")
    return WorkloadRequest(
        request_id=request_id,
        arrival_s=arrival_s,
        input_tokens=32,
        output_tokens=8,
        input_sha256=input_sha256,
        output_sha256=sha256_text(f"fixture:output:{input_sha256}"),
        artifacts=artifacts,
        controls=controls,
    )
