"""Evidence capture profiles compared by TraceBench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tracebench.model import (
    ARTIFACT_KINDS,
    CONTROL_FIELDS,
    EvidenceRecord,
    PolicyEvidence,
    TraceBenchError,
    WorkloadRequest,
    canonical_json_bytes,
    controls_subset,
)


@dataclass(frozen=True, slots=True)
class PolicyDefinition:
    name: str
    preserved_artifact_kinds: frozenset[str]
    referenced_artifact_kinds: frozenset[str]
    captured_controls: frozenset[str]
    preserve_input: bool
    preserve_output: bool
    description: str

    def __post_init__(self) -> None:
        artifact_names = self.preserved_artifact_kinds | self.referenced_artifact_kinds
        unsupported_artifacts = artifact_names - set(ARTIFACT_KINDS)
        if unsupported_artifacts:
            raise TraceBenchError(
                f"{self.name} has unsupported artifact kinds: {sorted(unsupported_artifacts)}"
            )
        unsupported_controls = self.captured_controls - set(CONTROL_FIELDS)
        if unsupported_controls:
            raise TraceBenchError(
                f"{self.name} has unsupported controls: {sorted(unsupported_controls)}"
            )
        overlap = self.preserved_artifact_kinds & self.referenced_artifact_kinds
        if overlap:
            raise TraceBenchError(
                f"{self.name} both preserves and references: {sorted(overlap)}"
            )

    def capture(self, requests: Iterable[WorkloadRequest]) -> PolicyEvidence:
        evidence = PolicyEvidence(policy=self.name)
        for request in requests:
            preserved = {
                artifact.sha256
                for artifact in request.artifacts
                if artifact.kind in self.preserved_artifact_kinds
            }
            referenced = {
                artifact.sha256
                for artifact in request.artifacts
                if artifact.kind in self.referenced_artifact_kinds
            }
            for artifact in request.artifacts:
                if artifact.sha256 in preserved:
                    evidence.add_blob(artifact.sha256, artifact.size_bytes)
            controls = controls_subset(request.controls_dict, self.captured_controls)
            metadata = {
                "schema": "tracebench/evidence-record/v1",
                "policy": self.name,
                "request_id": request.request_id,
                "arrival_s": round(request.arrival_s, 9),
                "preserved_artifacts": sorted(preserved),
                "referenced_artifacts": sorted(referenced),
                "controls": dict(controls),
                "input_sha256": request.input_sha256 if self.preserve_input else None,
                "output_sha256": request.output_sha256
                if self.preserve_output
                else None,
            }
            payload_bytes = (request.input_bytes if self.preserve_input else 0) + (
                request.output_bytes if self.preserve_output else 0
            )
            evidence.add_record(
                EvidenceRecord(
                    policy=self.name,
                    request_id=request.request_id,
                    preserved_hashes=frozenset(preserved),
                    referenced_hashes=frozenset(referenced),
                    controls=controls,
                    input_preserved=self.preserve_input,
                    output_preserved=self.preserve_output,
                    metadata_bytes=len(canonical_json_bytes(metadata)),
                    payload_bytes=payload_bytes,
                )
            )
        return evidence


POLICIES = (
    PolicyDefinition(
        name="input-output-only",
        preserved_artifact_kinds=frozenset(),
        referenced_artifact_kinds=frozenset(),
        captured_controls=frozenset(),
        preserve_input=True,
        preserve_output=True,
        description="Request and response payloads without dependency or execution pins.",
    ),
    PolicyDefinition(
        name="mlflow-reference",
        preserved_artifact_kinds=frozenset(),
        referenced_artifact_kinds=frozenset({"model", "config"}),
        captured_controls=frozenset({"random_seed"}),
        preserve_input=False,
        preserve_output=False,
        description=(
            "A benchmark-defined reference profile with model/config references and "
            "a seed; it is not a claim about universal MLflow behavior."
        ),
    ),
    PolicyDefinition(
        name="wandb-reference",
        preserved_artifact_kinds=frozenset(),
        referenced_artifact_kinds=frozenset({"model", "prompt", "config"}),
        captured_controls=frozenset({"random_seed"}),
        preserve_input=False,
        preserve_output=False,
        description=(
            "A benchmark-defined reference profile with model/prompt/config references "
            "and a seed; it is not a claim about universal W&B behavior."
        ),
    ),
    PolicyDefinition(
        name="full-artifact-dedup",
        preserved_artifact_kinds=frozenset(ARTIFACT_KINDS),
        referenced_artifact_kinds=frozenset(),
        captured_controls=frozenset({"random_seed"}),
        preserve_input=True,
        preserve_output=True,
        description=(
            "Content-address every required artifact once, but omit batching and exact "
            "runtime controls."
        ),
    ),
    PolicyDefinition(
        name="capsule",
        preserved_artifact_kinds=frozenset(ARTIFACT_KINDS),
        referenced_artifact_kinds=frozenset(),
        captured_controls=frozenset(CONTROL_FIELDS),
        preserve_input=True,
        preserve_output=True,
        description=(
            "Content-address all dependencies and bind seed, batch, and runtime controls."
        ),
    ),
)


def policy_by_name(name: str) -> PolicyDefinition:
    matches = [policy for policy in POLICIES if policy.name == name]
    if len(matches) != 1:
        raise TraceBenchError(f"unknown policy: {name}")
    return matches[0]
