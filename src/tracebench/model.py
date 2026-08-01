"""Core immutable records shared by workload, policy, and replay modules."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


ARTIFACT_KINDS = ("model", "prompt", "config", "index")
CONTROL_FIELDS = ("random_seed", "batch_fingerprint", "runtime_digest")


class TraceBenchError(ValueError):
    """Raised when an input violates a benchmark invariant."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the deterministic JSON encoding used for hashes and accounting."""

    def validate(item: Any, path: str) -> None:
        if item is None or isinstance(item, (bool, str, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise TraceBenchError(f"{path} contains a non-finite float")
            return
        if isinstance(item, list) or isinstance(item, tuple):
            for index, child in enumerate(item):
                validate(child, f"{path}[{index}]")
            return
        if isinstance(item, dict) or isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TraceBenchError(f"{path} contains a non-string key")
                validate(child, f"{path}.{key}")
            return
        raise TraceBenchError(f"{path} contains unsupported type {type(item).__name__}")

    validate(value, "document")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assert_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise TraceBenchError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ArtifactVersion:
    kind: str
    version: int
    sha256: str
    size_bytes: int
    mutable_uri: str

    def __post_init__(self) -> None:
        if self.kind not in ARTIFACT_KINDS:
            raise TraceBenchError(f"unsupported artifact kind: {self.kind}")
        if self.version < 0:
            raise TraceBenchError("artifact version must be non-negative")
        if self.size_bytes <= 0:
            raise TraceBenchError("artifact size must be positive")
        if not self.mutable_uri:
            raise TraceBenchError("artifact mutable_uri is required")
        assert_sha256(self.sha256, field_name="artifact sha256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "mutable_uri": self.mutable_uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class WorkloadRequest:
    request_id: str
    arrival_s: float
    input_tokens: int
    output_tokens: int
    input_sha256: str
    output_sha256: str
    artifacts: tuple[ArtifactVersion, ...]
    controls: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise TraceBenchError("request_id is required")
        if self.arrival_s < 0 or not math.isfinite(self.arrival_s):
            raise TraceBenchError("arrival_s must be finite and non-negative")
        if self.input_tokens <= 0 or self.output_tokens <= 0:
            raise TraceBenchError("token counts must be positive")
        assert_sha256(self.input_sha256, field_name="input_sha256")
        assert_sha256(self.output_sha256, field_name="output_sha256")
        kinds = tuple(artifact.kind for artifact in self.artifacts)
        if tuple(sorted(kinds)) != tuple(sorted(ARTIFACT_KINDS)):
            raise TraceBenchError(
                "each request must bind exactly one artifact of every kind"
            )
        control_names = tuple(name for name, _ in self.controls)
        if tuple(sorted(control_names)) != tuple(sorted(CONTROL_FIELDS)):
            raise TraceBenchError(
                "each request must bind every execution control exactly once"
            )

    @property
    def artifact_hashes(self) -> frozenset[str]:
        return frozenset(artifact.sha256 for artifact in self.artifacts)

    @property
    def controls_dict(self) -> dict[str, str]:
        return dict(self.controls)

    @property
    def input_bytes(self) -> int:
        return self.input_tokens * 4

    @property
    def output_bytes(self) -> int:
        return self.output_tokens * 4


@dataclass(frozen=True, slots=True)
class Snapshot:
    request_id: str
    horizon_days: int
    available_hashes: frozenset[str]

    def __post_init__(self) -> None:
        if self.horizon_days <= 0:
            raise TraceBenchError("horizon_days must be positive")


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    policy: str
    request_id: str
    preserved_hashes: frozenset[str]
    referenced_hashes: frozenset[str]
    controls: tuple[tuple[str, str], ...]
    input_preserved: bool
    output_preserved: bool
    metadata_bytes: int
    payload_bytes: int

    @property
    def controls_dict(self) -> dict[str, str]:
        return dict(self.controls)


@dataclass(slots=True)
class PolicyEvidence:
    policy: str
    records: dict[str, EvidenceRecord] = field(default_factory=dict)
    blobs: dict[str, int] = field(default_factory=dict)

    def add_blob(self, sha256: str, size_bytes: int) -> None:
        assert_sha256(sha256, field_name="blob sha256")
        existing = self.blobs.get(sha256)
        if existing is not None and existing != size_bytes:
            raise TraceBenchError("one digest cannot identify two blob sizes")
        self.blobs[sha256] = size_bytes

    def add_record(self, record: EvidenceRecord) -> None:
        if record.policy != self.policy:
            raise TraceBenchError("record policy does not match evidence policy")
        if record.request_id in self.records:
            raise TraceBenchError(f"duplicate evidence record: {record.request_id}")
        self.records[record.request_id] = record

    @property
    def blob_bytes(self) -> int:
        return sum(self.blobs.values())

    @property
    def metadata_bytes(self) -> int:
        return sum(record.metadata_bytes for record in self.records.values())

    @property
    def payload_bytes(self) -> int:
        return sum(record.payload_bytes for record in self.records.values())

    @property
    def total_bytes(self) -> int:
        return self.blob_bytes + self.metadata_bytes + self.payload_bytes


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    policy: str
    request_id: str
    horizon_days: int
    missing_artifact_hashes: tuple[str, ...]
    missing_controls: tuple[str, ...]
    missing_payloads: tuple[str, ...]
    artifact_complete: bool
    control_complete: bool
    payload_complete: bool
    replay_sufficient: bool


def controls_subset(
    controls: Mapping[str, str],
    names: Iterable[str],
) -> tuple[tuple[str, str], ...]:
    selected = [(name, controls[name]) for name in names]
    return tuple(sorted(selected))
