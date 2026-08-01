# TraceBench

[![CI](https://github.com/David-Wu1119/tracebench/actions/workflows/ci.yml/badge.svg)](https://github.com/David-Wu1119/tracebench/actions/workflows/ci.yml)

TraceBench measures whether the evidence retained for an ML decision is structurally sufficient to reconstruct its dependencies, execution controls, request, and reference output at a later audit horizon.

This repository is an alpha research artifact. The CPU demo does not execute a model, measure GPU nondeterminism, validate a product, or establish regulatory compliance.

## Reproduce the CPU demo

Requirements: GNU Make and Python 3.11 or 3.12. There are no runtime dependencies.

```bash
git clone https://github.com/David-Wu1119/tracebench.git
cd tracebench
make demo
```

The command deterministically regenerates:

- `results/demo/results.csv`, the complete result matrix;
- `results/demo/results.md`, a readable table;
- `results/demo/manifest.json`, configuration, workload summaries, scope labels, and result hashes.

Run the test suite and verify the checked-in demo has not drifted:

```bash
make check
```

## What the demo evaluates

Every synthetic request binds four artifact versions (`model`, `prompt`, `config`, and `index`), three execution controls (`random_seed`, `batch_fingerprint`, and `runtime_digest`), a request digest, and a reference-output digest.

TraceBench compares five benchmark-defined evidence policies:

| Policy | Artifact behavior | Controls | Request + reference output |
| --- | --- | --- | --- |
| `input-output-only` | None | None | Preserved |
| `mlflow-reference` | Mutable model/config references | Seed | Not preserved |
| `wandb-reference` | Mutable model/prompt/config references | Seed | Not preserved |
| `full-artifact-dedup` | All artifacts content-addressed once | Seed | Preserved |
| `capsule` | All artifacts content-addressed once | Seed, batch, runtime | Preserved |

The vendor-named rows are controlled reference profiles, not emulators, product audits, or claims about universal MLflow or W&B defaults. MLflow behavior varies by integration, and W&B artifact logging is explicit. The exact definitions are in [`src/tracebench/policies.py`](src/tracebench/policies.py) and the rationale is in [`docs/methodology.md`](docs/methodology.md).

## Current result

The checked-in demo produces 30 rows: two arrival regimes, five policies, and audit horizons of 30, 90, and 365 days. Its key result is deliberately strict:

- Reference-based profiles retain only 25% of required artifacts at 30 days and 0% by 90 days under the declared drift scenario.
- Content-addressing preserves artifact and payload coverage, but remains structurally insufficient when batch and runtime controls are absent.
- Only the declared capsule contains every required component, so it is the only policy with 100% structural replay sufficiency.

These are eligibility results, not proof that a replay will produce the same tokens. See the [full generated table](results/demo/results.md).

## Workload provenance

The Poisson baseline and burst regime are synthetic. Their arrival and token parameters are reproducibly derived from public [BurstGPT](https://github.com/HPMLL/BurstGPT) and [Azure LLM inference](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) traces. Source hashes and derivation outputs are pinned in [`configs/trace-calibration.json`](configs/trace-calibration.json).

Artifact drift cadences, virtual artifact sizes, the batch window, and audit horizons are scenario knobs. They are not presented as trace-derived facts.

## Repository map

- `src/tracebench/workload.py`: Poisson and Markov-modulated burst workloads, artifact drift, and audit snapshots.
- `src/tracebench/policies.py`: evidence-policy capture profiles and content-addressed accounting.
- `src/tracebench/replay.py`: structural replay-sufficiency evaluator.
- `src/tracebench/protocol.py`: generation-fenced lease, heartbeat, reclaim, and idempotent terminal-report semantics.
- `src/tracebench/calibration.py`: public-trace calibration and source-hash binding.
- `src/tracebench/analysis.py`: deterministic result and manifest generation.
- `tests/`: behavioral tests over the public seams.

## Scope boundary

Version `0.1.0a0` establishes a CPU-reproducible structural benchmark. It does not yet include the GPU decision-divergence experiment. The intended experiment and required evidence contract are documented in [`docs/gpu-path.md`](docs/gpu-path.md); no result from that document should be treated as measured until a GPU evidence packet is published.

TraceBench is the public research artifact. CloudTune remains a separate private testbed and is not required to run this repository.

## License and attribution

TraceBench code is MIT licensed. Public trace data is not redistributed here; the source projects retain their own licenses and attribution requirements. See [`NOTICE.md`](NOTICE.md).
