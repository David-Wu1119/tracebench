# Qwen2.5-0.5B RTX 2060 Batch-Variation Experiment

## Result

On the declared native-Windows RTX 2060 and Hugging Face Transformers scope,
changing registered batch membership, order, and position changed exact output
tokens for 4 of 80 greedy comparisons and 69 of 80 sampled comparisons. Repeating
the same registered execution context produced no divergence in 80 comparisons
per decoding mode.

Two clean runs reproduced every terminal execution outcome: 352 keys compared,
with zero status, output-digest, or error mismatches. This is an exact replication
of the fixed experiment, not a population-level result.

| Decoding | Condition | Valid / planned | Divergences | Rate | Wilson 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| Greedy | Uncontrolled batch variation | 80 / 80 | 4 | 5.000% | [1.961%, 12.162%] |
| Greedy | Registered context pinned | 80 / 80 | 0 | 0.000% | [0.000%, 4.582%] |
| Sampled | Uncontrolled batch variation | 80 / 80 | 69 | 86.250% | [77.033%, 92.145%] |
| Sampled | Registered context pinned | 80 / 80 | 0 | 0.000% | [0.000%, 4.582%] |

The checked-in schema calls the second condition `capsule_pinned_replay`. In this
v1 experiment, the runner manually pins the registered model snapshot, runtime,
seed, decoding controls, batch membership, order, and position. It does **not**
load and execute a serialized replay capsule. The defensible claim is therefore
that registered context pinning eliminated observed divergence in this fixed
experiment, not that a capsule implementation guarantees deterministic replay.

## Registered scope

- Experiment: `qwen25-05b-rtx2060-batching-v1`
- Plan SHA-256: `d813bd11bf60d9845438d9865b4592e426adeab7686a923a06b7409ad1da44cc`
- TraceBench commit: `885f9209a80cce3b5f230a3b59a0a8a4a5cd8e75`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` at revision
  `7ae557604adf67be50417f59c2c2f167def9a775`
- Model snapshot SHA-256:
  `8048fda149f96a2d1f38036ddea4adc0a52925c9332b316c66535e679b6d35a3`
- Engine: Hugging Face Transformers `generate`, version 5.5.0
- Runtime: Python 3.12.10, PyTorch 2.6.0+cu124, CUDA runtime 12.4
- Hardware: NVIDIA GeForce RTX 2060, 6144 MiB, compute capability 7.5
- Requests: 16 fixed synthetic prompts
- Comparisons: five batch variants and five pinned repetitions per decoding mode

The clean source runs were 178.253 and 178.633 seconds. Each contains 352 complete
execution records and zero failed records.

## Evidence map

- [`run2-public-evidence/`](run2-public-evidence/): first clean evidence bundle.
- [`run3-public-evidence/`](run3-public-evidence/): independent clean rerun.
- [`run2-analysis/`](run2-analysis/): verified analysis of the first clean run.
- [`run3-analysis/`](run3-analysis/): verified analysis of the rerun.
- [`replication-comparison/`](replication-comparison/): machine-generated
  outcome-by-outcome comparison.
- [`operator-log.md`](operator-log.md): full run chronology, including the valid
  but conservatively excluded first run and two aborted contention attempts.

Reverify and regenerate the analyses from the repository root:

```bash
PYTHONPATH=src python -m tracebench gpu-analyze \
  --evidence results/gpu/qwen25-05b-rtx2060-batching-v1/run2-public-evidence \
  --output /tmp/tracebench-run2-analysis

PYTHONPATH=src python -m tracebench gpu-analyze \
  --evidence results/gpu/qwen25-05b-rtx2060-batching-v1/run3-public-evidence \
  --output /tmp/tracebench-run3-analysis

PYTHONPATH=src python -m tracebench gpu-compare \
  --evidence \
    results/gpu/qwen25-05b-rtx2060-batching-v1/run2-public-evidence \
    results/gpu/qwen25-05b-rtx2060-batching-v1/run3-public-evidence \
  --output /tmp/tracebench-gpu-replication
```

## Public-copy redaction

The private raw bundles contain host-local identifiers in `nvidia-smi.txt`. The
public copies remove the GPU UUID, GPU PDI, and process-directory paths while
retaining executable basenames. Their outer `checksums.sha256` files were then
regenerated. The other seven payload files in each public bundle are byte-for-byte
identical to the raw source bundle.

Raw local archive identifiers:

| Source run | Raw archive SHA-256 |
| --- | --- |
| Run 2 | `516f2f387ba34e4fbe87254e8d5cd09ea2ba2af325c949ecf868db9ae229a31f` |
| Run 3 | `86150dfc831c04eed449ba7ac9dfd244b4f54018db1984f84ce6a8200e935259` |

Those hashes identify private raw archives; the raw archives are not published in
this repository. The redaction changes only host inventory, not the plan,
environment contract, requests, execution records, manifest, packages, or log.

## Limits

- This is one consumer GPU, one Windows host, one model, one engine, and one fixed
  prompt set. It is not evidence about vLLM, other GPUs, serving clusters, or LLMs
  generally.
- Sampled batch variation includes the interaction between a reset global random
  seed and changed batch membership/order. Its 86.25% rate is batch-context
  sensitivity, not proof of hardware nondeterminism.
- Greedy divergence is exact-token evidence of batch-context sensitivity in this
  declared stack; the experiment does not isolate the lower-level numerical cause.
- The Wilson intervals are descriptive. Repeated comparisons share 16 prompts and
  are not independent population samples.
- No run used CloudTune orchestration or its outbound-only runner. This result
  validates the public TraceBench runner/analyzer path only.
- Zero observed pinned divergence does not prove a zero true failure rate or a
  general determinism guarantee.
