# GPU Decision-Divergence Path

Status: preregistered experiment completed; two clean evidence bundles and an
outcome-by-outcome replication comparison are published in
[`results/gpu/qwen25-05b-rtx2060-batching-v1/`](../results/gpu/qwen25-05b-rtx2060-batching-v1/).

## Question

When the same request is replayed on a GPU under batching variation, how often
does the generated token sequence diverge, and how much divergence remains when
the registered observable execution context is repeated exactly?

## Registered engine and model

The registered host is native Windows with an NVIDIA GeForce RTX 2060. [vLLM does not support Windows natively](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/), and this host has no WSL distribution. The experiment therefore uses the proven Hugging Face Transformers `generate` implementation instead of a custom inference engine. That narrows every result to this engine and hardware; it is not evidence about vLLM or serving systems generally.

The first registered model is `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`. The plan binds the exact seven-file local snapshot with per-file SHA-256 values: weights, model and generation configs, tokenizer config, tokenizer JSON, vocabulary, and merges. Extra, missing, symlinked, or changed files abort the run. Qwen2.5-1.5B is a separate later experiment and must not be pooled into this preregistration.

The plan also fixes Python 3.12.10, PyTorch 2.6.0+cu124, Transformers 5.5.0, CUDA runtime 12.4, the normalized installed-package inventory (excluding the `pip` installer itself), GPU name, 6144 MiB VRAM, and compute capability 7.5. A mismatch aborts before a final evidence directory can be published.

The GPU runner preserves raw results locally and emits a self-contained evidence directory. This experiment is executed directly on the registered host; it is not evidence that CloudTune orchestration or an outbound-only runner path worked. The public analyzer consumes only the exported evidence contract.

## Experiment cells

`configs/gpu-experiment.json` fixes 16 synthetic prompts, two decoding modes, five batch-schedule variants, five pinned replays, seeds, batch sizes, runtime flags, and the model snapshot hashes before result inspection. Use the same fixed request set in every cell:

| Condition | Reference | Comparison |
| --- | --- | --- |
| Uncontrolled batch variation | Each request generated alone | Same request under each of five registered shuffled batch schedules |
| Capsule-pinned replay | The recorded `variant-00` execution | Five exact replays with the same model snapshot, runtime, seed, decoding controls, batch membership, order, and position |

The registered sample size is 80 comparisons per condition and decoding mode. This is exploratory, not a high-powered population estimate. The analyzer reports Wilson intervals and retains this limitation.

`capsule_pinned_replay` is the v1 schema label. The runner manually reconstructs
the registered context; it does not deserialize a capsule. This experiment can
support a context-pinning result, but not a claim that a serialized capsule
implementation has been validated.

## Measured result

The two reported clean runs produced identical terminal outcomes across all 352
registered execution keys:

| Decoding | Condition | Divergences / valid | Rate |
| --- | --- | ---: | ---: |
| Greedy | Uncontrolled batch variation | 4 / 80 | 5.000% |
| Greedy | Registered context pinned | 0 / 80 | 0.000% |
| Sampled | Uncontrolled batch variation | 69 / 80 | 86.250% |
| Sampled | Registered context pinned | 0 / 80 | 0.000% |

The sampled result includes the interaction between a reset global seed and
changed batch membership/order; it is not evidence of hardware nondeterminism.
The greedy result establishes exact-token batch-context sensitivity in this stack
without isolating its lower-level numerical cause. Full intervals, evidence,
redactions, failed-attempt history, and limitations are in the
[result packet](../results/gpu/qwen25-05b-rtx2060-batching-v1/README.md).

## Commands

Install the optional environment on a compatible Windows CUDA host:

```powershell
python -m pip install -r requirements-gpu-windows.txt
$env:PYTHONPATH = (Resolve-Path .\src)
```

Execute the registered plan with an already downloaded, revision-pinned model directory:

```powershell
python -m tracebench gpu-run --plan configs/gpu-experiment.json --model-path D:\path\to\pinned-model --implementation-commit <40-character-tracebench-commit> --output D:\path\to\new-evidence-directory
```

Verify checksums and produce the result table on a CPU-only machine:

```bash
PYTHONPATH=src python -m tracebench gpu-analyze --evidence /path/to/evidence-directory --output /path/to/new-analysis-directory
```

Verify two complete reruns and compare every registered execution outcome:

```bash
PYTHONPATH=src python -m tracebench gpu-compare \
  --evidence /path/to/first-evidence /path/to/second-evidence \
  --output /path/to/new-comparison-directory
```

## Required exported files

```text
gpu-evidence/
  checksums.sha256
  environment.json
  executions.jsonl
  manifest.json
  nvidia-smi.txt
  packages.txt
  plan.json
  requests.jsonl
  runner.log
```

The sidecar binds the exact eight-file payload; unbound extra files are rejected. The plan, environment, request rows, execution rows, manifest hashes, record counts, and TraceBench commit are then cross-validated rather than trusted independently.

Each execution row binds:

- request ID and input digest;
- the runtime digest, which binds model/tokenizer snapshot, software, hardware, and runtime settings;
- the registered seed and decoding mode;
- batch ID, ordered batch members, and position;
- the exact TraceBench implementation commit;
- output token IDs and output digest;
- batch duration, terminal status, and retained error text.

The runner writes to a `.partial` directory and only renames it to the final path after the registered record count, manifest, and checksums are complete. A failed run stays partial and is not valid evidence.

## Metrics

Primary metric:

```text
sequence_divergence_rate = replays with a different output-token digest / valid replays
```

Secondary metrics:

- normalized first-different-token position;
- invalid replay rate, reported separately from divergence;
- Wilson 95% confidence intervals.

The Wilson interval treats repeated comparisons as independent and is descriptive only. The 16 fixed prompts are not a population sample, so the interval must not be presented as population-level uncertainty.

Do not use semantic similarity as the headline metric. It can hide exact decision changes and introduces another model into the measurement chain.

## Acceptance gate

The GPU result is publishable only when:

1. a fresh machine can verify every checksum and parse every row;
2. the request set and sample size were fixed before result inspection;
3. invalid runs are retained and accounted for rather than silently dropped;
4. at least one independent rerun reproduces the direction of the result;
5. the analyzer and a small non-sensitive evidence sample are public;
6. the paper states hardware and engine scope instead of generalizing to all GPUs or serving systems.

The published v1 packet satisfies this artifact gate for the declared
Transformers/RTX 2060 scope. It does not validate CloudTune orchestration, a
serialized capsule implementation, vLLM, or serving systems generally.
