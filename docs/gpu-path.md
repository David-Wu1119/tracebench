# GPU Decision-Divergence Path

Status: experiment contract only. Version `0.1.0a0` does not implement or report this experiment.

## Question

When the same request is replayed on a GPU under batching variation, how often does the generated token sequence diverge, and how much of that divergence remains when a replay capsule pins the observable execution context?

## Engine and models

Use a proven serving engine rather than a custom scheduler. The planned reference path is a pinned vLLM container with Qwen2.5-0.5B first and Qwen2.5-1.5B second. Record the exact container digest, vLLM version, model revision, tokenizer revision, CUDA runtime, driver, GPU model, and decoding configuration.

The GPU machine must pull work outbound, preserve raw results locally, and emit a self-contained evidence directory. CloudTune may orchestrate the run, but the public analyzer must consume only the exported evidence contract.

## Experiment cells

Use the same fixed request set in every cell:

| Cell | Batch schedule | Capsule pins |
| --- | --- | --- |
| A | Dynamic, varied arrival order | Input and declared model reference only |
| B | Dynamic, repeated with a second order | Input and declared model reference only |
| C | Recorded batch membership and order | Model/tokenizer/runtime/seed/decoding/batch |
| D | Exact replay of cell C | Model/tokenizer/runtime/seed/decoding/batch |

Run enough repetitions to report a binomial confidence interval, not one anecdotal mismatch. The final sample size must be chosen before inspecting divergence results and recorded in the experiment manifest.

## Required exported files

```text
gpu-evidence/
  manifest.json
  requests.jsonl
  executions.jsonl
  environment.json
  nvidia-smi.txt
  container-image.txt
  logs/
  checksums.sha256
```

Each execution row must bind:

- request ID and input digest;
- model and tokenizer revisions;
- seed and all decoding parameters;
- batch ID, ordered batch members, and position;
- runtime/container digest and GPU identity;
- output token IDs and output digest;
- start/end timestamps and terminal status.

The manifest must distinguish recorded values from inferred values. Missing fields remain missing; the exporter must not fill them from a later environment.

## Metrics

Primary metric:

```text
sequence_divergence_rate = replays with a different output-token digest / valid replays
```

Secondary metrics:

- normalized first-different-token position;
- output-length difference;
- invalid replay rate, reported separately from divergence;
- divergence by batch size and batch position;
- Wilson 95% confidence intervals.

Do not use semantic similarity as the headline metric. It can hide exact decision changes and introduces another model into the measurement chain.

## Acceptance gate

The GPU result is publishable only when:

1. a fresh machine can verify every checksum and parse every row;
2. the request set and sample size were fixed before result inspection;
3. invalid runs are retained and accounted for rather than silently dropped;
4. at least one independent rerun reproduces the direction of the result;
5. the analyzer and a small non-sensitive evidence sample are public;
6. the paper states hardware and engine scope instead of generalizing to all GPUs or serving systems.

Until those conditions hold, TraceBench has no measured GPU nondeterminism result.
