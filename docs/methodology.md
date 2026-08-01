# Methodology

## Research question

TraceBench asks a narrow question: given the evidence retained at decision time and the state of mutable stores at a later audit horizon, is every component needed to reconstruct and compare that decision still available?

Structural sufficiency is necessary but not sufficient for behavioral equivalence. A structurally sufficient record can still diverge when executed because of nondeterministic kernels, hardware, scheduling, implementation defects, or hidden dependencies.

## Decision record

For request `r`, the benchmark defines:

- required artifacts `A(r) = {model, prompt, config, index}`;
- required controls `C(r) = {random_seed, batch_fingerprint, runtime_digest}`;
- required comparison payloads `P(r) = {request_input, reference_output}`.

An evidence policy can preserve an artifact by content hash, retain only a mutable reference, capture a control, and preserve either payload. Content-addressed blobs are counted once per policy, even when many requests share them.

At audit horizon `T`, a mutable reference resolves to the original artifact only if that version is still current or remains inside the configured half-open retention window. With zero retention, an artifact becomes unavailable at the exact drift boundary.

A request is structurally replay-sufficient only when all three conditions hold:

```text
A(r) is available AND C(r) is captured exactly AND P(r) is preserved
```

The result table also reports component coverage. Coverage explains a failed result; it does not weaken the binary sufficiency criterion.

## Evidence profiles

The five profiles are declared benchmark treatments:

| Profile | Preserved artifacts | Mutable references | Controls | Payloads |
| --- | --- | --- | --- | --- |
| `input-output-only` | None | None | None | Input and output |
| `mlflow-reference` | None | Model, config | Seed | None |
| `wandb-reference` | None | Model, prompt, config | Seed | None |
| `full-artifact-dedup` | Model, prompt, config, index | None | Seed | Input and output |
| `capsule` | Model, prompt, config, index | None | Seed, batch, runtime | Input and output |

The vendor names are orientation labels, not measured integrations. MLflow's own documentation shows that autologging differs across libraries; for LangChain, trace logging is enabled by default while model logging is optional. W&B documents artifacts as objects users explicitly create and log. These sources justify treating experiment tracking and artifact preservation as separate choices, but they do not justify a claim that either product has one universal default:

- [MLflow automatic logging](https://mlflow.org/docs/latest/ml/tracking/autolog/)
- [MLflow LangChain autologging defaults](https://mlflow.org/docs/latest/genai/flavors/langchain/autologging/)
- [W&B Artifacts overview](https://docs.wandb.ai/models/artifacts)
- [W&B artifact construction and logging](https://docs.wandb.ai/models/artifacts/construct-an-artifact)

## Workload generation

The baseline uses exponential interarrival times, producing a homogeneous Poisson process. The burst treatment uses a two-state Markov-modulated Poisson process: a quiet state at the baseline rate and a burst state at `burst_multiplier` times that rate. Quiet and burst durations are exponentially distributed.

Request and response token counts are sampled independently from bounded log-normal distributions. Prompt contents are never downloaded or synthesized; request records contain deterministic digests and token counts only.

The approach is motivated by two public trace families:

- [BurstGPT](https://github.com/HPMLL/BurstGPT) reports 10.31 million Azure OpenAI traces across 213 days and explicitly documents burstiness, token lengths, and scaling of average RPS for evaluation. The associated [BurstGPT paper](https://arxiv.org/abs/2401.17644) describes diverse concurrency patterns and realistic workload variation.
- The [Azure LLM inference trace](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) publishes invocation timestamps plus context and generated token counts from multiple services. It accompanies the [Splitwise paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/12/Splitwise_ISCA24.pdf).

## Public-trace calibration

`configs/trace-calibration.json` is generated from three downloaded CSV files by:

```bash
python -m tracebench calibrate \
  --burstgpt /path/to/BurstGPT_1.csv \
  --azure-code /path/to/AzureLLMInferenceTrace_code.csv \
  --azure-conversation /path/to/AzureLLMInferenceTrace_conv.csv \
  --output /tmp/trace-calibration.json
```

The checked-in calibration binds each input with SHA-256 and records row counts and summary statistics. The demo parameters are derived as follows:

- baseline RPS: BurstGPT mean RPS rounded down to one decimal;
- burst multiplier: rounded geometric center of BurstGPT's 10-second p95/mean and p99/mean count ratios;
- burst duration: median duration of BurstGPT high-state runs, where high means a 10-second bin at or above p95;
- quiet duration: p75 duration of the complementary quiet runs;
- token medians: nearest powers of two to the geometric centers of positive per-source medians;
- shared log-space sigma: rounded combination of median input and output sigmas.

The calibration file, not this prose, is the authoritative numeric record.

## Scenario knobs

The default drift schedule is model 90 days, prompt 7 days, config 30 days, and index 1 day. Audit horizons are 30, 90, and 365 days. Snapshot retention is zero days. These values are explicit experimental scenarios, not estimates from BurstGPT or Azure.

Artifact sizes are virtual: model 1.25 GB, prompt 16 KiB, config 8 KiB, and index 250 MB. They drive storage accounting only. The CPU demo never allocates blobs of those sizes.

## Determinism and output integrity

The generator uses a fixed seed and Python's standard-library pseudorandom generator. JSON encodings are sorted and reject non-finite numbers. The result manifest records hashes for the CSV and Markdown table, the complete configuration, workload summaries, policy names, and explicit measurement-scope labels.

`make check` reruns the tests and demo, then fails if the checked-in result directory changes.

## Limitations

- No model executes in the CPU demo.
- Replay sufficiency is structural, not behavioral.
- Artifact bytes and payload bytes are simulated by size and preservation state.
- The benchmark does not measure storage outages, authorization failures, or retention-policy enforcement by a real provider.
- The policy profiles are not adapters to MLflow or W&B.
- The reference runner protocol is in-memory semantics, not a distributed queue implementation.
- Cryptographic signing, receipt verification, and GPU decision divergence are outside version `0.1.0a0`.
