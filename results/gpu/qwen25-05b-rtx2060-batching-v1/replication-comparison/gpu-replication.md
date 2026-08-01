# TraceBench GPU Replication Comparison

- Registered plan: `d813bd11bf60d9845438d9865b4592e426adeab7686a923a06b7409ad1da44cc`
- Execution records compared: 352
- Terminal-outcome mismatches: 0
- Identical aggregate analysis rows: True

| Run | Mode | Condition | Valid / planned | Divergences | Rate |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | greedy | uncontrolled_batch_variation | 80 / 80 | 4 | 5.000% |
| 1 | greedy | capsule_pinned_replay | 80 / 80 | 0 | 0.000% |
| 1 | sampled | uncontrolled_batch_variation | 80 / 80 | 69 | 86.250% |
| 1 | sampled | capsule_pinned_replay | 80 / 80 | 0 | 0.000% |
| 2 | greedy | uncontrolled_batch_variation | 80 / 80 | 4 | 5.000% |
| 2 | greedy | capsule_pinned_replay | 80 / 80 | 0 | 0.000% |
| 2 | sampled | uncontrolled_batch_variation | 80 / 80 | 69 | 86.250% |
| 2 | sampled | capsule_pinned_replay | 80 / 80 | 0 | 0.000% |

The comparison verifies exact registered execution outcomes. It does not turn the fixed prompts or repeated runs into a population sample.
