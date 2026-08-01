# TraceBench GPU Divergence Results

Exact token-sequence divergence for the declared experiment scope.

| Mode | Condition | Valid / planned | Divergences | Rate | Wilson 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| greedy | uncontrolled_batch_variation | 80 / 80 | 4 | 5.000% | [1.961%, 12.162%] |
| greedy | capsule_pinned_replay | 80 / 80 | 0 | 0.000% | [0.000%, 4.582%] |
| sampled | uncontrolled_batch_variation | 80 / 80 | 69 | 86.250% | [77.033%, 92.145%] |
| sampled | capsule_pinned_replay | 80 / 80 | 0 | 0.000% | [0.000%, 4.582%] |

Invalid comparisons are excluded from the rate and reported separately.
