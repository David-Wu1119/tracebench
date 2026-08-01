# TraceBench CPU Demo Results

These are deterministic structural sufficiency results over virtual artifact sizes. They are not GPU decision-divergence measurements.

| Workload | Policy | Horizon | Artifact coverage | Control coverage | Payload coverage | Replay sufficient | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| poisson | input-output-only | 30d | 0.000% | 0.000% | 100.000% | 0.000% | 683.25 KiB |
| poisson | input-output-only | 90d | 0.000% | 0.000% | 100.000% | 0.000% | 683.25 KiB |
| poisson | input-output-only | 365d | 0.000% | 0.000% | 100.000% | 0.000% | 683.25 KiB |
| poisson | mlflow-reference | 30d | 25.000% | 33.333% | 0.000% | 0.000% | 36.57 KiB |
| poisson | mlflow-reference | 90d | 0.000% | 33.333% | 0.000% | 0.000% | 36.57 KiB |
| poisson | mlflow-reference | 365d | 0.000% | 33.333% | 0.000% | 0.000% | 36.57 KiB |
| poisson | wandb-reference | 30d | 25.000% | 33.333% | 0.000% | 0.000% | 42.76 KiB |
| poisson | wandb-reference | 90d | 0.000% | 33.333% | 0.000% | 0.000% | 42.76 KiB |
| poisson | wandb-reference | 365d | 0.000% | 33.333% | 0.000% | 0.000% | 42.76 KiB |
| poisson | full-artifact-dedup | 30d | 100.000% | 33.333% | 100.000% | 0.000% | 1.40 GiB |
| poisson | full-artifact-dedup | 90d | 100.000% | 33.333% | 100.000% | 0.000% | 1.40 GiB |
| poisson | full-artifact-dedup | 365d | 100.000% | 33.333% | 100.000% | 0.000% | 1.40 GiB |
| poisson | capsule | 30d | 100.000% | 100.000% | 100.000% | 100.000% | 1.40 GiB |
| poisson | capsule | 90d | 100.000% | 100.000% | 100.000% | 100.000% | 1.40 GiB |
| poisson | capsule | 365d | 100.000% | 100.000% | 100.000% | 100.000% | 1.40 GiB |
| burst | input-output-only | 30d | 0.000% | 0.000% | 100.000% | 0.000% | 561.68 KiB |
| burst | input-output-only | 90d | 0.000% | 0.000% | 100.000% | 0.000% | 561.68 KiB |
| burst | input-output-only | 365d | 0.000% | 0.000% | 100.000% | 0.000% | 561.68 KiB |
| burst | mlflow-reference | 30d | 25.000% | 33.333% | 0.000% | 0.000% | 36.34 KiB |
| burst | mlflow-reference | 90d | 0.000% | 33.333% | 0.000% | 0.000% | 36.34 KiB |
| burst | mlflow-reference | 365d | 0.000% | 33.333% | 0.000% | 0.000% | 36.34 KiB |
| burst | wandb-reference | 30d | 25.000% | 33.333% | 0.000% | 0.000% | 42.52 KiB |
| burst | wandb-reference | 90d | 0.000% | 33.333% | 0.000% | 0.000% | 42.52 KiB |
| burst | wandb-reference | 365d | 0.000% | 33.333% | 0.000% | 0.000% | 42.52 KiB |
| burst | full-artifact-dedup | 30d | 100.000% | 33.333% | 100.000% | 0.000% | 1.40 GiB |
| burst | full-artifact-dedup | 90d | 100.000% | 33.333% | 100.000% | 0.000% | 1.40 GiB |
| burst | full-artifact-dedup | 365d | 100.000% | 33.333% | 100.000% | 0.000% | 1.40 GiB |
| burst | capsule | 30d | 100.000% | 100.000% | 100.000% | 100.000% | 1.40 GiB |
| burst | capsule | 90d | 100.000% | 100.000% | 100.000% | 100.000% | 1.40 GiB |
| burst | capsule | 365d | 100.000% | 100.000% | 100.000% | 100.000% | 1.40 GiB |

See `results.csv` for exact byte accounting and `manifest.json` for the configuration.
