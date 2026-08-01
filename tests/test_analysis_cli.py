from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tracebench.analysis import run_benchmark, write_results
from tracebench.cli import main
from tracebench.workload import WorkloadConfig


class AnalysisAndCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = WorkloadConfig(seed=5, requests_per_regime=3)

    def test_benchmark_has_one_row_per_workload_policy_and_horizon(self) -> None:
        rows = run_benchmark(self.config, horizons=(365, 30, 90, 30))

        self.assertEqual(len(rows), 30)
        self.assertEqual(rows, run_benchmark(self.config, horizons=(30, 90, 365)))
        capsule_rows = [row for row in rows if row["policy"] == "capsule"]
        self.assertEqual(len(capsule_rows), 6)
        self.assertTrue(
            all(row["replay_sufficient_pct"] == 100.0 for row in capsule_rows)
        )
        full_rows = [row for row in rows if row["policy"] == "full-artifact-dedup"]
        self.assertTrue(all(row["artifact_complete_pct"] == 100.0 for row in full_rows))
        self.assertTrue(all(row["replay_sufficient_pct"] == 0.0 for row in full_rows))
        mlflow_rows = [row for row in rows if row["policy"] == "mlflow-reference"]
        self.assertEqual(
            [row["artifact_coverage_pct"] for row in mlflow_rows[:3]],
            [25.0, 0.0, 0.0],
        )
        self.assertTrue(
            all(row["control_coverage_pct"] == 33.333 for row in mlflow_rows)
        )

    def test_invalid_horizons_are_rejected_instead_of_coerced(self) -> None:
        for horizons in ((), (0,), (True,), (30.5,)):
            with self.subTest(horizons=horizons), self.assertRaises(ValueError):
                run_benchmark(self.config, horizons=horizons)  # type: ignore[arg-type]

    def test_written_artifacts_are_self_verifying_and_explicitly_structural(
        self,
    ) -> None:
        demo_config = WorkloadConfig(seed=20260801, requests_per_regime=96)
        rows = run_benchmark(demo_config)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            manifest = write_results(
                output,
                rows=rows,
                config=demo_config,
                horizons=(30, 90, 365),
            )

            csv_bytes = (output / "results.csv").read_bytes()
            markdown_bytes = (output / "results.md").read_bytes()
            parsed_manifest = json.loads((output / "manifest.json").read_text())
            parsed_rows = list(csv.DictReader(io.StringIO(csv_bytes.decode())))

            self.assertEqual(manifest, parsed_manifest)
            self.assertEqual(len(parsed_rows), 30)
            self.assertEqual(
                hashlib.sha256(csv_bytes).hexdigest(),
                manifest["results_csv_sha256"],
            )
            self.assertEqual(
                hashlib.sha256(markdown_bytes).hexdigest(),
                manifest["results_markdown_sha256"],
            )
            self.assertIn("virtual artifact sizes", markdown_bytes.decode())
            self.assertIn("no model was executed", manifest["interpretation"])
            self.assertEqual(manifest["measurement_scope"]["model_execution"], "none")
            self.assertLess(
                manifest["workload_summaries"]["burst"]["elapsed_s"],
                manifest["workload_summaries"]["poisson"]["elapsed_s"],
            )

    def test_demo_command_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({"seed": 11, "requests_per_regime": 2}),
                encoding="utf-8",
            )
            output = root / "results"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                main(
                    [
                        "demo",
                        "--config",
                        str(config_path),
                        "--output",
                        str(output),
                    ]
                )

            self.assertIn("wrote 30 result rows", stdout.getvalue())
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"manifest.json", "results.csv", "results.md"},
            )


if __name__ == "__main__":
    unittest.main()
