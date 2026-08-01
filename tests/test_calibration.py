from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tracebench.calibration import derive_calibration, write_calibration
from tracebench.model import TraceBenchError


class CalibrationTests(unittest.TestCase):
    def write_csv(
        self, path: Path, fieldnames: list[str], rows: list[dict[str, object]]
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(
                destination, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)

    def make_fixture_traces(self, root: Path) -> tuple[Path, Path, Path]:
        burst = root / "burst.csv"
        self.write_csv(
            burst,
            ["Timestamp", "Request tokens", "Response tokens"],
            [
                {"Timestamp": 0, "Request tokens": 16, "Response tokens": 4},
                {"Timestamp": 1, "Request tokens": 32, "Response tokens": 8},
                {"Timestamp": 100, "Request tokens": 64, "Response tokens": 16},
            ],
        )
        azure_code = root / "code.csv"
        azure_conversation = root / "conversation.csv"
        for path, offset in ((azure_code, 0), (azure_conversation, 10)):
            self.write_csv(
                path,
                ["TIMESTAMP", "ContextTokens", "GeneratedTokens"],
                [
                    {
                        "TIMESTAMP": f"2023-11-16 18:17:{offset:02d}.0000000",
                        "ContextTokens": 128,
                        "GeneratedTokens": 32,
                    },
                    {
                        "TIMESTAMP": f"2023-11-16 18:17:{offset + 1:02d}.0000000",
                        "ContextTokens": 256,
                        "GeneratedTokens": 64,
                    },
                ],
            )
        return burst, azure_code, azure_conversation

    def test_calibration_is_deterministic_and_binds_source_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            burst, code, conversation = self.make_fixture_traces(root)

            first = derive_calibration(
                burstgpt_path=burst,
                azure_code_path=code,
                azure_conversation_path=conversation,
            )
            second = derive_calibration(
                burstgpt_path=burst,
                azure_code_path=code,
                azure_conversation_path=conversation,
            )

            self.assertEqual(first, second)
            self.assertEqual(first["schema"], "tracebench/public-trace-calibration/v1")
            expected_hashes = [
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (burst, code, conversation)
            ]
            self.assertEqual(
                [source["sha256"] for source in first["sources"]],
                expected_hashes,
            )
            self.assertGreater(first["derived_demo_parameters"]["base_rate_rps"], 0)
            self.assertIn(
                "8345c824bf744e21692186af2835521ba75e5f6d",
                first["sources"][0]["url"],
            )

            output = root / "calibration.json"
            write_calibration(output, first)
            self.assertEqual(json.loads(output.read_text()), first)

    def test_checked_in_demo_matches_checked_in_calibration(self) -> None:
        root = Path(__file__).resolve().parents[1]
        demo = json.loads((root / "configs" / "demo.json").read_text())
        calibration = json.loads(
            (root / "configs" / "trace-calibration.json").read_text()
        )

        for name, value in calibration["derived_demo_parameters"].items():
            with self.subTest(parameter=name):
                self.assertEqual(demo[name], value)

    def test_missing_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            burst, code, conversation = self.make_fixture_traces(root)
            burst.write_text("Timestamp,Request tokens\n0,1\n1,2\n", encoding="utf-8")

            with self.assertRaisesRegex(TraceBenchError, "missing required columns"):
                derive_calibration(
                    burstgpt_path=burst,
                    azure_code_path=code,
                    azure_conversation_path=conversation,
                )


if __name__ == "__main__":
    unittest.main()
