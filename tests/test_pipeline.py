"""End-to-end smoke pipeline tests."""

import json
import tempfile
import unittest
from pathlib import Path

from temporal_alignment.pipeline import run_smoke_pipeline

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_smoke_pipeline_emits_labeled_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "run.json"
            result = run_smoke_pipeline(ROOT / "configs" / "smoke.json", output)
            persisted = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["evidence_class"], "smoke_test")
        self.assertEqual(result, persisted)
        self.assertEqual(result["example_count"], 7)
        reports = {item["encoder_name"]: item for item in result["encoders"]}
        self.assertEqual(reports["published_ticktack"]["alias_collision_rate_60"], 1.0)
        self.assertEqual(reports["hybrid"]["alias_collision_rate_60"], 0.0)


if __name__ == "__main__":
    unittest.main()
