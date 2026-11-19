"""Check presentation-neutral dashboard status."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from temporal_alignment.ui import (
    LineageError,
    build_dashboard_view,
    load_latest_checkpoint,
)


class DashboardViewTests(unittest.TestCase):
    def test_scaffold_never_claims_measured_results(self) -> None:
        view = build_dashboard_view()
        self.assertIn("Scaffold", view.warning)
        self.assertNotIn("measured", {gate.evidence_class for gate in view.gates})
        self.assertEqual(len(view.rows()), len(view.gates))
        checkpoint = Path(__file__).resolve().parents[1] / view.lineage.path
        self.assertTrue(checkpoint.is_file())
        self.assertIn(view.lineage.version, checkpoint.name)
        self.assertEqual(len(view.lineage.sha256), 64)
        self.assertEqual(
            view.lineage.sha256,
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )

    def test_lineage_selects_latest_version_and_hashes_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            general = root / "Documents" / "General"
            general.mkdir(parents=True)
            project_name = "long-range-temporal-alignment"
            first = (
                "- Project: long-range-temporal-alignment\n"
                "- Phiên bản: V1\n"
                "- Mức bằng chứng: ignored\n"
            ).encode()
            latest = (
                "- Project: long-range-temporal-alignment\n"
                "- Phiên bản: V2\n"
                "- Mức bằng chứng: `static_check`; không phải benchmark\n"
                "- Training: not-run\n"
            ).encode()
            (general / f"{project_name}_Checkpoints_25-08-26_V1.md").write_bytes(first)
            (general / f"{project_name}_Checkpoints_25-08-26_V2.md").write_bytes(latest)

            lineage = load_latest_checkpoint(root)

        self.assertEqual(lineage.version, "V2")
        self.assertEqual(lineage.sha256, hashlib.sha256(latest).hexdigest())
        self.assertIn("static_check", lineage.evidence_class)
        self.assertEqual(len(lineage.warnings), 2)

    def test_lineage_fails_closed_without_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(LineageError):
                load_latest_checkpoint(Path(temporary))

    def test_lineage_fails_closed_when_body_version_mismatches_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            general = root / "Documents" / "General"
            general.mkdir(parents=True)
            checkpoint = general / ("long-range-temporal-alignment_Checkpoints_25-08-26_V2.md")
            checkpoint.write_text(
                "- Project: long-range-temporal-alignment\n"
                "- Phiên bản: V1\n"
                "- Mức bằng chứng: static_check\n",
                encoding="utf-8",
            )

            with self.assertRaises(LineageError):
                load_latest_checkpoint(root)


if __name__ == "__main__":
    unittest.main()
