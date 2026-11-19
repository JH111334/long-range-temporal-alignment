"""Regression tests for repository validation helpers."""

import tempfile
import unittest
from pathlib import Path

from scripts.validate_project import (
    contains_mojibake,
    general_log_violations,
    html_contract_violations,
)


class TextEncodingValidationTests(unittest.TestCase):
    """Distinguish valid Vietnamese text from common encoding damage."""

    def test_valid_vietnamese_a_circumflex_is_not_mojibake(self) -> None:
        self.assertFalse(contains_mojibake("Âm lịch Việt Nam"))

    def test_common_broken_tone_sequence_is_mojibake(self) -> None:
        broken = "".join(chr(codepoint) for codepoint in (0x00E1, 0x00BB))
        self.assertTrue(contains_mojibake(broken))

    def test_complete_html_contract_has_no_violations(self) -> None:
        html = "<!doctype html><html><meta charset='utf-8'><title>x</title></html>"
        self.assertEqual(html_contract_violations(html), ())

    def test_incomplete_html_contract_reports_missing_markers(self) -> None:
        violations = html_contract_violations("<html><title>x</title></html>")
        self.assertIn("<!doctype html", violations)
        self.assertIn("<meta charset=", violations)

    def test_general_log_validator_rejects_version_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            general = Path(temporary)
            (general / "RULE.md").write_text("rule", encoding="utf-8")
            content = "Project: x\nThời điểm: x\nPhiên bản: x\nMức bằng chứng: x\nKết quả: PASS\n"
            (general / "long-range-temporal-alignment_Checkpoints_25-08-26_V2.md").write_text(
                content, encoding="utf-8"
            )
            violations = general_log_violations(general)
        self.assertTrue(any("continuous" in item for item in violations))

    def test_general_log_validator_accepts_complete_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            general = Path(temporary)
            content = (
                "Project: x\nThời điểm: x\nPhiên bản: V1\nExpected: x\nActual: y\n"
                "Nguyên nhân: confirmed\nKhắc phục: z\nRegression: PASS\n"
            )
            (general / "long-range-temporal-alignment_Error_25-08-26_V1.md").write_text(
                content, encoding="utf-8"
            )
            violations = general_log_violations(general)
        self.assertEqual(violations, ())

    def test_general_log_validator_rejects_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            general = Path(temporary)
            (general / "nested").mkdir()
            violations = general_log_violations(general)
        self.assertIn("non-file entry in Documents/General: nested", violations)


if __name__ == "__main__":
    unittest.main()
