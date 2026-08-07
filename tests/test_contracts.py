"""Tests for fail-closed data contracts."""

import unittest

from temporal_alignment.contracts import DatasetManifest, HistoricalYear, TemporalInterval
from temporal_alignment.enums import AssetStatus, DataRole, Era
from temporal_alignment.errors import ContractError, EvidenceGateError


class HistoricalYearTests(unittest.TestCase):
    def test_astronomical_boundary(self) -> None:
        self.assertEqual(HistoricalYear(Era.BCE, 1).astronomical, 0)
        self.assertEqual(HistoricalYear(Era.BCE, 2).astronomical, -1)
        self.assertEqual(HistoricalYear(Era.CE, 1).astronomical, 1)

    def test_round_trip(self) -> None:
        for value in (-100, -1, 0, 1, 2025):
            self.assertEqual(HistoricalYear.from_astronomical(value).astronomical, value)

    def test_reject_year_zero(self) -> None:
        with self.assertRaises(ContractError):
            HistoricalYear(Era.CE, 0)

    def test_interval_crosses_bce_ce_without_display_zero(self) -> None:
        interval = TemporalInterval.one_year(HistoricalYear(Era.BCE, 1))
        self.assertEqual(interval.end_exclusive, HistoricalYear(Era.CE, 1))


class DatasetManifestTests(unittest.TestCase):
    def test_hold_dataset_fails_gate(self) -> None:
        manifest = DatasetManifest(
            dataset_id="candidate",
            official_url="https://example.org",
            accessed_on="2026-08-07",
            license_or_terms="unknown",
            reviewer="reviewer",
            revision="v1",
            sha256="unknown",
            role=DataRole.PROBE,
            status=AssetStatus.HOLD,
        )
        with self.assertRaises(EvidenceGateError):
            manifest.assert_ingestible()

    def test_approved_dataset_requires_all_fields(self) -> None:
        manifest = DatasetManifest(
            dataset_id="bad-approved",
            official_url="",
            accessed_on="2026-08-07",
            license_or_terms="MIT",
            reviewer="reviewer",
            revision="v1",
            sha256="abc",
            role=DataRole.PROBE,
            status=AssetStatus.APPROVED,
        )
        with self.assertRaises(EvidenceGateError):
            manifest.assert_ingestible()


if __name__ == "__main__":
    unittest.main()
