"""Tests for the Vietnam-first planning scaffold and fail-closed gates."""

import unittest

from temporal_alignment.application import TemporalQARequest, TemporalQAResponse
from temporal_alignment.contracts import (
    DatasetManifest,
    HistoricalDate,
    HistoricalYear,
    TemporalInterval,
)
from temporal_alignment.data_management import DataLifecyclePlan
from temporal_alignment.enums import (
    AssetStatus,
    DataRole,
    Era,
    Granularity,
    TemporalKind,
)
from temporal_alignment.errors import ContractError, EvidenceGateError
from temporal_alignment.normalization import parse_explicit_vietnamese_year
from temporal_alignment.workflow import ResearchWorkflowPlan, default_research_stages


class VietnameseNormalizationTests(unittest.TestCase):
    """Verify that year parsing never guesses an omitted era."""

    def test_explicit_tcn_and_scn(self) -> None:
        self.assertEqual(parse_explicit_vietnamese_year("111 TCN"), HistoricalYear(Era.BCE, 111))
        self.assertEqual(
            parse_explicit_vietnamese_year("Năm 938 sau Công nguyên"),
            HistoricalYear(Era.CE, 938),
        )

    def test_ambiguous_year_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            parse_explicit_vietnamese_year("năm 0")
        with self.assertRaises(ContractError):
            parse_explicit_vietnamese_year("năm 938")


class LifecycleAndApplicationTests(unittest.TestCase):
    """Verify that data collection and application output require evidence."""

    def test_hold_manifest_blocks_collection(self) -> None:
        manifest = DatasetManifest(
            dataset_id="held",
            official_url="https://example.org",
            accessed_on="2026-08-23",
            license_or_terms="unknown",
            reviewer="project-team",
            revision="paper-only",
            sha256="not-collected",
            role=DataRole.PROBE,
            status=AssetStatus.HOLD,
        )
        plan = DataLifecyclePlan(manifest, ("event_id",), "review yearly", "project-team")
        with self.assertRaises(EvidenceGateError):
            plan.assert_collection_ready()

    def test_answer_requires_evidence(self) -> None:
        with self.assertRaises(ContractError):
            TemporalQAResponse("q1", "938 SCN", (), 0.8)
        abstained = TemporalQAResponse("q1", None, (), 0.0, abstention_reason="thiếu nguồn")
        self.assertEqual(abstained.abstention_reason, "thiếu nguồn")

    def test_temporal_contract_requires_both_axes(self) -> None:
        interval = TemporalInterval(
            HistoricalDate(HistoricalYear(Era.CE, 1945), 9, 2),
            HistoricalDate(HistoricalYear(Era.CE, 1945), 9, 3),
        )
        request = TemporalQARequest(
            "q2",
            "Sự kiện xảy ra khi nào?",
            "vi",
            interval,
            Granularity.DAY,
            TemporalKind.POINT,
        )
        self.assertEqual(request.target_kind, TemporalKind.POINT)
        with self.assertRaises(ContractError):
            TemporalQARequest("q3", "Khi nào?", "vi", interval)


class WorkflowPlanTests(unittest.TestCase):
    """Verify the minimum lifecycle sequence."""

    def test_default_plan_is_complete(self) -> None:
        plan = ResearchWorkflowPlan(
            primary_use_case="Vietnamese historical temporal QA",
            stages=default_research_stages(),
        )
        plan.assert_complete()


if __name__ == "__main__":
    unittest.main()
