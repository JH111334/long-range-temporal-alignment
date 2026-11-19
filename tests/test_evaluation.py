"""Tests for representation and output-format evaluation."""

import unittest

from temporal_alignment.contracts import HistoricalYear
from temporal_alignment.encoders import (
    HybridCycleMonotonicEncoder,
    PublishedTicktackEncoder,
)
from temporal_alignment.enums import Era
from temporal_alignment.evaluation import (
    alias_collision_rate,
    invalid_output_rate,
    parse_choice,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.years = [HistoricalYear(Era.CE, value) for value in (1965, 2025)]

    def test_alias_collision_is_reported(self) -> None:
        self.assertEqual(alias_collision_rate(self.years, PublishedTicktackEncoder().encode), 1.0)
        self.assertEqual(
            alias_collision_rate(self.years, HybridCycleMonotonicEncoder().encode), 0.0
        )

    def test_choice_parser_rejects_ambiguous_output(self) -> None:
        self.assertEqual(parse_choice("Đáp án: B"), "B")
        self.assertIsNone(parse_choice("Có thể A hoặc B"))

    def test_invalid_output_rate_is_separate_metric(self) -> None:
        self.assertEqual(invalid_output_rate(["A", "không rõ", "B hoặc C"]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
