"""Tests for deterministic group-aware splitting."""

import unittest

from temporal_alignment.splits import assign_group_split


class SplitTests(unittest.TestCase):
    def test_assignment_is_deterministic(self) -> None:
        first = assign_group_split("event-123", seed="fixed")
        second = assign_group_split("event-123", seed="fixed")
        self.assertEqual(first, second)

    def test_assignment_returns_known_split(self) -> None:
        self.assertIn(assign_group_split("source-abc"), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
