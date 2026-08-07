"""Tests exposing published-equation behavior and candidate invariants."""

import unittest

from temporal_alignment.contracts import HistoricalYear
from temporal_alignment.encoders import (
    FourierTimeEncoder,
    GregorianScalarEncoder,
    HybridCycleMonotonicEncoder,
    OperationalRadialTicktackEncoder,
    PublishedTicktackEncoder,
)
from temporal_alignment.enums import Era


class EncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.year_1965 = HistoricalYear(Era.CE, 1965)
        self.year_2025 = HistoricalYear(Era.CE, 2025)

    def test_published_equations_collide_at_sixty_years(self) -> None:
        encoder = PublishedTicktackEncoder()
        self.assertEqual(encoder.encode(self.year_1965), encoder.encode(self.year_2025))

    def test_operational_radial_assumption_breaks_alias(self) -> None:
        encoder = OperationalRadialTicktackEncoder()
        self.assertNotEqual(encoder.encode(self.year_1965), encoder.encode(self.year_2025))

    def test_hybrid_preserves_nonperiodic_axis(self) -> None:
        encoder = HybridCycleMonotonicEncoder()
        first = encoder.encode(self.year_1965)
        second = encoder.encode(self.year_2025)
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(len(first), 9)

    def test_fourier_dimension(self) -> None:
        self.assertEqual(len(FourierTimeEncoder(periods=(10, 60)).encode(self.year_1965)), 4)

    def test_scalar_is_monotonic(self) -> None:
        encoder = GregorianScalarEncoder()
        self.assertLess(encoder.encode(self.year_1965)[0], encoder.encode(self.year_2025)[0])


if __name__ == "__main__":
    unittest.main()
