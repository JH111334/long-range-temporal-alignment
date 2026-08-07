"""Calendar conversion and sexagenary indexing."""

from .contracts import HistoricalYear


def published_sexagenary_index(year: HistoricalYear) -> int:
    """Implement Equation (1) exactly as printed, using signed BCE/CE years."""
    signed = year.year if year.era.value == "CE" else -year.year
    if signed < 0:
        return (60 - abs(signed) - 2) % 60
    if 0 < signed < 4:
        return (60 - abs(signed - 3)) % 60
    return (signed - 3) % 60


def anchored_cycle_index(
    year: HistoricalYear,
    *,
    anchor_year: int = 1984,
    anchor_index: int = 0,
    period: int = 60,
) -> int:
    """Return a documented modulo index under astronomical numbering."""
    if period < 2:
        raise ValueError("period must be >= 2")
    return (year.astronomical - anchor_year + anchor_index) % period
