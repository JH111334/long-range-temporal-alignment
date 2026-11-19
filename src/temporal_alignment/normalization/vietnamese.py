"""Normalize only explicit Vietnamese BCE/CE year labels at Tier 0."""

from __future__ import annotations

import re
from enum import StrEnum

from ..contracts import HistoricalYear
from ..enums import Era
from ..errors import ContractError


class VietnameseCalendarSystem(StrEnum):
    """Track calendar systems that need distinct provenance and conversion."""

    GREGORIAN = "gregorian"
    LUNAR = "vietnamese_lunar"
    REIGN_YEAR = "reign_year"
    SEXAGENARY = "sexagenary"


_YEAR_PATTERN = re.compile(
    r"^\s*(?:năm\s+)?(?P<year>[1-9]\d*)\s*(?P<era>tcn|trước\s+công\s+nguyên|scn|sau\s+công\s+nguyên)\s*$",
    flags=re.IGNORECASE,
)


def parse_explicit_vietnamese_year(value: str) -> HistoricalYear:
    """Parse an explicit Vietnamese BCE/CE label without assuming an era.

    Raises:
        ContractError: If the label omits an era or uses an unsupported calendar.
    """
    match = _YEAR_PATTERN.fullmatch(value)
    if match is None:
        raise ContractError(
            "year label must state TCN/SCN or trước/sau Công nguyên; "
            "lunar, reign-year, and sexagenary conversion remain scaffold-only"
        )
    era_text = re.sub(r"\s+", " ", match.group("era").strip().casefold())
    era = Era.BCE if era_text in {"tcn", "trước công nguyên"} else Era.CE
    return HistoricalYear(era=era, year=int(match.group("year")))
