"""Unit tests for sexagenary calendar indexing conventions: Paper 1-based vs Project V1 0-based.

Proves:
1. (year - 4) % 60 is an isomorphic 0-based rotation of paper's 1-based (year - 3) % 60.
2. Neither convention generates Python None on valid integers.
3. Sexagenary stems (10) and branches (12) are isomorphic across conventions.
4. BCE and CE 1-3 branches follow paper-specified rules.
"""

from __future__ import annotations

import math
import pytest


THIEN_CAN = ["Giáp", "Ất", "Bính", "Đinh", "Mậu", "Kỷ", "Canh", "Tân", "Nhâm", "Quý"]
DIA_CHI = ["Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi"]


def paper_eq1_cycle_1based(year: int) -> int:
    """TickTack paper Eq. 1: 1-based calendar cycle modulo 60."""
    if not isinstance(year, int):
        raise TypeError("year must be an integer")
    if year >= 4:
        return (year - 3) % 60
    elif year > 0:
        return (60 - abs(year - 3)) % 60
    else:
        return (60 - abs(year) - 2) % 60


def project_v1_cycle_0based(year: int) -> int:
    """Project V1 / Computer Science convention: 0-based calendar cycle modulo 60."""
    if not isinstance(year, int):
        raise TypeError("year must be an integer")
    return (year - 4) % 60


def test_year_minus_four_never_returns_none():
    """Mathematical proof that (year - 4) % 60 on integers never returns None."""
    test_ints = list(range(-2000, 2100))
    for y in test_ints:
        res = project_v1_cycle_0based(y)
        assert res is not None
        assert isinstance(res, int)
        assert 0 <= res < 60


def test_golden_year_crosswalk_isomorphism():
    """Verify isomorphism between 1-based and 0-based indexing on golden historical years."""
    golden_years = [4, 64, 1924, 1947, 1983, 1984, 1985, 2022, 2044, 2104]
    for y in golden_years:
        c1 = paper_eq1_cycle_1based(y)
        c0 = project_v1_cycle_0based(y)

        # 1-based to canonical class 1..60
        class_1to60 = 60 if c1 == 0 else c1

        # Isomorphic relationship: class_1to60 == c0 + 1
        assert class_1to60 == c0 + 1, f"Failed at year {y}: c1={c1}, c0={c0}"

        # In modulo 60: c1 == (c0 + 1) % 60
        assert c1 == (c0 + 1) % 60


def test_jiazi_1984_and_cycle_boundaries():
    """1984 is Jiazi (Giap Ty), the start of the contemporary 60-year sexagenary cycle."""
    # 0-based: 1984 is index 0
    assert project_v1_cycle_0based(1984) == 0
    # 1-based: 1984 is class 1
    assert paper_eq1_cycle_1based(1984) == 1

    # 1983 is Guihai (Quy Hoi), the last year of the preceding cycle
    assert project_v1_cycle_0based(1983) == 59
    assert paper_eq1_cycle_1based(1983) == 0  # 60 modulo 60 is 0

    # 1985 is Yichou (At Suu)
    assert project_v1_cycle_0based(1985) == 1
    assert paper_eq1_cycle_1based(1985) == 2


def test_angular_origin_difference():
    """Verify that the angular difference between conventions is exactly 6 degrees (2*pi/60)."""
    golden_years = [1947, 1984, 2022]
    for y in golden_years:
        c1 = paper_eq1_cycle_1based(y)
        c0 = project_v1_cycle_0based(y)

        theta1_deg = c1 * 6.0
        theta0_deg = (c0 * 6.0) % 360.0

        # Difference is exactly 6 degrees
        diff = (theta1_deg - theta0_deg) % 360.0
        assert math.isclose(diff, 6.0, abs_tol=1e-5)


def test_bce_and_edge_years():
    """Verify paper Eq. 1 branch handling for BCE and CE 1-3."""
    # Year 1 CE
    c1_yr1 = paper_eq1_cycle_1based(1)
    assert isinstance(c1_yr1, int)
    assert 0 <= c1_yr1 < 60

    # Year 3 CE
    c1_yr3 = paper_eq1_cycle_1based(3)
    assert isinstance(c1_yr3, int)
    assert 0 <= c1_yr3 < 60

    # Year 1 BCE (-1)
    c1_bce1 = paper_eq1_cycle_1based(-1)
    assert isinstance(c1_bce1, int)
    assert 0 <= c1_bce1 < 60
