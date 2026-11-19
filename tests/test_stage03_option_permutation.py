"""Unit tests for Stage 03 Online Option Permutation."""

from __future__ import annotations

import pytest

from temporal_alignment.training.option_permutation import OnlineOptionPermutation


def test_default_off_is_identity():
    perm = OnlineOptionPermutation(enabled=False)
    options = {
        "A": "Năm 1945",
        "B": "Năm 1954",
        "C": "Năm 1975",
        "D": "Năm 1986",
    }
    gold = "C"

    new_opts, new_gold, fwd, inv = perm.permute_mcqa_options(options, gold)
    assert new_opts == options
    assert new_gold == gold
    assert fwd == {"A": "A", "B": "B", "C": "C", "D": "D"}
    assert inv == {"A": "A", "B": "B", "C": "C", "D": "D"}


def test_permutation_bijection_and_inversion_on_4_samples():
    perm = OnlineOptionPermutation(enabled=True)

    samples = [
        ({"A": "Opt 1", "B": "Opt 2", "C": "Opt 3", "D": "Opt 4"}, "A"),
        ({"A": "Warsaw 1955", "B": "NATO 1949", "C": "UN 1945", "D": "EU 1993"}, "B"),
        ({"A": "Đổi mới 1986", "B": "Chiến dịch HCM 1975", "C": "Điện Biên Phủ 1954", "D": "Cách mạng T8 1945"}, "D"),
        ({"A": "Khởi nghĩa Lam Sơn", "B": "Bạch Đằng", "C": "Chi Lăng", "D": "Đống Đa"}, "C"),
    ]

    for idx, (opts, gold) in enumerate(samples):
        new_opts, new_gold, fwd, inv = perm.permute_mcqa_options(opts, gold, seed=idx + 42)

        # Check all original keys exist
        assert set(new_opts.keys()) == set(opts.keys())

        # Check bijection
        for k in opts.keys():
            assert inv[fwd[k]] == k
            assert fwd[inv[k]] == k

        # Check target semantic text is exactly preserved
        assert new_opts[new_gold] == opts[gold]


def test_deterministic_reproducibility():
    perm1 = OnlineOptionPermutation(enabled=True, seed=100)
    perm2 = OnlineOptionPermutation(enabled=True, seed=100)

    options = {"A": "First", "B": "Second", "C": "Third", "D": "Fourth"}
    gold = "B"

    opts1, gold1, fwd1, inv1 = perm1.permute_mcqa_options(options, gold, seed=999)
    opts2, gold2, fwd2, inv2 = perm2.permute_mcqa_options(options, gold, seed=999)

    assert opts1 == opts2
    assert gold1 == gold2
    assert fwd1 == fwd2
    assert inv1 == inv2
