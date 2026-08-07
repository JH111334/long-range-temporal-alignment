"""Tests for objective semantics and reproducibility risks."""

import unittest

from temporal_alignment.errors import ContractError
from temporal_alignment.objectives import (
    ewc_penalty,
    normalized_alignment_loss,
    published_ticktack_loss,
)


class ObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vectors = [(1.0, 0.0), (0.8, 0.2), (0.5, 0.5), (0.2, 0.8)]
        self.labels = [0, 0, 1, 1]

    def test_published_sum_changes_when_batch_is_duplicated(self) -> None:
        base = published_ticktack_loss(self.vectors, self.labels, delta=0.5)
        duplicated = published_ticktack_loss(
            self.vectors + self.vectors,
            self.labels + self.labels,
            delta=0.5,
        )
        self.assertNotAlmostEqual(base.total, duplicated.total)

    def test_normalized_loss_is_stable_under_full_duplication(self) -> None:
        vectors = [(1.0, 0.0), (1.0, 0.0), (0.6, 0.8), (0.6, 0.8)]
        labels = [0, 0, 1, 1]
        base = normalized_alignment_loss(vectors, labels)
        duplicated = normalized_alignment_loss(vectors + vectors, labels + labels)
        self.assertAlmostEqual(base.total, duplicated.total)

    def test_delta_one_discards_inter_term(self) -> None:
        result = published_ticktack_loss(self.vectors, self.labels, delta=1.0)
        self.assertAlmostEqual(result.total, result.intra)
        self.assertGreater(result.inter_pairs, 0)

    def test_ewc_requires_explicit_trainable_parameters(self) -> None:
        with self.assertRaises(ContractError):
            ewc_penalty({}, {}, {}, coefficient=1.0, trainable_parameters=())

    def test_ewc_diagonal_penalty(self) -> None:
        penalty = ewc_penalty(
            {"adapter": (2.0, 4.0)},
            {"adapter": (1.0, 2.0)},
            {"adapter": (2.0, 0.5)},
            coefficient=3.0,
            trainable_parameters=("adapter",),
        )
        self.assertAlmostEqual(penalty, 6.0)


if __name__ == "__main__":
    unittest.main()
