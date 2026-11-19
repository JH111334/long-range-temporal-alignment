"""Published and normalized temporal-alignment objectives."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from .errors import ContractError

Vector = Sequence[float]


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Return cosine similarity and reject incompatible or zero vectors."""
    if len(left) != len(right) or not left:
        raise ContractError("vectors must have equal non-zero dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ContractError("cosine similarity is undefined for a zero vector")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


@dataclass(frozen=True, slots=True)
class AlignmentLoss:
    """Inspectable breakdown of a temporal alignment loss."""

    total: float
    intra: float
    inter: float
    intra_pairs: int
    inter_pairs: int


def published_ticktack_loss(
    vectors: Sequence[Vector],
    labels: Sequence[int],
    *,
    delta: float = 1.0,
    include_self_pairs: bool = True,
) -> AlignmentLoss:
    """Implement Equation (8) as printed, including unnormalized ordered sums.

    The paper does not state whether self-pairs are excluded. The default keeps
    them; callers must record any alternative as an operational assumption.
    """
    _validate_batch(vectors, labels, delta)
    intra_sum = 0.0
    inter_sum = 0.0
    intra_pairs = 0
    inter_pairs = 0
    for left_index, left in enumerate(vectors):
        for right_index, right in enumerate(vectors):
            if left_index == right_index and not include_self_pairs:
                continue
            similarity = cosine_similarity(left, right)
            if labels[left_index] == labels[right_index]:
                intra_sum += similarity
                intra_pairs += 1
            else:
                inter_sum += similarity
                inter_pairs += 1
    intra = 1.0 - intra_sum
    inter = inter_sum
    return AlignmentLoss(
        total=delta * intra + (1.0 - delta) * inter,
        intra=intra,
        inter=inter,
        intra_pairs=intra_pairs,
        inter_pairs=inter_pairs,
    )


def normalized_alignment_loss(
    vectors: Sequence[Vector],
    labels: Sequence[int],
    *,
    delta: float = 0.5,
    inter_margin: float = 0.0,
) -> AlignmentLoss:
    """Average unique pairs to reduce class-count and batch-size dependence."""
    _validate_batch(vectors, labels, delta)
    intra_terms: list[float] = []
    inter_terms: list[float] = []
    for left_index, right_index in combinations(range(len(vectors)), 2):
        similarity = cosine_similarity(vectors[left_index], vectors[right_index])
        if labels[left_index] == labels[right_index]:
            intra_terms.append(1.0 - similarity)
        else:
            inter_terms.append(max(0.0, similarity - inter_margin))
    intra = _mean(intra_terms)
    inter = _mean(inter_terms)
    return AlignmentLoss(
        total=delta * intra + (1.0 - delta) * inter,
        intra=intra,
        inter=inter,
        intra_pairs=len(intra_terms),
        inter_pairs=len(inter_terms),
    )


def ewc_penalty(
    current: Mapping[str, Sequence[float]],
    reference: Mapping[str, Sequence[float]],
    fisher_diagonal: Mapping[str, Sequence[float]],
    *,
    coefficient: float,
    trainable_parameters: Iterable[str],
) -> float:
    """Calculate diagonal EWC only for explicitly named trainable parameters."""
    if coefficient < 0:
        raise ContractError("EWC coefficient must be non-negative")
    selected = tuple(trainable_parameters)
    if not selected:
        raise ContractError("EWC requires explicit trainable parameter names")
    total = 0.0
    for name in selected:
        if name not in current or name not in reference or name not in fisher_diagonal:
            raise ContractError(f"missing EWC state for parameter: {name}")
        values = current[name]
        old_values = reference[name]
        fisher = fisher_diagonal[name]
        if not (len(values) == len(old_values) == len(fisher)):
            raise ContractError(f"incompatible EWC shapes for parameter: {name}")
        total += sum(
            importance * (value - old_value) ** 2
            for value, old_value, importance in zip(values, old_values, fisher, strict=True)
        )
    return 0.5 * coefficient * total


def _validate_batch(vectors: Sequence[Vector], labels: Sequence[int], delta: float) -> None:
    if len(vectors) != len(labels) or len(vectors) < 2:
        raise ContractError("vectors and labels must have the same length >= 2")
    if not 0.0 <= delta <= 1.0:
        raise ContractError("delta must be in [0, 1]")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
