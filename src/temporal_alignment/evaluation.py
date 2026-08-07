"""Representation and output-format metrics for controlled evaluations."""

from __future__ import annotations

import math
import re
from itertools import combinations
from typing import Callable, Sequence

from .contracts import HistoricalYear
from .errors import ContractError
from .objectives import cosine_similarity

Vector = Sequence[float]


def euclidean_distance(left: Vector, right: Vector) -> float:
    if len(left) != len(right) or not left:
        raise ContractError("vectors must have equal non-zero dimensions")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def alias_collision_rate(
    years: Sequence[HistoricalYear],
    encode: Callable[[HistoricalYear], Vector],
    *,
    period: int = 60,
    tolerance: float = 1e-12,
) -> float:
    """Measure exact/near collisions for distinct years separated by a period."""
    pairs = [
        (left, right)
        for left, right in combinations(years, 2)
        if abs(left.astronomical - right.astronomical) % period == 0
    ]
    if not pairs:
        return 0.0
    collisions = sum(
        euclidean_distance(encode(left), encode(right)) <= tolerance for left, right in pairs
    )
    return collisions / len(pairs)


def temporal_distance_correlation(
    years: Sequence[HistoricalYear],
    encode: Callable[[HistoricalYear], Vector],
) -> float:
    """Pearson correlation between chronological and representation distances."""
    chronological: list[float] = []
    represented: list[float] = []
    for left, right in combinations(years, 2):
        chronological.append(float(abs(left.astronomical - right.astronomical)))
        represented.append(euclidean_distance(encode(left), encode(right)))
    if len(chronological) < 2:
        return 0.0
    return _pearson(chronological, represented)


def mean_off_diagonal_similarity(vectors: Sequence[Vector]) -> float:
    pairs = [cosine_similarity(vectors[i], vectors[j]) for i, j in combinations(range(len(vectors)), 2)]
    return sum(pairs) / len(pairs) if pairs else 0.0


def parse_choice(output: str, allowed: Sequence[str] = ("A", "B", "C", "D")) -> str | None:
    """Parse one multiple-choice label without silently accepting prose ambiguity."""
    matches = re.findall(r"(?<![A-Z])([A-Z])(?![A-Z])", output.upper())
    valid = [match for match in matches if match in allowed]
    return valid[0] if len(set(valid)) == 1 else None


def invalid_output_rate(outputs: Sequence[str], allowed: Sequence[str] = ("A", "B", "C", "D")) -> float:
    if not outputs:
        raise ContractError("outputs must not be empty")
    invalid = sum(parse_choice(output, allowed) is None for output in outputs)
    return invalid / len(outputs)


def exact_match_accuracy(predictions: Sequence[str | None], references: Sequence[str]) -> float:
    if len(predictions) != len(references) or not predictions:
        raise ContractError("predictions and references must have equal non-zero length")
    return sum(prediction == reference for prediction, reference in zip(predictions, references, strict=True)) / len(references)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)
