"""Deterministic group-aware split utilities."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable

from .contracts import TemporalExample
from .errors import ContractError


def assign_group_split(
    group_id: str,
    *,
    seed: str = "temporal-alignment-v1",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> str:
    """Assign all members of a group to one stable split."""
    if not group_id.strip():
        raise ContractError("group_id must not be empty")
    if not seed.strip():
        raise ContractError("seed must not be empty")
    if not 0.0 < train_ratio < 1.0 or not 0.0 <= validation_ratio < 1.0:
        raise ContractError("invalid split ratios")
    if train_ratio + validation_ratio >= 1.0:
        raise ContractError("train_ratio + validation_ratio must be < 1")
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + validation_ratio:
        return "validation"
    return "test"


def split_examples(
    examples: Iterable[TemporalExample], *, seed: str
) -> dict[str, list[TemporalExample]]:
    splits: dict[str, list[TemporalExample]] = defaultdict(list)
    for example in examples:
        splits[assign_group_split(example.group_id, seed=seed)].append(example)
    validate_group_isolation(splits)
    return dict(splits)


def validate_group_isolation(splits: dict[str, list[TemporalExample]]) -> None:
    owners: dict[str, str] = {}
    for split_name, examples in splits.items():
        for example in examples:
            prior = owners.setdefault(example.group_id, split_name)
            if prior != split_name:
                raise ContractError(
                    f"group {example.group_id} leaks between {prior} and {split_name}"
                )
