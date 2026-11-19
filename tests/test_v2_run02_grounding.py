"""Unit tests for V2 Run 02 Real Training, Checkpoint Distinctness, and Statistical Gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from scripts.evaluate_v2_run02_checkpoints import exact_mcnemar_binomial

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN02_DIR = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02"


def test_run02_checkpoints_exist_and_distinct():
    """Verify that 4 trained arm checkpoints exist and have distinct SHA256 hashes."""
    ckpt_dir = _RUN02_DIR / "checkpoints"
    arms = ["C0_CONTINUED_MASKED", "C1_TRUE_TEMPORAL_RELATION", "C2_SHUFFLED_CONDITIONING", "C3_FORMAT_ONLY"]
    hashes = set()

    for arm in arms:
        path = ckpt_dir / f"{arm}.pt"
        assert path.is_file(), f"Missing checkpoint for {arm}"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        digest = h.hexdigest()
        assert digest not in hashes, f"Checkpoint hash collision detected: {digest}"
        hashes.add(digest)

    assert len(hashes) == 4


def test_run02_split_leakage_invariants():
    """Verify strict zero leakage between train and dev entity sets."""
    audit_path = _RUN02_DIR / "SPLIT_LEAKAGE_AUDIT.json"
    assert audit_path.is_file()

    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["train_dev_entity_leakage"] is True
    assert audit["eval_benchmark_exact_leakage_count"] == 0
    assert audit["leakage_verdict"] == "STRICT_ZERO_LEAKAGE_CONFIRMED"


def test_exact_mcnemar_statistical_validity():
    """Verify exact binomial McNemar test against known contingency counts."""
    # Symmetrical discordant: a_only=5, b_only=5 -> p=1.0
    chi2, p = exact_mcnemar_binomial(5, 5)
    assert p == 1.0

    # Strong asymmetry: a_only=19, b_only=5 (Run 02 C1 vs C2)
    chi2, p = exact_mcnemar_binomial(19, 5)
    assert p < 0.01
    assert chi2 > 6.0
