"""Inference-Only Evaluation Runner for Kaggle GPU / Local Execution.

Does NOT launch any training.
Evaluates checkpoints across P00, P01, P10, P11 protocols on TempUN 180 MCQA items.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from temporal_alignment.models.run_m0_fc40 import build_toy_smoke_model


def run_inference_study():
    print("=" * 80)
    print("M0-FC40 FACTORED INFERENCE PROTOCOL STUDY (INFERENCE ONLY)")
    print("=" * 80)
    # Execution logic is handled by evaluate_m0_fc40_parity.py
    print("Executable runner registered at scripts/evaluate_m0_fc40_parity.py")


if __name__ == "__main__":
    run_inference_study()
