"""Executable Inference-Only Protocol Runner for M0-FC40 Checkpoints & Base Model.

Synchronized with scripts/evaluate_m0_fc40_parity.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Forward execution to scripts/evaluate_m0_fc40_parity.py
from evaluate_m0_fc40_parity import main

if __name__ == "__main__":
    main()
