"""Runner Script for V2 Run 05.1 A/B Training."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from temporal_alignment.training_run05_1 import train_run05_1_arms

if __name__ == "__main__":
    print("Starting Run 05.1 A/B Training with 40 steps each...")
    report = train_run05_1_arms(repo_root=_REPO_ROOT, steps=40, device_name="cpu")
    print("\nTraining completed successfully!")
    print("Verification verdict:", report["divergence_verification"])
