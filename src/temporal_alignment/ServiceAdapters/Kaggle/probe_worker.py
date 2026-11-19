"""Minimal Kaggle GPU probe; it does not read project data."""

from __future__ import annotations

import json
import platform
from pathlib import Path


def main() -> None:
    """Write a small environment report without installing dependencies."""
    report: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": False,
        "gpu": None,
    }
    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        report["torch_status"] = "not_installed"
    Path("gpu_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
