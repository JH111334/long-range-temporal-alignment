import os
import re
import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()
    dest = Path("artifacts/quarantine/M0_FC40/kaggle_3seed").resolve()
    dest.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading JSON/LOG artifacts from qh0205/lrr-fc105-m0-fc40-3seed to {dest}...")
    files, _ = api.kernels_output(
        "qh0205/lrr-fc105-m0-fc40-3seed",
        path=str(dest),
        file_pattern=r"\.(json|log)$",
        force=True,
        quiet=False
    )
    print(f"Successfully downloaded {len(files)} files.")

if __name__ == "__main__":
    main()
