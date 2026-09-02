"""One-shot local pipeline sanity check: data prep -> real TF-IDF baseline -> tiny
CPU transformer training run -> evaluation -> inference, to catch bugs before
spending any Kaggle GPU quota on a real run.

Usage:
    python scripts/smoke_test.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SMOKE_CONFIG = "Fine_Grained"
SMOKE_OUTPUT_DIR = ROOT / "outputs" / f"smoke-test-{SMOKE_CONFIG.lower()}"


def run(*args):
    cmd = [sys.executable, *args]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=SRC, check=True)


def main():
    if not (ROOT / "data" / "processed" / SMOKE_CONFIG / "dataset_dict.json").exists():
        run("data_prep.py")

    run("train_baseline.py", "--config", SMOKE_CONFIG)
    run("train_classifier.py", "--config", SMOKE_CONFIG, "--smoke-test")
    run(
        "evaluate.py",
        "--model-dir",
        str(SMOKE_OUTPUT_DIR),
        "--config",
        SMOKE_CONFIG,
        "--split",
        "validation",
        # Scratch path -- never overwrite the checked-in results/metrics.json with
        # fake smoke-test numbers.
        "--output",
        str(SMOKE_OUTPUT_DIR / "metrics.json"),
    )
    run(
        "inference.py",
        "--model-dir",
        str(SMOKE_OUTPUT_DIR),
        "--text",
        "Yeh tou bohat hi bakwaas cheez hai yaar",
    )
    print("\nSmoke test passed -- the pipeline is wired up correctly.")


if __name__ == "__main__":
    main()
