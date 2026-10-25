"""Quick sanity check of the data pipeline: sizes, scaling, stratification and leakage.

Usage:  python scripts/00_check_splits.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import make_splits, check_splits  # noqa: E402


def main():
    S = make_splits()
    report = check_splits(S)
    for key, value in report.items():
        print(f"{key:>22}: {value}")
    print("All split checks passed.")


if __name__ == "__main__":
    main()
