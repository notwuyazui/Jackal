"""Compatibility entry for the unified model evaluator."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.test_model import evaluate, main, parse_args

__all__ = ["evaluate", "main", "parse_args"]


if __name__ == "__main__":
    main()
