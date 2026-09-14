"""Canonical command-line entry point for all Jackal training algorithms."""

from __future__ import annotations

import argparse
from importlib import import_module
import sys
from typing import Sequence


TRAINERS = {
    "dqn": "training.DQN_train",
    "drqn": "training.DRQN_train",
    "qmix": "training.train_qmix_marl2",
}


def main(argv: Sequence[str] | None = None) -> None:
    cli_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m training",
        description="Train a Jackal agent with a supported algorithm.",
    )
    parser.add_argument("algorithm", choices=TRAINERS, nargs="?")
    if not cli_args or cli_args[0] in {"-h", "--help"}:
        parser.print_help()
        return
    algorithm = cli_args.pop(0)
    if algorithm not in TRAINERS:
        parser.error(f"unknown algorithm: {algorithm}")
    trainer = import_module(TRAINERS[algorithm])
    trainer.main(cli_args, prog=f"python -m training {algorithm}")


if __name__ == "__main__":
    main()
