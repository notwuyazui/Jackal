"""Compatibility CLI for unified model evaluation and keyboard demonstrations."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "interactive":
        from training.run_game_keyboard import main as run_keyboard
        return run_keyboard(args[1:])
    if args and args[0] in {"qmix", "dqn", "drqn"}:
        args.pop(0)
    from training.test_model import main as evaluate_model
    return evaluate_model(args)


if __name__ == "__main__":
    main()
