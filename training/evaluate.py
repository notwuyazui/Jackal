"""Dispatch model evaluation commands to implementations owned by training."""
import argparse
import sys
from importlib import import_module
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = {'qmix': 'test_qmix_marl2', 'dqn': 'DQN_test_render',
           'drqn': 'DRQN_test_render', 'interactive': 'test'}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='python -m training.evaluate')
    parser.add_argument('algorithm', choices=MODULES, nargs='?')
    if not args or args[0] in {'-h', '--help'}:
        parser.print_help()
        return
    name = args.pop(0)
    if name not in MODULES:
        parser.error(f'Unknown algorithm: {name}')
    import_module('training.' + MODULES[name]).main(args)


if __name__ == '__main__':
    main()
