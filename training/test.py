"""Keyboard evaluation entry, with simulation and UI owned by environment."""
from pathlib import Path
import sys
import argparse

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.interactive import KeyboardSession, create_keyboard_world

__all__ = ['init_game', 'parse_args', 'main']


def init_game(map_name="test_map"):
    """Retain the historical setup helper for existing integrations."""
    return create_keyboard_world(map_name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Jackal keyboard battle evaluation')
    parser.add_argument('map_name', nargs='?', default='test_map',
                        choices=['test_map', 'big_map_test'])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    KeyboardSession(args.map_name).run()

if __name__ == "__main__":
    main()
