"""Sprite choices and resource paths belong solely to the 2D renderer."""
from pathlib import Path

ASSET_ROOT = Path(__file__).resolve().parent / 'assets'
TILES = {letter: f'Map/{kind}Tile/{kind.lower()}.png' for letter, kind in
         [('o', 'Flat'), ('x', 'Barrier'), ('s', 'Sand'), ('w', 'Water'), ('t', 'Trap')]}


def unit_sprites(unit_type, team):
    folder = unit_type.capitalize()
    enemy = 'enemy_' if team == 'ENEMY' else ''
    return (f'Unit/{folder}/{enemy}{unit_type}.png',
            f'Unit/{folder}/{enemy if unit_type != "tank" else ""}turret.png')


def projectile_sprite(ammunition, exploded=False):
    folder, shell, blast = {'normal_shell': ('NormalShell', 'normalshell.png', 'normalshell.png'),
                           'rocket_shell': ('RocketShell', 'RocketShell.png', 'RocketShellExplosion.png'),
                           'heavy_shell': ('HeavyShell', 'heavyShell.png', 'HeavyShellExplosion.png')}[ammunition]
    return f'Bullet/{folder}/{blast if exploded else shell}'
