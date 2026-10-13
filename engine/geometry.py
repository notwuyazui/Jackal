"""Integer rectangle geometry. Keep Pygame rounding/edge semantics during migration.

This module never initializes SDL or creates surfaces.
"""
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
from pygame.rect import Rect

__all__ = ["Rect"]
