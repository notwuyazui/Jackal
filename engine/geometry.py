"""Integer rectangle geometry. Keep Pygame rounding/edge semantics during migration.

This module never initializes SDL or creates surfaces.
"""
import os
import warnings
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
with warnings.catch_warnings():
    if os.environ.get("JACKAL_RAW_DEPENDENCY_WARNINGS") != "1":
        # Pygame 2.6's resource loader has a filesystem fallback when setuptools
        # removes pkg_resources. Keep its behavior and silence only this known
        # upstream deprecation, not resource failures or other UserWarnings.
        warnings.filterwarnings(
            "ignore", message=r"^pkg_resources is deprecated as an API\.",
            category=UserWarning, module=r"^pygame\.pkgdata$", append=True,
        )
    from pygame.rect import Rect

__all__ = ["Rect"]
