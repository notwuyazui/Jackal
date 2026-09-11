"""OpenCV-backed video helpers with lazy dependency loading."""

from typing import Any, Tuple

import numpy as np


def create_video_writer(
    path: str,
    fps: int,
    frame_size: Tuple[int, int],
) -> Any:
    """Create a writer only when episode recording is explicitly enabled."""
    import cv2

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, frame_size)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert a Pygame RGB frame to the channel order expected by OpenCV."""
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
