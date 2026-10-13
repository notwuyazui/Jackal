"""ImageIO-backed video helpers with lazy dependency loading."""

from typing import Any, Tuple
import math

import numpy as np


class ImageIOVideoWriter:
    """Expose the small write/release interface used by JackalEnv."""

    def __init__(self, writer: Any, frame_size: Tuple[int, int]) -> None:
        self._writer = writer
        width, height = frame_size
        self._expected_shape = (int(height), int(width), 3)

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError('Video writer is closed')
        rgb_frame = np.asarray(frame, dtype=np.uint8)
        if rgb_frame.shape != self._expected_shape:
            raise ValueError(
                f"Video frame shape {rgb_frame.shape} does not match "
                f"expected {self._expected_shape}"
            )
        self._writer.append_data(np.ascontiguousarray(rgb_frame))

    def release(self) -> None:
        if self._writer is not None:
            writer, self._writer = self._writer, None
            writer.close()


def create_video_writer(
    path: str,
    fps: float,
    frame_size: Tuple[int, int],
) -> Any:
    """Create a writer only when episode recording is explicitly enabled."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Video fps must be finite and > 0')
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Video recording requires imageio and imageio-ffmpeg. "
            "Install the project requirements before enabling video."
        ) from exc

    writer = imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=float(fps),
        codec="libx264",
        macro_block_size=None,
    )
    return ImageIOVideoWriter(writer, frame_size)
