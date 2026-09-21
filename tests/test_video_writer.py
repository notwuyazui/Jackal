import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from environment.rendering.video import create_video_writer


class ImageIOVideoWriterTests(unittest.TestCase):
    def test_writer_keeps_rgb_frames_and_closes_backend(self) -> None:
        backend = Mock()
        imageio_v2 = types.ModuleType("imageio.v2")
        imageio_v2.get_writer = Mock(return_value=backend)
        imageio_package = types.ModuleType("imageio")
        imageio_package.__path__ = []
        imageio_package.v2 = imageio_v2

        with patch.dict(
            sys.modules,
            {"imageio": imageio_package, "imageio.v2": imageio_v2},
        ):
            writer = create_video_writer("episode.mp4", 25, (4, 2))

        frame = np.zeros((2, 4, 3), dtype=np.uint8)
        frame[0, 0] = (255, 10, 20)
        writer.write(frame)
        writer.release()
        writer.release()

        imageio_v2.get_writer.assert_called_once_with(
            "episode.mp4",
            format="FFMPEG",
            mode="I",
            fps=25,
            codec="libx264",
            macro_block_size=None,
        )
        written_frame = backend.append_data.call_args.args[0]
        np.testing.assert_array_equal(written_frame, frame)
        backend.close.assert_called_once_with()

    def test_writer_rejects_unexpected_frame_size(self) -> None:
        backend = Mock()
        imageio_v2 = types.ModuleType("imageio.v2")
        imageio_v2.get_writer = Mock(return_value=backend)
        imageio_package = types.ModuleType("imageio")
        imageio_package.__path__ = []
        imageio_package.v2 = imageio_v2

        with patch.dict(
            sys.modules,
            {"imageio": imageio_package, "imageio.v2": imageio_v2},
        ):
            writer = create_video_writer("episode.mp4", 25, (4, 2))

        with self.assertRaisesRegex(ValueError, "does not match"):
            writer.write(np.zeros((3, 4, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
