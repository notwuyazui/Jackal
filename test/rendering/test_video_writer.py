import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from render.render2D.video import create_video_writer, ImageIOVideoWriter


class ImageIOVideoWriterTests(unittest.TestCase):
    def test_fractional_fps_preserves_simulation_time_and_invalid_fps_fails_early(self):
        imageio = types.ModuleType('imageio.v2')
        imageio.get_writer = Mock()
        package = types.ModuleType('imageio')
        package.__path__ = []
        package.v2 = imageio
        with patch.dict(sys.modules, {'imageio': package, 'imageio.v2': imageio}):
            factory = imageio.get_writer
            writer = create_video_writer('episode.mp4', .5, (4, 2))
            self.assertEqual(factory.call_args.kwargs['fps'], .5)
            writer.release()
            factory.reset_mock()
            for fps in (0, -1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    create_video_writer('episode.mp4', fps, (4, 2))
            factory.assert_not_called()

    def test_failed_finalization_is_not_retried_and_closed_writes_are_rejected(self):
        backend = Mock()
        backend.close.side_effect = RuntimeError('encoder failed')
        writer = ImageIOVideoWriter(backend, (4, 2))
        with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
            writer.release()
        writer.release()
        backend.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            writer.write(np.zeros((2, 4, 3), dtype=np.uint8))

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
