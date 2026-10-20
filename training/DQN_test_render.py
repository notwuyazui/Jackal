"""Compatibility entry; model evaluation is implemented in training.test_model."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.test_model import main, parse_args


def test_model(model_path, episodes=3, device_name="auto", video_dir="artifacts/videos/dqn_eval"):
    args = ["--checkpoint", str(model_path), "--episodes", str(episodes), "--device", device_name]
    if video_dir is not None:
        args.extend(["--video-dir", str(video_dir)])
    return main(args)


if __name__ == "__main__":
    main()
