"""Compatibility entry; model evaluation is implemented in training.test_model."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.model_evaluation import find_run_config
from training.test_model import main, parse_args


def resolve_run_config_path(model_path):
    path = find_run_config(Path(model_path), "drqn")
    return str(path) if path is not None else None


def test_drqn_model(model_path, episodes=3, video_dir="artifacts/videos/drqn_eval",
                    max_steps=None, fixed_delta_time=None, device_name="auto"):
    args = ["--checkpoint", str(model_path), "--episodes", str(episodes), "--device", device_name]
    if video_dir is not None:
        args.extend(["--video-dir", str(video_dir)])
    if max_steps is not None:
        args.extend(["--max-steps", str(max_steps)])
    if fixed_delta_time is not None:
        args.extend(["--fixed-delta-time", str(fixed_delta_time)])
    return main(args)


if __name__ == "__main__":
    main()
