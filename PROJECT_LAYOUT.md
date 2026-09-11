# Project Layout

The project is split into three code areas:

- `game/`: Pygame battle engine, units, bullets, maps, constants, and art assets.
- `environment/`: RL-facing Jackal environment built on top of the game engine.
  - `jackal_env.py`: episode lifecycle, action execution, rendering, and the stable environment API.
  - `observation/`: local observations, centralized state, map features, and feature normalization.
  - `rendering/`: optional video helpers and lazily loaded rendering dependencies.
  - `reward/`: reward defaults, shared battle statistics, and aim-mode reward functions.
- `training/`: training and evaluation code, configs, MARL framework, DQN, and DRQN modules.
- `artifacts/`: generated outputs such as checkpoints, TensorBoard events, plots, logs, videos, and temporary runs.

Common entry points:

- Train QMIX or EDT-QMIX: `python train_qmix_marl2.py --config training/configs/marl2/<config>.json`
- Evaluate QMIX checkpoint: `python test_qmix_marl2.py --config training/configs/marl2/<config>.json --checkpoint <path>`
- Import the environment: `from environment import JackalEnv`

Artifact folders:

- `artifacts/checkpoints/marl2/`: QMIX and EDT-QMIX checkpoints.
- `artifacts/checkpoints/dqn/`: DQN checkpoints and run configs.
- `artifacts/checkpoints/drqn/`: DRQN checkpoints and run configs.
- `artifacts/tensorboard/marl2/by_map/<map>/`: TensorBoard event files grouped by map.
- `artifacts/plots/marl2/`: exported plots and CSV summaries.
- `artifacts/logs/marl2/`: analysis logs and derived summaries.
- `artifacts/videos/`: evaluation and render videos.
- `artifacts/tmp/`: temporary checkpoint experiments.

Compatibility notes:

- `JackalEnv.py`, `train_qmix_marl2.py`, and `test_qmix_marl2.py` remain as thin root-level shims.
- Old config paths beginning with `configs/` are redirected to `training/configs/` by the config loader.
