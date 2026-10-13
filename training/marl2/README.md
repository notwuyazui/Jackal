# Jackal PyMARL2-Style Framework

This folder provides a lightweight PyMARL2-style MARL training/testing pipeline.

## Current Components

- Env wrapper: `environment/marl_adapter.py` (registered in `training/marl2/registry.py`)
- Controller (shared policy): `training/marl2/controllers/basic_mac.py`
- Agent network: `training/marl2/modules/agents/rnn_agent.py`
- EDT agent network: `training/marl2/modules/agents/etd_rnn_agent.py`
- Mixer: `training/marl2/modules/mixers/qmix.py`
- Learner: `training/marl2/learners/qmix_learner.py`
- Replay buffer: `training/marl2/components/replay_buffer.py`
- Runner: `training/marl2/runners/episode_runner.py`
- Registry: `training/marl2/registry.py`

## Extension Points

1. Add a new environment:
   - Implement a wrapper with methods used by `EpisodeRunner`.
   - Register it in `ENV_REGISTRY` in `training/marl2/registry.py`.

2. Add a new learner/algorithm:
   - Implement learner class with `train/save_models/load_models`.
   - Register it in `LEARNER_REGISTRY` in `training/marl2/registry.py`.

3. Add new task config:
   - Create a JSON in `training/configs/marl2/`.
   - Keep sections: `env`, `agent`, `algo`, `train`, `experiment`.

## EDT Map Fusion

`ETDRNNAgent` supports three map-fusion modes through `agent.etd_map_fusion`:

- `tokens`: legacy behavior; local map tokens join entity self-attention.
- `gated`: map CNN is pooled into a separate context and gated into the entity context.
- `none`: ignore map features inside EDT even if the environment emits them.

For cluttered maps such as Spindle, prefer `gated` before increasing model size.

TensorBoard logs are stored by map under
`artifacts/tensorboard/marl2/by_map/<map>/<experiment>`.
