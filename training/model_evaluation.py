"""Checkpoint identification, evaluation configuration and greedy policy adapters."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import inspect
import json
from pathlib import Path
import pickle
from typing import Protocol

import torch

from environment.jackal_env import JackalEnv
from training.DQN_Test.network import QNetwork
from training.DRQN_Test.network import DRQNNetwork
from training.marl2.controllers.basic_mac import BasicMAC
from training.marl2.utils.config import merge_dict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARL_CONFIG = ROOT / "training/configs/marl2/jackal_autoaim_3v3_qmix.json"


def resolve_path(value: str | Path, *, config: bool = False) -> Path:
    path = Path(value).expanduser()
    if config and not path.suffix:
        path = path.with_suffix(".json")
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, ROOT / path]
    if config and not path.is_absolute():
        candidates.extend([ROOT / "training" / path, DEFAULT_MARL_CONFIG.parent / path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"找不到{'配置' if config else '检查点'}：{value}")


def load_checkpoint(path: Path) -> tuple[str, Mapping[str, torch.Tensor]]:
    """Load tensors on CPU; inference never restores optimizers or target nets."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as exc:
        raise ValueError(f"无法读取检查点权重：{path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("检查点必须包含网络 state_dict 或 MARL 的 agent 权重")
    is_marl = "agent" in payload
    weights = payload.get("agent", payload.get("state_dict", payload.get("model_state_dict", payload)))
    if not isinstance(weights, Mapping) or not weights or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in weights.items()):
        raise ValueError("检查点的网络权重必须是非空的 Tensor 字典")
    if is_marl:
        kind = "qmix"
    elif all(key in weights for key in ("chassis_head.weight", "turret_head.weight", "fire_head.weight")):
        kind = "drqn"
    elif all(key in weights for key in ("fc1.weight", "fc2.weight", "fc3.weight")):
        kind = "dqn"
    elif "rnn.weight_ih" in weights and "fc2.weight" in weights:
        kind = "qmix"
    else:
        raise ValueError("无法识别检查点：支持 DQN、因子化 DRQN 和 QMIX 系列的 agent 权重")
    return kind, weights


def find_run_config(checkpoint: Path, kind: str) -> Path | None:
    """Resolve run configs beside final/periodic and historical checkpoints."""
    directories = [checkpoint.parent]
    if checkpoint.parent.name == "checkpoints":
        directories.append(checkpoint.parent.parent)
    names = ["dqn_run_config.json", "run_config.json"] if kind == "dqn" else ["run_config.json"]
    if kind == "drqn":
        stem = checkpoint.stem
        for marker in ("_model_final", "_model_ep"):
            if marker in stem:
                names.append(stem.split(marker, 1)[0] + "_run_config.json")
                break
        names.append("drqn_run_config.json")
    names.append("config.json")
    return next((directory / name for directory in directories for name in names
                 if (directory / name).is_file()), None)


def load_evaluation_config(value: str | Path | None, checkpoint: Path,
                           kind: str) -> tuple[dict, Path | None]:
    path = resolve_path(value, config=True) if value is not None else find_run_config(checkpoint, kind)
    if path is None and kind == "qmix":
        path = DEFAULT_MARL_CONFIG
    if path is None:
        return {}, None
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict) or config.get("format") == "jackal-game-state":
        raise ValueError("--config 需要训练/运行配置 JSON，不能直接使用游戏状态存档")
    algorithm = config.get("algo")
    if isinstance(algorithm, str) and algorithm.lower() in {"dqn", "drqn"} and algorithm.lower() != kind:
        raise ValueError(f"配置算法 {algorithm} 与检查点类型 {kind} 不一致")
    return config, path.resolve()


def build_env_config(config: dict, kind: str, weights: Mapping[str, torch.Tensor],
                     *, render_mode: str = "none", video_dir: str | None = None,
                     max_steps: int | None = None, fixed_delta_time: float | None = None) -> dict:
    if "env" in config:
        if not isinstance(config["env"], dict):
            raise ValueError("配置中的 env 必须是对象")
        env_config = deepcopy(config["env"])
    else:
        fields = inspect.signature(JackalEnv).parameters
        env_config = {key: deepcopy(value) for key, value in config.items() if key in fields and key != "self"}
    train_config = config.get("train", {})
    if not isinstance(train_config, dict):
        raise ValueError("配置中的 train 必须是对象")
    override = config.get("eval_env", train_config.get("eval_env", {}))
    if not isinstance(override, dict):
        raise ValueError("eval_env 必须是对象")
    env_config = merge_dict(env_config, override)
    if kind in {"dqn", "drqn"}:
        action_dim = (int(weights["fc3.weight"].shape[0]) if kind == "dqn" else
                      int(weights["chassis_head.weight"].shape[0]) * int(weights["turret_head.weight"].shape[0]) + 1)
        if action_dim not in {10, 28}:
            raise ValueError(f"不支持的 {kind.upper()} 动作维度：{action_dim}")
        env_config.setdefault("auto_aim", action_dim == 10)
    env_config["headless"] = render_mode not in {"human", "2D"}
    env_config["render_mode"] = None if render_mode == "none" else render_mode
    env_config["use_video"] = video_dir is not None
    if video_dir is not None:
        env_config["video_dir"] = video_dir
    if max_steps is not None:
        env_config["max_steps"] = max_steps
    if fixed_delta_time is not None:
        env_config["fixed_delta_time"] = fixed_delta_time
    return env_config


class EvaluationPolicy(Protocol):
    def reset(self) -> None: ...

    def select_actions(self, obs, state, avail_actions) -> list[int]: ...


def _load_weights(network: torch.nn.Module, weights: Mapping[str, torch.Tensor]) -> None:
    try:
        network.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        raise ValueError("检查点网络结构或输入/输出维度与 --config 不一致，请选择该模型的训练配置") from exc
    network.eval()
    network.requires_grad_(False)


def _single_agent_dimensions(weights, config, env_info, output_dim):
    if env_info["n_agents"] != 1:
        raise ValueError("DQN/DRQN 检查点使用全局状态控制一个智能体，需要 n_agents=1 的环境配置")
    expected = {"state_dim": int(weights["fc1.weight"].shape[1]), "action_dim": output_dim}
    actual = {"state_dim": env_info["state_shape"], "action_dim": env_info["n_actions"]}
    for name, value in expected.items():
        if actual[name] != value or (name in config and int(config[name]) != value):
            raise ValueError(f"{name} 不匹配：env={actual[name]}, checkpoint={value}；请选择匹配的 --config")


class DQNPolicy:
    def __init__(self, weights, config, env_info, device):
        action_dim = int(weights["fc3.weight"].shape[0])
        _single_agent_dimensions(weights, config, env_info, action_dim)
        self.device = device
        self.network = QNetwork(env_info["state_shape"], action_dim).to(device)
        _load_weights(self.network, weights)

    def reset(self) -> None:
        """DQN has no recurrent state to carry between episodes."""

    def select_actions(self, obs, state, avail_actions) -> list[int]:
        inputs = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            q_values = self.network(inputs)[0]
            mask = torch.as_tensor(avail_actions[0], device=self.device) > 0
            return [int(q_values.masked_fill(~mask, -torch.inf).argmax().item())]


class DRQNPolicy:
    def __init__(self, weights, config, env_info, device):
        if not config.get("factorized_action", True):
            raise ValueError("DRQN 检查点需要 factorized_action=True")
        dimensions = {"hidden_dim": int(weights["fc1.weight"].shape[0]),
                      "chassis_dim": int(weights["chassis_head.weight"].shape[0]),
                      "turret_dim": int(weights["turret_head.weight"].shape[0]),
                      "fire_dim": int(weights["fire_head.weight"].shape[0])}
        for name, value in dimensions.items():
            if name in config and int(config[name]) != value:
                raise ValueError(f"{name} 与 DRQN 检查点不一致")
        self.fire_action_id = dimensions["chassis_dim"] * dimensions["turret_dim"]
        if dimensions["fire_dim"] != 2 or int(config.get("fire_action_id", self.fire_action_id)) != self.fire_action_id:
            raise ValueError("DRQN 射击动作编号或 fire_dim 与环境的因子化动作空间不一致")
        _single_agent_dimensions(weights, config, env_info, self.fire_action_id + 1)
        self.device = device
        self.network = DRQNNetwork(env_info["state_shape"], env_info["n_actions"], **dimensions).to(device)
        _load_weights(self.network, weights)
        self.hidden_state: torch.Tensor | None = None

    def reset(self) -> None:
        self.hidden_state = self.network.init_hidden(batch_size=1, device=self.device)

    def select_actions(self, obs, state, avail_actions) -> list[int]:
        if self.hidden_state is None:
            raise RuntimeError("Call policy.reset() before evaluating an episode")
        inputs = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            chassis, turret, fire, self.hidden_state = self.network(inputs, self.hidden_state)
            mask = torch.as_tensor(avail_actions[0], device=self.device) > 0
            if int(fire[0].argmax().item()) == 1 and bool(mask[self.fire_action_id].item()):
                return [self.fire_action_id]
            movement_q = (chassis[0].repeat(self.network.turret_dim)
                          + turret[0].repeat_interleave(self.network.chassis_dim))
            return [int(movement_q.masked_fill(~mask[:self.fire_action_id], -torch.inf).argmax().item())]


class MARLPolicy:
    def __init__(self, weights, config, env_info, device):
        agent_config = config.get("agent")
        if not isinstance(agent_config, dict):
            raise ValueError("QMIX 系列检查点需要包含 env 和 agent 的训练配置")
        self.mac = BasicMAC(env_info["n_agents"], env_info["n_actions"], env_info["obs_shape"], agent_config, device)
        _load_weights(self.mac.agent, weights)

    def reset(self) -> None:
        self.mac.init_hidden(batch_size=1)

    def select_actions(self, obs, state, avail_actions) -> list[int]:
        return self.mac.select_actions(obs, avail_actions, epsilon=0.0, test_mode=True)


POLICY_REGISTRY = {"dqn": DQNPolicy, "drqn": DRQNPolicy, "qmix": MARLPolicy}


def create_policy(kind: str, weights, config: dict, env_info: dict,
                  device: torch.device) -> EvaluationPolicy:
    return POLICY_REGISTRY[kind](weights, config, env_info, device)
