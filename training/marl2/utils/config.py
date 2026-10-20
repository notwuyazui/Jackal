import json
from pathlib import Path
import os
import random
from copy import deepcopy

import numpy as np
import torch
from training.utils.device import seed_torch


def load_config(config_path):
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        if str(path).startswith('configs/'):
            path = Path('training') / path
        path = Path(__file__).resolve().parents[3] / path
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def merge_dict(base, override):
    """Recursively merge override into base and return a new dict."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_global_seeds(seed, device=None):
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed, device)

    # Deterministic mode is optional; we keep benchmark enabled for speed.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
