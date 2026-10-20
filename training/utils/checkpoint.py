"""Portable model saves: serialized tensors use standard CPU storage."""
from copy import copy


def _cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        # Preserve OrderedDict state_dict metadata used by module loaders.
        result = copy(value)
        result.update({key: _cpu_tree(item) for key, item in value.items()})
        return result
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        items = [_cpu_tree(item) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    return value


def save_checkpoint(payload, path) -> None:
    """Keep checkpoint layouts intact and readable across CPU/CUDA/MPS/NPU."""
    import torch
    torch.save(_cpu_tree(payload), path)
