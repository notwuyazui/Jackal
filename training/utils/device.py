"""Device selection and RNG operations for optional PyTorch accelerators.

Argument helpers do not import torch or initialize an accelerator.
TorchNPU is registered when selecting or detecting an NPU backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

DEVICE_HELP = "auto/cpu/cuda/cuda:N/mps/npu/npu:N（NPU 使用昇腾 torch_npu）"


def validate_device_name(value) -> str:
    name = str(value).strip().lower()
    if name not in {"auto", "cpu", "cuda", "mps", "npu"} and not re.fullmatch(r"(?:cuda|npu):[0-9]+", name):
        raise ValueError(f"不支持的设备 {value!r}；可选 {DEVICE_HELP}")
    return name


def add_device_argument(parser) -> None:
    parser.add_argument("--device", type=validate_device_name, default="auto", metavar="DEVICE", help=DEVICE_HELP)


def _load_npu(*, required: bool = True) -> Any:
    import torch
    backend = getattr(torch, "npu", None)
    if backend is not None:
        return backend
    try:
        importlib.import_module("torch_npu")
    except ModuleNotFoundError as exc:
        if not required and exc.name == "torch_npu":
            return None
        raise RuntimeError("NPU 后端不可用：请安装与 PyTorch/CANN 匹配的 torch_npu，并加载 CANN 环境变量") from exc
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError("torch_npu 加载失败：请检查 PyTorch、torch_npu、CANN 与驱动的版本及环境变量") from exc
    backend = getattr(torch, "npu", None)
    if backend is None:
        raise RuntimeError("torch_npu 未注册 torch.npu；请检查 PyTorch 与 torch_npu 的版本是否匹配")
    return backend


def get_device(device_name="auto"):
    """Select a device; auto prefers CUDA, then NPU, MPS and CPU."""
    name = validate_device_name(device_name)
    import torch
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        else:
            npu = _load_npu(required=False)
            if npu is not None and npu.is_available():
                name = "npu"
            elif torch.backends.mps.is_available():
                name = "mps"
            else:
                name = "cpu"
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("指定 mps，但是当前环境 MPS 不可用")
        return torch.device("mps")

    kind, _, index_text = name.partition(":")
    backend = _load_npu() if kind == "npu" else torch.cuda
    if not backend.is_available():
        raise RuntimeError(f"指定 {kind}，但是当前环境 {kind.upper()} 不可用，请检查设备及运行时配置")
    if index_text or kind == "npu":
        index = int(index_text) if index_text else 0
        count = backend.device_count()
        if not 0 <= index < count:
            raise ValueError(f"{kind.upper()} 设备不存在：{name}；当前可见设备数为 {count}")
        backend.set_device(index)
        if kind == "npu":
            from training.utils.runtime_warnings import report_npu_installation
            report_npu_installation()
        return torch.device(f"{kind}:{index}")
    return torch.device(kind)


def _backend(device_type: str) -> Any:
    if device_type == "cpu":
        return None
    if device_type == "npu":
        return _load_npu()
    import torch
    return getattr(torch, device_type)


def seed_torch(seed: int, device: torch.device | None = None) -> None:
    """Seed CPU plus the selected accelerator; CPU workers never probe NPU."""
    import torch
    seed = int(seed)
    torch.random.default_generator.manual_seed(seed)
    kind = device.type if device is not None else "cpu"
    backend = _backend(kind)
    if kind in {"cuda", "npu"}:
        backend.manual_seed_all(seed)
    elif kind == "mps":
        backend.manual_seed(seed)


@dataclass(frozen=True)
class TorchRNGState:
    cpu: Any
    backend: str
    accelerator: Any = None


def capture_torch_rng_state(device: torch.device | None = None) -> TorchRNGState:
    """Capture state changed by seed_torch for restoration after evaluation."""
    import torch
    kind = device.type if device is not None else "cpu"
    backend = _backend(kind)
    state = None
    if kind in {"cuda", "npu"}:
        state = backend.get_rng_state_all()
    elif kind == "mps":
        state = backend.get_rng_state()
    return TorchRNGState(torch.random.get_rng_state(), kind, state)


def restore_torch_rng_state(state: TorchRNGState) -> None:
    import torch
    torch.random.set_rng_state(state.cpu)
    if state.accelerator is not None:
        backend = _backend(state.backend)
        if state.backend in {"cuda", "npu"}:
            backend.set_rng_state_all(state.accelerator)
        else:
            backend.set_rng_state(state.accelerator)


def synchronize_device(device: torch.device) -> None:
    backend = _backend(device.type)
    if device.type == "mps":
        backend.synchronize()
    elif backend is not None:
        backend.synchronize(device)


def print_device_info(device):
    print(f"[Device] Using {device}")
    if device.type in {"cuda", "npu"}:
        label = "NPU" if device.type == "npu" else "GPU"
        print(f"[Device] {label}: {_backend(device.type).get_device_name(device)}")
    elif device.type == "mps":
        print("[Device] Apple Silicon GPU (MPS)")
