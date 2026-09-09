import torch


def get_device(device_name="auto"):
    """
    统一设备管理
    支持: cpu cuda mps auto
    auto优先级: CUDA > MPS > CPU
    """

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定cuda，但是当前环境CUDA不可用")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("指定mps，但是当前环境MPS不可用")
        return torch.device("mps")
    if device_name == "cpu":
        return torch.device("cpu")
    
    # auto模式
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def print_device_info(device):
    """
    打印设备信息
    """

    print(f"[Device] Using {device}")

    if device.type == "cuda":
        print(
            f"[Device] GPU: "
            f"{torch.cuda.get_device_name(device)}"
        )

    elif device.type == "mps":
        print(
            "[Device] Apple Silicon GPU (MPS)"
        )