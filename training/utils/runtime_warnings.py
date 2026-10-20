"""Narrow diagnostics for shared CANN installations; never import torch here."""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat
import sys
import warnings

_reported_installations: set[str] = set()


def _shared_cann_installation() -> tuple[str, str, int, int] | None:
    root = os.environ.get("ASCEND_HOME_PATH")
    if not root or not hasattr(os, "getuid"):
        return None
    path = os.path.abspath(root)
    try:
        info = os.stat(path)
    except OSError:
        return None
    uid = os.getuid()
    if (path == os.path.sep or not stat.S_ISDIR(info.st_mode)
            or info.st_uid == uid or info.st_mode & stat.S_IWOTH
            or not os.access(path, os.R_OK | os.X_OK)):
        return None
    return path, os.path.realpath(path), info.st_uid, uid


def configure_runtime_warnings() -> None:
    """Recognize a readable shared SDK without disabling its permission checks.

    This must run before torch: recent versions auto-import torch_npu. Explicit
    Python -W policies take priority; the raw-diagnostics switch bypasses us.
    """
    if os.environ.get("JACKAL_RAW_DEPENDENCY_WARNINGS") == "1":
        return
    installation = _shared_cann_installation()
    if installation is None:
        return
    paths = "|".join(re.escape(p) for p in dict.fromkeys(installation[:2]))
    # collect_env separately checks ownership and readability. Only its known
    # ownership notice for this SDK is replaced; missing/unreadable paths and
    # all other warnings/errors still go through the original SDK checks.
    warnings.filterwarnings(
        "ignore",
        message=(rf"^Warning: The (?:{paths})(?:/[^\r\n]*ascend_[^/\r\n]*_install\.info)? "
                 r"owner does not match the current (?:owner|user)\.$"),
        category=UserWarning,
        module=r"^torch_npu\.utils\.collect_env$",
        append=True,
    )


def report_npu_installation() -> None:
    """Describe shared ownership once in the NPU learner, not in CPU workers."""
    if os.environ.get("JACKAL_RAW_DEPENDENCY_WARNINGS") == "1":
        return
    installation = _shared_cann_installation()
    if installation is None:
        return
    path, resolved, owner, uid = installation
    if resolved not in _reported_installations:
        _reported_installations.add(resolved)
        print(f"[NPU] CANN 共享安装：{Path(path)}（属主 UID={owner}，当前 UID={uid}）。"
              "目录可读；属主不同的重复提示已合并，SDK 的权限及运行时检查仍保留。",
              file=sys.stderr, flush=True)
