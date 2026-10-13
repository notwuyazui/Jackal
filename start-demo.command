#!/bin/zsh
set -euo pipefail
setopt NO_BG_NICE
cd -P -- "$(dirname "$0")"
demo_python="$PWD/.venv/bin/python"
if [[ ! -x "$demo_python" ]]; then
  print '请先创建环境：python3 -m venv .venv'
  print '然后安装：.venv/bin/python -m pip install -e ".[training,video]"'
  exit 1
fi
demo_node="${JACKAL_NODE:-}"
if [[ -z "$demo_node" ]]; then
  if command -v node >/dev/null 2>&1; then demo_node="$(command -v node)"
  else demo_node="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"; fi
fi
if [[ ! -x "$demo_node" || ! -f render/render3D/node_modules/vite/bin/vite.js ]]; then
  print '需要 Node.js。请在 render/render3D 目录执行 npm ci。也可通过 JACKAL_NODE 指定 Node 路径。'
  exit 1
fi
# Editable package registration makes imports independent of the current directory.
if ! "$demo_python" -c 'import render.api, training.test_model' >/dev/null 2>&1; then
  "$demo_python" -m pip install -e . --no-build-isolation --no-deps
fi
# The unified rendering API owns both local servers and their cleanup.
# Automatic ports allow independent battles and avoid stale-service reuse.
demo_browser_args=()
if [[ "${JACKAL_NO_BROWSER:-0}" == 1 ]]; then demo_browser_args+=(--no-open-browser); fi
print '正在运行 CoT 3D 与右上角 Jackal 2D；关闭本次战斗窗口即可退出。'
exec "$demo_python" -m training.test_model --render 2D3D "${demo_browser_args[@]}" "$@"
