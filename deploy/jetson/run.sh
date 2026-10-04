#!/usr/bin/env bash
# systemd 的 roomind@<名字> 都从这里启动。也可以手动执行：deploy/jetson/run.sh perception
set -euo pipefail
cd "$(dirname "$0")/../.."

case "${1:-}" in
  perception) exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 ;;
  web|agent)
    # Node 装在 nvm 里时，systemd 的 PATH 里看不到它。
    if [[ -s "$HOME/.nvm/nvm.sh" ]]; then source "$HOME/.nvm/nvm.sh"; fi
    cd "$1"
    if [[ "$1" == web ]]; then exec npm start; else exec npm run server; fi
    ;;
  *) echo "用法: run.sh perception|web|agent" >&2; exit 1 ;;
esac
