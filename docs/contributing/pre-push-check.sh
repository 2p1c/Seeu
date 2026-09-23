#!/usr/bin/env bash
# 推送前检查：暂存区不能有密钥和依赖，并跑不需要 GPU 的测试。
# 用法：./contributing/pre-push-check.sh
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

branch="$(git branch --show-current)"
if [[ "$branch" == "main" || "$branch" == "master" ]]; then
  echo "当前在 ${branch}。请先开功能分支再推送。" >&2
  exit 1
fi

echo "当前分支: ${branch}"

staged="$(git diff --cached --name-only)"
if [[ -n "$staged" ]]; then
  blocked="$(printf '%s\n' "$staged" | grep -E '(^|/)\.env$|node_modules/|(^|/)\.venv/|__pycache__/|\.pyc$|^models/|\.pt$|egg-info/' || true)"
  if [[ -n "$blocked" ]]; then
    echo "暂存区里有不应提交的文件:" >&2
    printf '%s\n' "$blocked" >&2
    echo "用 git restore --staged <文件> 撤出暂存区后再检查。" >&2
    exit 1
  fi
fi

if [[ -d agent/node_modules ]]; then
  echo "运行 Agent 测试"
  (cd agent && npm test)
else
  echo "跳过 Agent 测试：还没有 agent/node_modules。需要时执行 cd agent && npm install && npm test"
fi

if [[ -d web/node_modules ]]; then
  echo "检查感知页类型"
  (cd web && npx tsc --noEmit)
else
  echo "跳过感知页类型检查：还没有 web/node_modules。需要时执行 cd web && npm install && npx tsc --noEmit"
fi

echo "检查通过。请再按 README 把这次改动的功能在页面或接口上点一次，然后再 git push。"
