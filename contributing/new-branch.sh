#!/usr/bin/env bash
# 从最新的 origin/main 开出一条功能分支。
# 用法：./contributing/new-branch.sh feat/你的功能名
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "用法: ./contributing/new-branch.sh feat/你的功能名" >&2
  exit 1
fi

name="$1"
if [[ "$name" == "main" || "$name" == "master" ]]; then
  echo "不要用 main 作为功能分支名。" >&2
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

git fetch origin
git switch main
git pull --ff-only origin main
git switch -c "$name"
echo "当前分支: $(git branch --show-current)"
