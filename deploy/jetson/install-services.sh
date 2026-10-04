#!/usr/bin/env bash
# 在 Jetson 上把感知、页面、Agent 注册成 systemd 服务。用法: sudo bash install-services.sh
set -euo pipefail
user=${SUDO_USER:?请用 sudo 运行，服务会以当前登录用户的身份启动}
root=$(cd "$(dirname "$0")/../.." && pwd)
[[ -x "$root/.venv/bin/python" ]] || { echo "缺少 $root/.venv，先按 README 建虚拟环境" >&2; exit 1; }

# 物体记忆库：PostgreSQL + pgvector。数据在 Docker 卷 roomind-pg 里，不在仓库目录，部署同步不会删掉它。
# 板子连不上 Docker Hub 时，在电脑上 docker pull --platform linux/arm64 pgvector/pgvector:pg17，
# 再 docker save pgvector/pgvector:pg17 | ssh <板子> docker load。
docker start roomind-db >/dev/null 2>&1 || docker run -d --name roomind-db --restart unless-stopped \
  -e POSTGRES_USER=roomind -e POSTGRES_PASSWORD=roomind -e POSTGRES_DB=roomind -e TZ=Asia/Shanghai \
  -p 127.0.0.1:5432:5432 -v roomind-pg:/var/lib/postgresql/data \
  pgvector/pgvector:pg17 -c shared_buffers=64MB -c max_connections=20

sed -e "s|@ROOT@|$root|g" -e "s|@USER@|$user|g" \
  "$root/deploy/jetson/roomind@.service.in" > /etc/systemd/system/roomind@.service
systemctl daemon-reload
systemctl enable --now roomind@perception roomind@web roomind@agent
systemctl enable --now ollama 2>/dev/null || echo "没有 ollama.service，VLM 需要另装 Ollama 并 ollama pull qwen3-vl:2b-instruct"

cat <<EOF
感知   http://127.0.0.1:8000/api/deploy
页面   http://127.0.0.1:8080
Agent  http://127.0.0.1:8001
编译引擎: cd $root && .venv/bin/python -m app.inference.trt
EOF
