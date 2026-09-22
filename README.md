<h1 align="center">RoomMind</h1>

目标环境是 Jetson Orin Nano，带 GPU 的笔记本可以联调。感知（Python FastAPI）和 Agent（TypeScript）两套进程，数据库和前端还没接。

<p align="center">
  <img src="docs/architecture.jpg" alt="RoomMind 技术架构">
</p>

**进度**

| 模块 | 现状 |
| --- | --- |
| YOLO | 可用。`POST /api/yolo/detect` 收图；`python -m app.yolo` 接摄像头 + ByteTrack。输出类别、框、track id。 |
| VLM | 可用。`POST /api/vlm/analyze`，Ollama `qwen3-vl:2b-instruct`，只生成描述。 |
| Agent | 可用。`:8001` 上 `GET /health`、`POST /complete`、`POST /compact`。唯一工具 `status`，数据库是 stub，查不到就如实说。 |
| 数据库 / 前端 / 预处理流水线 / PTZ | 未做。 |

**目录**

```text
app/main.py              感知服务入口，挂 YOLO / VLM / SAM 路由
app/camera/              摄像头打开、列设备和读帧。Linux 优先 V4L2
app/yolo/                检测、跟踪
app/vlm/                 Ollama 适配和描述生成
app/sam/                 SAM 2.1 Tiny 自动分割
agent/src/agent.ts       ReAct 循环
agent/src/server.ts      HTTP
agent/src/tools/status.ts
agent/src/db/            SpaceStatusStore，目前是 stub
docs/architecture.jpg
```

**本地跑**

Python 3.10+、Node.js 20+、本机 Ollama。两边服务可以只起一个。

```bash
curl -fsSL https://ollama.com/install.sh | sh    # Windows 用官网安装包
ollama pull qwen3-vl:2b-instruct
```

YOLO 权重 `yolo26s.pt` 第一次检测会下到 `models/`。NVIDIA 笔记本先按 [pytorch.org](https://pytorch.org) 装 CUDA 版 PyTorch，再 `pip install -r requirements.txt`，避免被 CPU 版 torch 占住。Jetson 用板上已有的 PyTorch，需要 `numpy>=1.23.5,<2`。Apple Silicon 直接装依赖。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

python3 -m app.yolo --list
python3 -m app.yolo --source 0
python3 -m app.yolo --source 0 --show          # SSH 上板子改 --save-dir

cd agent
cp .env.example .env    # OPENAI_API_KEY、MODEL；兼容端点再设 OPENAI_BASE_URL
npm install && npm test && npm run server
```

```bash
curl -s http://127.0.0.1:8000/api/yolo/detect -F image=@photo.jpg
curl -s http://127.0.0.1:8001/complete \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"客厅现在有什么？"}]}'
```
