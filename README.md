<h1 align="center">RoomMind</h1>

目标环境是 Jetson Orin Nano。电脑（最好是M芯片Mac或者带GPU的电脑）可以按下面的步骤联调。感知（Python FastAPI）、页面（Hono）和 Agent（TypeScript）分开跑，数据库还没接。

<p align="center">
  <img src="docs/architecture.jpg" alt="RoomMind 技术架构">
</p>

**进度**

| 模块 | 现状 |
| --- | --- |
| YOLO | 可用。`POST /api/yolo/detect` 收图；`python -m app.yolo` 接摄像头 + ByteTrack。页面不调用它。 |
| SAM | 可用。`POST /api/sam/segment`，SAM 2.1 Tiny 自动分割。可设每边点数和每批点数。 |
| DINO | 可用。`POST /api/dino/detect`，Grounding DINO Tiny。英文短语检测，返回框和置信度。没有检出时页面不显示带框图。 |
| VLM | 可用。`POST /api/vlm/analyze`，Ollama `qwen3-vl:2b-instruct`。物体用自然语言填写。 |
| 前端 | 可用。感知页 Hono `:8080`。摄像头预览，或上传图片跑 VLM / DINO / SAM，也可按顺序跑。 |
| Agent | `:8001` 提供对话页，工具有 `status` 和 `fire_status`；空间数据库仍是 stub。 |
| 火警监测 | 后台摄像头检测、持续规则、事件列表和浏览器语音提醒；可异步追加 Qwen / Agent 分析。配置见下文。 |
| 数据库 / 预处理流水线 / PTZ | 未做。 |

**目录**

```text
app/main.py              感知服务，端口 8000
app/cli.py               roomind 命令
app/camera/              摄像头。Linux 优先 V4L2
app/yolo/                检测、跟踪
app/sam/                 SAM 2.1 Tiny
app/dino/                Grounding DINO Tiny
app/vlm/                 Ollama 描述
web/                     页面，端口 8080，结果图从 test/tmp/ 读取
agent/                   Agent 服务和对话页，端口 8001
agent/.env               本机密钥，不进 git
test/tmp/                推理结果图和 boxes JSON
models/                  本机权重，不进 git
```

## 在电脑上测试

需要 Python 3.10+、Node.js 22+（当前 Agent 锁定的依赖要求）。测 VLM 还要本机 Ollama。三个进程各开一个终端，都从仓库根目录进入。

### 1. 安装依赖

NVIDIA 显卡先按 [pytorch.org](https://pytorch.org) 装 CUDA 版 PyTorch，再装下面的依赖，避免被 CPU 版 torch 占住。Apple Silicon 直接装，会走 MPS。没有独显也能跑，SAM 和 DINO 会慢。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch          # NVIDIA 机器换成 pytorch.org 给出的 CUDA 安装命令
pip install -r requirements.txt
pip install -e .           # 注册 roomind。之后改代码不用重装
```

`requirements.txt` 把 `transformers` 限制在 5.0 之前，Grounding DINO 的接口还停在 4.x。

页面和 Agent：

```bash
cd web && npm install
cd ../agent && npm install
```

### 2. 下载模型

在仓库根目录、已激活的虚拟环境里执行。YOLO 和 SAM 第一次推理也会自己下；DINO 必须事先下好，代码只读本地文件。

```bash
# YOLO，约几十 MB，下到 models/yolo26s.pt
mkdir -p models
wget -O models/yolo26s.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt

# SAM 2.1 Tiny，下到 models/sam/。访问不了 huggingface.co 时用镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download facebook/sam2.1-hiera-tiny \
  --local-dir models/sam \
  --endpoint "$HF_ENDPOINT"

# Grounding DINO Tiny，下到 models/dino/ 的 Hugging Face 缓存结构
huggingface-cli download IDEA-Research/grounding-dino-tiny \
  --cache-dir "$PWD/models/dino"

# VLM
curl -fsSL https://ollama.com/install.sh | sh    # Windows 用官网安装包
ollama pull qwen3-vl:2b-instruct
```

目录里要能看到这些文件才算下完：`models/yolo26s.pt`、`models/sam/config.json` 和同目录的 `*.safetensors`、`models/dino/` 下的 `models--IDEA-Research--grounding-dino-tiny`。

### 3. 开启服务

终端 A，感知服务。模型在第一次对应请求时加载，第一次会久一些。

```bash
source .venv/bin/activate
roomind
# 等同 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

终端 B，页面。浏览器打开 http://127.0.0.1:8080 。

```bash
cd web
npm start
# 感知服务不在本机时：ROOMIND_API=http://<主机>:8000 npm start
```

页面每 5 秒请求一次感知服务的 `/openapi.json`，用来显示「感知服务已连接」。这是正常日志，不是推理。

### Agent 环境变量

在 `agent/` 目录复制示例文件再改。`npm run server` 会读取 `agent/.env`，这个文件已经在 `.gitignore` 里。

```bash
cd agent
cp .env.example .env
```

| 变量 | 是否必填 | 作用 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 必填 | 调用模型的密钥。兼容端点若不做校验，填一个非空占位即可。 |
| `MODEL` | 必填 | 模型名，例如 `gpt-4o`。没设的话进程启动时直接退出。 |
| `OPENAI_BASE_URL` | 可选 | 兼容 OpenAI 的接口地址。不设就走官方 `api.openai.com`。 |
| `PORT` | 可选 | Agent 端口，默认 `8001`。 |
| `AGENT_LOOP_LOG` | 可选 | `1` 打印每步工具调用，`0` 关闭。默认在非 `production` 时打开。 |
| `NODE_ENV` | 可选 | 设为 `production` 时，默认不再打印循环日志。 |

官方接口示例：

```bash
OPENAI_API_KEY=sk-...
MODEL=gpt-4o
# PORT=8001
```

改完 `.env` 要重启 `npm run server`。

终端 C 只在要测 Agent 时开。先写好上面的 `agent/.env`，再启动。浏览器打开 http://127.0.0.1:8001 ，这是 Agent 自己的页面，和 `:8080` 的感知页分开。

```bash
cd agent
npm test && npm run server
```

页面上直接对话。问空间里有什么时，Agent 会调用 `status`；库还是空的，回答会说明查不到。不满 10 轮用户消息时，「压缩上下文」会跳过。

### 4. 用页面测

1. 右上角应显示「感知服务已连接」。
2. **摄像头**：选设备，打开预览。这里只看画面和分辨率、帧率、编码，不跑模型。
3. **上传图片**：选一张图，再选服务。
   - VLM：物体用自然语言，多个用顿号或逗号分开，例如 `穿粉色衣服的小女孩、饮水机`。只返回文字。
   - DINO：英文短语，例如 `a chair. a sofa.`。有检出时显示带框图、置信度和框；数量为 0 时不显示图片。
   - SAM：默认每边 16 点（采样 256 点）、每批 8 点。返回 mask 数量、叠加图和 mask 图。
   - **顺序全部**：同一张图按 VLM → DINO → SAM 依次跑，某一步失败就停。8GB 内存上不要和 Ollama 同时占着，见下面的显存错误。

结果图在 `test/tmp/`，页面从这里读 `*_annotated.jpg`、`*_overlay.jpg`、`*_masks.png`。

### 5. 用 curl 测

`-F image=@` 读的是执行 curl 这台机器上的文件。

```bash
curl -s http://127.0.0.1:8000/api/vlm/analyze \
  -F image=@photo.jpg \
  -F prompt="描述这些物体在画面中的位置和状态。" \
  -F time="2026-09-22T22:00" \
  -F location="客厅" \
  -F objects="穿粉色衣服的小女孩、饮水机"

curl -s http://127.0.0.1:8000/api/dino/detect \
  -F image=@photo.jpg \
  -F prompt="a chair. a sofa."

curl -s "http://127.0.0.1:8000/api/sam/segment?points_per_crop=16&points_per_batch=8" \
  -F image=@photo.jpg

curl -s http://127.0.0.1:8000/api/yolo/detect -F image=@photo.jpg

curl -s http://127.0.0.1:8001/complete \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"客厅现在有什么？"}]}'
```

不经过 HTTP 时：

```bash
python3 -m app.dino photo.jpg --prompt "a chair. a sofa."
python3 -m app.sam photo.jpg --points-per-crop 16 --points-per-batch 8
python3 -m app.yolo --list
python3 -m app.yolo --source 0 --show
```

## 可能遇到的错误

**感知服务未连接，或页面打不开结果图。** `roomind` 和 `npm start` 要同时在跑，并且都使用这份仓库。页面读的是仓库里的 `test/tmp/`。改完 Python 后要重启 `roomind`，改完页面后刷新浏览器。

**`本地没有完整的 Grounding DINO 权重。`** 还没执行上面的 `huggingface-cli download ... --cache-dir`。DINO 不会在推理时自动下载。

**`无法下载 facebook/sam2.1-hiera-tiny。`** 网络访问不了 Hugging Face。先 `export HF_ENDPOINT=https://hf-mirror.com` 再重试。SAM 进程启动时就会把这个镜像写成默认值。

**`无法下载 yolo26s.pt` 或文件不完整。** 手动执行报错里的 `wget`。权重大于 5MB 才会被当成有效文件。

**`加载 Grounding DINO 时显存不足。`** 8GB 统一内存上，Ollama 的 VLM 还占着内存。先 `ollama ps`，再 `ollama stop qwen3-vl:2b-instruct`，用 `free -h` 确认至少还有约 2GB 可用。

**`Grounding DINO 显存不足。请把 max_size 降到 640 或 512。`** 推理阶段内存不够。页面目前用默认最长边 800。curl 加上 `?max_size=640`。

**`SAM AMG 显存不足。`** 把每批点数降到 4 或 1，或把 `max_size` 降到 768。

**`请填写要描述的物体`。** VLM 的物体是空的。

**`empty image` / `cannot decode image`。** 没有上传文件，或文件不是图片。

**Ollama 导入即报错，和 SOCKS、`socksio` 有关。** 本机 `11434` 不应走代理。代码会在导入时暂时拿掉 `ALL_PROXY` 一类变量；若仍失败，先在当前终端 `unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY` 再启动。

**`模型未找到` 或 VLM 请求一直转圈。** `ollama list` 里要有 `qwen3-vl:2b-instruct`，并且 `ollama serve` 在跑。第一次加载会到几分钟。

**摄像头 `409` 或打不开。** 同一时间只能开一路。先关掉页面预览或正在跑的 `python -m app.yolo`。macOS 上设备一般是 `0`；Linux 上是 `/dev/video*`。

**Jetson 上 `numpy 2.x`。** 板上 PyTorch 需要 `numpy>=1.23.5,<2`。电脑上这条检查不会触发。

**终端不停刷 `GET /openapi.json`。** 感知页开着就会每 5 秒探一次活。关掉该页面即停。Agent 页不会打这行日志。

**`MODEL environment variable is not set.`** `agent/.env` 里没有 `MODEL`，或服务不是从这份配置启动的。写上后重启。

**对话返回 401、invalid api key，或连接被拒绝。** 核对 `OPENAI_API_KEY`。用兼容接口时还要核对 `OPENAI_BASE_URL`，Ollama 一般是 `http://127.0.0.1:11434/v1`，并且 `ollama serve` 已在跑。


## 火焰与烟雾主动提醒

模型：[e1250/safety_detection](https://huggingface.co/e1250/safety_detection)，模型卡标明是 YOLOv26 微调权重，类别为 `fire` 和 `smoke`。这条流程与普通 YOLO 物体检测分开，不使用 `yolo26s.pt` 替代火警权重。

### 下载与运行

安装原有 Python 依赖后执行：

```bash
python -m app.fire.download
# 官方站点无法访问时：
HF_ENDPOINT=https://hf-mirror.com python -m app.fire.download
```

当前核实的仓库权重文件是 `yolo_smoke_fire.pt`。脚本先查询仓库真实文件列表：只有一个 `.pt` 时自动选择；存在多个时会列出文件，按提示加 `--filename 仓库内文件路径`。脚本固定仓库提交版本下载，保存到 `models/fire/model.pt`，并将来源写入同目录 JSON。已有权重不会被覆盖。可设置 `HF_ENDPOINT` 使用自己的镜像。

也可以自行下载该仓库权重，启动前指定：

```bash
export FIRE_MODEL_PATH=/绝对路径/火警权重.pt
roomind
```

按原有方式启动 `web`。页面“火焰与烟雾监测”中填写位置，选择上方摄像头，关闭摄像头预览后点击“开始监测”。等待状态变为“监测中”；缺权重、模型类别不符或摄像头故障会明确显示错误。监测独占摄像头，不能同时使用原有预览或其他摄像头进程。

点击“启用语音提醒”测试声音，保持浏览器页面打开。后台检测不依赖浏览器持续打开，但本版声音由浏览器播放，关闭页面后不会从 Jetson 扬声器播报，也不发送联系人消息。刷新页面只展示旧事件，不重新播报旧事件。

“检测已上传图片”可检查火警权重对单张图的输出；它不会更新实时监测的持续时间，也不会触发主动提醒。

### 规则与事件

- 每秒最多处理一帧；实际频率受设备推理耗时限制。
- `fire` / `smoke` 分别判断。置信度低于 0.4 不计入候选；达到 0.8 立即提醒。
- 0.4–0.8 的候选需要连续检出至少 5 秒且至少两次观测；中间未检出会重新计时。
- 同类持续事件每 60 秒最多提醒一次；连续 10 秒未检出后结束这一事件，再次出现视为新事件。
- 两次检测间隔超过 10 秒时，不将缺失画面算作持续证据。界面标出画面过期；断流或推理失败进入故障状态。
- 事件含类型、位置、开始时间、检测时间、持续秒数、框与置信度、触发原因、建议动作和提醒文字。全部标为“疑似／待确认”。

这些阈值是第一版工程默认值，集中在 `app/fire/rules.py`，需要结合实际摄像头数据校准。模型置信度不是火灾发生概率。模型卡说明其未完成生产安全关键场景验证；本功能不能代替烟感或消防报警器。

事件和持续状态目前保存在进程内，最多保留最近 100 条提醒，页面显示最近 20 条。重启后清空，不自动恢复监测；这里没有接入持久化数据库或空间 Objects Memory。

### Qwen 与 Agent

启用“场景分析”后，规则提醒先立即发布，独立后台线程再调用 Ollama 的 `qwen3-vl:2b-instruct` 描述事件帧，然后请求 Agent 的 `POST /fire/review`，结合描述、事件持续时间和最近提醒生成分析。不会因为模型慢、不可用或分析失败而取消规则提醒。分析线程忙时，新事件仍提醒，但跳过该次场景分析。

需另行启动 Ollama 与 Agent；Agent 使用原有 `agent/.env` 模型配置。可在启动 Python 服务前设置：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `FIRE_MODEL_PATH` | `models/fire/model.pt` | 火警权重路径 |
| `FIRE_OLLAMA_URL` | `http://127.0.0.1:11434` | 火警场景描述服务 |
| `FIRE_AGENT_URL` | `http://127.0.0.1:8001` | 火警分析服务 |

在 `agent/.env` 中可设置 `ROOMIND_API`，让 `fire_status` 工具查询其他主机的感知后端，默认 `http://127.0.0.1:8000`。

当前火警专用规则负责“忽略／观察／提醒”；Agent 补充解释，不能撤销已发出的规则提醒。尚未实现其余雨天开窗、夜间异常等风险场景。

### 接口与验证

| 接口 | 用途 |
| --- | --- |
| `POST /api/fire/start` | JSON：`source`、`location`、`semantics`，启动后台监测 |
| `POST /api/fire/stop` | 请求停止，当前模型调用结束后释放摄像头 |
| `GET /api/fire/status` | 读取运行状态、当前候选和最近事件 |
| `POST /api/fire/detect` | multipart `image`，独立单图检测 |
| Agent `POST /fire/review` | 根据 `events`、`semantic`、`history` 补充分析 |

感知后端应以单进程运行；多 worker 会各有独立状态和摄像头锁。

```bash
python3 -m unittest discover -s test/fire -v
python3 -m compileall -q app/fire app/main.py
cd agent
npm test
npm run typecheck
cd ../web
npm run typecheck
```

自动测试使用模拟检测和模型回复，不证明真实权重的识别效果。实机验收需要：普通无火画面、火焰／烟雾测试图片、弱信号持续、关闭并重新启动监测、断开摄像头，以及停掉 Ollama / Agent 后确认检测提醒仍正常。
