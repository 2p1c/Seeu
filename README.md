<h1 align="center">Seeu</h1>

目标环境是 Jetson Orin Nano。电脑（最好是M芯片Mac或者带GPU的电脑）可以按下面的步骤联调。感知（Python FastAPI）、页面（Hono）和 Agent（TypeScript）分开跑，数据库还没接。

<p align="center">
  <img src="docs/architecture.jpg" alt="Seeu 技术架构">
</p>

**进度**

| 模块 | 现状 |
| --- | --- |
| YOLO | 可用。`POST /api/yolo/detect` 收图；`python -m app.yolo` 接摄像头 + ByteTrack。页面不调用它。 |
| SAM | 可用。`POST /api/sam/segment`，SAM 2.1 Tiny 自动分割。可设每边点数和每批点数。 |
| DINO | 可用。`POST /api/dino/detect`，Grounding DINO Tiny。英文短语检测，返回框和置信度。没有检出时页面不显示带框图。 |
| VLM | 可用。`POST /api/vlm/analyze`，Ollama `qwen3-vl:2b-instruct`。物体用自然语言填写。 |
| 场景流水线 | 可用。`POST /api/scene`。SAM 分割后，对每个物体依次做 SigLIP2 分类、DINOv3 向量和 VLM 描述。模型逐个加载，用完即释放。 |
| 前端 | 可用。感知页 Hono `:8080`。摄像头预览，或上传图片跑 VLM / DINO / SAM，也可按顺序跑。 |
| Agent | 可用。`:8001` 同时提供对话页。唯一工具 `status`，通过 `GET /api/memory/latest` 读感知服务里最新的画面。 |
| 物体记忆库 | 可用。PostgreSQL + pgvector，跑在 Docker 容器 `roomind-db` 里，只有感知服务直连。每次 `POST /api/scene` 成功后存一个画面和它的物体。还没有跨画面判断同一物体。 |
| 预处理流水线 / PTZ | 未做。 |

**目录**

```text
app/main.py              感知服务，端口 8000
app/cli.py               roomind 命令
app/camera/              摄像头。Linux 优先 V4L2
app/yolo/                检测、跟踪
app/inference/sam/       SAM 2.1 Tiny
app/inference/dino/      Grounding DINO Tiny
app/inference/vlm/       Ollama 描述
app/inference/siglip/    SigLIP2 零样本类别
app/inference/dinov3/    DINOv3 图像向量
app/inference/scene/     一张图里各物体的分割、类别、向量和描述
web/                     页面，端口 8080，结果图从 tests/tmp/ 读取
agent/                   Agent 服务和对话页，端口 8001
agent/.env               本机密钥，不进 git
tests/tmp/               推理结果图和 boxes JSON
app/object_memory.py     物体记忆库（PostgreSQL + pgvector）
models/                  本机权重，不进 git
```

## 在电脑上测试

需要 Python 3.10+、Node.js 20+。测 VLM 还要本机 Ollama。三个进程各开一个终端，都从仓库根目录进入。

### 1. 安装依赖

NVIDIA 显卡先按 [pytorch.org](https://pytorch.org) 装 CUDA 版 PyTorch，再装下面的依赖，避免被 CPU 版 torch 占住。Apple Silicon 直接装，会走 MPS。没有独显也能跑，SAM 和 DINO 会慢。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch          # NVIDIA 机器换成 pytorch.org 给出的 CUDA 安装命令
pip install -r requirements.txt
pip install -e .           # 注册 roomind。之后改代码不用重装
```

`requirements.txt` 把 `transformers` 限制在 5.0 之前，Grounding DINO 的接口还停在 4.x。DINOv3 需要 4.56 及以上。

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

# SigLIP2，下到 models/siglip/
huggingface-cli download google/siglip2-base-patch16-224 \
  --local-dir models/siglip \
  --endpoint "$HF_ENDPOINT"

# DINOv3 ViT-S/16，下到 models/dinov3/。这个仓库要先在网页上同意协议
huggingface-cli download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir models/dinov3 \
  --endpoint "$HF_ENDPOINT"

# VLM
curl -fsSL https://ollama.com/install.sh | sh    # Windows 用官网安装包
ollama pull qwen3-vl:2b-instruct
```

目录里要能看到这些文件才算下完：`models/yolo26s.pt`、`models/sam/config.json` 和同目录的 `*.safetensors`、`models/dino/` 下的 `models--IDEA-Research--grounding-dino-tiny`、`models/siglip/config.json`、`models/dinov3/config.json`。后两个只在跑 `POST /api/scene` 时需要。

### 3. 开启服务

先起物体记忆库。只有 `POST /api/scene` 和 `GET /api/memory/latest` 用到它，其他接口不开也能测。

```bash
docker run -d --name roomind-db --restart unless-stopped \
  -e POSTGRES_USER=roomind -e POSTGRES_PASSWORD=roomind -e POSTGRES_DB=roomind -e TZ=Asia/Shanghai \
  -p 127.0.0.1:5432:5432 -v roomind-pg:/var/lib/postgresql/data pgvector/pgvector:pg17
```

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

结果图在 `tests/tmp/`，页面从这里读 `*_annotated.jpg`、`*_overlay.jpg`、`*_masks.png`。

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

curl -s http://127.0.0.1:8000/api/scene \
  -F image=@photo.jpg \
  -F time="2026-09-23T14:00:00+08:00"

curl -s http://127.0.0.1:8001/complete \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"客厅现在有什么？"}]}'
```

不经过 HTTP 时：

```bash
python3 -m app.inference.dino photo.jpg --prompt "a chair. a sofa."
python3 -m app.inference.sam photo.jpg --points-per-crop 16 --points-per-batch 8
python3 -m app.yolo --list
python3 -m app.yolo --source 0 --show
```

`POST /api/scene` 先用 SAM 得到每个物体的 mask 和框，再按框裁剪。裁剪图依次送给 SigLIP2（top 3 类别）、DINOv3（向量）和 Qwen3-VL（物体本身和位置）。同一时间只留一个 PyTorch 模型，进入 VLM 之前会先卸掉 Ollama 里的模型，描述完再释放。框的坐标在 SAM 缩放过的画面上，响应里的 `width` 和 `height` 就是这张画面。面积太小或互相遮挡严重的框会丢掉，默认最多 12 个。SigLIP 的候选类别在 `app/inference/siglip/labels.py`，分数是各自的 sigmoid，不是加起来等于 1 的概率。

返回的 JSON 以这一张图为一条记录：`timestamp`、`filename`、`width`、`height`，`objects` 里每个物体有 `mask`（COCO RLE）、`bounding_box`、`embedding`、`crop`（JPEG base64）、`class`、`description`。同一份文件写在 `tests/tmp/<图片名>_scene.json`。

同时写进物体记忆库：`frames` 表一个画面一行，`objects` 表一个物体一行。向量是 pgvector 的 `vector(384)`，可以直接用 `<=>` 算余弦距离；裁剪图存成 JPEG 字节（`bytea`）。连接串由 `ROOMIND_DATABASE_URL` 决定，默认 `postgresql://roomind:roomind@127.0.0.1:5432/roomind`。第一次连接时自动建表。`curl -s http://127.0.0.1:8000/api/memory/latest` 返回最新画面的物体（类别、框、九宫格位置、描述，不含 mask、向量和裁剪图）；还没有画面时返回 404。Agent 的 `status` 读的就是这个接口，地址由 `ROOMIND_API` 决定。

## 可能遇到的错误

**感知服务未连接，或页面打不开结果图。** `roomind` 和 `npm start` 要同时在跑，并且都使用这份仓库。页面读的是仓库里的 `tests/tmp/`。改完 Python 后要重启 `roomind`，改完页面后刷新浏览器。

**`本地没有完整的 Grounding DINO 权重。`** 还没执行上面的 `huggingface-cli download ... --cache-dir`。DINO 不会在推理时自动下载。

**`无法下载 facebook/sam2.1-hiera-tiny。`** 网络访问不了 Hugging Face。先 `export HF_ENDPOINT=https://hf-mirror.com` 再重试。SAM 进程启动时就会把这个镜像写成默认值。

**`无法下载 yolo26s.pt` 或文件不完整。** 手动执行报错里的 `wget`。权重大于 5MB 才会被当成有效文件。

**`加载 Grounding DINO 时显存不足。`** 8GB 统一内存上，Ollama 的 VLM 还占着内存。先 `ollama ps`，再 `ollama stop qwen3-vl:2b-instruct`，用 `free -h` 确认至少还有约 2GB 可用。`POST /api/scene` 会自己按 SAM、SigLIP2、DINOv3、VLM 的顺序装卸模型；不要在它还没跑完时再开别的推理请求。

**`无法下载 google/siglip2-base-patch16-224。` 或 `无法下载 facebook/dinov3-vits16-pretrain-lvd1689m。`** 先 `export HF_ENDPOINT=https://hf-mirror.com`。DINOv3 还要在模型页面同意协议，并 `huggingface-cli login`。

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
