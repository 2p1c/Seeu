# PyTorch → ONNX → TensorRT 这条链路怎么写

这份文档只讲一件事：RoomMind 里视觉模型是怎么从 PyTorch 权重变成 TensorRT 引擎，再在请求里跑起来的。代码主要在 `app/inference/trt/export.py`（导出和编译）和 `app/inference/trt/engine.py`（加载和推理）。

在开发板上执行：

```bash
python -m app.inference.trt --list          # 只看计划，不编译
python -m app.inference.trt --only dinov3   # 只编一个
python -m app.inference.trt                 # 三个都编
python -m app.inference.trt --force         # 引擎已存在也重编
```

入口是 `app/inference/trt/__main__.py`，它只做一件事：调用 `export.main()`。

这条链有三步，每一步的产物都能单独检查：

```text
PyTorch 权重 (.pt / safetensors)
    │  torch.onnx.export   记录网络做了哪些运算
    ▼
ONNX 文件 (.onnx)            通用计算图，和 GPU 无关
    │  trtexec --fp16       针对这块 GPU 选最快的实现
    ▼
TensorRT 引擎 (.engine)      只能在这块 GPU、这一版 TensorRT 上加载
    │  deserialize + execute_async_v3
    ▼
推理结果
```

YOLO 不走我们自己写的 ONNX，Ultralytics 的 `export(format="engine")` 内部也是先转 ONNX 再调 TensorRT，只是把两步包在一起了。

Mac 上没有 CUDA，`main()` 会直接退出。引擎必须在 Jetson 上编。

---

## 1. 加载模型：用 FP32、eager attention

SigLIP 和 DINOv3 共用 `_load()`：

```python
model = AutoModel.from_pretrained(
    path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager"
).eval()
```

这几个参数是导出时的标准写法：

| 参数 | 含义 | 为什么这样写 |
| --- | --- | --- |
| `local_files_only=True` | 只读本地权重，不访问网络 | 板上编译时不应该再下载 |
| `torch_dtype=torch.float32` | 权重按 FP32 加载 | ONNX 用 FP32 追踪最稳。降到 FP16 交给后面的 `trtexec --fp16`，不要在导出这一步降 |
| `attn_implementation="eager"` | 用普通的 attention 实现 | Transformers 默认可能用 FlashAttention 这类融合实现，ONNX 追踪经常失败。eager 是一层层算的普通实现，最容易导出 |
| `.eval()` | 切到推理模式 | 关掉 Dropout 这类训练时才有的随机层。不写的话，导出的图和线上推理不是同一个网络 |

`processor` 是预处理，不进引擎。它负责把图片缩放、归一化成模型吃的张量。引擎只认张量，不认 JPEG。

---

## 2. 决定导出网络的哪一段

同一个模型不一定整网导出。`_to_onnx` 的第三个参数 `forward(model, pixel_values)` 就是“只导出这一段”。

**SigLIP 只导出图像塔：**

```python
_to_onnx(model, processor, lambda m, x: _as_tensor(m.get_image_features(pixel_values=x)), SIGLIP_ONNX)
```

`get_image_features` 是图像塔的入口。文本塔在导出前已经跑过一遍，向量存进 `siglip2-text.npz`，所以引擎里没有文本塔。`_as_tensor` 是因为不同版本的 transformers 有时返回张量，有时返回一个带 `.image_embeds` 的对象，这里统一成张量，ONNX 才能记下来。

**DINOv3 导出“整张图 → 一个向量”：**

```python
def embed(m, x):
    out = m(pixel_values=x)
    return out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
```

模型原始输出是一串 token。分类和检索要用的是池化后的那个向量（没有 pooler 时取第 0 个 token，也就是 CLS）。把这一步写进 `forward`，引擎直接吐向量，线上就不用再在 Python 里挑输出。

---

## 3. 包一层 `nn.Module`，再 `torch.onnx.export`

这是整条链路里最常用的一段：

```python
class Wrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = model

    def forward(self, pixel_values):
        return forward(self.inner, pixel_values)

sample = processor(images=Image.new("RGB", (224, 224)), return_tensors="pt")["pixel_values"].float()

options = dict(
    input_names=["pixel_values"],
    output_names=["embedding"],
    opset_version=17,
    dynamo=False,
)
with torch.no_grad():
    torch.onnx.export(Wrapper().eval(), sample, str(onnx), **options)
```

### 为什么要包一层

`torch.onnx.export` 要求一个 `nn.Module`，并且只追踪 `forward`。原始模型的 `forward` 返回的是一个复杂对象（`pooler_output`、`last_hidden_state` 都在里面），ONNX 不知道该把哪个当输出。Wrapper 把“我真正要的那个张量”变成 `forward` 的返回值。

`self.inner = model` 必须写成模块属性。如果只是闭包里抓一个局部变量，导出时权重可能不被记进图里。

### 样例输入决定形状

`Image.new("RGB", (224, 224))` 是一张纯黑图，像素值不重要。`processor` 把它变成 `pixel_values`，形状一般是 `(1, 3, 224, 224)`：1 张图、3 个通道、224×224。

ONNX 追踪是“拿这张样例走一遍，把走过的运算记下来”。样例的形状就是引擎的输入形状。所以这里写 224，线上也必须喂 224。颜色用黑图是因为导出不看数值，只看形状和运算。

### 四个导出参数

- **`input_names=["pixel_values"]`**：给输入起名。后面 TensorRT 用这个名字绑定显存地址。
- **`output_names=["embedding"]`**：给输出起名。
- **`opset_version=17`**：ONNX 算子版本。太老会缺算子，太新 TensorRT 可能不认。17 是目前视觉模型比较稳的一档。
- **`dynamo=False`**：用老的 trace 导出，不用新的 dynamo 导出器。dynamo 还在变，视觉模型上 trace 更熟。旧版 PyTorch 没有这个参数，所以代码里 `TypeError` 时删掉再导一次。

### 为什么是 `no_grad`，不是 `inference_mode`

两者都表示“不用为反向传播记账”。`inference_mode()` 更彻底，会关掉张量的版本计数。ONNX 追踪依赖这些信息，用 `inference_mode` 经常直接失败。导出只用 `torch.no_grad()`。线上推理可以、也应该用 `inference_mode()`，那时候已经不追踪了。

### 导出完立刻丢掉模型

```python
_to_onnx(...)
del model
release_cuda()
_trtexec(...)
```

`trtexec` 编译时要在 GPU 上试很多种实现，自己就要一块内存。如果 FP32 权重还占着显存，8GB 板上编译很容易失败。`del model` 必须写在调用 `_to_onnx` 的地方：函数内部的 `del` 删不掉外面还握着的那个引用。

---

## 4. `trtexec`：把 ONNX 编译成引擎

```python
command = [
    trtexec,
    f"--onnx={onnx.resolve()}",
    f"--saveEngine={engine.resolve()}",
    "--fp16",
    "--memPoolSize=workspace:512M",
    "--builderOptimizationLevel=3",
]
subprocess.run(command, check=True)
```

这是 TensorRT 自带的命令行，JetPack 里一般在 `/usr/src/tensorrt/bin/trtexec`。等价于：

```bash
trtexec \
  --onnx=models/trt/dinov3-vits16.onnx \
  --saveEngine=models/trt/dinov3-vits16-fp16.engine \
  --fp16 \
  --memPoolSize=workspace:512M \
  --builderOptimizationLevel=3
```

| 参数 | 做什么 | 为什么这样设 |
| --- | --- | --- |
| `--onnx` | 读入计算图 | 上一步的产物 |
| `--saveEngine` | 把编译结果写成 `.engine` | 运行时只加载这个文件，不再需要 ONNX 和 PyTorch 权重 |
| `--fp16` | 能用 FP16 的层就用 FP16 | 权重和激活大约减半，Orin 的 Tensor Core 吃 FP16。个别层数值范围不够时，TensorRT 会自己留在 FP32 |
| `--memPoolSize=workspace:512M` | 编译时的临时内存上限 | TensorRT 会试每种卷积的多种实现，试的时候要草稿空间。不限制的话，编译进程本身会把 8GB 吃满 |
| `--builderOptimizationLevel=3` | 搜索 kernel 的仔细程度，0 到 5 | 5 最慢、也最吃内存。3 是速度和编译成本的折中 |

编译要几分钟，因为它在**这块 GPU 上**把候选实现实际跑一遍再挑最快的。所以 `.engine` 不能拷到另一台机器，也不能换 TensorRT 版本后继续用。

`subprocess.run(..., check=True)` 表示命令非 0 退出就抛异常，不会假装编译成功。

### YOLO 的对应写法

```python
YOLO(str(weights)).export(
    format="engine", imgsz=YOLO_IMGSZ, half=True, device=0, workspace=1
)
```

- `format="engine"`：内部走 PyTorch → ONNX → TensorRT。
- `imgsz=640`：输入边长，和 `YOLO_IMGSZ` 是同一个常量。请求里改了边长会被拒绝。
- `half=True`：就是 FP16，对应 `--fp16`。
- `device=0`：用第 0 块 GPU。
- `workspace=1`：编译临时内存 1GB。YOLO 的导出器单位是 GB，我们自己的 `trtexec` 用的是 `512M`，都是在限制编译时的内存。

---

## 5. 加载引擎

运行时不再经过 PyTorch 的模型，只反序列化 `.engine`：

```python
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
self.context = self.engine.create_execution_context()
```

三个对象各管一件事：

- **Runtime**：读文件。`deserialize_cuda_engine` 把字节变回 GPU 上的引擎。文件和当前 GPU 或 TensorRT 版本不匹配时返回 `None`，代码直接报错，提示不能从别的机器拷。
- **Engine**：模型本身，权重和选好的 kernel 都在这里。一个引擎可以建多个执行上下文，本项目每次只建一个。
- **Execution Context**：这一次推理用的中间内存。释放显存时必须把它和 engine 都丢掉。只调 `torch.cuda.empty_cache()` 还不回去，因为那块内存不是 PyTorch 申请的。

接着按名字找出唯一的输入和输出，并记下形状和数据类型：

```python
names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
```

`num_io_tensors`、`get_tensor_name`、`execute_async_v3` 是 TensorRT 8.5 以后按张量名字绑定的接口。JetPack 6.0（TensorRT 8.6）和 6.1 以后（TensorRT 10）都能用。更老的接口是按位置绑指针的 `execute_async_v2`，这里没有用。

静态引擎的 `input_shape` 里没有 `-1`。`-1` 表示动态维度，那是 optimization profile 的写法，本项目三个引擎都是静态的。

---

## 6. 一次推理在做什么

调用方只写一行，例如 DINOv3：

```python
vector = self.engine.infer(inputs["pixel_values"])
```

`inputs["pixel_values"]` 是 processor 产出的 CPU 张量。`infer` 里面是 TensorRT 推理的标准五步：

```python
if tuple(values.shape) != self.input_shape:
    raise RuntimeError(...)

blob = values.to(device="cuda", dtype=self.input_dtype).contiguous()
self.context.set_input_shape(self.input_name, self.input_shape)
output = torch.empty(out_shape, device="cuda", dtype=self.output_dtype)

self.context.set_tensor_address(self.input_name, blob.data_ptr())
self.context.set_tensor_address(self.output_name, output.data_ptr())

stream = torch.cuda.current_stream()
self.context.execute_async_v3(stream.cuda_stream)
stream.synchronize()
return output.float().cpu()
```

按顺序解释：

1. **对形状。** 静态引擎不会跟着请求改尺寸。对不上就立刻失败，比让 TensorRT 在 GPU 上报一个含糊的错更容易查。
2. **搬到 GPU，并转成引擎的 dtype。** 导出是 FP32，`trtexec --fp16` 之后输入常常是 FP16。`.to(device="cuda", dtype=...)` 同时完成拷贝和类型转换。
3. **`.contiguous()`。** 张量在内存里必须是连续的一块。有些切片在逻辑上是一个矩阵，物理上是跳着存的。TensorRT 只收一个起始地址，不连续就会读错。
4. **`set_input_shape`。** 即使形状是静态的，执行前也要再声明一次这次的形状，上下文才能算出输出形状。
5. **准备输出张量。** `torch.empty` 只分配、不初始化。结果会被 TensorRT 整个写上，不需要先填 0。
6. **`set_tensor_address` + `data_ptr()`。** 把“名叫 `pixel_values` 的输入，数据在这块 GPU 内存”告诉 TensorRT。`data_ptr()` 是张量首地址的整数。TensorRT 和 PyTorch 共用同一块 CUDA 内存，所以不用再拷一次。
7. **`execute_async_v3`。** 把计算放进 CUDA stream（GPU 的任务队列）就返回，CPU 不等它算完。
8. **`stream.synchronize()`。** CPU 在这里等 GPU 算完。不写的话，下一步 `.cpu()` 可能读到还没写完的结果。
9. **`.float().cpu()`。** 转回 FP32 再拷回 CPU。后面 SigLIP 的点积、写数据库都在 CPU 上，用 FP32 更省事。这一步会同步，所以上一步的 `synchronize` 也可以理解成“在读结果之前明确等一次”。

SigLIP 拿到图像向量之后，仍然在 PyTorch 里和缓存的文本向量做点积：

```python
image_features = self.engine.infer(inputs["pixel_values"])
logits = image_features @ text_features.T
```

引擎只替换了最重的图像塔。类别打分还是普通矩阵乘法，不值得再编进引擎。

---

## 7. 什么时候走引擎

```python
def use_engine(path: Path) -> bool:
    return tensorrt_enabled() and path.is_file() and cuda_available()
```

三个条件都成立才用引擎：没把 `ROOMIND_TENSORRT` 设成 `0`、`.engine` 文件在、这台机器有 CUDA。缺任何一个就用原来的 PyTorch。所以同一份代码在 Mac 上和在还没编译的 Jetson 上都能跑。

YOLO 的切换更早，发生在选权重文件的时候：有 `yolo26s.engine` 就把它交给 Ultralytics，否则用 `.pt`。Ultralytics 自己会加载 TensorRT 引擎，所以 YOLO 没有走 `TrtEngine`。

---

## 自己再写一个模型时的顺序

1. 写一个 `nn.Module`，`forward` 只返回你要的那个张量。
2. 用真实的 processor 做一张样例，确认形状。
3. `torch.no_grad()` 里 `torch.onnx.export`，输入输出都起名字，`opset_version=17`。
4. 先用 ONNX Runtime 跑同一张图，和 PyTorch 对一下数值。这一步项目里还没写。对不上就是导出坏了，不要急着编 TensorRT。
5. `trtexec --fp16`，并限制 workspace。
6. 运行时：反序列化 → 建 context → 张量 `.contiguous()` 放到 CUDA → `set_tensor_address` → `execute_async_v3` → `synchronize`。
7. 用完把 `context` 和 `engine` 设成 `None`，再 `empty_cache()`。

面试时可以收成一句：ONNX 只负责把网络记成一张和硬件无关的图，TensorRT 再针对这块 GPU 把图编译成引擎。导出用 FP32 和 eager attention 是为了让追踪成功，FP16 放到 `trtexec`。运行时 TensorRT 不认识 PyTorch 的张量，只认显存地址，所以用 `data_ptr()` 把两边接起来。
