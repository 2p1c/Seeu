from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger("roommind.trt")


def _system_tensorrt_dir() -> Path | None:
    extra = Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")
    if (extra / "tensorrt").is_dir():
        return extra
    return None


def _import_tensorrt():
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        extra = _system_tensorrt_dir()
        if extra is not None and str(extra) not in sys.path:
            # Jetson 的 python3-libnvinfer 装在系统 dist-packages 里，conda/venv 默认看不见。
            # 接到 path 末尾，避免盖掉环境里的 numpy。
            sys.path.append(str(extra))
            log.info("using system TensorRT at %s", extra)
            try:
                import tensorrt as trt
                return trt
            except ImportError:
                pass
        raise RuntimeError(
            "找不到 TensorRT Python 包。Jetson 上应安装 python3-libnvinfer，不要 pip install tensorrt-cu12。"
        ) from None


class TrtEngine:
    """TensorRT 8.5+ 的静态引擎，一个输入一个输出。输入输出地址用 PyTorch 的 CUDA 张量。"""

    def __init__(self, path: Path) -> None:
        trt = _import_tensorrt()
        import torch

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"无法加载 {path}。引擎绑定构建时的 GPU 和 TensorRT 版本，不能从别的机器拷过来。")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output_name = next(n for n in names if n != self.input_name)
        dtypes = {trt.float16: torch.float16, trt.float32: torch.float32}
        self.input_dtype = dtypes[self.engine.get_tensor_dtype(self.input_name)]
        self.output_dtype = dtypes[self.engine.get_tensor_dtype(self.output_name)]
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))

    def infer(self, values):
        """values 是 CPU 上的 torch 张量，返回 CPU 上的 float32 张量。"""
        import torch

        if tuple(values.shape) != self.input_shape:
            raise RuntimeError(f"引擎输入形状是 {self.input_shape}，这次是 {tuple(values.shape)}。改尺寸要重新导出。")
        blob = values.to(device="cuda", dtype=self.input_dtype).contiguous()
        self.context.set_input_shape(self.input_name, self.input_shape)
        output = torch.empty(
            tuple(self.context.get_tensor_shape(self.output_name)), device="cuda", dtype=self.output_dtype
        )
        self.context.set_tensor_address(self.input_name, blob.data_ptr())
        self.context.set_tensor_address(self.output_name, output.data_ptr())
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT 执行失败")
        stream.synchronize()
        return output.float().cpu()

    def close(self) -> None:
        # empty_cache 只归还 PyTorch 缓存分配器里的块。引擎占用要丢掉 context 才还得回去。
        self.context = None
        self.engine = None
        from app.inference.memory import release_cuda

        release_cuda()
