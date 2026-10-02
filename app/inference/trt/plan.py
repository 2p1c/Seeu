from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from app.inference.trt.cache import read_text_cache

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENGINE_DIR = PROJECT_ROOT / "models" / "trt"
SIGLIP_ENGINE = ENGINE_DIR / "siglip2-image-fp16.engine"
SIGLIP_ONNX = ENGINE_DIR / "siglip2-image.onnx"
SIGLIP_CACHE = ENGINE_DIR / "siglip2-text.npz"
DINOV3_ENGINE = ENGINE_DIR / "dinov3-vits16-fp16.engine"
DINOV3_ONNX = ENGINE_DIR / "dinov3-vits16.onnx"
YOLO_IMGSZ = 640

# Orin Nano 8GB 是 CPU/GPU 共用的统一内存。下面的保留给系统、页面、Agent 和空载的 Ollama。
BOARD_MB = 8192
RESERVED_MB = 1800


@dataclass(frozen=True)
class ModelPlan:
    name: str
    runtime: str
    budget_mb: int
    reason: str
    artifact: str = ""


PLANS: tuple[ModelPlan, ...] = (
    ModelPlan(
        "yolo26s", "tensorrt", 400,
        "输入边长固定，Ultralytics 能直接导出 engine，摄像头会反复调用。",
        "models/yolo26s.engine",
    ),
    ModelPlan(
        "siglip2-base", "tensorrt", 700,
        "类别表是固定的，文本向量可以提前算好，引擎只跑图像塔。",
        "models/trt/siglip2-image-fp16.engine",
    ),
    ModelPlan(
        "dinov3-vits16", "tensorrt", 500,
        "纯视觉、输入尺寸固定，输出就是一个向量。",
        "models/trt/dinov3-vits16-fp16.engine",
    ),
    ModelPlan(
        "sam2.1-tiny", "pytorch", 2800,
        "耗显存的是逐点循环和每批点数，只把图像编码器换成引擎解决不了尖峰。",
    ),
    ModelPlan(
        "grounding-dino-tiny", "pytorch", 2200,
        "每次提示词长度不同，网络里还有 deformable attention，不适合做成静态引擎。",
    ),
    ModelPlan(
        "qwen3-vl-2b", "ollama", 3400,
        "多模态大模型继续交给 Ollama 装卸。8GB 上再维护一份 TensorRT-LLM 引擎不划算。",
    ),
)


def slot_mb() -> int:
    return BOARD_MB - RESERVED_MB


def tensorrt_enabled() -> bool:
    return os.environ.get("ROOMIND_TENSORRT", "1").strip().lower() not in {"0", "off", "false"}


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def use_engine(path: Path) -> bool:
    """有 CUDA、有引擎文件、没被 ROOMIND_TENSORRT=0 关掉，才用引擎；否则回退到 PyTorch。"""
    return tensorrt_enabled() and path.is_file() and cuda_available()


def select_yolo_weights(pt_path: Path) -> Path:
    engine = pt_path.with_suffix(".engine")
    return engine if use_engine(engine) else pt_path


def engine_ready(item: ModelPlan) -> bool | None:
    if item.runtime != "tensorrt":
        return None
    if item.name == "siglip2-base":
        from app.inference.siglip.labels import PROMPTS, ROOM_LABELS

        if read_text_cache(SIGLIP_CACHE, ROOM_LABELS, PROMPTS) is None:
            return False
    return (PROJECT_ROOT / item.artifact).is_file()


def deployment_status() -> dict:
    return {
        "board": "jetson-orin-nano",
        "memory_mb": BOARD_MB,
        "reserved_mb": RESERVED_MB,
        "slot_mb": slot_mb(),
        "tensorrt": tensorrt_enabled(),
        "policy": "exclusive",
        "models": [{**asdict(item), "engine_ready": engine_ready(item)} for item in PLANS],
    }
