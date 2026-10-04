from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

from app.inference.memory import release_cuda
from app.inference.trt.cache import read_text_cache, save_text_cache
from app.inference.trt.lease import GpuLease, release_remote
from app.inference.trt.plan import (
    DINOV3_ENGINE,
    DINOV3_ONNX,
    SIGLIP_CACHE,
    SIGLIP_ENGINE,
    SIGLIP_ONNX,
    cuda_available,
    deployment_status,
)

log = logging.getLogger("roommind.trt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在 Jetson 上把 SigLIP 图像塔和 DINOv3 编成 TensorRT 引擎")
    parser.add_argument("--list", action="store_true", help="只打印部署计划")
    parser.add_argument("--only", choices=("siglip", "dinov3"), help="只构建一个")
    parser.add_argument("--force", action="store_true", help="引擎已经存在时也重新构建")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.list:
        print(json.dumps(deployment_status(), ensure_ascii=False, indent=2))
        return 0
    if not cuda_available():
        log.error("构建引擎需要开发板上的 CUDA。这台机器没有 CUDA，推理会继续走 PyTorch。")
        return 1
    steps = {"siglip": _export_siglip, "dinov3": _export_dinov3}
    with GpuLease("trt-export", timeout_s=120):
        release_remote()
        for name in [args.only] if args.only else steps:
            log.info("build %s", name)
            steps[name](args.force)
            release_cuda()
    print(json.dumps(deployment_status(), ensure_ascii=False, indent=2))
    return 0


def _export_siglip(force: bool) -> None:
    import torch
    from transformers import AutoProcessor

    from app.inference.siglip.classifier import LOCAL_MODEL_DIR, MODEL_ID, _as_tensor
    from app.inference.siglip.labels import PROMPTS, ROOM_LABELS

    if SIGLIP_ENGINE.is_file() and read_text_cache(SIGLIP_CACHE, ROOM_LABELS, PROMPTS) and not force:
        log.info("已有 %s，跳过", SIGLIP_ENGINE)
        return
    model, processor = _load(MODEL_ID, LOCAL_MODEL_DIR, AutoProcessor)
    text = processor(text=list(PROMPTS), padding="max_length", max_length=64, return_tensors="pt")
    with torch.no_grad():
        features = _as_tensor(
            model.get_text_features(input_ids=text["input_ids"], attention_mask=text.get("attention_mask"))
        )
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    save_text_cache(
        SIGLIP_CACHE,
        features.numpy(),
        float(model.logit_scale.exp()),
        model.logit_bias.detach().numpy(),
        ROOM_LABELS,
        PROMPTS,
    )
    _to_onnx(model, processor, lambda m, x: _as_tensor(m.get_image_features(pixel_values=x)), SIGLIP_ONNX)
    del model
    release_cuda()
    _trtexec(SIGLIP_ONNX, SIGLIP_ENGINE)


def _export_dinov3(force: bool) -> None:
    from transformers import AutoImageProcessor

    from app.inference.dinov3.encoder import LOCAL_MODEL_DIR, MODEL_ID

    if DINOV3_ENGINE.is_file() and not force:
        log.info("已有 %s，跳过", DINOV3_ENGINE)
        return

    def embed(m, x):
        out = m(pixel_values=x)
        return out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]

    model, processor = _load(MODEL_ID, LOCAL_MODEL_DIR, AutoImageProcessor)
    _patch_dinov3_rope_for_tensorrt()
    _to_onnx(model, processor, embed, DINOV3_ONNX)
    del model
    release_cuda()
    _trtexec(DINOV3_ONNX, DINOV3_ENGINE)


def _patch_dinov3_rope_for_tensorrt() -> None:
    """DINOv3 的 `angles.tile(2)` 会导出成 ONNX If，then/else 形状是 [2] 和 [1]。
    TensorRT 10.3 不允许这种 If。对二维 angles 来说 `tile((1, 2))` 数值相同，且不会生成 If。
    """
    import torch
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTRopePositionEmbedding

    if getattr(DINOv3ViTRopePositionEmbedding.forward, "_roomind_trt", False):
        return
    original = DINOv3ViTRopePositionEmbedding.forward

    def forward(self, pixel_values):
        orig_tile = torch.Tensor.tile

        def tile(tensor, *dims):
            if dims == (2,):
                return orig_tile(tensor, (1, 2))
            return orig_tile(tensor, *dims)

        torch.Tensor.tile = tile
        try:
            return original(self, pixel_values)
        finally:
            torch.Tensor.tile = orig_tile

    forward._roomind_trt = True
    DINOv3ViTRopePositionEmbedding.forward = forward


def _load(model_id: str, local_dir: Path, processor_cls):
    import torch
    from transformers import AutoModel

    from app.inference.weights import ensure_local_model

    path = ensure_local_model(model_id, local_dir)
    # eager attention 导出 ONNX 最稳；FP32 导出，精度交给 trtexec --fp16 去降。
    model = AutoModel.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    return model, processor_cls.from_pretrained(path, local_files_only=True)


def _to_onnx(model, processor, forward, onnx: Path) -> None:
    """forward(model, pixel_values) 决定导出的是网络的哪一段。"""
    import torch
    from PIL import Image

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = model

        def forward(self, pixel_values):
            return forward(self.inner, pixel_values)

    sample = processor(images=Image.new("RGB", (224, 224)), return_tensors="pt")["pixel_values"].float()
    onnx.parent.mkdir(parents=True, exist_ok=True)
    options = dict(input_names=["pixel_values"], output_names=["embedding"], opset_version=17, dynamo=False)
    log.info("onnx %s shape=%s", onnx, tuple(sample.shape))
    # inference_mode 会让 ONNX 追踪失败，这里只能用 no_grad。
    with torch.no_grad():
        try:
            torch.onnx.export(Wrapper().eval(), sample, str(onnx), **options)
        except TypeError:  # 旧版 torch 没有 dynamo 参数
            options.pop("dynamo")
            torch.onnx.export(Wrapper().eval(), sample, str(onnx), **options)


def _trtexec(onnx: Path, engine: Path) -> None:
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    command = [
        trtexec,
        f"--onnx={onnx.resolve()}",
        f"--saveEngine={engine.resolve()}",
        "--fp16",
        "--memPoolSize=workspace:512M",
        "--builderOptimizationLevel=3",
    ]
    log.info("%s", " ".join(command))
    subprocess.run(command, check=True)
