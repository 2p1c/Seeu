from __future__ import annotations

import logging
import os
from pathlib import Path

from PIL import Image

from app.inference.memory import accelerator, inference_dtype, is_oom, release_cuda
from app.inference.weights import ensure_local_model, load_auto_model

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "dinov3"
log = logging.getLogger("roommind.dinov3")


class Dinov3Encoder:
    def __init__(self, model_id: str | None = None) -> None:
        model_id = model_id or os.environ.get("DINOV3_MODEL") or MODEL_ID
        local_model = ensure_local_model(model_id, LOCAL_MODEL_DIR)
        self.device = accelerator()
        self.dtype = inference_dtype(self.device)
        self.model = None
        self.engine = None
        from transformers import AutoImageProcessor

        self.processor = AutoImageProcessor.from_pretrained(local_model, local_files_only=True)
        if self._try_tensorrt():
            return
        log.info(
            "loading DINOv3 model=%s path=%s device=%s dtype=%s",
            model_id,
            local_model,
            self.device,
            str(self.dtype).replace("torch.", ""),
        )
        try:
            self.model = load_auto_model(local_model, self.device, self.dtype)
        except RuntimeError as exc:
            if not is_oom(exc):
                raise
            raise RuntimeError(
                "加载 DINOv3 时显存不足。请先停掉其他模型，确认还有空闲内存后再试。"
            ) from exc
        log.info("DINOv3 ready")

    def _try_tensorrt(self) -> bool:
        from app.inference.trt.engine import TrtEngine
        from app.inference.trt.plan import DINOV3_ENGINE, use_engine

        if not use_engine(DINOV3_ENGINE):
            return False
        self.engine = TrtEngine(DINOV3_ENGINE)
        log.info("DINOv3 TensorRT ready %s", DINOV3_ENGINE)
        return True

    def embed(self, images: list[Image.Image]) -> list[list[float]]:
        if not images:
            return []
        import torch

        vectors: list[list[float]] = []
        for image in images:
            inputs = self.processor(images=image, return_tensors="pt")
            if self.engine is not None:
                vector = self.engine.infer(inputs["pixel_values"])[0].tolist()
                vectors.append([round(float(value), 6) for value in vector])
                continue
            pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
            try:
                with torch.inference_mode():
                    outputs = self.model(pixel_values=pixel_values)
            except RuntimeError as exc:
                if not is_oom(exc):
                    raise
                raise RuntimeError("DINOv3 显存不足。") from exc
            pooled = outputs.pooler_output
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            vector = pooled[0].detach().float().cpu().tolist()
            vectors.append([round(float(value), 6) for value in vector])
        return vectors

    def release(self) -> None:
        model = getattr(self, "model", None)
        engine = self.engine
        self.model = None
        self.engine = None
        self.processor = None
        if engine is not None:
            engine.close()
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
            release_cuda()
