from __future__ import annotations

import logging
import os
from pathlib import Path

from PIL import Image

from app.inference.memory import accelerator, inference_dtype, is_oom, release_cuda
from app.inference.siglip.labels import PROMPTS, ROOM_LABELS
from app.inference.siglip.schema import ClassScore
from app.inference.weights import ensure_local_model, load_auto_model

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "google/siglip2-base-patch16-224"
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "siglip"
TOP_K = 3
log = logging.getLogger("roommind.siglip")


def _as_tensor(value):
    import torch

    if torch.is_tensor(value):
        return value
    for name in ("pooler_output", "image_embeds", "text_embeds"):
        item = getattr(value, name, None)
        if torch.is_tensor(item):
            return item
    raise RuntimeError("无法读取 SigLIP 输出的向量")


class SiglipClassifier:
    def __init__(self, model_id: str | None = None) -> None:
        model_id = model_id or os.environ.get("SIGLIP_MODEL") or MODEL_ID
        local_model = ensure_local_model(model_id, LOCAL_MODEL_DIR)
        from transformers import AutoProcessor

        self.device = accelerator()
        self.dtype = inference_dtype(self.device)
        log.info(
            "loading SigLIP model=%s path=%s device=%s dtype=%s",
            model_id,
            local_model,
            self.device,
            str(self.dtype).replace("torch.", ""),
        )
        try:
            self.processor = AutoProcessor.from_pretrained(local_model, local_files_only=True)
            self.model = load_auto_model(local_model, self.device, self.dtype)
        except RuntimeError as exc:
            if not is_oom(exc):
                raise
            raise RuntimeError(
                "加载 SigLIP2 时显存不足。请先停掉其他模型，确认还有空闲内存后再试。"
            ) from exc
        self._text_features = None
        self._logit_scale = None
        self._logit_bias = None
        log.info("SigLIP ready labels=%d", len(ROOM_LABELS))

    def classify(self, images: list[Image.Image], top_k: int = TOP_K) -> list[list[ClassScore]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not images:
            return []
        import torch

        self._ensure_text_features()
        found: list[list[ClassScore]] = []
        for image in images:
            inputs = self.processor(images=image, return_tensors="pt")
            payload = {
                key: inputs[key]
                for key in ("pixel_values", "pixel_attention_mask", "spatial_shapes")
                if key in inputs
            }
            payload = _to_device(payload, self.device, self.dtype)
            try:
                with torch.inference_mode():
                    image_features = _as_tensor(self.model.get_image_features(**payload))
            except RuntimeError as exc:
                if not is_oom(exc):
                    raise
                raise RuntimeError("SigLIP2 显存不足。") from exc
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            text_features = self._text_features.to(dtype=image_features.dtype)
            logits = image_features @ text_features.T
            logits = logits * self._logit_scale.to(dtype=logits.dtype) + self._logit_bias.to(
                dtype=logits.dtype
            )
            probs = torch.sigmoid(logits)[0]
            k = min(top_k, int(probs.shape[0]))
            values, indices = torch.topk(probs, k)
            found.append(
                [
                    ClassScore(name=ROOM_LABELS[int(index)], score=round(float(score), 4))
                    for score, index in zip(values.tolist(), indices.tolist())
                ]
            )
        return found

    def _ensure_text_features(self) -> None:
        if self._text_features is not None:
            return
        import torch

        text_inputs = self.processor(
            text=list(PROMPTS),
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].to(self.device)
        attention_mask = text_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.inference_mode():
            text_features = _as_tensor(
                self.model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            )
        self._text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        self._logit_scale = self.model.logit_scale.detach().exp()
        self._logit_bias = self.model.logit_bias.detach()

    def release(self) -> None:
        model = getattr(self, "model", None)
        self.model = None
        self.processor = None
        self._text_features = None
        self._logit_scale = None
        self._logit_bias = None
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
        release_cuda()


def _to_device(payload: dict, device: str, dtype) -> dict:
    moved = {}
    for key, value in payload.items():
        if not hasattr(value, "to"):
            moved[key] = value
        elif key == "pixel_values":
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device)
    return moved
