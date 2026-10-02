from __future__ import annotations

from pathlib import Path

import numpy as np


def save_text_cache(path: Path, text_features, logit_scale: float, logit_bias, labels, prompts) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        text_features=np.asarray(text_features, dtype=np.float32),
        logit_scale=np.float32(logit_scale),
        logit_bias=np.asarray(logit_bias, dtype=np.float32),
        labels=np.array(labels),
        prompts=np.array(prompts),
    )


def read_text_cache(path: Path, labels, prompts) -> tuple[np.ndarray, float, np.ndarray] | None:
    """返回 (文本向量, logit_scale, logit_bias)。类别表和缓存对不上时返回 None。"""
    if not path.is_file():
        return None
    with np.load(path) as data:
        if data["labels"].tolist() != list(labels) or data["prompts"].tolist() != list(prompts):
            return None
        return data["text_features"], float(data["logit_scale"]), data["logit_bias"]
