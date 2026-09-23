from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SegmentInstance:
    mask: Any
    score: float
    bbox: list[int]


def bbox_from_mask(mask: Any) -> list[int] | None:
    import numpy as np

    binary = np.asarray(mask, dtype=bool)
    ys, xs = np.where(binary)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_to_rle(mask: Any) -> dict[str, Any]:
    """COCO 未压缩 RLE。size 为 [高, 宽]，counts 按列展开，背景行程在前。"""
    import numpy as np

    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("mask must be a 2D array")
    flat = binary.reshape(-1, order="F")
    height, width = binary.shape
    if flat.size == 0:
        return {"size": [int(height), int(width)], "counts": []}
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.concatenate(([0], changes, [flat.size])))
    if flat[0] == 1:
        runs = np.concatenate(([0], runs))
    return {"size": [int(height), int(width)], "counts": [int(value) for value in runs]}


def score_at(scores: list[Any] | Any, index: int) -> float:
    try:
        length = len(scores)
    except TypeError:
        return 0.0
    if index >= length:
        return 0.0
    value = scores[index]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item") and not isinstance(value, str):
        value = value.item()
    return round(float(value), 4)
