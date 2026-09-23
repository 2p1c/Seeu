from __future__ import annotations

DEFAULT_MAX_OBJECTS = 12
DEFAULT_MIN_AREA_RATIO = 0.002
DEFAULT_BOX_IOU = 0.7
MIN_BOX_AREA = 64


def box_area(box: list[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def box_iou(left: list[int], right: list[int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    if intersection == 0:
        return 0.0
    union = box_area(left) + box_area(right) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def select_instances(
    instances: list,
    *,
    image_area: int,
    max_objects: int = DEFAULT_MAX_OBJECTS,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
    box_iou_threshold: float = DEFAULT_BOX_IOU,
) -> list:
    if max_objects < 1:
        raise ValueError("max_objects must be >= 1")
    if not 0 <= min_area_ratio <= 1:
        raise ValueError("min_area_ratio must be between 0 and 1")
    min_area = max(MIN_BOX_AREA, int(image_area * min_area_ratio))
    ranked = [item for item in instances if box_area(item.bbox) >= min_area]
    ranked.sort(key=lambda item: box_area(item.bbox), reverse=True)
    kept = []
    for item in ranked:
        if any(box_iou(item.bbox, other.bbox) >= box_iou_threshold for other in kept):
            continue
        kept.append(item)
        if len(kept) >= max_objects:
            break
    return kept
