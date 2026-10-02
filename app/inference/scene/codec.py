from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

JPEG_QUALITY = 85


def crop_image(image: Image.Image, bbox: list[int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), image.width - 1))
    y1 = max(0, min(int(y1), image.height - 1))
    x2 = max(x1, min(int(x2), image.width - 1))
    y2 = max(y1, min(int(y2), image.height - 1))
    return image.crop((x1, y1, x2 + 1, y2 + 1))


def jpeg_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_labeled_image(image: Image.Image, boxes: list[tuple[list[int], str]]) -> Image.Image:
    """在画面上画选中物体的框和标签。框坐标含两端像素。"""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    width = max(2, annotated.width // 320)
    for bbox, label in boxes:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=(20, 108, 84), width=width)
        draw.text((x1 + width, max(0, y1 - 12)), label, fill=(20, 108, 84))
    return annotated


def save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=JPEG_QUALITY)
