from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps


def open_rgb(data: bytes) -> Image.Image:
    """按照片里的方向标记转正，再转成 RGB。手机相册看到的方向和像素方向会因此一致。"""
    image = ImageOps.exif_transpose(Image.open(BytesIO(data)))
    return image.convert("RGB")
