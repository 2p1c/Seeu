from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

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
