from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.inference.sam.generator import (
    DEFAULT_MAX_SIZE,
    DEFAULT_POINTS_PER_BATCH,
    DEFAULT_POINTS_PER_CROP,
)
from app.inference.sam.service import SAMService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAM 2.1 Tiny Automatic Mask Generation（面向 Jetson Orin Nano 8GB）",
    )
    parser.add_argument("image", type=Path, help="输入图片路径")
    parser.add_argument("--model", help="本地模型目录或 Hugging Face repo id")
    parser.add_argument("--points-per-batch", type=int, default=DEFAULT_POINTS_PER_BATCH)
    parser.add_argument("--points-per-crop", type=int, default=DEFAULT_POINTS_PER_CROP)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    args = parser.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"image not found: {args.image}")

    result = SAMService(args.model).segment(
        args.image.read_bytes(),
        args.image.name,
        points_per_batch=args.points_per_batch,
        points_per_crop=args.points_per_crop,
        max_size=args.max_size,
    )
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
