from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.dino.detector import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_MAX_SIZE,
    DEFAULT_TEXT_THRESHOLD,
)
from app.dino.service import DinoService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grounding DINO Tiny 开放词汇检测（面向 Jetson Orin Nano 8GB）",
    )
    parser.add_argument("image", type=Path, help="输入图片路径")
    parser.add_argument(
        "--prompt",
        required=True,
        help='英文短语，小写并用句号分隔，例如 "a chair. a sofa."',
    )
    parser.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    args = parser.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"image not found: {args.image}")

    result = DinoService().detect(
        args.image.read_bytes(),
        args.prompt,
        args.image.name,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_size=args.max_size,
    )
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
