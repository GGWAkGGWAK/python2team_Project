"""Extract OCR words and normalized boxes from the generated training cards.

This creates an intermediate JSON file for human label review.  It does not
modify the source images and does not guess field labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from cardocr.ocr_engine import PaddleOCREngine


ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
OUTPUT = ROOT / "synthetic_ocr_draft.json"


def normalize_box(box: list[list[int]], width: int, height: int) -> list[int]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return [
        max(0, min(1000, round(min(xs) / width * 1000))),
        max(0, min(1000, round(min(ys) / height * 1000))),
        max(0, min(1000, round(max(xs) / width * 1000))),
        max(0, min(1000, round(max(ys) / height * 1000))),
    ]


def main() -> None:
    paths = sorted(IMAGES.glob("synthetic_card_*.png"))
    if not paths:
        raise SystemExit(f"합성 명함을 찾지 못했습니다: {IMAGES}")

    engine = PaddleOCREngine()
    rows: list[dict] = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] OCR: {path.name}", flush=True)
        # cv2.imread may fail on Korean Windows paths.  Reading bytes first
        # keeps the extractor independent from the path's character set.
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"이미지를 열 수 없습니다: {path}")
        height, width = image.shape[:2]
        lines = engine.recognize(image)
        lines.sort(
            key=lambda line: (
                min(point[1] for point in line.box),
                min(point[0] for point in line.box),
            )
        )
        rows.append(
            {
                "image": f"images/{path.name}",
                "width": width,
                "height": height,
                "words": [line.text for line in lines],
                "boxes": [normalize_box(line.box, width, height) for line in lines],
                "confidences": [line.confidence for line in lines],
            }
        )

    OUTPUT.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"OCR 초안 저장 완료: {OUTPUT}")


if __name__ == "__main__":
    main()
