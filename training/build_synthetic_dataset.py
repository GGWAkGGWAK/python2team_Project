"""Build reviewed LayoutXLM JSONL files from synthetic OCR extraction.

Run ``extract_synthetic_ocr.py`` first.  The labels below were manually
reviewed against the six generated cards and deliberately keep OCR mistakes:
the field classifier receives OCR output at inference time as well.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DRAFT = ROOT / "synthetic_ocr_draft.json"
TRAIN = ROOT / "train.jsonl"
VALIDATION = ROOT / "validation.jsonl"


LABELS_BY_IMAGE = {
    "images/synthetic_card_01_korean_corporate.png": [
        "B-COMPANY", "B-NAME", "B-TELEPHONE", "B-MOBILE",
        "B-POSITION", "B-FAX", "O", "B-POSITION", "B-EMAIL",
        "B-ADDRESS", "B-WEBSITE",
    ],
    "images/synthetic_card_02_english_tech.png": [
        "B-COMPANY", "B-MOBILE", "B-NAME", "B-TELEPHONE", "B-FAX",
        "B-POSITION", "B-EMAIL", "B-ADDRESS", "B-WEBSITE",
    ],
    "images/synthetic_card_03_bilingual_consulting.png": [
        "B-COMPANY", "I-COMPANY", "B-NAME", "B-POSITION",
        "B-TELEPHONE", "O", "B-MOBILE", "O", "B-FAX", "O",
        "B-EMAIL", "B-ADDRESS",
    ],
    "images/synthetic_card_04_korean_startup_vertical.png": [
        "B-COMPANY", "B-NAME", "B-POSITION", "B-POSITION",
        "B-MOBILE", "O", "B-TELEPHONE", "O", "B-FAX", "O", "O",
        "B-EMAIL", "B-ADDRESS", "B-WEBSITE",
    ],
    "images/synthetic_card_05_bilingual_bio.png": [
        "B-COMPANY", "I-COMPANY", "I-COMPANY", "I-COMPANY",
        "B-NAME", "I-NAME", "B-POSITION", "I-POSITION", "O",
        "B-TELEPHONE", "O", "O", "B-MOBILE", "O", "O", "B-FAX",
        "O", "B-EMAIL", "B-ADDRESS", "I-ADDRESS", "I-ADDRESS",
    ],
    "images/synthetic_card_06_creative_studio.png": [
        "B-COMPANY", "B-NAME", "B-COMPANY", "B-POSITION",
        "B-TELEPHONE", "O", "B-MOBILE", "O", "B-FAX", "O",
        "B-EMAIL", "O", "B-ADDRESS", "B-WEBSITE",
    ],
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    draft_rows = json.loads(DRAFT.read_text(encoding="utf-8"))
    output_rows: list[dict] = []
    for row in draft_rows:
        image = row["image"]
        if image not in LABELS_BY_IMAGE:
            raise ValueError(f"검수 라벨이 없는 이미지입니다: {image}")
        labels = LABELS_BY_IMAGE[image]
        if not (len(row["words"]) == len(row["boxes"]) == len(labels)):
            raise ValueError(
                f"{image}: words={len(row['words'])}, boxes={len(row['boxes'])}, "
                f"labels={len(labels)}"
            )
        if any(len(box) != 4 or min(box) < 0 or max(box) > 1000 for box in row["boxes"]):
            raise ValueError(f"{image}: 0~1000 범위를 벗어난 좌표가 있습니다.")
        output_rows.append(
            {
                "image": image,
                "words": row["words"],
                "boxes": row["boxes"],
                "labels": labels,
            }
        )

    # Deterministic 5:1 split.  The visually distinct creative-studio card is
    # kept unseen for validation.
    write_jsonl(TRAIN, output_rows[:5])
    write_jsonl(VALIDATION, output_rows[5:])
    print(f"학습 데이터: {TRAIN} ({len(output_rows[:5])}장)")
    print(f"검증 데이터: {VALIDATION} ({len(output_rows[5:])}장)")


if __name__ == "__main__":
    main()
