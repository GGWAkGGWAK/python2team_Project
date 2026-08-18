"""Fine-tune LayoutXLM for business-card token classification.

Input JSONL rows contain: image, words, boxes (0..1000), labels.
Run this on a CUDA-capable Linux/Colab environment when possible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    LayoutLMv2ImageProcessor,
    LayoutXLMProcessor,
    LayoutXLMTokenizerFast,
    Trainer,
    TrainingArguments,
)


BASE_MODEL = "microsoft/layoutxlm-base"
LABELS = [
    "O",
    "B-NAME", "I-NAME",
    "B-COMPANY", "I-COMPANY",
    "B-POSITION", "I-POSITION",
    "B-ADDRESS", "I-ADDRESS",
    "B-TELEPHONE", "I-TELEPHONE",
    "B-MOBILE", "I-MOBILE",
    "B-FAX", "I-FAX",
    "B-EMAIL", "I-EMAIL",
    "B-WEBSITE", "I-WEBSITE",
]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            lengths = {len(row[key]) for key in ("words", "boxes", "labels")}
            if len(lengths) != 1:
                raise ValueError(f"{path}:{line_number} words/boxes/labels 길이가 다릅니다.")
            unknown = set(row["labels"]) - set(LABEL2ID)
            if unknown:
                raise ValueError(f"{path}:{line_number} 알 수 없는 라벨: {sorted(unknown)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"학습 데이터가 비어 있습니다: {path}")
    return rows


class CardDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path, processor) -> None:
        self.rows = rows
        self.root = root
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(self.root / row["image"]).convert("RGB")
        encoded = self.processor(
            images=image,
            text=row["words"],
            boxes=row["boxes"],
            word_labels=[LABEL2ID[label] for label in row["labels"]],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/business-card-layoutxlm"))
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    tokenizer = LayoutXLMTokenizerFast.from_pretrained(BASE_MODEL)
    image_processor = LayoutLMv2ImageProcessor(apply_ocr=False)
    # LayoutLMv2Processor only accepts LayoutLMv2 tokenizers.  LayoutXLM has
    # its own processor even though it reuses LayoutLMv2ImageProcessor.
    processor = LayoutXLMProcessor(image_processor, tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_rows = load_rows(args.train)
    validation_rows = load_rows(args.validation) if args.validation else []
    train_dataset = CardDataset(train_rows, args.train.parent, processor)
    validation_dataset = (
        CardDataset(validation_rows, args.validation.parent, processor)
        if validation_rows
        else None
    )

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if validation_dataset else "no",
        load_best_model_at_end=bool(validation_dataset),
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    processor.save_pretrained(str(args.output))
    print(f"학습 모델 저장 완료: {args.output.resolve()}")


if __name__ == "__main__":
    main()
