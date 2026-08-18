"""Optional LayoutXLM field classifier layered on top of OCR and rules."""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .ocr_engine import OCRLine
from .parser import ParsedCard


AI_FIELDS = {"name", "company", "job_title", "address"}
LABEL_TO_FIELD = {
    "NAME": "name",
    "PERSON": "name",
    "COMPANY": "company",
    "ORGANIZATION": "company",
    "ORG": "company",
    "POSITION": "job_title",
    "TITLE": "job_title",
    "JOB_TITLE": "job_title",
    "ADDRESS": "address",
}


@dataclass(frozen=True, slots=True)
class FieldCandidate:
    value: str
    confidence: float


def normalize_box(box: Iterable[Iterable[int]], width: int, height: int) -> list[int]:
    """Convert a four-point OCR polygon to LayoutXLM's 0..1000 rectangle."""
    points = [list(point) for point in box]
    if not points or width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x0 = max(0, min(1000, round(min(xs) * 1000 / width)))
    y0 = max(0, min(1000, round(min(ys) * 1000 / height)))
    x1 = max(x0, min(1000, round(max(xs) * 1000 / width)))
    y1 = max(y0, min(1000, round(max(ys) * 1000 / height)))
    return [x0, y0, x1, y1]


def _field_from_label(label: str) -> tuple[str, bool]:
    normalized = str(label or "").strip().upper()
    begins = normalized.startswith("B-")
    if normalized.startswith(("B-", "I-")):
        normalized = normalized[2:]
    return LABEL_TO_FIELD.get(normalized, ""), begins


def merge_ai_candidates(
    parsed: ParsedCard,
    candidates: dict[str, FieldCandidate],
    *,
    fill_threshold: float = 0.72,
    override_threshold: float = 0.88,
) -> ParsedCard:
    """Use AI for ambiguous fields while retaining deterministic contact rules."""
    result = replace(parsed)
    for field, candidate in candidates.items():
        if field not in AI_FIELDS or not candidate.value.strip():
            continue
        current = str(getattr(result, field, "") or "").strip()
        required = override_threshold if current else fill_threshold
        if candidate.confidence >= required:
            setattr(result, field, candidate.value.strip())
    return result


class LayoutXLMRuntime:
    """Lazy Hugging Face runtime for a business-card fine-tuned LayoutXLM model."""

    def __init__(self, model_dir: Path) -> None:
        from transformers import (
            AutoModelForTokenClassification,
            LayoutLMv2ImageProcessor,
            LayoutXLMProcessor,
            LayoutXLMTokenizerFast,
        )

        self.model_dir = model_dir
        tokenizer = LayoutXLMTokenizerFast.from_pretrained(str(model_dir))
        image_processor = LayoutLMv2ImageProcessor(apply_ocr=False)
        # LayoutXLM uses LayoutLMv2's image processor, but its tokenizer must
        # be paired with the LayoutXLM-specific processor.
        self.processor = LayoutXLMProcessor(image_processor, tokenizer)
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
        self.model.eval()

    def classify(
        self, image: np.ndarray, lines: list[OCRLine]
    ) -> dict[str, FieldCandidate]:
        import torch
        from PIL import Image

        if not lines:
            return {}
        height, width = image.shape[:2]
        words = [line.text for line in lines]
        boxes = [normalize_box(line.box, width, height) for line in lines]
        rgb_image = Image.fromarray(image[:, :, ::-1])
        encoding = self.processor(
            images=rgb_image,
            text=words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        model_inputs = {
            key: value for key, value in encoding.items() if hasattr(value, "to")
        }
        with torch.inference_mode():
            logits = self.model(**model_inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1)

        token_predictions: dict[int, tuple[str, float]] = {}
        for token_index, word_index in enumerate(word_ids):
            if word_index is None or word_index in token_predictions:
                continue
            label_index = int(probabilities[token_index].argmax().item())
            label = str(self.model.config.id2label.get(label_index, "O"))
            confidence = float(probabilities[token_index, label_index].item())
            token_predictions[word_index] = (label, confidence)

        segments: list[tuple[str, list[str], list[float]]] = []
        for word_index in sorted(token_predictions):
            label, confidence = token_predictions[word_index]
            field, begins = _field_from_label(label)
            if not field:
                continue
            text = words[word_index].strip()
            if not text:
                continue
            if begins or not segments or segments[-1][0] != field:
                segments.append((field, [text], [confidence]))
            else:
                segments[-1][1].append(text)
                segments[-1][2].append(confidence)

        candidates: dict[str, FieldCandidate] = {}
        for field, values, scores in segments:
            candidate = FieldCandidate(
                value=" ".join(values),
                confidence=round(sum(scores) / len(scores), 4),
            )
            prior = candidates.get(field)
            if prior is None or candidate.confidence > prior.confidence:
                candidates[field] = candidate
        return candidates


class HybridFieldClassifier:
    """Apply LayoutXLM when configured and fall back to the existing parser."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        runtime_factory: Callable[[Path], LayoutXLMRuntime] = LayoutXLMRuntime,
    ) -> None:
        configured = model_dir or os.environ.get("CARDOCR_LAYOUT_MODEL_DIR", "")
        self.model_dir = Path(configured).expanduser() if configured else None
        self.runtime_factory = runtime_factory
        self._runtime: LayoutXLMRuntime | None = None
        self._lock = threading.Lock()
        self._last_error = ""

    def configured(self) -> bool:
        return bool(self.model_dir and (self.model_dir / "config.json").is_file())

    @staticmethod
    def dependencies_installed() -> bool:
        return all(
            importlib.util.find_spec(package) is not None
            for package in ("torch", "transformers", "PIL")
        )

    def _get_runtime(self) -> LayoutXLMRuntime:
        if not self.configured():
            raise RuntimeError(
                "학습된 LayoutXLM 모델 경로가 없습니다. "
                "CARDOCR_LAYOUT_MODEL_DIR 환경변수를 설정해 주세요."
            )
        if not self.dependencies_installed():
            raise RuntimeError("requirements-ai.txt의 AI 의존성을 설치해 주세요.")
        if self._runtime is None:
            with self._lock:
                if self._runtime is None:
                    self._runtime = self.runtime_factory(self.model_dir)
        return self._runtime

    def status(self) -> dict[str, object]:
        return {
            "mode": "layoutxlm-hybrid" if self.configured() else "rules-only",
            "configured": self.configured(),
            "dependencies_installed": self.dependencies_installed(),
            "ready": self._runtime is not None,
            "model_dir": str(self.model_dir) if self.model_dir else "",
            "error": self._last_error,
        }

    def enhance(
        self, image: np.ndarray, lines: list[OCRLine], parsed: ParsedCard
    ) -> tuple[ParsedCard, dict[str, object]]:
        if not self.configured():
            return parsed, {**self.status(), "used": False, "predictions": {}}
        try:
            candidates = self._get_runtime().classify(image, lines)
            enhanced = merge_ai_candidates(parsed, candidates)
            self._last_error = ""
            return enhanced, {
                **self.status(),
                "used": True,
                "predictions": {
                    field: {"value": item.value, "confidence": item.confidence}
                    for field, item in candidates.items()
                },
            }
        except Exception as exc:
            self._last_error = str(exc)
            logging.getLogger("cardocr.ai_fields").warning(
                "LayoutXLM 분류 실패, 규칙 기반 결과를 사용합니다: %s", exc
            )
            return parsed, {**self.status(), "used": False, "predictions": {}}


field_classifier = HybridFieldClassifier()
