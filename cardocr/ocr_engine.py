"""Lazy EasyOCR integration so the web app can start before model loading."""

from __future__ import annotations

import importlib.util
import os
import re
import threading
from dataclasses import dataclass
from statistics import mean

import cv2
import numpy as np


class OCRUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class OCRLine:
    text: str
    confidence: float
    box: list[list[int]]


class EasyOCREngine:
    def __init__(self) -> None:
        self._reader = None
        self._lock = threading.Lock()

    @staticmethod
    def installed() -> bool:
        return importlib.util.find_spec("easyocr") is not None

    def _get_reader(self):
        if not self.installed():
            raise OCRUnavailableError(
                "EasyOCR이 설치되어 있지 않습니다. requirements.txt를 설치한 뒤 다시 시도하세요."
            )
        if self._reader is None:
            with self._lock:
                if self._reader is None:
                    import easyocr

                    gpu = os.environ.get("EASYOCR_GPU", "false").lower() in {"1", "true", "yes"}
                    self._reader = easyocr.Reader(["ko", "en"], gpu=gpu, verbose=False)
        return self._reader

    def recognize(self, image: np.ndarray) -> list[OCRLine]:
        try:
            reader = self._get_reader()
            stable_image = cv2.fastNlMeansDenoising(image, None, 8, 7, 21)

            # Run the focused logo pass first. EasyOCR's beam decoder is most
            # reliable for the large Korean heading before the dense full-card
            # passes are evaluated.
            height, width = image.shape[:2]
            upper_top = int(height * 0.10)
            upper_left = int(width * 0.10)
            upper_results = reader.readtext(
                image[upper_top : int(height * 0.38), upper_left : int(width * 0.75)],
                detail=1,
                paragraph=False,
                decoder="beamsearch",
                contrast_ths=0.05,
                adjust_contrast=0.7,
                text_threshold=0.4,
                low_text=0.2,
                link_threshold=0.3,
                mag_ratio=1.8,
            )
            adjusted_upper_results = []
            for box, text, confidence in upper_results:
                adjusted_box = [
                    [point[0] + upper_left, point[1] + upper_top] for point in box
                ]
                adjusted_upper_results.append((adjusted_box, text, confidence))

            # The normal pass is good at large labels such as names and titles.
            normal_results = reader.readtext(stable_image, detail=1, paragraph=False)
            # A second, more sensitive pass recovers small address/contact text.
            sensitive_results = reader.readtext(
                stable_image,
                detail=1,
                paragraph=False,
                decoder="beamsearch",
                contrast_ths=0.05,
                adjust_contrast=0.7,
                text_threshold=0.45,
                low_text=0.25,
                link_threshold=0.3,
                mag_ratio=1.5,
            )

            # Names, titles and departments frequently mix small Korean text
            # with abbreviations such as AI, R&D or CEO. Re-reading the centre
            # band at a higher scale gives those short Latin runs enough pixels.
            middle_top = int(height * 0.20)
            middle_left = int(width * 0.18)
            middle_results = reader.readtext(
                image[middle_top : int(height * 0.68), middle_left : int(width * 0.88)],
                detail=1,
                paragraph=False,
                decoder="beamsearch",
                contrast_ths=0.03,
                adjust_contrast=0.8,
                text_threshold=0.4,
                low_text=0.2,
                link_threshold=0.25,
                mag_ratio=2.0,
            )
            adjusted_middle_results = []
            for box, text, confidence in middle_results:
                adjusted_box = [
                    [point[0] + middle_left, point[1] + middle_top] for point in box
                ]
                adjusted_middle_results.append((adjusted_box, text, confidence))
            # Business cards commonly place the smallest address/contact text
            # in the lower half. Cropping that region makes it large enough for
            # EasyOCR's detector without destabilising the name/company pass.
            crop_top = int(height * 0.48)
            lower_results = reader.readtext(
                stable_image[crop_top : int(height * 0.96), :],
                detail=1,
                paragraph=False,
                decoder="beamsearch",
                contrast_ths=0.05,
                adjust_contrast=0.7,
                text_threshold=0.45,
                low_text=0.25,
                link_threshold=0.3,
                mag_ratio=1.5,
            )
            adjusted_lower_results = []
            for box, text, confidence in lower_results:
                adjusted_box = [[point[0], point[1] + crop_top] for point in box]
                adjusted_lower_results.append((adjusted_box, text, confidence))

            results = (
                adjusted_upper_results
                + normal_results
                + sensitive_results
                + adjusted_middle_results
                + adjusted_lower_results
            )
        except OCRUnavailableError:
            raise
        except Exception as exc:
            raise OCRUnavailableError(f"OCR 엔진 실행에 실패했습니다: {exc}") from exc

        lines: list[OCRLine] = []
        seen: set[str] = set()
        for box, text, confidence in results:
            cleaned = str(text).strip()
            box_height = max(float(point[1]) for point in box) - min(
                float(point[1]) for point in box
            )
            large_korean_heading = (
                len(re.findall(r"[가-힣]", cleaned)) >= 4
                and box_height >= 28
                and float(confidence) >= 0.03
            )
            if not cleaned or (float(confidence) < 0.15 and not large_korean_heading):
                continue
            identity = "".join(cleaned.lower().split())
            if identity in seen:
                continue
            seen.add(identity)
            lines.append(
                OCRLine(
                    text=cleaned,
                    confidence=round(float(confidence), 4),
                    box=[[int(round(x)), int(round(y))] for x, y in box],
                )
            )
        return lines


class PaddleOCREngine:
    """Lazy local PP-OCRv5 Korean/English inference."""

    def __init__(self) -> None:
        self._reader = None
        self._lock = threading.Lock()

    @staticmethod
    def installed() -> bool:
        return (
            importlib.util.find_spec("paddle") is not None
            and importlib.util.find_spec("paddleocr") is not None
        )

    def _get_reader(self):
        if not self.installed():
            raise OCRUnavailableError("PaddleOCR가 설치되어 있지 않습니다.")
        if self._reader is None:
            with self._lock:
                if self._reader is None:
                    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
                    from paddleocr import PaddleOCR

                    # PaddlePaddle 3.3 on Windows CPU has a known oneDNN/PIR
                    # incompatibility for the v5 detector, so use the stable
                    # generic CPU path.
                    self._reader = PaddleOCR(
                        lang="korean",
                        ocr_version="PP-OCRv5",
                        device="cpu",
                        enable_mkldnn=False,
                        cpu_threads=4,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                    )
        return self._reader

    def recognize(self, image: np.ndarray) -> list[OCRLine]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise OCRUnavailableError("PP-OCRv5에는 3채널 컬러 이미지가 필요합니다.")
        try:
            pages = list(self._get_reader().predict(image))
        except OCRUnavailableError:
            raise
        except Exception as exc:
            raise OCRUnavailableError(f"PP-OCRv5 실행에 실패했습니다: {exc}") from exc

        lines: list[OCRLine] = []
        for page in pages:
            texts = list(page.get("rec_texts", []))
            scores = list(page.get("rec_scores", []))
            polygons = list(page.get("rec_polys", []))
            for text, score, polygon in zip(texts, scores, polygons):
                cleaned = str(text).strip()
                confidence = float(score)
                if not cleaned or confidence < 0.2:
                    continue
                lines.append(
                    OCRLine(
                        text=cleaned,
                        confidence=round(confidence, 4),
                        box=[
                            [int(round(float(point[0]))), int(round(float(point[1])))]
                            for point in polygon
                        ],
                    )
                )
        return lines


class HybridOCREngine:
    """Prefer PP-OCRv5 and fall back to EasyOCR when results are weak."""

    def __init__(self) -> None:
        self.paddle = PaddleOCREngine()
        self.easy = EasyOCREngine()

    def installed(self) -> bool:
        return self.paddle.installed() or self.easy.installed()

    def available_engines(self) -> list[str]:
        available: list[str] = []
        if self.paddle.installed():
            available.append("PP-OCRv5 Korean")
        if self.easy.installed():
            available.append("EasyOCR ko/en")
        return available

    @staticmethod
    def _paddle_result_is_strong(lines: list[OCRLine]) -> bool:
        if len(lines) < 4:
            return False
        average_confidence = mean(line.confidence for line in lines)
        raw_text = "\n".join(line.text for line in lines)
        has_identity_text = bool(re.search(r"[가-힣A-Za-z]{2,}", raw_text))
        has_contact_text = bool(
            re.search(r"@|(?:^|\D)0\d{1,2}[\s.\-]?\d", raw_text)
        )
        return average_confidence >= 0.65 and has_identity_text and has_contact_text

    def recognize(self, image: np.ndarray) -> list[OCRLine]:
        paddle_lines: list[OCRLine] = []
        paddle_error: Exception | None = None
        if self.paddle.installed():
            try:
                paddle_lines = self.paddle.recognize(image)
                if self._paddle_result_is_strong(paddle_lines):
                    return paddle_lines
            except OCRUnavailableError as exc:
                paddle_error = exc

        easy_lines: list[OCRLine] = []
        if self.easy.installed():
            try:
                from .image_processing import prepare_for_ocr

                easy_lines = self.easy.recognize(prepare_for_ocr(image))
            except OCRUnavailableError as exc:
                if not paddle_lines:
                    raise exc from paddle_error

        if paddle_lines and easy_lines:
            seen = {"".join(line.text.lower().split()) for line in paddle_lines}
            paddle_lines.extend(
                line
                for line in easy_lines
                if "".join(line.text.lower().split()) not in seen
            )
        if paddle_lines or easy_lines:
            return paddle_lines or easy_lines
        if paddle_error:
            raise OCRUnavailableError(str(paddle_error))
        raise OCRUnavailableError(
            "PP-OCRv5 또는 EasyOCR를 requirements.txt로 설치해 주세요."
        )


engine = HybridOCREngine()
