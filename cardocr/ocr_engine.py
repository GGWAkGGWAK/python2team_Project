"""Lazy EasyOCR integration so the web app can start before model loading."""

from __future__ import annotations

import importlib.util
import logging
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
                "EasyOCR이 설치되어 있지 않습니다. "
                "requirements.txt를 설치한 뒤 다시 시도하세요."
            )

        if self._reader is None:
            with self._lock:
                if self._reader is None:
                    import easyocr

                    gpu = os.environ.get(
                        "EASYOCR_GPU", "false"
                    ).lower() in {"1", "true", "yes"}

                    self._reader = easyocr.Reader(
                        ["ko", "en"],
                        gpu=gpu,
                        verbose=False,
                    )

        return self._reader

    def recognize(self, image: np.ndarray) -> list[OCRLine]:
        try:
            reader = self._get_reader()
            stable_image = cv2.fastNlMeansDenoising(
                image,
                None,
                8,
                7,
                21,
            )

            height, width = image.shape[:2]

            # 명함 상단의 한글 회사명을 집중적으로 인식합니다.
            upper_top = int(height * 0.10)
            upper_left = int(width * 0.10)

            upper_results = reader.readtext(
                image[
                    upper_top : int(height * 0.38),
                    upper_left : int(width * 0.75),
                ],
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
                    [
                        point[0] + upper_left,
                        point[1] + upper_top,
                    ]
                    for point in box
                ]

                adjusted_upper_results.append(
                    (adjusted_box, text, confidence)
                )

            # 일반적인 크기의 이름, 직책 등을 인식합니다.
            normal_results = reader.readtext(
                stable_image,
                detail=1,
                paragraph=False,
            )

            # 작은 주소와 연락처를 민감하게 인식합니다.
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

            # 이름, 직책, 부서 및 한글·영문 혼합 문자를 인식합니다.
            middle_top = int(height * 0.20)
            middle_left = int(width * 0.18)

            middle_results = reader.readtext(
                image[
                    middle_top : int(height * 0.68),
                    middle_left : int(width * 0.88),
                ],
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
                    [
                        point[0] + middle_left,
                        point[1] + middle_top,
                    ]
                    for point in box
                ]

                adjusted_middle_results.append(
                    (adjusted_box, text, confidence)
                )

            # 명함 하단의 작은 주소, 전화번호, 이메일을 인식합니다.
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
                adjusted_box = [
                    [point[0], point[1] + crop_top]
                    for point in box
                ]

                adjusted_lower_results.append(
                    (adjusted_box, text, confidence)
                )

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
            raise OCRUnavailableError(
                f"OCR 엔진 실행에 실패했습니다: {exc}"
            ) from exc

        lines: list[OCRLine] = []
        seen: set[str] = set()

        for box, text, confidence in results:
            cleaned = str(text).strip()

            box_height = (
                max(float(point[1]) for point in box)
                - min(float(point[1]) for point in box)
            )

            large_korean_heading = (
                len(re.findall(r"[가-힣]", cleaned)) >= 4
                and box_height >= 28
                and float(confidence) >= 0.03
            )

            if not cleaned:
                continue

            if (
                float(confidence) < 0.15
                and not large_korean_heading
            ):
                continue

            identity = "".join(cleaned.lower().split())

            if identity in seen:
                continue

            seen.add(identity)

            lines.append(
                OCRLine(
                    text=cleaned,
                    confidence=round(
                        float(confidence),
                        4,
                    ),
                    box=[
                        [
                            int(round(x)),
                            int(round(y)),
                        ]
                        for x, y in box
                    ],
                )
            )

        return lines


class PaddleOCREngine:
    """Lazy local PP-OCRv5 Korean/English inference."""

    def __init__(self) -> None:
        self._reader = None
        self._lock = threading.Lock()
        self._warmup_thread: threading.Thread | None = None
        self._last_error = ""

    @staticmethod
    def installed() -> bool:
        return (
            importlib.util.find_spec("paddle") is not None
            and importlib.util.find_spec("paddleocr") is not None
        )

    def _get_reader(self):
        if not self.installed():
            raise OCRUnavailableError(
                "PaddleOCR가 설치되어 있지 않습니다."
            )

        if self._reader is None:
            with self._lock:
                if self._reader is None:
                    os.environ.setdefault(
                        "PADDLE_PDX_MODEL_SOURCE",
                        "BOS",
                    )

                    from paddleocr import PaddleOCR

                    # Windows CPU 환경에서 안정적인 설정을 사용합니다.
                    try:
                        self._reader = PaddleOCR(
                            text_detection_model_name="PP-OCRv5_mobile_det",
                            text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
                            device="cpu",
                            enable_mkldnn=False,
                            cpu_threads=4,
                            use_doc_orientation_classify=False,
                            use_doc_unwarping=False,
                            use_textline_orientation=False,
                        )

                        self._last_error = ""

                    except Exception as exc:
                        self._last_error = str(exc)
                        raise

        return self._reader

    def start_warmup(self) -> None:
        """Load the Paddle models before the first OCR request."""

        if not self.installed():
            return

        if self._reader is not None:
            return

        if (
            self._warmup_thread is not None
            and self._warmup_thread.is_alive()
        ):
            return

        def load_models() -> None:
            logger = logging.getLogger("cardocr.ocr")

            logger.info(
                "OCR 모델 준비를 시작합니다 "
                "(PP-OCRv5 mobile, CPU)."
            )

            try:
                self._get_reader()

            except Exception:
                logger.exception(
                    "OCR 모델 준비에 실패했습니다."
                )

            else:
                logger.info(
                    "OCR 모델 준비가 완료되었습니다."
                )

        self._warmup_thread = threading.Thread(
            target=load_models,
            name="cardocr-ocr-warmup",
            daemon=True,
        )

        self._warmup_thread.start()

    def runtime_status(self) -> dict[str, str | bool]:
        warming = (
            self._warmup_thread is not None
            and self._warmup_thread.is_alive()
        )

        return {
            "ready": self._reader is not None,
            "warming": warming,
            "error": self._last_error,
        }

    def recognize(
        self,
        image: np.ndarray,
    ) -> list[OCRLine]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise OCRUnavailableError(
                "PP-OCRv5에는 3채널 컬러 이미지가 필요합니다."
            )

        try:
            pages = list(
                self._get_reader().predict(image)
            )

        except OCRUnavailableError:
            raise

        except Exception as exc:
            raise OCRUnavailableError(
                f"PP-OCRv5 실행에 실패했습니다: {exc}"
            ) from exc

        lines: list[OCRLine] = []

        for page in pages:
            texts = list(
                page.get("rec_texts", [])
            )

            scores = list(
                page.get("rec_scores", [])
            )

            polygons = list(
                page.get("rec_polys", [])
            )

            for text, score, polygon in zip(
                texts,
                scores,
                polygons,
            ):
                cleaned = str(text).strip()
                confidence = float(score)

                if not cleaned or confidence < 0.2:
                    continue

                lines.append(
                    OCRLine(
                        text=cleaned,
                        confidence=round(
                            confidence,
                            4,
                        ),
                        box=[
                            [
                                int(
                                    round(
                                        float(point[0])
                                    )
                                ),
                                int(
                                    round(
                                        float(point[1])
                                    )
                                ),
                            ]
                            for point in polygon
                        ],
                    )
                )

        return lines


class HybridOCREngine:
    """Prefer PP-OCRv5 and fall back to EasyOCR."""

    def __init__(self) -> None:
        self.paddle = PaddleOCREngine()
        self.easy = EasyOCREngine()

    def installed(self) -> bool:
        return (
            self.paddle.installed()
            or self.easy.installed()
        )

    def available_engines(self) -> list[str]:
        available: list[str] = []

        if self.paddle.installed():
            available.append(
                "PP-OCRv5 Korean"
            )

        if self.easy.installed():
            available.append(
                "EasyOCR ko/en"
            )

        return available

    def start_warmup(self) -> None:
        self.paddle.start_warmup()

    def runtime_status(
        self,
    ) -> dict[str, str | bool]:
        if self.paddle.installed():
            return self.paddle.runtime_status()

        return {
            "ready": self.easy.installed(),
            "warming": False,
            "error": "",
        }

    @staticmethod
    def _paddle_result_is_strong(
        lines: list[OCRLine],
    ) -> bool:
        if len(lines) < 4:
            return False

        average_confidence = mean(
            line.confidence
            for line in lines
        )

        raw_text = "\n".join(
            line.text
            for line in lines
        )

        has_identity_text = bool(
            re.search(
                r"[가-힣A-Za-z]{2,}",
                raw_text,
            )
        )

        has_contact_text = bool(
            re.search(
                r"@|(?:^|\D)0\d{1,2}[\s.\-]?\d",
                raw_text,
            )
        )

        return (
            average_confidence >= 0.65
            and has_identity_text
            and has_contact_text
        )

    def recognize(
        self,
        image: np.ndarray,
    ) -> list[OCRLine]:
        paddle_lines: list[OCRLine] = []
        paddle_error: Exception | None = None

        if self.paddle.installed():
            try:
                paddle_lines = (
                    self.paddle.recognize(image)
                )

                if self._paddle_result_is_strong(
                    paddle_lines
                ):
                    return paddle_lines

            except OCRUnavailableError as exc:
                paddle_error = exc

        easy_lines: list[OCRLine] = []

        if self.easy.installed():
            try:
                from .image_processing import (
                    prepare_for_ocr,
                )

                prepared_image = prepare_for_ocr(
                    image
                )

                easy_lines = self.easy.recognize(
                    prepared_image
                )

            except OCRUnavailableError as exc:
                if not paddle_lines:
                    raise exc from paddle_error

        if paddle_lines and easy_lines:
            seen = {
                "".join(
                    line.text.lower().split()
                )
                for line in paddle_lines
            }

            paddle_lines.extend(
                line
                for line in easy_lines
                if "".join(
                    line.text.lower().split()
                )
                not in seen
            )

        if paddle_lines or easy_lines:
            return paddle_lines or easy_lines

        if paddle_error:
            raise OCRUnavailableError(
                str(paddle_error)
            )

        raise OCRUnavailableError(
            "PP-OCRv5 또는 EasyOCR를 "
            "requirements.txt로 설치해 주세요."
        )


engine = HybridOCREngine()