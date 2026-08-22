"""Optional Gemini post-processor for OCR business-card fields.

The API key is read only from the server environment. The classifier receives
OCR text and layout coordinates, not the original image, and classifies every
business-card field. Local code only validates format and OCR evidence.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import threading
from dataclasses import replace
from typing import Any, Callable

from .ocr_engine import OCRLine
from .parser import ParsedCard


FIELDS = (
    "name",
    "company",
    "job_title",
    "phone",
    "phone2",
    "fax",
    "email",
    "website",
    "address",
)
SEMANTIC_FIELDS = {"name", "company", "job_title", "address"}
CONTACT_FIELDS = {"phone", "phone2", "fax", "email", "website"}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        field: {
            "type": "string",
            "description": f"Extracted business-card {field}; empty if absent",
        }
        for field in FIELDS
    },
    "required": list(FIELDS),
}


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _evidence_key(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.lower())


def _has_evidence(value: str, raw_text: str) -> bool:
    candidate = _evidence_key(value)
    source = _evidence_key(raw_text)
    return bool(candidate and candidate in source)


def _valid_contact(field: str, value: str) -> bool:
    if field in {"phone", "phone2", "fax"}:
        digits = re.sub(r"\D", "", value)
        return 9 <= len(digits) <= 13
    if field == "email":
        return bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value))
    if field == "website":
        return bool(re.fullmatch(r"(?:https?://)?(?:www\.)?[a-z0-9.-]+\.[a-z]{2,}(?:/\S*)?", value, re.I))
    return False


def merge_gemini_fields(parsed: ParsedCard, values: dict[str, Any]) -> ParsedCard:
    """Build all fields from Gemini while rejecting unsupported OCR values."""
    result = replace(parsed)
    raw_text = parsed.raw_text
    for field in FIELDS:
        value = str(values.get(field, "") or "").strip()
        if not value or not _has_evidence(value, raw_text):
            continue
        if field in CONTACT_FIELDS:
            if _valid_contact(field, value):
                setattr(result, field, value)
            continue
        if field in SEMANTIC_FIELDS:
            setattr(result, field, value)
    return result


class GeminiFieldClassifier:
    """Lazy Google Gen AI client used as an optional OCR-text classifier."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        api_key: str | None = None,
        model: str | None = None,
        client_factory: Callable[[str], Any] | None = None,
        config_factory: Callable[[], Any] | None = None,
        part_factory: Callable[[bytes, str], Any] | None = None,
        send_image: bool | None = None,
    ) -> None:
        self.enabled = (
            _enabled(os.environ.get("CARDOCR_GEMINI_ENABLED"))
            if enabled is None
            else enabled
        )
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model or os.environ.get("CARDOCR_GEMINI_MODEL", "gemini-2.5-flash")
        self.client_factory = client_factory or self._default_client
        self.config_factory = config_factory or self._default_config
        self.part_factory = part_factory or self._default_image_part
        self.send_image = (
            _enabled(os.environ.get("CARDOCR_GEMINI_SEND_IMAGE"))
            if send_image is None
            else send_image
        )
        self._client: Any | None = None
        self._ready = False
        self._lock = threading.Lock()
        self._last_error = ""

    @staticmethod
    def dependencies_installed() -> bool:
        try:
            return importlib.util.find_spec("google.genai") is not None
        except (ImportError, ModuleNotFoundError):
            return False

    @staticmethod
    def _default_client(api_key: str) -> Any:
        from google import genai

        return genai.Client(api_key=api_key)

    @staticmethod
    def _default_config() -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        )

    @staticmethod
    def _default_image_part(data: bytes, mime_type: str) -> Any:
        from google.genai import types

        return types.Part.from_bytes(data=data, mime_type=mime_type)

    def configured(self) -> bool:
        return self.enabled and bool(self.api_key)

    def _get_client(self) -> Any:
        if not self.configured():
            raise RuntimeError("Gemini가 비활성화되었거나 GEMINI_API_KEY가 없습니다.")
        if not self.dependencies_installed():
            raise RuntimeError("requirements-llm.txt를 설치해 주세요.")
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self.client_factory(self.api_key)
        return self._client

    def status(self) -> dict[str, object]:
        return {
            "mode": (
                "gemini-multimodal"
                if self.configured() and self.send_image
                else "gemini-text-hybrid" if self.configured() else "disabled"
            ),
            "enabled": self.enabled,
            "configured": self.configured(),
            "dependencies_installed": self.dependencies_installed(),
            "ready": self._ready,
            "model": self.model,
            "sends_original_image": self.send_image,
            "error": self._last_error,
        }

    @staticmethod
    def _prompt(lines: list[OCRLine], parsed: ParsedCard) -> str:
        layout = [
            {
                "text": line.text,
                "confidence": round(float(line.confidence), 4),
                "box": line.box,
            }
            for line in lines
        ]
        return (
            "You classify OCR text from one Korean/English business card. "
            "Return only values literally supported by OCR_INPUT; never guess or invent. "
            "Use labels T/Tel/전화=office phone, M/Mobile/휴대폰=mobile, "
            "F/Fax/팩스=fax, E/Email/이메일=email. Separate two phone numbers. "
            "Use visual icons too: telephone handset=office phone, smartphone=mobile, "
            "printer/fax-machine=fax, envelope=email, globe=website, map pin=address. "
            "Put a mobile number in phone first, an office number in phone2, and never "
            "place a fax number in phone or phone2. "
            "A URL is website, not phone. For duplicate Korean and English company names, "
            "prefer the larger Korean line. Preserve original spelling. Empty means absent.\n"
            f"CURRENT_RESULT={json.dumps(parsed.to_dict(), ensure_ascii=False)}\n"
            f"OCR_INPUT={json.dumps(layout, ensure_ascii=False)}"
        )

    def enhance(
        self,
        lines: list[OCRLine],
        parsed: ParsedCard,
        *,
        image_bytes: bytes | None = None,
        image_mime_type: str = "image/jpeg",
    ) -> tuple[ParsedCard, dict[str, object]]:
        if not self.configured():
            return parsed, {
                **self.status(),
                "used": False,
                "image_used": False,
                "predictions": {},
            }
        try:
            prompt = self._prompt(lines, parsed)
            contents: Any = prompt
            image_used = bool(self.send_image and image_bytes)
            if image_used:
                contents = [
                    self.part_factory(image_bytes, image_mime_type),
                    prompt,
                ]
            response = self._get_client().models.generate_content(
                model=self.model,
                contents=contents,
                config=self.config_factory(),
            )
            values = json.loads(str(response.text or "{}"))
            if not isinstance(values, dict):
                raise ValueError("Gemini 응답이 JSON 객체가 아닙니다.")
            enhanced = merge_gemini_fields(parsed, values)
            self._last_error = ""
            self._ready = True
            return enhanced, {
                **self.status(),
                "used": True,
                "image_used": image_used,
                "predictions": {field: str(values.get(field, "") or "") for field in FIELDS},
            }
        except Exception as exc:
            self._last_error = str(exc)
            self._ready = False
            logging.getLogger("cardocr.gemini_fields").warning(
                "Gemini 분류 실패, 로컬 결과를 사용합니다: %s", exc
            )
            return parsed, {
                **self.status(),
                "used": False,
                "image_used": False,
                "predictions": {},
            }


gemini_classifier = GeminiFieldClassifier()
