import json

from cardocr.gemini_field_classifier import GeminiFieldClassifier, merge_gemini_fields
from cardocr.ocr_engine import OCRLine
from cardocr.parser import ParsedCard


def test_merge_only_fills_missing_values_with_ocr_evidence():
    parsed = ParsedCard(
        name="",
        company="기존 회사",
        phone="010-1234-5678",
        raw_text="김서연\n기존 회사\n새 회사\n010-1234-5678\nname@example.com",
    )
    result = merge_gemini_fields(
        parsed,
        {
            "name": "김서연",
            "company": "새 회사",
            "phone": "010-0000-0000",
            "email": "name@example.com",
        },
    )
    assert result.name == "김서연"
    assert result.company == "새 회사"
    assert result.phone == "010-1234-5678"
    assert result.email == "name@example.com"


def test_gemini_classifier_uses_structured_response_without_real_api():
    class FakeModels:
        def generate_content(self, **_kwargs):
            values = {field: "" for field in (
                "name", "company", "job_title", "phone", "phone2", "fax",
                "email", "website", "address",
            )}
            values["name"] = "김서연"
            return type("Response", (), {"text": json.dumps(values, ensure_ascii=False)})()

    class FakeClient:
        models = FakeModels()

    classifier = GeminiFieldClassifier(
        enabled=True,
        api_key="test-only",
        client_factory=lambda _key: FakeClient(),
        config_factory=lambda: {"response_mime_type": "application/json"},
    )
    classifier.dependencies_installed = lambda: True
    line = OCRLine("김서연", 0.99, [[0, 0], [50, 0], [50, 20], [0, 20]])
    result, metadata = classifier.enhance(
        [line], ParsedCard(raw_text="김서연")
    )
    assert result.name == "김서연"
    assert metadata["used"] is True
    assert metadata["model"] == "gemini-2.5-flash"
    assert metadata["ready"] is True


def test_classifier_is_disabled_without_explicit_enablement():
    classifier = GeminiFieldClassifier(enabled=False, api_key="test-only")
    parsed = ParsedCard(name="규칙 결과")
    result, metadata = classifier.enhance([], parsed)
    assert result == parsed
    assert metadata["used"] is False
    assert metadata["mode"] == "disabled"


def test_multimodal_classifier_sends_inline_image_and_prompt():
    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            values = {field: "" for field in (
                "name", "company", "job_title", "phone", "phone2", "fax",
                "email", "website", "address",
            )}
            values["fax"] = "02-1234-1245"
            return type("Response", (), {"text": json.dumps(values)})()

    class FakeClient:
        models = FakeModels()

    classifier = GeminiFieldClassifier(
        enabled=True,
        api_key="test-only",
        send_image=True,
        client_factory=lambda _key: FakeClient(),
        config_factory=lambda: {},
        part_factory=lambda data, mime: {"data": data, "mime_type": mime},
    )
    classifier.dependencies_installed = lambda: True
    line = OCRLine(
        "02-1234-1245", 0.99, [[0, 0], [100, 0], [100, 20], [0, 20]]
    )
    result, metadata = classifier.enhance(
        [line],
        ParsedCard(raw_text="02-1234-1245"),
        image_bytes=b"jpeg-data",
    )

    assert result.fax == "02-1234-1245"
    assert captured["contents"][0]["data"] == b"jpeg-data"
    assert "printer/fax-machine" in captured["contents"][1]
    assert metadata["image_used"] is True
    assert metadata["mode"] == "gemini-multimodal"
    assert metadata["sends_original_image"] is True
