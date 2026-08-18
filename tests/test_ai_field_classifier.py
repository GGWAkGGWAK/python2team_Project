import numpy as np

from cardocr.ai_field_classifier import (
    FieldCandidate,
    HybridFieldClassifier,
    merge_ai_candidates,
    normalize_box,
)
from cardocr.ocr_engine import OCRLine
from cardocr.parser import ParsedCard


def test_normalize_box_uses_layoutxlm_coordinate_range():
    assert normalize_box([[10, 20], [90, 20], [90, 60], [10, 60]], 100, 100) == [
        100,
        200,
        900,
        600,
    ]


def test_ai_overrides_ambiguous_fields_but_not_contact_fields():
    parsed = ParsedCard(
        name="오인식",
        company="기존 회사",
        phone="010-1234-5678",
        email="name@example.com",
    )
    merged = merge_ai_candidates(
        parsed,
        {
            "name": FieldCandidate("김재진", 0.97),
            "company": FieldCandidate("대구디지털혁신진흥원", 0.95),
            "phone": FieldCandidate("010-0000-0000", 0.99),
        },
    )
    assert merged.name == "김재진"
    assert merged.company == "대구디지털혁신진흥원"
    assert merged.phone == "010-1234-5678"


def test_classifier_falls_back_when_no_trained_model_is_configured():
    classifier = HybridFieldClassifier(model_dir=None)
    parsed = ParsedCard(name="규칙 결과")
    result, metadata = classifier.enhance(
        np.zeros((100, 200, 3), dtype=np.uint8),
        [OCRLine("규칙 결과", 0.9, [[0, 0], [10, 0], [10, 10], [0, 10]])],
        parsed,
    )
    assert result.name == "규칙 결과"
    assert metadata["used"] is False
    assert metadata["mode"] == "rules-only"


def test_classifier_uses_injected_layout_runtime(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    class FakeRuntime:
        def classify(self, _image, _lines):
            return {"name": FieldCandidate("AI 이름", 0.99)}

    classifier = HybridFieldClassifier(
        model_dir=tmp_path, runtime_factory=lambda _path: FakeRuntime()
    )
    classifier.dependencies_installed = lambda: True
    result, metadata = classifier.enhance(
        np.zeros((100, 200, 3), dtype=np.uint8),
        [OCRLine("AI 이름", 0.9, [[0, 0], [10, 0], [10, 10], [0, 10]])],
        ParsedCard(name="규칙 이름"),
    )
    assert result.name == "AI 이름"
    assert metadata["used"] is True
