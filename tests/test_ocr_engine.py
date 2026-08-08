import numpy as np

from cardocr.ocr_engine import HybridOCREngine, OCRLine


BOX = [[0, 0], [100, 0], [100, 20], [0, 20]]


class FakeEngine:
    def __init__(self, lines):
        self.lines = lines
        self.calls = 0

    @staticmethod
    def installed():
        return True

    def recognize(self, _image):
        self.calls += 1
        return self.lines


def test_hybrid_uses_strong_paddle_result_without_easyocr():
    hybrid = HybridOCREngine()
    hybrid.paddle = FakeEngine(
        [
            OCRLine("대구디지털혁신진흥원", 0.98, BOX),
            OCRLine("김재진", 0.99, BOX),
            OCRLine("010-1234-5678", 0.99, BOX),
            OCRLine("name@example.com", 0.99, BOX),
        ]
    )
    hybrid.easy = FakeEngine([OCRLine("잘못된 결과", 0.9, BOX)])

    result = hybrid.recognize(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result[0].text == "대구디지털혁신진흥원"
    assert hybrid.paddle.calls == 1
    assert hybrid.easy.calls == 0


def test_hybrid_uses_easyocr_when_paddle_result_is_weak():
    hybrid = HybridOCREngine()
    hybrid.paddle = FakeEngine([OCRLine("김", 0.3, BOX)])
    hybrid.easy = FakeEngine(
        [
            OCRLine("테스트 회사", 0.9, BOX),
            OCRLine("010-1234-5678", 0.9, BOX),
        ]
    )

    result = hybrid.recognize(np.zeros((100, 200, 3), dtype=np.uint8))

    assert any(line.text == "테스트 회사" for line in result)
    assert hybrid.easy.calls == 1
