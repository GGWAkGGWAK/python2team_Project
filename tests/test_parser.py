from cardocr.parser import parse_business_card
from cardocr.ocr_engine import OCRLine


def test_parse_korean_business_card():
    parsed = parse_business_card(
        [
            "주식회사 카드플로우",
            "홍길동 영업팀 과장",
            "Mobile 010 1234 5678",
            "hong.gildong@cardflow.co.kr",
            "www.cardflow.co.kr",
            "서울특별시 중구 세종대로 110",
        ]
    )

    assert parsed.name == "홍길동"
    assert parsed.company == "주식회사 카드플로우"
    assert parsed.job_title == "영업팀 과장"
    assert parsed.phone == "010-1234-5678"
    assert parsed.email == "hong.gildong@cardflow.co.kr"
    assert parsed.website == "www.cardflow.co.kr"
    assert parsed.address == "서울특별시 중구 세종대로 110"


def test_parse_english_business_card():
    parsed = parse_business_card(
        [
            "JANE DOE",
            "BLUE OCEAN SYSTEMS INC.",
            "Product Manager",
            "+82 10-9876-5432",
            "jane.doe@blueocean.com",
        ]
    )

    assert parsed.name == "JANE DOE"
    assert parsed.company == "BLUE OCEAN SYSTEMS INC."
    assert parsed.job_title == "Product Manager"
    assert parsed.phone == "010-9876-5432"
    assert parsed.email == "jane.doe@blueocean.com"


def test_empty_text_returns_empty_fields():
    parsed = parse_business_card([])
    assert parsed.name == ""
    assert parsed.email == ""
    assert parsed.raw_text == ""


def test_parse_noisy_daegu_business_card_ocr():
    parsed = parse_business_card(
        [
            "애 대구되지털학신진홍원",
            "Dacgu Dioital Innovation Promoton",
            "김 재 진|",
            "선임",
            "시인자양성팀",
            "42250 | 대구강역시 수성구 알다시티i로 170 (9중동)",
            "T 053-215-3606",
            "F 053-655-5635",
            "M 010-6616-2882",
            "E reentry@dip orkr",
            "WWw dip orK",
        ]
    )

    assert parsed.name == "김재진"
    assert "대구디지털혁신진흥원" in parsed.company
    assert parsed.job_title == "선임 / AI인재양성팀"
    assert parsed.phone == "010-6616-2882"
    assert parsed.phone2 == "053-215-3606"
    assert parsed.email == "reentry@dip.or.kr"
    assert parsed.website == "www.dip.or.kr"
    assert "대구광역시 수성구 알파시티1로 170" in parsed.address


def test_noisy_english_company_is_not_replaced_by_department():
    parsed = parse_business_card(
        [
            "Dacgu Dloltal Innovatlon Promotlon",
            "Aooncy",
            "김 재 진",
            "선임",
            "시인자양성팀",
            "M 010-6616-2882",
        ]
    )

    assert parsed.company == "Daegu Digital Innovation Promotion Agency"
    assert parsed.job_title == "선임 / AI인재양성팀"


def test_two_phones_do_not_fill_website_and_fax_is_ignored():
    parsed = parse_business_card(
        [
            "T 053-215.3606",
            "F 053-655-5635",
            "M 010 -6616-2882",
            "E reentry@dip or kr",
        ]
    )

    assert parsed.phone == "010-6616-2882"
    assert parsed.phone2 == "053-215-3606"
    assert parsed.website == "www.dip.or.kr"
    assert "655-5635" not in parsed.phone2


def test_larger_korean_company_name_wins_over_small_english_translation():
    parsed = parse_business_card(
        [
            OCRLine(
                "대구디지터최신진하원",
                0.8,
                [[100, 100], [650, 100], [650, 160], [100, 160]],
            ),
            OCRLine(
                "Daegu Digital Innovation Promotion Agency",
                0.99,
                [[150, 170], [600, 170], [600, 194], [150, 194]],
            ),
            OCRLine("김 재 진", 0.99, [[300, 250], [500, 250], [500, 320], [300, 320]]),
        ]
    )

    assert parsed.company == "대구디지털혁신진흥원"
