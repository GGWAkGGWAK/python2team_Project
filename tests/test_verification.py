import cv2
from cardocr.ocr_engine import OCRLine
from cardocr.parser import map_field_confidence, parse_business_card
from cardocr.security import extract_qr_url, inspect_url
from cardocr.validator import classify_job, find_similar_contact, valid_biz_no, verify_contact
from cardocr.online_validator import check_email_mx, check_website


def test_business_number_and_typo_validation():
    assert valid_biz_no("220-81-62517")
    assert not valid_biz_no("220-81-62518")
    result = verify_contact({"email": "hong@gmial.com", "biz_no": "2208162518"})
    checks = {item["id"]: item for item in result["checks"]}
    assert checks["email_typo"]["suggestion"] == "hong@gmail.com"
    assert checks["biz_no_checksum"]["state"] == "warn"


def test_url_security_is_conservative_and_detects_mismatch():
    result = inspect_url("http://abc-login.xyz/path", ["https://abc.com"])
    checks = {item["id"]: item for item in result}
    assert checks["qr_https"]["state"] == "warn"
    assert checks["qr_domain_mismatch"]["state"] == "warn"
    unknown = inspect_url("")
    assert unknown[0]["state"] == "unknown"


def test_qr_detection_and_field_confidence_mapping():
    encoder = cv2.QRCodeEncoder_create()
    qr = encoder.encode("https://cardflow.example")
    assert extract_qr_url(qr) == "https://cardflow.example"
    lines = [OCRLine("홍길동", 0.93, [[0, 0], [20, 0], [20, 10], [0, 10]])]
    parsed = parse_business_card(lines)
    mapped = map_field_confidence(parsed, lines)
    assert mapped["name"]["confidence"] == 0.93
    assert mapped["name"]["box"]


def test_similar_contact_uses_name_and_company():
    duplicate = find_similar_contact(
        {"name": "홍길동", "company": "(주) 카드플로우"},
        [{"id": 7, "name": "홍길동", "company": "주식회사 카드플로우"}],
    )
    assert duplicate["contact_id"] == 7
    assert duplicate["similarity"] >= 0.9


def test_job_category_and_neutral_network_failures(monkeypatch):
    import requests

    assert classify_job("AI 연구개발팀 선임") == "기술"
    assert classify_job("브랜드 마케팅 매니저") == "마케팅"
    monkeypatch.setattr("cardocr.online_validator.requests.head",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.Timeout()))
    assert check_website("example.com")["state"] == "unknown"
    assert check_email_mx("")["state"] == "unknown"
