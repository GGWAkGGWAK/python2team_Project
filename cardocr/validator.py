"""Offline contact normalization, validation and three-axis scoring."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher, get_close_matches
from statistics import mean
from typing import Any, Iterable

from .security import hostname, inspect_url, registrable_hint


COMMON_DOMAINS = ["gmail.com", "naver.com", "daum.net", "hanmail.net", "kakao.com",
                  "outlook.com", "hotmail.com", "icloud.com", "yahoo.com"]
JOB_CATEGORIES = {
    "영업": ("영업", "sales", "account", "business development", "고객"),
    "기술": ("개발", "엔지니어", "engineer", "developer", "cto", "기술", "it"),
    "연구": ("연구", "research", "r&d", "scientist"),
    "마케팅": ("마케팅", "marketing", "홍보", "브랜드", "brand"),
    "경영": ("대표", "ceo", "cfo", "임원", "이사", "사장", "경영"),
}


def normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    return "0" + digits[2:] if digits.startswith("82") else digits


def normalize_company(value: str) -> str:
    value = re.sub(r"주식회사|\(주\)|㈜|co\.?\s*,?\s*ltd\.?|inc\.?|corp(?:oration)?\.?", "", value or "", flags=re.I)
    return re.sub(r"[^0-9a-z가-힣]", "", value.lower())


def classify_job(job_title: str) -> str:
    lowered = (job_title or "").lower()
    if not lowered:
        return "미분류"
    for category, keywords in JOB_CATEGORIES.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "기타"


def valid_biz_no(value: str) -> bool:
    digits = [int(char) for char in value if char.isdigit()]
    if len(digits) != 10:
        return False
    weights = [1, 3, 7, 1, 3, 7, 1, 3, 5]
    total = sum(a * b for a, b in zip(digits[:9], weights)) + (digits[8] * 5) // 10
    return (10 - total % 10) % 10 == digits[9]


def _check(identifier: str, label: str, state: str, message: str, field: str, suggestion=None) -> dict:
    return {"id": identifier, "label": label, "state": state, "message": message,
            "field": field, "suggestion": suggestion}


def _phone_check(value: str, field: str) -> dict:
    digits = normalize_phone(value)
    if not digits:
        return _check(field + "_format", "전화번호 형식", "unknown", "전화번호가 없습니다", field)
    valid = bool(re.fullmatch(r"(?:02\d{7,8}|0(?:1[016789]|3[1-3]|4[1-4]|5[1-5]|6[1-4]|70)\d{7,8}|1(?:544|566|577|588|599)\d{4})", digits))
    return _check(field + "_format", "전화번호 형식", "ok" if valid else "warn",
                  "정상" if valid else "국번 또는 자릿수가 올바르지 않습니다", field)


def _email_checks(email: str) -> list[dict]:
    if not email:
        return [_check("email_format", "이메일 형식", "unknown", "이메일이 없습니다", "email")]
    valid = bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[A-Za-z]{2,}", email))
    checks = [_check("email_format", "이메일 형식", "ok" if valid else "warn",
                     "정상" if valid else "이메일 형식이 올바르지 않습니다", "email")]
    if valid:
        local, domain = email.rsplit("@", 1)
        match = get_close_matches(domain.lower(), COMMON_DOMAINS, n=1, cutoff=0.82)
        if match and match[0] != domain.lower():
            suggestion = f"{local}@{match[0]}"
            checks.append(_check("email_typo", "이메일 오탈자", "warn",
                                 f"{domain} → {match[0]} 으로 보입니다", "email", suggestion))
    return checks


def find_similar_contact(fields: dict[str, Any], contacts: Iterable[dict[str, Any]], exclude_id=None) -> dict | None:
    name = str(fields.get("name", "")).strip().lower()
    company = normalize_company(str(fields.get("company", "")))
    best: tuple[float, dict] | None = None
    for contact in contacts:
        if exclude_id is not None and contact.get("id") == exclude_id:
            continue
        other_name = str(contact.get("name", "")).strip().lower()
        other_company = normalize_company(str(contact.get("company", "")))
        if not name or not company or not other_name or not other_company:
            continue
        score = (SequenceMatcher(None, name, other_name).ratio() * 0.55 +
                 SequenceMatcher(None, company, other_company).ratio() * 0.45)
        if score >= 0.82 and (best is None or score > best[0]):
            best = (score, contact)
    if best is None:
        return None
    return {"contact_id": best[1]["id"], "similarity": round(best[0], 2), "reason": "이름+회사 유사"}


def verify_contact(fields: dict[str, Any], field_confidence: dict[str, dict] | None = None,
                   contacts: Iterable[dict[str, Any]] = (), exclude_id=None) -> dict:
    checks = [_phone_check(str(fields.get("phone", "")), "phone")]
    checks += _email_checks(str(fields.get("email", "")))
    website = str(fields.get("website", ""))
    checks.append(_check("website_format", "웹사이트 형식", "ok" if hostname(website) else "unknown",
                         "정상" if hostname(website) else "웹사이트를 확인할 수 없습니다", "website"))
    biz_no = str(fields.get("biz_no", ""))
    if biz_no:
        valid = valid_biz_no(biz_no)
        checks.append(_check("biz_no_checksum", "사업자등록번호", "ok" if valid else "warn",
                             "체크섬이 유효합니다" if valid else "체크섬이 일치하지 않습니다", "biz_no"))
    else:
        checks.append(_check("biz_no_checksum", "사업자등록번호", "unknown", "번호가 없습니다", "biz_no"))
    email_host = hostname(str(fields.get("email", "")).split("@")[-1]) if "@" in str(fields.get("email", "")) else ""
    web_host = hostname(website)
    if email_host and web_host and registrable_hint(email_host) != registrable_hint(web_host) and email_host not in COMMON_DOMAINS:
        checks.append(_check("contact_domain_mismatch", "연락처 도메인", "warn",
                             f"이메일({email_host})과 웹사이트({web_host}) 도메인이 다릅니다", "website"))
    checks += inspect_url(str(fields.get("qr_url", "")), [email_host, web_host])
    confidence_values = [float(item.get("confidence", 0)) for item in (field_confidence or {}).values()]
    accuracy = round(mean(confidence_values) * 100) if confidence_values else 0
    evaluated = [item for item in checks if item["state"] != "unknown" and not item["id"].startswith("qr_")]
    consistency = max(0, 100 - 20 * sum(item["state"] == "warn" for item in evaluated)) if evaluated else 0
    safety = "warn" if any(item["state"] == "warn" and item["id"].startswith("qr_") for item in checks) else ("ok" if fields.get("qr_url") else "unknown")
    return {"scores": {"accuracy": accuracy, "consistency": consistency, "safety": safety},
            "checks": checks, "duplicate": find_similar_contact(fields, contacts, exclude_id)}


def append_online_checks(verification: dict, checks: list[dict]) -> dict:
    verification["checks"].extend(checks)
    evaluated = [item for item in verification["checks"]
                 if item["state"] != "unknown" and not item["id"].startswith("qr_")]
    verification["scores"]["consistency"] = (
        max(0, 100 - 20 * sum(item["state"] == "warn" for item in evaluated))
        if evaluated else 0
    )
    verification["online"] = True
    return verification


def verification_storage(verification: dict) -> dict[str, Any]:
    scores = verification["scores"]
    return {"score_accuracy": scores["accuracy"], "score_consistency": scores["consistency"],
            "score_safety": scores["safety"], "verify_json": json.dumps(verification["checks"], ensure_ascii=False)}
