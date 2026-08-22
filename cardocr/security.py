"""Offline QR extraction and conservative URL safety checks."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

import cv2
import numpy as np


SHORTENER_HOSTS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly",
    "han.gl", "vo.la", "url.kr", "me2.do",
}


def extract_qr_url(*images: np.ndarray) -> str:
    """Try each image once and return the first decoded HTTP(S) QR payload."""
    detector = cv2.QRCodeDetector()
    for image in images:
        if image is None or not getattr(image, "size", 0):
            continue
        variants = [image]
        if min(image.shape[:2]) < 300:
            scale = max(2, int(320 / min(image.shape[:2])))
            enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            border = max(16, scale * 4)
            variants.append(cv2.copyMakeBorder(enlarged, border, border, border, border,
                                                cv2.BORDER_CONSTANT, value=255))
        for candidate in variants:
            try:
                value, _points, _straight = detector.detectAndDecode(candidate)
            except cv2.error:
                continue
            value = value.strip()
            if value and urlparse(_with_scheme(value)).scheme in {"http", "https"}:
                return value
    return ""


def _with_scheme(value: str) -> str:
    return value if re.match(r"^[a-z][a-z0-9+.-]*://", value, re.I) else "https://" + value


def hostname(value: str) -> str:
    try:
        host = (urlparse(_with_scheme(value)).hostname or "").rstrip(".").lower()
        return host[4:] if host.startswith("www.") else host
    except ValueError:
        return ""


def registrable_hint(host: str) -> str:
    """Return a dependency-free comparison hint (adequate for conservative warnings)."""
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    if parts[-2] in {"co", "or", "go", "ac", "ne"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def inspect_url(url: str, expected_domains: list[str] | None = None) -> list[dict]:
    checks: list[dict] = []
    if not url:
        return [{"id": "qr_presence", "label": "QR 코드", "state": "unknown",
                 "message": "QR 코드를 확인할 수 없습니다", "field": "qr_url", "suggestion": None}]
    parsed = urlparse(_with_scheme(url))
    host = hostname(url)
    checks.append({"id": "qr_presence", "label": "QR 코드", "state": "ok",
                   "message": "QR 링크를 인식했습니다", "field": "qr_url", "suggestion": None})
    if parsed.scheme.lower() != "https":
        checks.append({"id": "qr_https", "label": "QR 암호화 연결", "state": "warn",
                       "message": "QR 링크가 HTTPS를 사용하지 않습니다", "field": "qr_url", "suggestion": None})
    else:
        checks.append({"id": "qr_https", "label": "QR 암호화 연결", "state": "ok",
                       "message": "HTTPS 링크입니다", "field": "qr_url", "suggestion": None})
    if host in SHORTENER_HOSTS:
        checks.append({"id": "qr_shortener", "label": "단축 URL", "state": "warn",
                       "message": "최종 목적지를 숨기는 단축 URL입니다", "field": "qr_url", "suggestion": None})
    suspicious = "xn--" in host or any(ord(char) > 127 for char in host)
    try:
        ipaddress.ip_address(host)
        suspicious = True
    except ValueError:
        pass
    if suspicious:
        checks.append({"id": "qr_suspicious_host", "label": "QR 도메인", "state": "warn",
                       "message": "IP 주소 또는 국제화 도메인이 사용되어 확인이 필요합니다", "field": "qr_url", "suggestion": None})
    expected = [hostname(item) for item in (expected_domains or []) if hostname(item)]
    if expected and host and all(registrable_hint(host) != registrable_hint(item) for item in expected):
        checks.append({"id": "qr_domain_mismatch", "label": "QR 링크 도메인", "state": "warn",
                       "message": f"명함 도메인({expected[0]})과 QR 링크({host})가 다릅니다",
                       "field": "qr_url", "suggestion": None})
    return checks
