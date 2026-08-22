"""Opt-in network validation with strict timeouts and neutral failures."""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlparse

import requests

from .security import hostname


TIMEOUT_SECONDS = 2.0


def _check(identifier: str, label: str, state: str, message: str, field: str) -> dict:
    return {"id": identifier, "label": label, "state": state, "message": message,
            "field": field, "suggestion": None, "online": True}


def check_email_mx(email: str, timeout: float = TIMEOUT_SECONDS) -> dict:
    domain = email.rsplit("@", 1)[-1].strip().lower() if "@" in email else ""
    if not domain:
        return _check("email_mx", "이메일 도메인", "unknown", "조회할 이메일 도메인이 없습니다", "email")
    try:
        import dns.exception
        import dns.resolver
    except ImportError:
        return _check("email_mx", "이메일 도메인", "unknown", "DNS 검증 모듈을 사용할 수 없습니다", "email")
    try:
        resolver = dns.resolver.Resolver()
        resolver.timeout = timeout
        resolver.lifetime = timeout
        answers = resolver.resolve(domain, "MX")
        if answers:
            return _check("email_mx", "이메일 도메인", "ok", "메일 수신 서버(MX)가 확인되었습니다", "email")
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
        return _check("email_mx", "이메일 도메인", "warn", "메일 수신 서버(MX)를 찾을 수 없습니다", "email")
    except (dns.exception.Timeout, dns.resolver.NoNameservers, OSError):
        pass
    return _check("email_mx", "이메일 도메인", "unknown", "네트워크 문제로 MX를 확인하지 못했습니다", "email")


def _safe_public_url(value: str) -> str:
    if not value:
        return ""
    url = value if "://" in value else "https://" + value
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme not in {"http", "https"} or not host or host.lower() == "localhost":
        return ""
    try:
        if ipaddress.ip_address(host).is_private:
            return ""
    except ValueError:
        pass
    return url


def check_website(value: str, timeout: float = TIMEOUT_SECONDS) -> dict:
    url = _safe_public_url(value)
    if not value:
        return _check("website_reachable", "웹사이트 접속", "unknown", "조회할 웹사이트가 없습니다", "website")
    if not url:
        return _check("website_reachable", "웹사이트 접속", "warn", "안전하게 조회할 수 없는 주소입니다", "website")
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout,
                                 headers={"User-Agent": "CardFlow-Validator/1.0"})
        if response.status_code in {405, 501}:
            response = requests.get(url, allow_redirects=True, timeout=timeout, stream=True,
                                    headers={"User-Agent": "CardFlow-Validator/1.0"})
        if response.status_code < 400:
            final_host = hostname(response.url)
            return _check("website_reachable", "웹사이트 접속", "ok",
                          f"접속 확인됨{f' ({final_host})' if final_host else ''}", "website")
        return _check("website_reachable", "웹사이트 접속", "warn",
                      f"웹사이트가 HTTP {response.status_code}을 반환했습니다", "website")
    except requests.RequestException:
        return _check("website_reachable", "웹사이트 접속", "unknown",
                      "네트워크 문제로 접속 여부를 확인하지 못했습니다", "website")


def run_online_checks(fields: dict[str, Any], timeout: float = TIMEOUT_SECONDS) -> list[dict]:
    return [check_email_mx(str(fields.get("email", "")), timeout),
            check_website(str(fields.get("website", "")), timeout)]
