"""Heuristics for converting Korean/English OCR lines into business-card fields."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable


EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?<!\d)(?:(?:\+?82[\s.\-]?(?:\(0\)[\s.\-]?)?)|0)"
    r"(?:1[016789]|2|3[1-3]|4[1-4]|5[1-5]|6[1-4]|70)"
    r"[\s.\-]?\d{3,4}[\s.\-]?\d{4}(?!\d)"
)
WEBSITE_RE = re.compile(
    r"\b(?:https?://)?(?:www\.)?[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)+"
    r"(?:/[\w\-./?%&=+#]*)?",
    re.IGNORECASE,
)

COMPANY_MARKERS = (
    "주식회사", "(주)", "㈜", "회사", "기업", "그룹", "산업", "테크", "솔루션",
    "시스템", "연구소", "대학교", "센터", "진흥원", "공사", "재단", "협회", "기관",
    "company", "corp", "corporation", "co.,", "inc", "ltd", "group", "agency", "bank",
    "innovation", "promotion", "systems", "system",
)
TITLE_MARKERS = (
    "대표이사", "대표", "사장", "부사장", "전무", "상무", "이사", "부장", "차장", "과장",
    "대리", "주임", "선임", "책임", "사원", "팀장", "실장", "본부장", "매니저", "컨설턴트",
    "연구원", "ceo", "cto", "cfo", "manager", "director", "engineer", "consultant",
)
DEPARTMENT_MARKERS = ("팀", "부", "실", "본부", "센터", "연구소")
ADDRESS_MARKERS = (
    "특별시", "광역시", "특별자치시", "특별자치도", "시", "도", "구", "군", "읍", "면",
    "로", "길", "동", "번지", "층", "빌딩",
)
LABEL_RE = re.compile(
    r"^(?:tel|전화|mobile|mob|휴대폰|fax|e-?mail|web|website|주소|address|[TEMF])\s*[:.]?\s*",
    re.IGNORECASE,
)
POSTAL_ADDRESS_RE = re.compile(r"(?:^|\s)\d{5}\s*[|I1]?\s*.*(?:시|도|구|군|로|길|동)")
KOREAN_ADDRESS_RE = re.compile(r"(?:특별시|광역시|특별자치[시도]|[가-힣]+[시도])\s*.*(?:구|군|로|길|동)")


@dataclass(slots=True)
class ParsedCard:
    name: str = ""
    company: str = ""
    job_title: str = ""
    phone: str = ""
    phone2: str = ""
    email: str = ""
    website: str = ""
    address: str = ""
    raw_text: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _normalize_lines(lines: Iterable[object]) -> tuple[list[str], dict[str, float]]:
    normalized: list[str] = []
    heights: dict[str, float] = {}
    for source in lines:
        source_text = str(getattr(source, "text", source))
        box = getattr(source, "box", None)
        height = 0.0
        if isinstance(box, (list, tuple)) and box:
            try:
                y_values = [float(point[1]) for point in box]
                height = max(y_values) - min(y_values)
            except (IndexError, TypeError, ValueError):
                height = 0.0
        for part in source_text.replace("\r", "\n").split("\n"):
            line = re.sub(r"\s+", " ", part).strip(" |·•")
            if line and line not in normalized:
                normalized.append(line)
            if line:
                heights[line] = max(heights.get(line, 0.0), height)
    return normalized, heights


def _normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if digits.startswith("82"):
        digits = "0" + digits[2:]
    if len(digits) == 9 and digits.startswith("02"):
        return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"
    if len(digits) == 10:
        if digits.startswith("02"):
            return f"{digits[:2]}-{digits[2:6]}-{digits[6:]}"
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    if len(digits) == 11:
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    return value.strip()


def _repair_domain(value: str) -> str:
    domain = re.sub(r"\s+", "", value.lower()).strip(".,;:|/")
    domain = re.sub(r"(?i)(co|or|go|ac)kr$", r".\1.kr", domain)
    domain = re.sub(r"(?i)(co|or|go|ac)k$", r".\1.kr", domain)
    if "." not in domain:
        domain = re.sub(r"(?i)(com|net|org|kr)$", r".\1", domain)
    domain = re.sub(r"^www(?=[a-z0-9])", "www.", domain)
    domain = domain.replace("dlp.or.kr", "dip.or.kr")
    return domain


def _extract_email(lines: list[str]) -> str:
    joined = "\n".join(lines)
    direct = EMAIL_RE.search(joined)
    if direct:
        return direct.group(0).lower()

    for line in lines:
        if "@" not in line:
            continue
        candidate = LABEL_RE.sub("", line).replace(" ", "")
        candidate = re.sub(r"[^A-Za-z0-9._%+@\-]", "", candidate)
        if candidate.count("@") != 1:
            continue
        local, domain = candidate.split("@", 1)
        repaired = f"{local}@{_repair_domain(domain)}"
        if EMAIL_RE.fullmatch(repaired):
            return repaired.lower()
    return ""


def _phone_matches(line: str) -> list[str]:
    matches = [match.group(0) for match in PHONE_RE.finditer(line)]
    loose_pattern = re.compile(r"(?<!\d)(?:\+?82|0)[\d\s.\-()*]{7,18}\d(?!\d)")
    matches.extend(match.group(0) for match in loose_pattern.finditer(line))
    phones: list[str] = []
    for value in matches:
        normalized = _normalize_phone(value)
        digits = re.sub(r"\D", "", normalized)
        valid_prefix = digits.startswith(
            ("02", "010", "011", "016", "017", "018", "019", "031", "032", "033",
             "041", "042", "043", "044", "051", "052", "053", "054", "055", "061",
             "062", "063", "064", "070")
        )
        if valid_prefix and 9 <= len(digits) <= 11 and normalized not in phones:
            phones.append(normalized)
    return phones


def _extract_phones(lines: list[str]) -> tuple[str, str]:
    candidates: dict[str, int] = {}
    for index, line in enumerate(lines):
        lowered = line.lower()
        is_fax = bool(
            re.search(r"(?:^|\s)(?:f|fax)\s*[:.]?", lowered)
            or re.match(r"^5\s+0", lowered)
        )
        if is_fax:
            continue
        for phone in _phone_matches(line):
            lowered = line.lower()
            digits = re.sub(r"\D", "", phone)
            score = 0
            if digits.startswith("010"):
                score += 100
            if re.search(r"(?:^|\s)(?:m|mobile|mob|휴대폰)\s*[:.]?", lowered):
                score += 40
            if re.search(r"(?:^|\s)(?:t|tel|전화)\s*[:.]?", lowered):
                score += 35
            score -= index
            candidates[phone] = max(candidates.get(phone, -999), score)
    ordered = sorted(candidates, key=lambda phone: candidates[phone], reverse=True)
    primary = ordered[0] if ordered else ""
    primary_is_mobile = re.sub(r"\D", "", primary).startswith("01")
    secondary = next(
        (
            candidate
            for candidate in ordered[1:]
            if re.sub(r"\D", "", candidate).startswith("01") != primary_is_mobile
        ),
        ordered[1] if len(ordered) > 1 else "",
    )
    return (
        primary,
        secondary,
    )


def _is_address(line: str) -> bool:
    return bool(POSTAL_ADDRESS_RE.search(line) or KOREAN_ADDRESS_RE.search(line))


def _is_contact_line(line: str) -> bool:
    return bool("@" in line or _phone_matches(line) or WEBSITE_RE.fullmatch(line))


def _normalize_name(line: str) -> str:
    line = line.strip(" |·•,-")
    if re.fullmatch(r"[가-힣](?:\s+[가-힣]){1,4}", line):
        return re.sub(r"\s+", "", line)
    return line


def _looks_like_name(line: str) -> bool:
    cleaned = re.sub(r"\s", "", line.strip(" |·•,-"))
    if any(marker.lower() in line.lower() for marker in COMPANY_MARKERS + TITLE_MARKERS):
        return False
    if _is_contact_line(line) or _is_address(line):
        return False
    if re.fullmatch(r"[가-힣]{2,5}", cleaned):
        return True
    return bool(re.fullmatch(r"[A-Za-z]{2,20}(?:\s+[A-Za-z]{2,20}){1,2}", line))


def _repair_korean_text(value: str) -> str:
    replacements = {
        "되지털": "디지털",
        "디지탤": "디지털",
        "디지터": "디지털",
        "되지털": "디지털",
        "학신": "혁신",
        "최신": "혁신",
        "진홍원": "진흥원",
        "진하원": "진흥원",
        "강역시": "광역시",
        "광외시": "광역시",
        "광액시": "광역시",
        "강의시": "광역시",
        "광의시": "광역시",
        "알다시티": "알파시티",
        "알따시티": "알파시티",
        "안파시티": "알파시티",
        "대구디지터최신진하원": "대구디지털혁신진흥원",
        "대구디지터혁신진하원": "대구디지털혁신진흥원",
        "시인자양성팀": "AI인재양성팀",
        "시인재양성팀": "AI인재양성팀",
        "선인자양성팀": "선임 / AI인재양성팀",
    }
    repaired = value
    for wrong, correct in replacements.items():
        repaired = repaired.replace(wrong, correct)
    repaired = re.sub(r"(?<=시티)[itI](?=로)", "1", repaired)
    return repaired


def _repair_english_company(value: str) -> str:
    replacements = {
        "Dacgu": "Daegu",
        "Docgu": "Daegu",
        "Dloltal": "Digital",
        "Dloltol": "Digital",
        "Dloftal": "Digital",
        "Dloital": "Digital",
        "Dioital": "Digital",
        "Innovatlon": "Innovation",
        "Promotlon": "Promotion",
        "Promoton": "Promotion",
        "Aooncy": "Agency",
        "Kooncy": "Agency",
    }
    repaired = value
    for wrong, correct in replacements.items():
        repaired = re.sub(re.escape(wrong), correct, repaired, flags=re.IGNORECASE)
    return repaired


def _company_score(line: str, index: int, height: float = 0.0) -> float:
    if _is_contact_line(line) or _is_address(line) or _looks_like_name(line):
        return -999
    lowered = line.lower()
    if any(marker.lower() in lowered for marker in TITLE_MARKERS):
        return -999
    has_marker = any(marker.lower() in lowered for marker in COMPANY_MARKERS)
    has_long_korean = bool(re.search(r"[가-힣]{4,}", line))
    has_english_org_shape = bool(
        re.fullmatch(r"[A-Za-z0-9]{2,}(?:\s+[A-Za-z0-9]{2,}){3,}", line)
    )
    if not (has_marker or has_long_korean or has_english_org_shape):
        return -999
    score = 0
    score += 40 * sum(marker.lower() in lowered for marker in COMPANY_MARKERS)
    if re.search(r"[가-힣]{4,}", line):
        score += 80
    if re.search(r"(?:원|사|교|소|터)$", line.strip()):
        score += 15
    if re.search(r"\b(?:innovation|promotion|agency|company|systems?)\b", lowered):
        score += 12
    score += max(0, 10 - index)
    if len(line) > 60:
        score -= 30
    score += height * 4
    return score


def parse_business_card(lines: Iterable[object]) -> ParsedCard:
    clean_lines, line_heights = _normalize_lines(lines)
    raw_text = "\n".join(clean_lines)

    email = _extract_email(clean_lines)
    phone, phone2 = _extract_phones(clean_lines)

    website = ""
    for line in clean_lines:
        if "@" in line or _phone_matches(line):
            continue
        compact = re.sub(r"\s+", "", LABEL_RE.sub("", line)).lower()
        compact = _repair_domain(compact)
        match = WEBSITE_RE.search(compact)
        if match:
            website = match.group(0).rstrip(".,")
            break
    if not website and email:
        website = "www." + email.split("@", 1)[1]

    title_lines = [
        line for line in clean_lines
        if len(line) <= 40
        and not _is_contact_line(line)
        and not _is_address(line)
        and any(marker.lower() in line.lower() for marker in TITLE_MARKERS)
    ]
    job_title = title_lines[0] if title_lines else ""

    name = ""
    if job_title:
        combined = re.match(r"^([가-힣]{2,5})\s+(.+)$", job_title)
        if combined and any(
            marker.lower() in combined.group(2).lower() for marker in TITLE_MARKERS
        ):
            name = combined.group(1)
            job_title = combined.group(2).strip()
        for marker in TITLE_MARKERS:
            if name:
                break
            marker_index = job_title.lower().find(marker.lower())
            if marker_index > 0:
                prefix = job_title[:marker_index].strip(" /|·•,-")
                if _looks_like_name(prefix):
                    name = _normalize_name(prefix)
                    job_title = job_title[marker_index:].strip()
                    break
    if not name:
        name = next(
            (_normalize_name(line) for line in clean_lines[:10] if _looks_like_name(line)),
            "",
        )

    departments = [
        line for line in clean_lines
        if line not in {name, job_title}
        and len(line) <= 30
        and not _is_contact_line(line)
        and not _is_address(line)
        and any(line.endswith(marker) for marker in DEPARTMENT_MARKERS)
        and not any(marker.lower() in line.lower() for marker in COMPANY_MARKERS)
    ]
    if departments:
        department = departments[0]
        if job_title and department not in job_title:
            job_title = f"{job_title} / {department}"
        elif not job_title:
            job_title = department

    company_candidates = [
        (_company_score(line, index, line_heights.get(line, 0.0)), index, line)
        for index, line in enumerate(clean_lines[:10])
        if line not in {name, job_title} and line not in departments
    ]
    best_company = max(company_candidates, default=(-999, 0, ""), key=lambda item: item[0])
    company = _repair_korean_text(best_company[2]) if best_company[0] > 0 else ""
    company = _repair_english_company(company)
    if company and re.search(r"[A-Za-z]", company) and "agency" not in company.lower():
        next_index = best_company[1] + 1
        if next_index < len(clean_lines) and re.fullmatch(
            r"[A-Za-z]{3,10}", clean_lines[next_index]
        ) and re.search(r"(?:gency|ooncy)$", clean_lines[next_index], re.IGNORECASE):
            company = f"{company} Agency"
    company = re.sub(r"^[^가-힣A-Za-z0-9]+", "", company).strip()
    company = re.sub(r"^[가-힣][^가-힣A-Za-z0-9]+\s*", "", company).strip()
    if re.match(r"^[가-힣]\s+[가-힣]{5,}", company) and any(
        marker in company for marker in ("진흥원", "공사", "재단", "협회", "대학교")
    ):
        company = company.split(" ", 1)[1]

    job_title = _repair_korean_text(job_title)
    job_title = re.sub(
        r"^(선임|책임|주임|대리|과장|차장|부장|팀장)\s*/\s*\1(?:\s*/\s*)?",
        r"\1 / ",
        job_title,
    ).strip(" /·")

    address = next(
        (
            _repair_korean_text(LABEL_RE.sub("", line)).strip()
            for line in clean_lines
            if _is_address(line) and len(line) >= 8
        ),
        "",
    )

    return ParsedCard(
        name=name,
        company=company,
        job_title=job_title,
        phone=phone,
        phone2=phone2,
        email=email,
        website=website,
        address=address,
        raw_text=raw_text,
    )
