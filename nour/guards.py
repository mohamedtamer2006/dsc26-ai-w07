"""PII detection and redaction for Egyptian national IDs and phone numbers."""
from __future__ import annotations

import datetime
import re

ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
PERSIAN = "۰۱۲۳۴۵۶۷۸۹"
DIGIT_MAP = {ord(c): str(i) for i, c in enumerate(ARABIC_INDIC)}
DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate(PERSIAN)})

GOV_CODES = {
    "01", "02", "03", "04",
    "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "31", "32", "33", "34", "35", "88",
}

NID_RE = re.compile(r"(?<!\d)(\d{14})(?!\d)")

PHONE_RE = re.compile(
    r"(?<![\d+])"
    r"("
    r"(?:(?:\+[\s-]?2|00[\s-]?2|2)[\s-]?)?"
    r"0[\s-]?1[\s-]?[0125]"
    r"(?:[\s-]?\d){8}"
    r")"
    r"(?!\d)"
)


def _to_ascii(text: str) -> str:
    return text.translate(DIGIT_MAP)


def find_pii(text: str) -> list[dict]:
    """Find Egyptian national IDs and mobile phone numbers in text.

    Returns a list of dicts: [{"type": "NATIONAL_ID" | "PHONE", "start": int, "end": int}],
    sorted by start offset. National IDs take precedence over overlapping phones.
    """
    ascii_text = _to_ascii(text)

    # 1. Detect National IDs
    national_ids: list[dict] = []
    for match in NID_RE.finditer(ascii_text):
        s = match.group(1)
        first_digit = s[0]
        if first_digit not in ("2", "3"):
            continue

        century = 1900 if first_digit == "2" else 2000
        try:
            year = century + int(s[1:3])
            month = int(s[3:5])
            day = int(s[5:7])
            datetime.date(year, month, day)
        except ValueError:
            continue

        gov_code = s[7:9]
        if gov_code not in GOV_CODES:
            continue

        national_ids.append({
            "type": "NATIONAL_ID",
            "start": match.start(),
            "end": match.end(),
        })

    # 2. Detect Phone numbers
    phones: list[dict] = []
    for match in PHONE_RE.finditer(ascii_text):
        phones.append({
            "type": "PHONE",
            "start": match.start(),
            "end": match.end(),
        })

    # 3. Filter out phones overlapping with any national ID
    valid_phones: list[dict] = []
    for p in phones:
        overlaps = any(
            max(p["start"], nid["start"]) < min(p["end"], nid["end"])
            for nid in national_ids
        )
        if not overlaps:
            valid_phones.append(p)

    all_findings = national_ids + valid_phones
    all_findings.sort(key=lambda x: x["start"])
    return all_findings


def redact(text: str) -> tuple[str, list[dict]]:
    """Redact Egyptian national IDs and mobile phone numbers from text.

    Replaces occurrences with [NATIONAL_ID] or [PHONE].
    Returns (redacted_text, findings).
    """
    findings = find_pii(text)
    if not findings:
        return text, []

    redacted_text = text
    for item in reversed(findings):
        placeholder = f"[{item['type']}]"
        start = item["start"]
        end = item["end"]
        redacted_text = redacted_text[:start] + placeholder + redacted_text[end:]

    return redacted_text, findings
