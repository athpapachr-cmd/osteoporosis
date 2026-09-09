from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable


FORBIDDEN_FIELD_NAMES = {
    "patient_name",
    "full_name",
    "identity_number",
    "gesy_number",
    "phone",
    "email",
    "address",
    "date_of_birth",
    "dob",
}

_EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
_CONTIGUOUS_PHONE_RE = re.compile(r"(?<!\d)\+?\d{8,15}(?!\d)")
_SEPARATED_PHONE_RE = re.compile(r"(?<!\w)\+?\d{1,4}(?:[ ()\-.]\d){6,14}(?!\w)")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_DMY_DATE_RE = re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}\b")
_UUID_RE = re.compile(r"(?i)^\{?[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\}?$")


@dataclass(frozen=True)
class PrivacyFinding:
    code: str
    path: str


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def _path(parts: Iterable[str | int]) -> str:
    out = ""
    for part in parts:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += ("." if out else "") + part
    return out


def _is_bibliographic_numeric_exclusion(parts: tuple[str | int, ...]) -> bool:
    """Return True only for fields where numeric citation/locator patterns are expected.

    The exclusion suppresses only the generic phone-number-like heuristic. Explicit
    phone phrases, email, identity/GeSY, DOB and address detection still run in
    ``scan_text`` for these fields.
    """
    if not parts:
        return False
    last = str(parts[-1])
    text_parts = {str(part) for part in parts if not isinstance(part, int)}

    # LearningReferenceV1 stores the source citation in ``title`` for rich imports.
    # Bibliographic citations commonly contain year/volume/page ranges and other
    # numeric patterns that resemble phone numbers. Treat title as bibliographic
    # numeric content while retaining all explicit-identifier phrase checks.
    if last in {"title", "pmid", "doi", "url"} and "references" in text_parts:
        return True

    # L-1B fresh-resource recommendations are mutable learning locators. Only the
    # URL itself gets the numeric-locator exclusion; titles/rationales/providers
    # remain fully subject to the generic phone heuristic.
    if last == "url" and ("resources" in text_parts or "resource_recommendations" in text_parts):
        return True
    return False


def find_forbidden_structured_fields(value: Any, parts: tuple[str | int, ...] = ()) -> list[PrivacyFinding]:
    findings: list[PrivacyFinding] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_parts = (*parts, str(key))
            if str(key).casefold() in FORBIDDEN_FIELD_NAMES:
                findings.append(PrivacyFinding("direct_identifier_field_forbidden", _path(child_parts)))
            findings.extend(find_forbidden_structured_fields(child, child_parts))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            findings.extend(find_forbidden_structured_fields(child, (*parts, idx)))
    return findings


def _phone_like(text: str) -> bool:
    if _UUID_RE.match(text.strip()):
        return False
    # Dates are legitimate learning content. Remove common full-date shapes before
    # the generic phone-like heuristic so 2026-09-07 is not treated as a phone.
    scrubbed = _ISO_DATE_RE.sub(" ", text)
    scrubbed = _DMY_DATE_RE.sub(" ", scrubbed)
    return bool(_CONTIGUOUS_PHONE_RE.search(scrubbed) or _SEPARATED_PHONE_RE.search(scrubbed))


def scan_text(value: str, *, path: str, bibliographic_numeric_exclusion: bool = False) -> list[PrivacyFinding]:
    findings: list[PrivacyFinding] = []
    folded = _fold(value)

    if _EMAIL_RE.search(value):
        findings.append(PrivacyFinding("email_address_detected", path))

    identity_terms = (
        "identity number",
        "id number",
        "gesy number",
        "gesy no",
        "αριθμος ταυτοτητας",
        "αρ ταυτοτητας",
        "αριθμος γεσυ",
        "αρ γεσυ",
    )
    if any(term in folded for term in identity_terms) and re.search(r"\d{5,}", folded):
        findings.append(PrivacyFinding("explicit_identity_or_gesy_phrase_detected", path))

    dob_terms = ("date of birth", "dob", "ημερομηνια γεννησης", "ημερ γεννησης")
    if any(term in folded for term in dob_terms) and (_ISO_DATE_RE.search(value) or _DMY_DATE_RE.search(value)):
        findings.append(PrivacyFinding("full_date_of_birth_phrase_detected", path))

    address_terms = ("postal address", "home address", "διευθυνση κατοικιας", "ταχυδρομικη διευθυνση")
    if any(term in folded for term in address_terms):
        findings.append(PrivacyFinding("explicit_postal_address_phrase_detected", path))

    phone_terms = ("phone", "telephone", "mobile", "τηλεφων", "κινητο")
    if any(term in folded for term in phone_terms) and re.search(r"\d{6,}", folded):
        findings.append(PrivacyFinding("explicit_phone_phrase_detected", path))
    elif not bibliographic_numeric_exclusion and _phone_like(value):
        findings.append(PrivacyFinding("phone_number_like_sequence_detected", path))

    return list(dict.fromkeys(findings))


def scan_persistable_strings(value: Any, parts: tuple[str | int, ...] = ()) -> list[PrivacyFinding]:
    findings: list[PrivacyFinding] = []
    if isinstance(value, dict):
        for key, child in value.items():
            findings.extend(scan_persistable_strings(child, (*parts, str(key))))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            findings.extend(scan_persistable_strings(child, (*parts, idx)))
    elif isinstance(value, str):
        path = _path(parts)
        findings.extend(
            scan_text(
                value,
                path=path,
                bibliographic_numeric_exclusion=_is_bibliographic_numeric_exclusion(parts),
            )
        )
    return list(dict.fromkeys(findings))
