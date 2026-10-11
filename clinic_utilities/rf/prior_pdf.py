"""Read *provisional* values from the original native RF official A1/A2 PDF.

Only canonical template layouts are recognized; no OCR, file storage, patient
linkage or inference that a previously requested RF was actually performed.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import fitz

from .catalog import INDICATIONS, product_catalog_from_environment
from .pdf import (
    OFFICIAL_TEMPLATE_PATH, _PAGE1_TEXT, _INDICATION_CHECKS,
    _OTHER_AREA, _OTHER_DIAGNOSIS, _A1_VAS, _A1_LOCATION, _A2_HISTORY,
)

MAX_PRIOR_RF_PDF_BYTES = 20 * 1024 * 1024
MAX_PRIOR_RF_PDF_PAGES = 35


def _word_signature(page: fitz.Page) -> Counter:
    # The original unchanged form, including static words and their positions,
    # provides a stronger anchor than merely seeing "%PDF" or a Greek keyword.
    words = page.get_text("words") or []
    return Counter(
        (round(w[0] / 3), round(w[1] / 3), str(w[4]).casefold())
        for w in words if len(str(w[4])) >= 2
    )


def _template_match(page: fitz.Page, template_page: fitz.Page) -> float:
    if abs(page.rect.width - template_page.rect.width) > 2 or abs(
        page.rect.height - template_page.rect.height
    ) > 2:
        return 0.0
    static = _word_signature(template_page)
    if len(static) < 14:  # no textual provenance in an image-only template
        return 0.0
    return sum((static & _word_signature(page)).values()) / sum(static.values())


def _inside(page: fitz.Page, rect: fitz.Rect) -> str:
    # Native RF inserts the value into the given rectangles in the first place.
    words = page.get_text("words", clip=rect) or []
    return " ".join(str(word[4]) for word in sorted(words, key=lambda w: (round(w[1] / 3), w[0]))).strip()[:300]


def _marked(page: fitz.Page, template_page: fitz.Page, rect: fitz.Rect) -> bool:
    def clipped(candidate):
        return Counter((round(w[0] / 2), round(w[1] / 2), str(w[4]))
                       for w in (candidate.get_text("words", clip=rect) or []))
    return bool(clipped(page) - clipped(template_page))


def _field(value: str, limit: int = 200) -> str:
    return str(value or "").strip()[:limit]


def extract_prior_rf_pdf(data: bytes) -> dict[str, Any]:
    if not data.startswith(b"%PDF") or len(data) > MAX_PRIOR_RF_PDF_BYTES:
        raise ValueError("Η παλιά αίτηση πρέπει να είναι έγκυρο PDF έως 20 MB")
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError("Δεν μπορεί να διαβαστεί το προηγούμενο RF PDF") from exc
    try:
        if document.page_count < 3 or document.page_count > MAX_PRIOR_RF_PDF_PAGES:
            return {"status": "unrecognized", "fields": {},
                    "message": "Δεν αναγνωρίστηκε επίσημη προηγούμενη αίτηση RF"}
        if not OFFICIAL_TEMPLATE_PATH.is_file():
            raise ValueError("Δεν είναι διαθέσιμο το επίσημο πρότυπο RF")
        with fitz.open(OFFICIAL_TEMPLATE_PATH) as template:
            if _template_match(document[0], template[0]) < 0.65:
                return {"status": "unrecognized", "fields": {},
                        "message": "Το αρχείο δεν ταυτοποιήθηκε ως επίσημη αίτηση RF"}
            a1 = _template_match(document[1], template[1])
            a2 = _template_match(document[1], template[4])
            if max(a1, a2) < 0.65 or abs(a1 - a2) < 0.08:
                return {"status": "unrecognized", "fields": {},
                        "message": "Δεν διακρίνεται αξιόπιστα επίσημο A1 από A2"}
            pathway = "A1" if a1 > a2 else "A2"
            first = document[0]
            fields: dict[str, Any] = {}
            for key in ("patient_name", "identity_number", "gesy_number"):
                value = _field(_inside(first, _PAGE1_TEXT[key]))
                if value:
                    fields[key] = value
            age = _inside(first, _PAGE1_TEXT["patient_age"])
            if age.isdecimal() and 1 <= int(age) <= 130:
                fields["age"] = int(age)  # HISTORICAL, not today's age

            code = _inside(first, _PAGE1_TEXT["product_code"])
            try:
                candidates = product_catalog_from_environment()
            except ValueError:
                candidates = {}
            products = [key for key, profile in candidates.items() if profile.code == code]
            if len(products) == 1:
                fields["product_key"] = products[0]

            page = document[1]
            form = template[1 if pathway == "A1" else 4]
            checked = [k for k, rect in _INDICATION_CHECKS.items() if _marked(page, form, rect)]
            # 3 OTHER indications share one printed box; cannot recover a unique code
            # unless the very specific free-form diagnosis is unambiguous.
            others = {"OTHER_LATERAL_EPICONDYLITIS", "OTHER_DEQUERVAIN", "OTHER_CUSTOM"}
            distinct = sorted(set(checked) - others)
            if len(distinct) == 1 and not any(k in checked for k in others):
                fields["indication_code"] = distinct[0]
            elif checked and all(k in others for k in checked):
                area = _inside(page, _OTHER_AREA)
                diagnosis = _inside(page, _OTHER_DIAGNOSIS)
                if area:
                    fields["other_area"] = _field(area, 160)
                if diagnosis:
                    fields["other_diagnosis"] = _field(diagnosis, 240)
                normalized = diagnosis.casefold()
                if "de quervain" in normalized:
                    fields["indication_code"] = "OTHER_DEQUERVAIN"
                elif "επικονδυλ" in normalized:
                    fields["indication_code"] = "OTHER_LATERAL_EPICONDYLITIS"
                elif area and diagnosis:
                    fields["indication_code"] = "OTHER_CUSTOM"

            if pathway == "A1":
                location = _inside(document[2], _A1_LOCATION)
                if location:
                    fields["exact_location"] = _field(location, 300)
                for k, rect in _A1_VAS.items():
                    value = _inside(document[2], rect)
                    if k.endswith("_vas"):
                        if value.isdecimal() and 0 <= int(value) <= 10:
                            fields[k] = int(value)
                    elif value:
                        fields[k] = value
            else:
                previous = {}
                for k, rect in _A2_HISTORY.items():
                    value = _inside(page, rect)
                    if k.startswith("vas") or k.endswith("_vas"):
                        if value.isdecimal() and 0 <= int(value) <= 10:
                            previous[k] = int(value)
                    elif value:
                        previous[k] = value
                # This is explicitly a *candidate*, not verified actual procedure
                # or a procedure_history_id eligible for the server's A2 branch.
                if previous:
                    fields["previous_procedure_candidate"] = previous

            # The unilateral laterality is not a dedicated recoverable form field.
            # Only derive from an *unambiguous* written location, not the old checkbox.
            place = str(fields.get("exact_location", "")).casefold()
            if "αριστερ" in place and "δεξι" not in place:
                fields["laterality"] = "left"
            elif "δεξι" in place and "αριστερ" not in place:
                fields["laterality"] = "right"
            return {
                "status": "partial" if not all(fields.get(k) for k in (
                    "patient_name", "identity_number", "indication_code", "laterality"
                )) else "recognized",
                "pathway": pathway, "fields": fields,
                "message": "Τα στοιχεία προέρχονται από παλιό RF PDF και χρειάζονται ιατρικό έλεγχο.",
                "actual_procedure_verified": False,
                "attachments_saved": False,
            }
    finally:
        document.close()
