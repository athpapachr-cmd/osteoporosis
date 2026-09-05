from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

PRODUCT_LABELS = {"medikey": "Medikey", "diros": "DIROS", "thermedico": "Thermedico"}

INDICATIONS: dict[str, dict[str, Any]] = {
    "KNEE_OA_KL34": {"label": "Γόνατο — Οστεοαρθρίτιδα 3ου/4ου βαθμού Kellgren-Lawrence", "site_key": "knee", "official_other": False, "requires_intervention": False},
    "SI_DEGENERATIVE": {"label": "Ιερολαγόνια — Εκφυλιστική παθολογία", "site_key": "si", "official_other": False, "requires_intervention": True, "intervention_kind": "si_injection"},
    "HIP_OA_KL34": {"label": "Ισχίο — Οστεοαρθρίτιδα 3ου/4ου βαθμού Kellgren-Lawrence", "site_key": "hip", "official_other": False, "requires_intervention": True, "intervention_kind": "hip_diagnostic_block"},
    "MORTON_NEUROMA": {"label": "Νευρίνωμα Morton", "site_key": "morton", "official_other": False, "requires_intervention": False},
    "SHOULDER_OA_KL34": {"label": "Ώμος — Οστεοαρθρίτιδα 3ου/4ου βαθμού Kellgren-Lawrence", "site_key": "shoulder", "official_other": False, "requires_intervention": False},
    "SHOULDER_IRREPARABLE_CUFF": {"label": "Ώμος — Εκτεταμένη μη χειρουργικά αποκαταστάσιμη ρήξη στροφικού πετάλου", "site_key": "shoulder", "official_other": False, "requires_intervention": False},
    "OTHER_LATERAL_EPICONDYLITIS": {"label": "Άλλο — Αγκώνας / Έξω επικονδυλίτιδα", "site_key": "elbow", "official_other": True, "other_area": "Αγκώνας", "other_diagnosis": "Έξω επικονδυλίτιδα", "requires_intervention": False},
    "OTHER_DEQUERVAIN": {"label": "Άλλο — Καρπός / De Quervain", "site_key": "wrist", "official_other": True, "other_area": "Καρπός", "other_diagnosis": "De Quervain", "requires_intervention": False},
    "OTHER_CUSTOM": {"label": "Άλλο — Προσαρμοσμένη περιοχή / διάγνωση", "site_key": "other", "official_other": True, "requires_intervention": False},
}

RF_REASON_LABELS = {
    "FAILED_PHARMACOLOGIC": "Αποτυχία φαρμακευτικής αγωγής",
    "FAILED_CONSERVATIVE": "Αποτυχία συντηρητικής θεραπείας",
    "DECLINES_SURGERY": "Ο ασθενής δεν επιθυμεί χειρουργική αντιμετώπιση",
    "HIGH_SURGICAL_RISK": "Σοβαρά συνοδά προβλήματα υγείας / αυξημένος χειρουργικός κίνδυνος",
}

LATERALITY_LABELS = {"left": "Αριστερά", "right": "Δεξιά", "bilateral": "Αμφοτερόπλευρα", "midline": "Κεντρικά", "none": "Χωρίς πλευρά"}

@dataclass(frozen=True)
class DoctorProfile:
    name: str
    gesy_code: str
    specialty: str
    medical_center: str
    phone: str
    email: str

    @classmethod
    def from_environment(cls) -> "DoctorProfile":
        raw = os.getenv("RF_DOCTOR_PROFILE_JSON", "").strip()
        if not raw:
            raise ValueError("RF_DOCTOR_PROFILE_JSON is not configured")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("RF_DOCTOR_PROFILE_JSON is not valid JSON") from exc
        required = ("name", "gesy_code", "specialty", "medical_center", "phone", "email")
        values = {key: str(data.get(key) or "").strip() for key in required}
        if any(not value for value in values.values()):
            raise ValueError("RF_DOCTOR_PROFILE_JSON is incomplete")
        return cls(**values)

@dataclass(frozen=True)
class ProductProfile:
    key: str
    label: str
    code: str
    description: str
    quantity: str

def product_catalog_from_environment() -> dict[str, ProductProfile]:
    raw = os.getenv("RF_PRODUCT_CATALOG_JSON", "").strip()
    if not raw:
        raise ValueError("RF_PRODUCT_CATALOG_JSON is not configured")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("RF_PRODUCT_CATALOG_JSON is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("RF_PRODUCT_CATALOG_JSON must be an object")
    catalog: dict[str, ProductProfile] = {}
    for key, label in PRODUCT_LABELS.items():
        item = data.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"Missing RF product configuration: {key}")
        code = str(item.get("code") or "").strip()
        description = str(item.get("description") or "").strip()
        quantity = str(item.get("quantity") or "").strip()
        if not code or not description or not quantity:
            raise ValueError(f"Incomplete RF product configuration: {key}")
        catalog[key] = ProductProfile(key=key, label=label, code=code, description=description, quantity=quantity)
    return catalog
