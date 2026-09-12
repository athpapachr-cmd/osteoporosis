"""Server-authoritative Knee-OA product projection shared by Cockpit and local prototype.

This module owns deterministic validation, evidence projection, suggestions and
Greek referral projection for the reviewed Knee-OA product slice. It has no HTTP
server and no production registration side effect.
"""
from __future__ import annotations

import copy
import hashlib
import json
import uuid
from pathlib import Path

from clinic_utilities.physio_referral_runtime import get_cu1_engine, get_cu1_bundle
from clinic_utilities.physio_referral_formatter_el_v2 import CU1GreekReferralFormatter
from clinic_utilities.physio_referral_product.validate_knee_oa_template_contract_v1 import (
    load_yaml, render, copy_readiness_error,
)
from clinic_utilities.physio_referral_product.validate_knee_oa_evidence_interaction_v1 import (
    evidence_view, suggestion_candidates, apply_suggestion, export_readiness,
)
from clinic_utilities.physio_referral_product.prototype.qualifier_overlay import (
    apply_referral_overlay,
    clean_qualifiers,
    clinical_review_clues,
    empty_qualifiers,
    enrich_suggestion_facts,
    state_with_mapped_findings,
)

PRODUCT = Path(__file__).resolve().parent
PACKAGE = "knee-oa-prototype-1.1+step6a"
PINS = {
    "contracts/knee_oa_evidence_contract_v1.yaml": "8f4c657904ee028bba39d7a0557461a6acafadaf",
    "contracts/knee_oa_template_contract_v1.yaml": "e6c6a285ed4d2d64df1dca7b29630ef5053831e8",
    "contracts/knee_oa_evidence_interaction_v1.yaml": "08c60b1f244fa475a38ee7bbcc8d8bf1dcf7076f",
    "UX_CONTRACT_CURRENT.md": "f65343c0bc9652c715075853d3fc42c1910ff212",
    "validate_knee_oa_template_contract_v1.py": "aff39c675202c1711467de4566c2593d78a41886",
    "validate_knee_oa_evidence_interaction_v1.py": "3c7c029347a0e3545f6d5c67f11e04e1e24732c0",
}
for relative, expected in PINS.items():
    raw = (PRODUCT / relative).read_bytes()
    actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if actual != expected:
        raise RuntimeError("product_parent_identity_mismatch")

E = load_yaml(PRODUCT / "contracts/knee_oa_evidence_contract_v1.yaml")
T = load_yaml(PRODUCT / "contracts/knee_oa_template_contract_v1.yaml")
C = load_yaml(PRODUCT / "contracts/knee_oa_evidence_interaction_v1.yaml")
ENGINE = get_cu1_engine()
LANG = CU1GreekReferralFormatter(get_cu1_bundle()).language
SCOPE = T["product_supported_input_scope"]
HIDDEN = set(C["power_users"]["hidden_items"])
CATEGORIES = {"findings": "findings", "functional_impairments": "functional_impairments",
              "rehab_directions": "rehab_directions", "adjunct_options": "adjuncts", "goals": "goals"}
FLAGS = ["material_concern_unresolved", "infection_or_septic_joint_concern", "dvt_concern_unresolved"]
SAFETY_LABELS = {
    "material_concern_unresolved": "Ανεπίλυτη κλινική ανησυχία",
    "infection_or_septic_joint_concern": "Ανησυχία για λοίμωξη ή σηπτική άρθρωση",
    "dvt_concern_unresolved": "Ανεπίλυτη υποψία θρόμβωσης",
}

MIXED_COPY = {
    "manual_therapy": {
        "NICE_NG226_2022": "Να εξετάζεται μόνο μαζί με θεραπευτική άσκηση. Ανεπαρκή δεδομένα για μεμονωμένη χρήση.",
        "AAOS_OAK3_2021": "Πιθανή βελτίωση πόνου και λειτουργικότητας ως προσθήκη στην άσκηση. Περιορισμένη ισχύς σύστασης.",
        "ACR_AF_2019": "Υπό όρους σύσταση κατά της προσθήκης manual therapy στην άσκηση, σε σύγκριση με άσκηση μόνη της.",
    },
    "soft_tissue_techniques": {
        "NICE_NG226_2022": "Περιλαμβάνονται στη manual therapy και εξετάζονται μόνο μαζί με άσκηση.",
        "AAOS_OAK3_2021": "Η μάλαξη μπορεί να χρησιμοποιηθεί συμπληρωματικά. Περιορισμένη ισχύς σύστασης.",
        "ACR_AF_2019": "Υπό όρους σύσταση κατά της μάλαξης για μείωση συμπτωμάτων οστεοαρθρίτιδας.",
    },
    "acupuncture": {
        "NICE_NG226_2022": "Συστήνει να μην προσφέρεται βελονισμός για οστεοαρθρίτιδα.",
        "AAOS_OAK3_2021": "Πιθανή βελτίωση πόνου και λειτουργικότητας. Περιορισμένη σύσταση λόγω ασυνέπειας και ζητημάτων τυφλοποίησης.",
        "ACR_AF_2019": "Υπό όρους σύσταση υπέρ του βελονισμού.",
        "ACE_KNEE_OA_2026": "Να εξετάζεται συμπληρωματικά μετά από ανεπαρκή συμβατική ανταπόκριση ή λόγω προτίμησης ασθενούς.",
        "VA_DOD_OA_2026": "Ανεπαρκή δεδομένα για σύσταση υπέρ ή κατά.",
    },
}
MIXED_COPY.update(json.loads((PRODUCT / "prototype" / "source_copy_el.json").read_text(encoding="utf-8")))


def check(ok: bool) -> None:
    if not ok:
        raise ValueError("invalid_product_request")


def short_text(value, limit=800) -> str:
    check(isinstance(value, str) and len(value) <= limit)
    return " ".join(value.split())


def clean_request(payload: dict) -> dict:
    check(isinstance(payload, dict))
    check(set(payload) <= {"draft_id", "revision", "package_version", "synthetic_only", "state", "dismissed", "availability", "candidate"})
    check(payload.get("synthetic_only") is True and payload.get("package_version") == PACKAGE)
    check(isinstance(payload.get("draft_id"), str))
    try:
        uuid.UUID(payload["draft_id"])
    except (ValueError, TypeError, AttributeError):
        raise ValueError("invalid_product_request") from None
    revision = payload.get("revision")
    check(type(revision) is int and 0 <= revision < 1_000_000_000)
    state = payload.get("state")
    check(isinstance(state, dict))
    check(set(state) <= set(CATEGORIES) | {"laterality", "formal_assertion_state", "phenotype", "qualifiers", "explicit_restrictions", "clinician_free_text_optional", "safety_flags"})
    clean: dict = {}
    for key, category in CATEGORIES.items():
        values = state.get(key, [])
        check(isinstance(values, list) and len(values) <= 40)
        check(all(isinstance(v, str) and v in SCOPE[category] and v not in HIDDEN for v in values))
        clean[key] = list(dict.fromkeys(values))
    check(state.get("laterality") in {"not_stated", "left", "right", "bilateral"})
    check(state.get("formal_assertion_state") in {"yes", "not_stated"})
    clean.update(laterality=state["laterality"], formal_assertion_state=state["formal_assertion_state"])
    phenotype = state.get("phenotype", {})
    check(isinstance(phenotype, dict) and set(phenotype) <= set(T["product_overlay"]["allowed_fields"]))
    check(all(type(v) is bool for v in phenotype.values()))
    clean["phenotype"] = dict(phenotype)
    clean["qualifiers"] = clean_qualifiers(state.get("qualifiers"), {**clean, "findings": clean["findings"]})
    restrictions = state.get("explicit_restrictions", [])
    check(isinstance(restrictions, list) and len(restrictions) <= 6)
    clean["explicit_restrictions"] = []
    for restriction in restrictions:
        check(isinstance(restriction, dict) and set(restriction) == {"restriction_id", "state_or_value", "source"})
        check(restriction["restriction_id"] in LANG["restrictions"] and restriction["source"] == "clinician_entered")
        value = short_text(restriction["state_or_value"], 300)
        check(bool(value))
        clean["explicit_restrictions"].append({**restriction, "state_or_value": value})
    clean["clinician_free_text_optional"] = short_text(state.get("clinician_free_text_optional", ""))
    flags = state.get("safety_flags", [])
    check(isinstance(flags, list) and all(isinstance(v, str) and v in FLAGS for v in flags))
    clean["safety_flags"] = list(dict.fromkeys(flags))
    clean = state_with_mapped_findings(clean)
    dismissed = payload.get("dismissed", [])
    check(isinstance(dismissed, list) and len(dismissed) <= 100)
    check(all(isinstance(v, str) and len(v) < 2000 for v in dismissed))
    availability = payload.get("availability", {})
    check(isinstance(availability, dict) and set(availability) <= set(E["source_registry"]))
    check(all(isinstance(v, str) and v in C["availability"]["states"] for v in availability.values()))
    return {**payload, "state": clean, "dismissed": dismissed, "availability": availability}


def suggestion_draft(req: dict) -> dict:
    s = req["state"]
    facts = {key: "selected" for key in s["findings"] + s["functional_impairments"]}
    facts.update({key: "present" for key, value in s["phenotype"].items() if value})
    facts = enrich_suggestion_facts(facts, s.get("qualifiers") or empty_qualifiers())
    return {"draft_id": req["draft_id"], "draft_revision": req["revision"], "package_version": PACKAGE,
            "selected": s["rehab_directions"] + s["adjunct_options"], "facts": facts,
            "dismissed": req["dismissed"], "availability": req["availability"]}


def project(payload: dict) -> dict:
    req = clean_request(payload)
    if "candidate" in req:
        updated = apply_suggestion(C, E, T, suggestion_draft(req), req["candidate"])
        s = copy.deepcopy(req["state"])
        item = req["candidate"]["item_id"]
        target = "rehab_directions" if item in SCOPE["rehab_directions"] else "adjunct_options"
        s[target] = list(dict.fromkeys(s[target] + [item]))
        req.update(state=s, revision=updated["draft_revision"])
    s = req["state"]
    cu1 = {
        "contract_version": "cu1_referral_draft_v1", "body_region": "knee",
        "primary_problem": {"profile_id": "knee", "route_id": "knee_osteoarthritis", "wording_mode": "formal_diagnosis",
                            "formal_assertion_state_optional": s["formal_assertion_state"], "laterality": s["laterality"]},
        **{key: copy.deepcopy(s[key]) for key in CATEGORIES},
        "explicit_restrictions": s["explicit_restrictions"],
        "clinician_free_text_optional": s["clinician_free_text_optional"],
        "safety": {"input_flags": s["safety_flags"], "acknowledged_rule_ids": [], "clinician_disposition": "none_recorded"},
    }
    validation = ENGINE.validate(cu1)
    local_error = copy_readiness_error(T, s)
    blocked = any(row.formatter_blocked for row in validation.safety_results)
    allowed = not validation.formatter_blocked and local_error is None
    text = apply_referral_overlay(render(T, s, LANG), s) if allowed else None
    chosen = set(s["rehab_directions"] + s["adjunct_options"])
    views = {item: evidence_view(C, E, item, item in chosen, req["availability"])
             for item in list(SCOPE["rehab_directions"]) + list(SCOPE["adjuncts"]) if item not in HIDDEN}
    for item, view in views.items():
        for position in view["positions"]:
            position["summary_el"] = MIXED_COPY.get(item, {}).get(position["source_id"])
    candidates = suggestion_candidates(C, E, T, suggestion_draft(req))
    notes = [{"item_id": item, "label": view["evidence_label"] if view["all_sources_active"] else "Η τεκμηρίωση χρειάζεται έλεγχο"}
             for item, view in views.items() if item in chosen and
             (not view["all_sources_active"] or view["evidence_state"] in C["bubbles"]["selected_trigger_states"])]
    for item in E["suggestion_policy"]["core_omission_suggestions"]:
        if item not in chosen:
            notes.append({"item_id": item, "label": "Βασική επιλογή δεν έχει προστεθεί"})
    review_clues = clinical_review_clues(s.get("qualifiers") or empty_qualifiers())
    gate = {"allowed": allowed, "blocked": blocked, "draft_revision": req["revision"]}
    note_ids = [n["item_id"] for n in notes] + [clue["clue_id"] for clue in review_clues]
    readiness = export_readiness(C, req["revision"], req["revision"], gate, False, note_ids)
    return {"draft_id": req["draft_id"], "revision": req["revision"], "package_version": PACKAGE,
            "state": s, "text": text, "gate": gate, "readiness": readiness, "evidence": views,
            "suggestions": candidates, "notes": notes, "clinical_review_clues": review_clues,
            "safety": [{"rule_id": row.rule_id, "severity": row.severity, "blocked": row.formatter_blocked}
                       for row in validation.safety_results],
            "validation_errors": [row.error_id for row in validation.validation_errors]}


def bootstrap() -> dict:
    labels = {category: {item: LANG[category].get(item, item) for item in SCOPE[category] if item not in HIDDEN}
              for category in set(CATEGORIES.values())}
    return {"package_version": PACKAGE, "synthetic_only": True, "labels": labels,
            "states": C["states"], "defaults": E["default_plan"]["selected"],
            "restrictions": LANG["restrictions"], "safety_labels": SAFETY_LABELS,
            "reviewed_on": str(E["reviewed_on"]), "prototype_parent": "4e0e3206dd1e12a2e55a6abd18d2a6f7dbc3f7c6"}


__all__ = ["PACKAGE", "bootstrap", "project", "check"]
