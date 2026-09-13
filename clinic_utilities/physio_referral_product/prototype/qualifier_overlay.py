"""Bounded product-local Knee-OA symptom/examination qualifier overlay.

This module adds specificity to the Knee-OA product layer. It does not alter
frozen CU-1 taxonomy, international evidence states, safety rules or treatment
selection. Product-local examination detail remains ephemeral and clinician-entered.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

PAIN_LOCATIONS = {
    "medial_joint_line",
    "lateral_joint_line",
    "anterior_peripatellar",
    "pes_anserine_region",
    "posterior",
    "diffuse",
}
STIFFNESS_PATTERNS = {"morning", "after_inactivity"}
STIFFNESS_DURATIONS = {None, "le_30", "gt_30"}
# Legacy values remain accepted for transport/backward-test compatibility. The
# V5.1 UI exposes the three directional knee-strength values below.
WEAKNESS_DETAILS = {
    None,
    "objective",
    "quadriceps_exam",
    "knee_extension_exam",
    "knee_flexion_exam",
    "knee_extension_flexion_exam",
}
ATROPHY_LOCATIONS = {None, "quadriceps", "peri_knee_general"}
TENDERNESS_LOCATIONS = {
    "medial_joint_line",
    "lateral_joint_line",
    "medial_bony",
    "lateral_bony",
    "pes_anserine_region",
    "extensor_mechanism",
}
STABILITY_FINDINGS = {
    "valgus_instability",
    "varus_instability",
    "anterior_instability_acl",
    "posterior_instability_pcl",
}
CORE_DEFAULT_REHAB = {
    "therapeutic_exercise",
    "progressive_strengthening",
    "education_and_self_management",
}


def empty_qualifiers() -> dict[str, Any]:
    return {
        "pain_locations": [],
        "stiffness_patterns": [],
        "morning_stiffness_duration": None,
        "weakness_detail": None,
        "visible_atrophy": False,
        "atrophy_location": None,
        "rom_restriction_present": False,
        "active_flexion_restricted": False,
        "passive_flexion_restricted": False,
        "fixed_flexion_deformity": False,
        "fixed_flexion_deformity_deg": None,
        "crepitus": False,
        "focal_tenderness_locations": [],
        "stability_findings": [],
    }


def _check(ok: bool) -> None:
    if not ok:
        raise ValueError("invalid_prototype_request")


def clean_qualifiers(raw: Any, state: dict[str, Any]) -> dict[str, Any]:
    if raw is None:
        raw = {}
    _check(isinstance(raw, dict))
    _check(set(raw) <= set(empty_qualifiers()))
    q = empty_qualifiers()

    pain = raw.get("pain_locations", [])
    _check(isinstance(pain, list) and len(pain) <= 6 and all(v in PAIN_LOCATIONS for v in pain))
    q["pain_locations"] = list(dict.fromkeys(pain))
    _check(not ("diffuse" in q["pain_locations"] and len(q["pain_locations"]) > 1))

    stiffness = raw.get("stiffness_patterns", [])
    _check(isinstance(stiffness, list) and len(stiffness) <= 2 and all(v in STIFFNESS_PATTERNS for v in stiffness))
    q["stiffness_patterns"] = list(dict.fromkeys(stiffness))
    duration = raw.get("morning_stiffness_duration")
    _check(duration in STIFFNESS_DURATIONS)
    _check(duration is None or "morning" in q["stiffness_patterns"])
    q["morning_stiffness_duration"] = duration

    weakness = raw.get("weakness_detail")
    _check(weakness in WEAKNESS_DETAILS)
    q["weakness_detail"] = weakness
    atrophy = raw.get("visible_atrophy", False)
    _check(type(atrophy) is bool)
    q["visible_atrophy"] = atrophy
    atrophy_location = raw.get("atrophy_location")
    _check(atrophy_location in ATROPHY_LOCATIONS)
    _check(atrophy_location is None or atrophy)
    q["atrophy_location"] = atrophy_location

    rom_parent = raw.get("rom_restriction_present", False)
    active_flexion = raw.get("active_flexion_restricted", False)
    passive_flexion = raw.get("passive_flexion_restricted", False)
    _check(type(rom_parent) is bool)
    _check(type(active_flexion) is bool)
    _check(type(passive_flexion) is bool)
    q["active_flexion_restricted"] = active_flexion
    q["passive_flexion_restricted"] = passive_flexion

    ffd = raw.get("fixed_flexion_deformity", False)
    _check(type(ffd) is bool)
    q["fixed_flexion_deformity"] = ffd
    deg = raw.get("fixed_flexion_deformity_deg")
    # If a degree is supplied, it must describe a real positive deficit. An
    # unmeasured/unknown degree remains None rather than being invented as zero.
    _check(deg is None or (type(deg) is int and 1 <= deg <= 60))
    _check(deg is None or ffd)
    q["fixed_flexion_deformity_deg"] = deg

    # A specific ROM detail implies that the progressive parent is active. An
    # extension lag lives in canonical findings rather than the qualifier object.
    findings = set(state.get("findings") or [])
    q["rom_restriction_present"] = bool(
        rom_parent or active_flexion or passive_flexion or ffd or "extension_lag" in findings
    )

    crepitus = raw.get("crepitus", False)
    _check(type(crepitus) is bool)
    q["crepitus"] = crepitus

    tenderness = raw.get("focal_tenderness_locations", [])
    _check(isinstance(tenderness, list) and len(tenderness) <= 6 and all(v in TENDERNESS_LOCATIONS for v in tenderness))
    q["focal_tenderness_locations"] = list(dict.fromkeys(tenderness))

    stability = raw.get("stability_findings", [])
    _check(isinstance(stability, list) and len(stability) <= 4 and all(v in STABILITY_FINDINGS for v in stability))
    q["stability_findings"] = list(dict.fromkeys(stability))

    phenotype = state.get("phenotype") or {}
    _check(not q["pain_locations"] or "pain" in findings)
    _check(not (q["stiffness_patterns"] or duration) or phenotype.get("stiffness_symptom") is True)
    _check(not (weakness or atrophy or atrophy_location) or phenotype.get("weakness_symptom_or_context") is True)
    return q


def state_with_mapped_findings(state: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(state)
    q = result.get("qualifiers") or empty_qualifiers()
    # Preserve pre-existing canonical findings. Product-local detail only maps
    # when an existing CU-1 finding is semantically compatible.
    findings = list(result.get("findings", []))
    if q.get("pain_locations"):
        findings = [v for v in findings if v not in {"joint_line_pain", "anterior_peripatellar_pain"}]
        if "pain" not in findings:
            findings.append("pain")

    weakness_detail = q.get("weakness_detail")
    if weakness_detail is not None:
        findings = [v for v in findings if v not in {"objective_weakness", "quadriceps_weakness"}]
        if weakness_detail in {"quadriceps_exam", "knee_extension_exam"}:
            findings.append("quadriceps_weakness")
        elif weakness_detail in {"objective", "knee_flexion_exam", "knee_extension_flexion_exam"}:
            findings.append("objective_weakness")

    # Specific tenderness refines prose but retains the existing generic CU-1
    # tenderness finding. No tenderness location creates a second diagnosis.
    if q.get("focal_tenderness_locations"):
        findings.append("tenderness")

    # Flexion-specific ROM detail maps only to the matching generic active or
    # passive ROM restriction. Extension lag remains its existing canonical
    # finding; passive extension deficit remains the reviewed FFD qualifier.
    if q.get("active_flexion_restricted"):
        findings.append("active_rom_restricted")
    if q.get("passive_flexion_restricted"):
        findings.append("passive_rom_restricted")

    result["findings"] = list(dict.fromkeys(findings))
    return result


def enrich_suggestion_facts(facts: dict[str, str], qualifiers: dict[str, Any]) -> dict[str, str]:
    result = dict(facts)
    if qualifiers.get("fixed_flexion_deformity") or qualifiers.get("passive_flexion_restricted"):
        result["passive_rom_restricted"] = "abnormal"
    if qualifiers.get("active_flexion_restricted"):
        result["active_rom_restricted"] = "abnormal"
    return result


def _join_el(values: list[str]) -> str:
    values = [v for v in values if v]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} και {values[1]}"
    return ", ".join(values[:-1]) + f" και {values[-1]}"


def _pain_phrase(q: dict[str, Any]) -> str | None:
    values = q.get("pain_locations") or []
    if not values:
        return None
    if values == ["diffuse"]:
        return "διάχυτο πόνο στο γόνατο"
    phrases = {
        "medial_joint_line": "στην έσω μεσάρθρια περιοχή",
        "lateral_joint_line": "στην έξω μεσάρθρια περιοχή",
        "anterior_peripatellar": "πρόσθια ή περιεπιγονατιδικά",
        "pes_anserine_region": "στην περιοχή του χηνείου ποδός",
        "posterior": "οπίσθια στο γόνατο",
    }
    return "πόνο " + _join_el([phrases[v] for v in values if v in phrases])


def _stiffness_phrase(q: dict[str, Any]) -> str | None:
    patterns = q.get("stiffness_patterns") or []
    if not patterns:
        return None
    parts: list[str] = []
    if "morning" in patterns:
        duration = q.get("morning_stiffness_duration")
        if duration == "le_30":
            parts.append("πρωινή δυσκαμψία έως 30 λεπτά")
        elif duration == "gt_30":
            parts.append("πρωινή δυσκαμψία άνω των 30 λεπτών")
        else:
            parts.append("πρωινή δυσκαμψία")
    if "after_inactivity" in patterns:
        parts.append("δυσκαμψία μετά από ακινησία")
    return _join_el(parts)


def _weakness_phrase(q: dict[str, Any]) -> str | None:
    return {
        "knee_extension_exam": "αδυναμία έκτασης γόνατος / τετρακεφάλου κατά την εξέταση",
        "knee_flexion_exam": "αδυναμία κάμψης γόνατος / ισχιοκνημιαίων κατά την εξέταση",
        "knee_extension_flexion_exam": "αδυναμία κάμψης και έκτασης γόνατος κατά την εξέταση",
    }.get(q.get("weakness_detail"))


def _atrophy_suffix(q: dict[str, Any]) -> str:
    if not q.get("visible_atrophy"):
        return ""
    location = q.get("atrophy_location")
    if location == "quadriceps":
        return " με εμφανή ατροφία τετρακεφάλου"
    if location == "peri_knee_general":
        return " με εμφανή περιαρθρική μυϊκή ατροφία"
    return " με εμφανή μυϊκή ατροφία"


def _tenderness_phrase(q: dict[str, Any]) -> str | None:
    values = q.get("focal_tenderness_locations") or []
    if not values:
        return None
    phrases = {
        "medial_joint_line": "αρθρική ευαισθησία έσω",
        "lateral_joint_line": "αρθρική ευαισθησία έξω",
        "medial_bony": "οστική ευαισθησία έσω",
        "lateral_bony": "οστική ευαισθησία έξω",
        "pes_anserine_region": "ευαισθησία στην περιοχή του χηνείου ποδός",
        "extensor_mechanism": "ευαισθησία στον εκτατικό μηχανισμό",
    }
    return _join_el([phrases[v] for v in values if v in phrases])


def _rom_specificity(result: str, q: dict[str, Any]) -> str:
    active_flexion = q.get("active_flexion_restricted") is True
    passive_flexion = q.get("passive_flexion_restricted") is True
    if active_flexion and passive_flexion:
        result = result.replace(
            "περιορισμό ενεργητικού και παθητικού εύρους κίνησης",
            "περιορισμό ενεργητικής και παθητικής κάμψης",
            1,
        )
        result = result.replace("περιορισμό ενεργητικού εύρους κίνησης", "περιορισμό ενεργητικής κάμψης", 1)
        result = result.replace("περιορισμό παθητικού εύρους κίνησης", "περιορισμό παθητικής κάμψης", 1)
    elif active_flexion:
        result = result.replace("περιορισμό ενεργητικού εύρους κίνησης", "περιορισμό ενεργητικής κάμψης", 1)
    elif passive_flexion:
        result = result.replace("περιορισμό παθητικού εύρους κίνησης", "περιορισμό παθητικής κάμψης", 1)
    return result


def _extra_exam_phrases(q: dict[str, Any], state: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    findings = set(state.get("findings") or [])
    any_rom_detail = any((
        "extension_lag" in findings,
        q.get("fixed_flexion_deformity"),
        q.get("active_flexion_restricted"),
        q.get("passive_flexion_restricted"),
    ))
    if q.get("rom_restriction_present") and not any_rom_detail:
        phrases.append("περιορισμένο εύρος κίνησης")
    if q.get("fixed_flexion_deformity"):
        degrees = q.get("fixed_flexion_deformity_deg")
        phrase = "παθητικό έλλειμμα έκτασης"
        if degrees is not None:
            phrase += f" {degrees}°"
        phrases.append(phrase)
    if q.get("crepitus"):
        phrases.append("κριγμός κατά την κίνηση")
    stability_labels = {
        "valgus_instability": "αστάθεια σε βλαισότητα",
        "varus_instability": "αστάθεια σε ραιβότητα",
        "anterior_instability_acl": "πρόσθια αστάθεια / ΠΧΣ",
        "posterior_instability_pcl": "οπίσθια αστάθεια / ΟΧΣ",
    }
    phrases.extend(stability_labels[v] for v in q.get("stability_findings") or [] if v in stability_labels)
    return phrases


def _insert_exam_sentence(text: str, phrases: list[str]) -> str:
    if not phrases:
        return text
    verb = "καταγράφεται" if len(phrases) == 1 else "καταγράφονται"
    sentence = f"Κατά την εξέταση {verb} {_join_el(phrases)}."
    marker = " Λειτουργικά,"
    if marker in text:
        return text.replace(marker, " " + sentence + marker, 1)
    marker = " Παρακαλώ για"
    if marker in text:
        return text.replace(marker, " " + sentence + marker, 1)
    return text.rstrip() + " " + sentence


def _has_product_specific_signal(state: dict[str, Any]) -> bool:
    phenotype = state.get("phenotype") or {}
    qualifiers = state.get("qualifiers") or empty_qualifiers()
    qualifier_signal = any(
        value not in (None, False, [], {})
        for value in qualifiers.values()
    )
    return any((
        state.get("findings"),
        state.get("functional_impairments"),
        state.get("adjunct_options"),
        state.get("goals"),
        state.get("explicit_restrictions"),
        str(state.get("clinician_free_text_optional") or "").strip(),
        any(bool(v) for v in phenotype.values()),
        qualifier_signal,
        set(state.get("rehab_directions") or []) != CORE_DEFAULT_REHAB,
    ))


def _reframe_plan_as_priorities(text: str) -> str:
    prefix = "Παρακαλώ για ενεργητικό, εξατομικευμένο πρόγραμμα με "
    suffix = ", προσαρμοσμένο στην κλινική ανταπόκριση και στους λειτουργικούς στόχους."
    start = text.find(prefix)
    if start >= 0:
        end = text.find(suffix, start + len(prefix))
        if end >= 0:
            plan_list = text[start + len(prefix):end]
            replacement = (
                "Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένο ενεργητικό πρόγραμμα, "
                f"με ενδεικτικές προτεραιότητες {plan_list}, ανάλογα με τα ευρήματα της αξιολόγησης "
                "και τους λειτουργικούς στόχους."
            )
            text = text[:start] + replacement + text[end + len(suffix):]
    return text.replace(
        "Συμπληρωματικές επιλογές: ",
        "Πρόσθετες επιλογές προς φυσιοθεραπευτική αξιολόγηση: ",
    )


def _compact_low_information_output(text: str, state: dict[str, Any]) -> str:
    if _has_product_specific_signal(state):
        return text
    indication = text.split(". ", 1)[0].rstrip(".") + "."
    return (
        indication
        + " Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένο ενεργητικό πρόγραμμα, "
        + "με αρχικές προτεραιότητες θεραπευτική άσκηση, προοδευτική ενδυνάμωση και εκπαίδευση για αυτοδιαχείριση."
    )


def apply_referral_overlay(text: str, state: dict[str, Any]) -> str:
    q = state.get("qualifiers") or empty_qualifiers()
    result = text

    pain = _pain_phrase(q)
    if pain and "πόνο" in result:
        result = result.replace("πόνο", pain, 1)

    stiffness = _stiffness_phrase(q)
    if stiffness and "δυσκαμψία" in result:
        result = result.replace("δυσκαμψία", stiffness, 1)

    weakness = _weakness_phrase(q)
    if weakness:
        for phrase in (
            "αδυναμία τετρακεφάλου",
            "αντικειμενικά διαπιστωμένη μυϊκή αδυναμία",
            "μυϊκή αδυναμία",
        ):
            if phrase in result:
                result = result.replace(phrase, weakness, 1)
                break

    suffix = _atrophy_suffix(q)
    if suffix:
        weakness_phrases = (
            weakness,
            "αδυναμία τετρακεφάλου",
            "αντικειμενικά διαπιστωμένη μυϊκή αδυναμία",
            "μυϊκή αδυναμία",
        )
        for phrase in (p for p in weakness_phrases if p):
            if phrase in result:
                result = result.replace(phrase, phrase + suffix, 1)
                break

    tenderness = _tenderness_phrase(q)
    if tenderness and "ευαισθησία στην ψηλάφηση" in result:
        result = result.replace("ευαισθησία στην ψηλάφηση", tenderness, 1)

    result = _rom_specificity(result, q)
    result = _insert_exam_sentence(result, _extra_exam_phrases(q, state))
    result = _compact_low_information_output(result, state)
    return _reframe_plan_as_priorities(result)


def clinical_review_clues(qualifiers: dict[str, Any]) -> list[dict[str, str]]:
    clues: list[dict[str, str]] = []
    if qualifiers.get("morning_stiffness_duration") == "gt_30":
        clues.append({
            "clue_id": "morning_stiffness_over_30",
            "label": "Πρωινή δυσκαμψία >30′ · μη τυπικό χαρακτηριστικό",
            "detail": "Η διάρκεια αυτή βρίσκεται έξω από το τυπικό κλινικό πρότυπο OA που χρησιμοποιεί το NICE. Χρειάζεται περαιτέρω κλινική εκτίμηση για πιθανό πρόσθετο ή εναλλακτικό αίτιο, χωρίς αυτόματη αλλαγή διάγνωσης ή θεραπείας.",
            "source_label": "NICE NG226 · 2022",
            "source_url": "https://www.nice.org.uk/guidance/ng226/chapter/recommendations",
            "reviewed_on": "13/09/2026",
        })
    return clues
