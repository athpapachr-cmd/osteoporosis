"""Synthetic Step-4 design model. Not imported by the production application.

Checks referential integrity and specified interaction semantics, not rendered
accessibility, clinical efficacy, independent review or production readiness.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

PRODUCT = Path(__file__).resolve().parent
CONTRACTS = PRODUCT / "contracts"


def require(condition: bool, code: str) -> None:
    if not condition:
        raise AssertionError(code)


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "mapping_required")
    return data


def git_blob_id(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def validate_contract(c: dict, e: dict, t: dict) -> None:
    require(c["version"] == "knee_oa_evidence_interaction_v1", "wrong_version")
    require(c["scope"] == {"profile_id": "knee", "route_id": "knee_osteoarthritis"}, "wrong_scope")
    require(c["language"] == "el", "wrong_language")
    for key in ("runtime_authorized", "patient_data_persistence", "clinical_re_review_performed"):
        require(c[key] is False, "authority_expansion")
    require(set(c["states"]) == set(e["evidence_state_enum"]), "evidence_enum_drift")
    cues = [value["cue"] for value in c["states"].values()]
    require(len(set(cues)) == 6, "noncolour_cues_not_distinct")
    require(c["selection"]["cue"] not in cues, "selection_evidence_collision")
    require(all(value["label"] for value in c["states"].values()), "missing_evidence_label")
    require(c["selection"]["evidence_open_mutates_selection"] is False, "disclosure_selects_treatment")
    require(c["disclosure"]["per_item_control_count"] == 1, "duplicate_info_controls")
    require(c["disclosure"]["target_css_px"] >= 44, "product_target_too_small")
    require(c["disclosure"]["sheet_host_count"] == 1, "sheet_stack")
    require(c["disclosure"]["sheet_modal"] and c["disclosure"]["background_inert"], "modal_semantics")
    require(c["bubbles"]["max_expanded"] == 1, "bubble_stack")
    require(c["bubbles"]["auto_dismiss_ms"] is None, "timed_evidence")
    require(c["bubbles"]["focus_steal"] is False, "bubble_focus_theft")
    require(c["bubbles"]["dismiss_is_safety_ack"] is False, "dismissal_clears_safety")
    require(c["sheet"]["mixed_first_disclosure"] == "all_source_positions", "hidden_conflict")
    require(c["sheet"]["locator_precision"] == "source_level", "invented_locator_precision")
    require(c["sheet"]["recommendation_locator_invention"] is False, "invented_locator")
    require(set(c["scope_labels"]) == set(e["support_scope_enum"]), "claim_scope_drift")
    require(set(c["source_labels"]) == set(e["source_registry"]), "source_registry_drift")
    require(c["dates"]["ui_review_or_link_check_refreshes_clinical_date"] is False, "false_freshness")
    require(c["dates"]["implicit_review_interval"] is None, "invented_review_interval")
    require(c["availability"]["dropping_opposing_source_to_recompute_consensus"] is False, "silent_consensus")
    require(c["suggestions"]["auto_select"] is False, "automatic_treatment")
    require(c["readiness"]["evidence_interaction_can_clear_safety"] is False, "evidence_safety_bypass")
    require(set(c["readiness"]["applies_to"]) == {"copy", "print", "pdf", "share"}, "export_bypass")
    require(c["traceability"]["persistence"] == "none", "trace_persistence")
    require(c["privacy"]["patient_draft_storage"] == "none", "patient_persistence")
    require(c["accessibility_acceptance"]["actual_browser_checks_performed"] is False, "false_browser_claim")
    scope = t["product_supported_input_scope"]
    supported = set(scope["rehab_directions"]) | set(scope["adjuncts"])
    candidates = set(e["suggestion_policy"]["core_omission_suggestions"])
    for values in e["suggestion_policy"]["phenotype_suggestions"].values():
        candidates.update(values)
    require(set(c["suggestions"]["primary_source"]) == candidates, "suggestion_source_coverage")
    for item_id, source_id in c["suggestions"]["primary_source"].items():
        require(item_id in supported, "unsupported_suggestion")
        positions = e["items"][item_id]["positions"]
        anchors = [p for p in positions if p["source_id"] == source_id]
        require(len(anchors) == 1, "missing_or_duplicate_source_anchor")
        require(anchors[0]["direction"] in {"for", "strong_for", "weak_for", "conditional_for"}, "opposing_promotion_anchor")
    for item_id, item in e["items"].items():
        source_ids = [p["source_id"] for p in item["positions"]]
        require(len(source_ids) == len(set(source_ids)), "duplicate_position_identity")
        for position in item["positions"]:
            require(position["source_id"] in e["source_registry"], "unknown_source")
            require(position["support_scope"] in c["scope_labels"], "unknown_scope")
    for source_id, source in e["source_registry"].items():
        url = urlsplit(source["locator"])
        require(url.scheme == "https" and bool(url.hostname) and not url.username, "unsafe_source_url")
        require(not url.query, "unreviewed_query_parameters")
        require(str(source["publication_year"]) in c["source_labels"][source_id], "publication_year_missing")


def evidence_view(c: dict, e: dict, item_id: str, selected: bool, overrides: dict | None = None) -> dict:
    item = e["items"][item_id]
    overrides = overrides or {}
    require(set(overrides) <= set(e["source_registry"]), "unknown_availability_source")
    require(all(value in c["availability"]["states"] for value in overrides.values()), "invalid_availability")
    rows = []
    for p in item["positions"]:
        source = e["source_registry"][p["source_id"]]
        state = overrides.get(p["source_id"], source["status"])
        rows.append({
            "position_ref": f"{item_id}/{p['source_id']}",
            "source_id": p["source_id"],
            "source_label": c["source_labels"][p["source_id"]],
            "direction": p["direction"],
            "native_strength": p.get("native_strength"),
            "support_scope": p["support_scope"],
            "scope_label": c["scope_labels"][p["support_scope"]],
            "summary": p["summary"],
            "clinical_reviewed_on": str(source["reviewed_on"]),
            "availability": state,
            "locator": source["locator"],
            "locator_precision": "source_level",
            "standalone_strength_allowed": p["support_scope"] == "direct_item_recommendation",
        })
    active = all(p["availability"] == "active_reviewed" for p in rows)
    return {
        "item_id": item_id,
        "selected": selected,
        "evidence_state": item["app_state"],
        "evidence_label": c["states"][item["app_state"]]["label"],
        "endorsement": active and item["app_state"] == "recommended_or_supported",
        "all_sources_active": active,
        "purpose": item["purpose_summary_el"],
        "clinical_reviewed_on": str(e["reviewed_on"]),
        "positions": rows,
    }


def positive_facts(c: dict, t: dict, facts: dict) -> set[str]:
    scope = t["product_supported_input_scope"]
    allowed = set(scope["findings"]) | set(scope["functional_impairments"]) | set(t["product_overlay"]["allowed_fields"])
    require(set(facts) <= allowed, "unsupported_trigger_input")
    require(all(v in {"present", "abnormal", "selected", "absent", "normal", "not_assessed"} for v in facts.values()), "invalid_fact_state")
    result = {key for key, value in facts.items() if value in {"present", "abnormal", "selected"}}
    for original, canonical in c["suggestions"]["trigger_aliases"].items():
        if original in result:
            result.add(canonical)
    return result


def suggestion_candidates(c: dict, e: dict, t: dict, draft: dict) -> list[dict]:
    require(all(key in draft for key in ("draft_id", "draft_revision", "package_version", "selected")), "missing_draft_identity")
    selected = set(draft["selected"])
    scope = t["product_supported_input_scope"]
    supported = set(scope["rehab_directions"]) | set(scope["adjuncts"])
    require(selected <= supported, "unsupported_selected_item")
    facts = positive_facts(c, t, draft.get("facts", {}))
    policy = e["suggestion_policy"]
    reasons: dict[str, set[str]] = {}
    for item_id in policy["core_omission_suggestions"]:
        if item_id not in selected:
            reasons.setdefault(item_id, set()).add("core_omitted")
    for fact in sorted(facts):
        for item_id in policy["phenotype_suggestions"].get(fact, []):
            reasons.setdefault(item_id, set()).add("context:" + fact)
    excluded = set(policy["no_auto_promotion"]) | set(c["power_users"]["hidden_items"])
    result = []
    for item_id in e["items"]:
        if item_id not in reasons or item_id in selected or item_id in excluded or item_id not in supported:
            continue
        view = evidence_view(c, e, item_id, False, draft.get("availability"))
        if not view["all_sources_active"] or view["evidence_state"] not in c["suggestions"]["allowed_states"]:
            continue
        source_id = c["suggestions"]["primary_source"][item_id]
        anchor = next(row for row in view["positions"] if row["source_id"] == source_id)
        reason_signature = "|".join(sorted(reasons[item_id]))
        key = "::".join((draft["draft_id"], draft["package_version"], item_id, reason_signature))
        if key in draft.get("dismissed", []):
            continue
        result.append({
            "item_id": item_id,
            "reason_codes": sorted(reasons[item_id]),
            "reason_signature": reason_signature,
            "dismiss_key": key,
            "draft_id": draft["draft_id"],
            "draft_revision": draft["draft_revision"],
            "package_version": draft["package_version"],
            "source_position_refs": [row["position_ref"] for row in view["positions"]],
            "source_caption": anchor["scope_label"] + " · " + anchor["source_label"],
            "support_scope": anchor["support_scope"],
        })
    result.sort(key=lambda row: "core_omitted" not in row["reason_codes"])
    return result


def apply_suggestion(c: dict, e: dict, t: dict, draft: dict, candidate: dict) -> dict:
    current = suggestion_candidates(c, e, t, draft)
    matching = next((row for row in current if row == candidate), None)
    if matching is None:
        raise ValueError("stale_or_ineligible_suggestion")
    result = copy.deepcopy(draft)
    result["selected"] = list(dict.fromkeys(result["selected"] + [candidate["item_id"]]))
    result["draft_revision"] += 1
    return result


def export_readiness(c: dict, draft_revision: int, projection_revision: int | None, gate: dict | None, manual_pending: bool, note_items: list[str]) -> dict:
    rules = c["readiness"]
    if gate is not None and gate.get("blocked") is True:
        return {"allowed": False, "label": rules["safety_block_label"]}
    if gate is None or type(gate.get("allowed")) is not bool:
        return {"allowed": False, "label": rules["incomplete_label"]}
    if gate.get("draft_revision") != draft_revision or projection_revision != draft_revision:
        return {"allowed": False, "label": rules["stale_label"]}
    if gate["allowed"] is not True:
        return {"allowed": False, "label": rules["incomplete_label"]}
    if manual_pending:
        return {"allowed": False, "label": rules["manual_reconciliation_label"]}
    n = len(set(note_items))
    label = rules["ready_label"] if n == 0 else ("Έτοιμη · 1 σημείο για έλεγχο" if n == 1 else rules["ready_notes_label"].format(n=n))
    return {"allowed": True, "label": label}


def ui_transition(ui: dict, event: str, item_id: str | None = None) -> dict:
    result = copy.deepcopy(ui)
    if event == "show_note":
        require(bool(item_id), "missing_note_item")
        result["notes"] = sorted(set(result.get("notes", [])) | {item_id})
        result["bubble"] = item_id
    elif event == "dismiss_bubble":
        result["bubble"] = None
    elif event == "open_evidence":
        result["sheet"] = {"kind": "evidence", "item_id": item_id}
    elif event == "close_evidence":
        result["sheet"] = None
    elif event == "collapse_advanced":
        result["advanced_open"] = False
    else:
        raise ValueError("unsupported_ui_event")
    return result


def main() -> None:
    c = read_yaml(CONTRACTS / "knee_oa_evidence_interaction_v1.yaml")
    for relative, expected in c["parents"].items():
        require(git_blob_id(CONTRACTS / relative) == expected, "pinned_parent_changed")
    e = read_yaml(CONTRACTS / "knee_oa_evidence_contract_v1.yaml")
    t = read_yaml(CONTRACTS / "knee_oa_template_contract_v1.yaml")
    f = read_yaml(CONTRACTS / "knee_oa_evidence_interaction_fixtures_v1.yaml")
    require(f["synthetic_only"] is True and f["contract"] == c["version"], "invalid_fixture_authority")
    validate_contract(c, e, t)
    checked = 0
    core = e["default_plan"]["selected"]
    base = {"draft_id": "synthetic-1", "draft_revision": 1, "package_version": "step4-synthetic-v1", "selected": list(core), "facts": {}}
    for case in f["suggestion_cases"]:
        draft = copy.deepcopy(base)
        draft.update(copy.deepcopy(case["input"]))
        before = copy.deepcopy(draft)
        result = suggestion_candidates(c, e, t, draft)
        require([row["item_id"] for row in result] == case["expected_items"], case["id"])
        require(draft == before, "suggestion_mutated_input")
        if case.get("expected_scope"):
            require(result[0]["support_scope"] == case["expected_scope"], "source_scope_laundering")
        checked += 1
    for case in f["evidence_cases"]:
        view = evidence_view(c, e, case["item_id"], case["selected"], case.get("availability"))
        require(view["evidence_state"] == case["expected_state"], case["id"])
        require(view["endorsement"] == case["expected_endorsement"], case["id"])
        require(len(view["positions"]) == case["expected_source_count"], "source_position_loss")
        require(view["selected"] == case["selected"], "evidence_mutated_selection")
        if case.get("expected_directions"):
            require([p["direction"] for p in view["positions"]] == case["expected_directions"], "position_direction_loss")
        checked += 1
    for case in f["readiness_cases"]:
        result = export_readiness(c, **case["input"])
        require(result == case["expected"], case["id"])
        checked += 1
    # New or less common evidence states are exercised with synthetic copies only.
    for state in c["states"]:
        synthetic = copy.deepcopy(e)
        synthetic["items"]["therapeutic_exercise"]["app_state"] = state
        view = evidence_view(c, synthetic, "therapeutic_exercise", False)
        require(view["evidence_state"] == state and view["selected"] is False, "six_state_projection")
        checked += 1
    view = evidence_view(c, e, "progressive_strengthening", True)
    narrow = next(p for p in view["positions"] if p["source_id"] == "EULAR_CORE_2023_UPDATE")
    require(narrow["standalone_strength_allowed"] is False and narrow["native_strength"] is not None, "broad_strength_transfer")
    require("2023" in narrow["source_label"] and "2024" in narrow["source_label"], "eular_year_conflation")
    require(narrow["clinical_reviewed_on"] == "2026-09-11", "clinical_review_date_changed")
    checked += 1
    # Explicit add and stale-candidate rejection, including a new draft identity.
    draft = copy.deepcopy(base)
    draft["facts"] = {"stairs": "selected"}
    candidate = suggestion_candidates(c, e, t, draft)[0]
    updated = apply_suggestion(c, e, t, draft, candidate)
    require(candidate["item_id"] in updated["selected"] and candidate["item_id"] not in draft["selected"], "explicit_add_ownership")
    checked += 1
    for key, value in [("draft_revision", 2), ("draft_id", "synthetic-2"), ("package_version", "synthetic-v2"), ("facts", {})]:
        stale = copy.deepcopy(draft)
        stale[key] = value
        try:
            apply_suggestion(c, e, t, stale, candidate)
        except ValueError as error:
            require(str(error) == "stale_or_ineligible_suggestion", "unexpected_rejection")
        else:
            raise AssertionError("stale_candidate_accepted")
        checked += 1
    dismissed = copy.deepcopy(draft)
    dismissed["dismissed"] = [candidate["dismiss_key"]]
    require(not suggestion_candidates(c, e, t, dismissed), "dismissed_candidate_repeats")
    dismissed["facts"]["sit_to_stand"] = "selected"
    require(len(suggestion_candidates(c, e, t, dismissed)) == 1, "new_reason_not_available")
    checked += 1
    ui = {"bubble": None, "notes": [], "sheet": None, "advanced_open": True}
    clinical = copy.deepcopy(draft)
    clinical["manual_buffer"] = "Συνθετικό κείμενο, όχι κλινικό δεδομένο."
    before = copy.deepcopy(clinical)
    ui = ui_transition(ui, "show_note", "acupuncture")
    ui = ui_transition(ui, "show_note", "manual_therapy")
    require(ui["bubble"] == "manual_therapy" and len(ui["notes"]) == 2, "bubble_replacement_loses_notes")
    ui = ui_transition(ui, "dismiss_bubble")
    ui = ui_transition(ui, "open_evidence", "acupuncture")
    ui = ui_transition(ui, "collapse_advanced")
    require(ui["bubble"] is None and len(ui["notes"]) == 2 and clinical == before, "disclosure_mutates_clinical_state")
    selected_advanced = ["acupuncture", "acupuncture", "manual_therapy"]
    require(len(set(selected_advanced)) == 2, "advanced_count_is_not_unique")
    checked += 1
    # Fail-closed mutation oracles: these test the checker, not clinical recommendations.
    for mutation in ("cue", "scope", "auto_select", "timed_bubble"):
        bad_c, bad_e = copy.deepcopy(c), copy.deepcopy(e)
        if mutation == "cue":
            bad_c["states"]["not_yet_assessed"]["cue"] = bad_c["states"]["recommended_or_supported"]["cue"]
        elif mutation == "scope":
            bad_e["items"]["therapeutic_exercise"]["positions"][0]["support_scope"] = "invented_scope"
        elif mutation == "auto_select":
            bad_c["suggestions"]["auto_select"] = True
        else:
            bad_c["bubbles"]["auto_dismiss_ms"] = 2000
        try:
            validate_contract(bad_c, bad_e, t)
        except AssertionError:
            pass
        else:
            raise AssertionError("mutation_not_detected")
        checked += 1
    print(f"Step-4 design PASS: {checked} synthetic scenario/mutation checks; 3 pinned parent blobs; 6 evidence states.")
    print("NOT TESTED: browser rendering, pixel contrast, VoiceOver/Safari, usability, clinical source re-review, independent review.")


if __name__ == "__main__":
    main()
