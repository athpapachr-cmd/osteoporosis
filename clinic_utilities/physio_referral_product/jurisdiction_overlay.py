"""Deterministic jurisdiction overlay for the physiotherapy referral product.

The international evidence view remains authoritative.  This module can only add
reviewed local context alongside it.  It cannot change evidence state, treatment
selection, referral prose, safety, or patient data.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


PRODUCT = Path(__file__).resolve().parent
PROFILE_ENV = "PHYSIO_REFERRAL_JURISDICTION_PROFILE"
PROFILE_FILES = {
    "CY_GESY": PRODUCT / "jurisdictions" / "CY_GESY" / "knee_oa_overlay_v1.yaml",
}

POLICY_CLASSES = {
    "clinical_guidance",
    "administrative_access",
    "reimbursement",
    "resource_feasibility",
    "reimbursement_documentation",
    "system_lifecycle",
    "mixed",
    "unclear",
}
LOCAL_DIRECTIONS = {
    "for",
    "conditional_for",
    "against",
    "against_routine_use",
    "neutral_or_insufficient",
    "informational",
    "not_applicable",
}
LOCAL_STRENGTHS = {"mandatory", "strong", "weak", "not_explicit", "not_applicable"}
RELATIONSHIPS = {
    "agreement",
    "local_difference",
    "local_addition",
    "local_position_within_international_conflict",
    "administrative_only",
    "reimbursement_or_access_only",
    "status_unknown",
    "not_comparable",
}
CORE_STATES = {
    "recommended_or_supported",
    "conditional_or_context_dependent",
    "limited_or_insufficient_evidence",
    "guideline_conflict_or_mixed",
    "recommendation_against_routine_use",
    "not_yet_assessed",
    "not_in_current_contract",
}
GUIDELINE_STATUSES = {
    "draft_consultation",
    "published_active",
    "published_status_unclear",
    "superseded",
    "unknown",
}
PUBLIC_FINALITY = {"final_versioned", "draft_labelled", "metadata_conflict", "unknown"}
SYSTEM_STATUSES = {"active_verified", "planned", "not_applicable", "unknown"}
ENFORCEMENT_STATUSES = {"active_verified", "published_rule_only", "not_applicable", "unknown"}
ROUTINE_VISIBILITY = {"silent", "context_cue_only", "detail_only", "operational_detail_only", "never_routine"}
DISPLAY_SEMANTICS = {
    "local_agreement",
    "local_difference",
    "local_administrative_rule",
    "local_reimbursement_rule",
    "local_status_unknown",
}
REVIEW_STATES = {
    "audit_candidate",
    "clinically_reviewed",
    "product_reviewed",
    "approved_inactive",
    "approved_active",
    "retired",
}
SELECTION_SOURCES = {"explicit_account_configuration", "explicit_clinician_configuration"}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _mapping(value: Any, message: str) -> dict[str, Any]:
    _require(isinstance(value, dict), message)
    return value


def _validate_profile(data: Any) -> dict[str, Any]:
    root = _mapping(data, "jurisdiction_overlay_root_invalid")
    _require(root.get("schema_id") == "JurisdictionOverlayV1", "jurisdiction_overlay_schema_invalid")
    _require(root.get("schema_version") == 1, "jurisdiction_overlay_version_invalid")
    _require(root.get("diagnosis_vertical") == "knee_oa", "jurisdiction_overlay_vertical_invalid")

    profile = _mapping(root.get("profile"), "jurisdiction_profile_invalid")
    _require(profile.get("profile_id") == "CY_GESY", "jurisdiction_profile_id_invalid")
    _require(profile.get("country_code") == "CY", "jurisdiction_country_invalid")
    _require(profile.get("health_system_id") == "GESY", "jurisdiction_health_system_invalid")
    _require(isinstance(profile.get("label"), str) and profile["label"], "jurisdiction_label_invalid")
    _require(profile.get("profile_status") in {"reviewed_inactive", "active", "retired"}, "jurisdiction_profile_status_invalid")
    _require(profile.get("selection_source") in SELECTION_SOURCES, "jurisdiction_selection_source_invalid")

    positions = root.get("positions")
    _require(isinstance(positions, list) and positions, "jurisdiction_positions_invalid")
    seen: set[str] = set()
    for position in positions:
        p = _mapping(position, "jurisdiction_position_invalid")
        position_id = p.get("local_position_id")
        _require(isinstance(position_id, str) and position_id and position_id not in seen, "jurisdiction_position_id_invalid")
        seen.add(position_id)
        _require(p.get("jurisdiction_profile_id") == profile["profile_id"], "jurisdiction_position_profile_mismatch")
        _require(p.get("diagnosis_vertical") == "knee_oa", "jurisdiction_position_vertical_invalid")
        _require(p.get("policy_class") in POLICY_CLASSES, "jurisdiction_policy_class_invalid")
        _require(p.get("local_direction") in LOCAL_DIRECTIONS, "jurisdiction_direction_invalid")
        _require(p.get("local_strength") in LOCAL_STRENGTHS, "jurisdiction_strength_invalid")
        _require(p.get("relationship_to_core") in RELATIONSHIPS, "jurisdiction_relationship_invalid")
        _require(isinstance(p.get("normalized_local_position"), str) and p["normalized_local_position"], "jurisdiction_position_copy_invalid")
        _require(p.get("review_state") in REVIEW_STATES, "jurisdiction_review_state_invalid")

        core = _mapping(p.get("core_reference"), "jurisdiction_core_reference_invalid")
        _require(core.get("international_evidence_state") in CORE_STATES, "jurisdiction_core_state_invalid")
        _require(core.get("core_state_mutated_by_overlay") is False, "jurisdiction_core_mutation_forbidden")
        item_id = p.get("intervention_id")
        if item_id is not None:
            _require(isinstance(item_id, str) and item_id, "jurisdiction_intervention_id_invalid")
            _require(core.get("international_item_id") == item_id, "jurisdiction_item_reference_mismatch")

        operational = _mapping(p.get("operational_status"), "jurisdiction_operational_status_invalid")
        _require(operational.get("guideline_status") in GUIDELINE_STATUSES, "jurisdiction_guideline_status_invalid")
        _require(operational.get("public_artifact_finality") in PUBLIC_FINALITY, "jurisdiction_public_finality_invalid")
        _require(operational.get("information_system_status") in SYSTEM_STATUSES, "jurisdiction_system_status_invalid")
        _require(operational.get("reimbursement_enforcement_status") in ENFORCEMENT_STATUSES, "jurisdiction_enforcement_status_invalid")

        provenance = _mapping(p.get("source_provenance"), "jurisdiction_source_provenance_invalid")
        for key in ("source_owner", "source_title", "source_url", "source_type", "reviewed_on"):
            _require(isinstance(provenance.get(key), str) and provenance[key], "jurisdiction_source_field_invalid")
        _require(provenance["source_url"].startswith("https://"), "jurisdiction_source_url_invalid")

        display = _mapping(p.get("display_policy"), "jurisdiction_display_policy_invalid")
        _require(display.get("routine_visibility") in ROUTINE_VISIBILITY, "jurisdiction_visibility_invalid")
        _require(display.get("display_semantic") in DISPLAY_SEMANTICS, "jurisdiction_display_semantic_invalid")
        _require(display.get("may_change_selection_automatically") is False, "jurisdiction_auto_selection_forbidden")
        _require(display.get("may_change_referral_text_automatically") is False, "jurisdiction_auto_text_forbidden")
        _require(display.get("may_change_international_evidence_badge") is False, "jurisdiction_core_badge_mutation_forbidden")

    return root


def load_profile(profile_id: str | None) -> dict[str, Any] | None:
    """Load a reviewed active jurisdiction profile, otherwise fail closed to none."""
    if not profile_id:
        return None
    path = PROFILE_FILES.get(profile_id.strip())
    if path is None or not path.is_file():
        return None
    try:
        profile = _validate_profile(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, yaml.YAMLError, ValueError, TypeError):
        return None
    if profile["profile"].get("profile_status") != "active":
        return None
    return profile


def configured_profile() -> dict[str, Any] | None:
    return load_profile(os.environ.get(PROFILE_ENV, "").strip() or None)


def profile_public_view(profile: dict[str, Any] | None) -> dict[str, str] | None:
    if not profile:
        return None
    p = profile["profile"]
    return {
        "profile_id": p["profile_id"],
        "label": p["label"],
        "selection_source": p["selection_source"],
    }


def operational_positions(profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return non-clinical local policy separately from evidence resolution."""
    if not profile:
        return []
    return [copy.deepcopy(p) for p in profile["positions"] if p["policy_class"] != "clinical_guidance"]


def apply_evidence_overlay(projected: dict[str, Any], profile: dict[str, Any] | None) -> dict[str, Any]:
    """Attach reviewed local context while preserving international evidence semantics.

    Any profile/core mismatch fails closed by returning the original projected view
    (deep-copied) with no jurisdiction overlay.
    """
    result = copy.deepcopy(projected)
    if not profile:
        result["jurisdiction_profile"] = None
        return result

    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        result["jurisdiction_profile"] = None
        return result

    mapped: list[tuple[str, dict[str, Any]]] = []
    for position in profile["positions"]:
        if position["policy_class"] != "clinical_guidance":
            continue
        item_id = position.get("intervention_id")
        if not item_id or item_id not in evidence:
            continue
        view = evidence[item_id]
        if not isinstance(view, dict):
            return copy.deepcopy(projected)
        expected = position["core_reference"]["international_evidence_state"]
        if expected != view.get("evidence_state"):
            return copy.deepcopy(projected)
        mapped.append((item_id, position))

    for item_id, position in mapped:
        local = {
            key: copy.deepcopy(position[key])
            for key in (
                "local_position_id",
                "jurisdiction_profile_id",
                "policy_class",
                "local_direction",
                "local_strength",
                "certainty_or_evidence_quality",
                "normalized_local_position",
                "relationship_to_core",
                "operational_status",
                "source_provenance",
                "display_policy",
                "review_state",
            )
            if key in position
        }
        evidence[item_id]["jurisdiction"] = local

    result["jurisdiction_profile"] = profile_public_view(profile)
    return result


__all__ = [
    "PROFILE_ENV",
    "apply_evidence_overlay",
    "configured_profile",
    "load_profile",
    "operational_positions",
    "profile_public_view",
]
