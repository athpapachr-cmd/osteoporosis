from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
PRODUCT = ROOT / "clinic_utilities" / "physio_referral_product"
CONTRACT_PATH = PRODUCT / "contracts" / "knee_oa_evidence_contract_v1.yaml"
OPTION_CATALOG_PATH = ROOT / "clinic_utilities" / "contracts" / "cu1_option_catalog_v1.yaml"
REGISTRY_PATH = ROOT / "clinic_utilities" / "contracts" / "cu1_registry_v1.yaml"
UI_SCOPE_PATH = ROOT / "clinic_utilities" / "contracts" / "cu1_ui_relevance_scope_v1.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError(f"{path} must contain a mapping")
    return data


def assert_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise AssertionError(f"duplicate values in {label}")


def main() -> None:
    contract = load_yaml(CONTRACT_PATH)
    options = load_yaml(OPTION_CATALOG_PATH)
    registry = load_yaml(REGISTRY_PATH)
    ui_scope = load_yaml(UI_SCOPE_PATH)

    assert contract["version"] == "knee_oa_evidence_contract_v1"
    assert contract["domain"] == "physio_referral_product"
    assert contract["profile_id"] == "knee"
    assert contract["route_id"] == "knee_osteoarthritis"
    assert contract["runtime_authorized"] is False
    assert contract["patient_data_persistence"] is False

    knee_routes = registry["profiles"]["knee"]["routes"]
    assert "knee_osteoarthritis" in knee_routes

    states = list(contract["evidence_state_enum"])
    directions = list(contract["source_direction_enum"])
    scopes = list(contract["support_scope_enum"])
    assert_unique(states, "evidence_state_enum")
    assert_unique(directions, "source_direction_enum")
    assert_unique(scopes, "support_scope_enum")

    expected_states = {
        "recommended_or_supported",
        "conditional_or_context_dependent",
        "limited_or_insufficient_evidence",
        "guideline_conflict_or_mixed",
        "recommendation_against_routine_use",
        "not_yet_assessed",
    }
    assert set(states) == expected_states
    assert contract["resolution_policy"]["arithmetic_scoring_forbidden"] is True
    assert contract["resolution_policy"]["framework_conflict_must_surface"] is True
    assert contract["resolution_policy"]["insufficient_is_not_against"] is True
    assert contract["resolution_policy"]["source_claim_scope_retained"] is True

    source_registry = contract["source_registry"]
    assert isinstance(source_registry, dict) and len(source_registry) >= 6
    for source_id, source in source_registry.items():
        assert isinstance(source_id, str) and source_id
        assert source["publication_year"] >= 2019
        assert source["reviewed_on"]
        assert source["locator"].startswith("https://")
        assert source["status"] == "active_reviewed"

    valid_targets = set(options["common_rehab_direction_ids"]) | set(options["adjunct_ids"])
    items = contract["items"]
    assert isinstance(items, dict) and items

    for item_id, item in items.items():
        assert item["app_state"] in expected_states, item_id
        target_ids = item.get("cu1_target_ids", [])
        assert isinstance(target_ids, list), item_id
        for target_id in target_ids:
            assert target_id in valid_targets, f"unknown CU-1 target {target_id} in {item_id}"
        positions = item.get("positions", [])
        assert positions, f"{item_id} has no source positions"
        for position in positions:
            assert position["source_id"] in source_registry, (item_id, position)
            assert position["direction"] in directions, (item_id, position)
            assert position["support_scope"] in scopes, (item_id, position)
            # Prevent laundering a broad recommendation's native strength into a narrow item.
            if position["support_scope"] != "direct_item_recommendation":
                assert position["direction"] != "strong_for", (item_id, position)

    implicit = contract["default_plan"]["implicit"]
    selected = contract["default_plan"]["selected"]
    for item_id in implicit + selected:
        assert item_id in items
        assert items[item_id]["default_selected"] is True
    assert selected == [
        "therapeutic_exercise",
        "progressive_strengthening",
        "education_and_self_management",
    ]

    core_omissions = contract["suggestion_policy"]["core_omission_suggestions"]
    assert core_omissions == selected

    for trigger, suggestions in contract["suggestion_policy"]["phenotype_suggestions"].items():
        assert isinstance(trigger, str) and trigger
        for item_id in suggestions:
            assert item_id in items, (trigger, item_id)

    for item_id in contract["suggestion_policy"]["no_auto_promotion"]:
        assert item_id in items
        assert items[item_id].get("auto_suggestion", False) is False

    assert items["acupuncture"]["app_state"] == "guideline_conflict_or_mixed"
    assert items["manual_therapy"]["app_state"] == "guideline_conflict_or_mixed"
    assert items["soft_tissue_techniques"]["app_state"] == "guideline_conflict_or_mixed"
    assert items["dry_needling"]["app_state"] == "recommendation_against_routine_use"
    assert items["dry_needling"]["product_selectable"] is False
    assert items["weight_management"]["cu1_target_ids"] == []
    assert items["weight_management"]["product_selectable"] is False
    assert items["weight_management"]["auto_trigger"] is False

    knee_ui = ui_scope["profiles"]["knee"]
    assert "walking_aid_assessment_and_training" not in knee_ui["rehab_directions"]
    assert contract["integration_findings"]["walking_aid"]["current_knee_ui_scope_exposes"] is False
    assert contract["integration_findings"]["weight_management"]["dedicated_cu1_machine_id_exists"] is False
    assert contract["integration_findings"]["dry_needling"]["knee_v1_1_excluded"] is True

    bubbles = contract["bubble_semantics"]
    assert bubbles["limited_or_insufficient_evidence"] == "Περιορισμένη τεκμηρίωση"
    assert bubbles["guideline_conflict_or_mixed"] == "Οι οδηγίες διαφέρουν"
    assert bubbles["recommendation_against_routine_use"] == "Δεν συνιστάται για συνήθη χρήση"

    invariants = set(contract["hard_invariants"])
    required_invariants = {
        "insufficient_evidence_is_not_evidence_against",
        "guideline_conflict_is_not_consensus",
        "source_year_is_not_reviewed_on",
        "source_native_strength_must_not_be_transferred_to_narrower_item_without_scope",
        "missing_context_is_not_negative_context",
        "no_exact_exercise_dose_is_invented",
        "no_adjunct_replaces_active_rehabilitation",
        "no_patient_identifier_persistence_is_introduced",
    }
    assert required_invariants <= invariants

    print(
        "Knee-OA evidence contract PASS: "
        f"{len(source_registry)} sources, {len(items)} items, {len(states)} evidence states"
    )


if __name__ == "__main__":
    main()
