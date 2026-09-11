from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
PRODUCT = ROOT / "clinic_utilities" / "physio_referral_product"
TEMPLATE_PATH = PRODUCT / "contracts" / "knee_oa_template_contract_v1.yaml"
FIXTURES_PATH = PRODUCT / "contracts" / "knee_oa_template_fixtures_v1.yaml"
EVIDENCE_PATH = PRODUCT / "contracts" / "knee_oa_evidence_contract_v1.yaml"
OPTION_CATALOG_PATH = ROOT / "clinic_utilities" / "contracts" / "cu1_option_catalog_v1.yaml"
REGISTRY_PATH = ROOT / "clinic_utilities" / "contracts" / "cu1_registry_v1.yaml"
ROUTE_REQUIREMENTS_PATH = ROOT / "clinic_utilities" / "contracts" / "cu1_route_requirements_v1.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError(f"{path} must contain a mapping")
    return data


def join_el(items: Iterable[str]) -> str:
    values = [str(v).strip() for v in items if isinstance(v, str) and str(v).strip()]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} και {values[1]}"
    return ", ".join(values[:-1]) + f" και {values[-1]}"


def render(template: dict[str, Any], state: dict[str, Any]) -> str:
    laterality = state["laterality"]
    laterality_phrase = template["laterality_phrases"][laterality]
    blocks: list[str] = [
        template["block_templates"]["indication"].format(laterality_phrase=laterality_phrase)
    ]

    findings = set(state.get("findings") or [])
    phenotype = state.get("phenotype") or {}

    # Clinical picture is a pure output projection. Functional-like findings route to function.
    clinical_keys: set[str] = {
        key
        for key in findings
        if key not in template["clinical_picture"]["routed_to_function_instead_of_clinical_picture"]
    }
    if phenotype.get("stiffness_symptom"):
        clinical_keys.add("stiffness_symptom")
    if phenotype.get("weakness_symptom_or_context"):
        clinical_keys.add("weakness_symptom_or_context")

    weakness_priority = template["clinical_picture"]["precedence"]["weakness"]["priority"]
    selected_weakness = next((key for key in weakness_priority if key in clinical_keys), None)
    for key in weakness_priority:
        clinical_keys.discard(key)
    if selected_weakness:
        clinical_keys.add(selected_weakness)

    rom = template["clinical_picture"]["precedence"]["rom"]
    if all(key in clinical_keys for key in rom["when_all"]):
        for key in rom["suppress"]:
            clinical_keys.discard(key)
        clinical_keys.add(rom["emit"])

    clinical_order = template["clinical_picture"]["order"]
    clinical_phrases = template["clinical_picture"]["phrases"]
    rendered_clinical = [clinical_phrases[key] for key in clinical_order if key in clinical_keys]
    if rendered_clinical:
        blocks.append(
            template["block_templates"]["clinical_picture"].format(
                clinical_list=join_el(rendered_clinical)
            )
        )

    function_keys = set(state.get("functional_impairments") or [])
    routed = template["clinical_picture"]["routed_to_function_instead_of_clinical_picture"]
    for finding_id, function_id in routed.items():
        if finding_id in findings:
            function_keys.add(function_id)
    function_order = template["functional_impact"]["order"]
    function_phrases = template["functional_impact"]["phrases"]
    rendered_function = [function_phrases[key] for key in function_order if key in function_keys]
    if rendered_function:
        blocks.append(
            template["block_templates"]["functional_impact"].format(
                functional_list=join_el(rendered_function)
            )
        )

    selected_rehab = set(state.get("rehab_directions") or [])
    plan_order = template["active_plan"]["order"]
    plan_specs = template["active_plan"]["phrases"]
    plan_phrases: list[str] = []
    for item_id in plan_order:
        if item_id not in selected_rehab:
            continue
        spec = plan_specs[item_id]
        phrase = spec["default"]
        for refinement in spec.get("refinements", []):
            if refinement.get("when_finding") in findings:
                phrase = refinement["phrase"]
                break
            any_findings = refinement.get("when_any_finding")
            if isinstance(any_findings, list) and any(key in findings for key in any_findings):
                phrase = refinement["phrase"]
                break
        if spec.get("contextual_builder") == "functional_task_retraining":
            task_spec = template["functional_task_retraining"]
            task_order = task_spec["task_order"]
            task_phrases = task_spec["task_phrases"]
            selected_tasks = [key for key in task_order if key in function_keys]
            if selected_tasks:
                phrase = task_spec["phrase_with_tasks"].format(
                    task_list=join_el([task_phrases[key] for key in selected_tasks])
                )
            else:
                phrase = task_spec["phrase_without_tasks"]
        plan_phrases.append(phrase)

    if plan_phrases:
        blocks.append(
            template["block_templates"]["active_plan_with_items"].format(
                plan_list=join_el(plan_phrases)
            )
        )
    else:
        blocks.append(template["block_templates"]["active_plan_fallback"])

    selected_goals = set(state.get("goals") or [])
    goal_map = template["goals"]["nonredundant_renderable"]
    rendered_goals = [goal_map[key] for key in goal_map if key in selected_goals]
    if rendered_goals:
        blocks.append(
            template["block_templates"]["nonredundant_goal_emphasis"].format(
                goal_list=join_el(rendered_goals)
            )
        )

    selected_adjuncts = set(state.get("adjunct_options") or [])
    adjunct_order = template["adjuncts"]["order"]
    adjunct_phrases = template["adjuncts"]["phrases"]
    rendered_adjuncts = [adjunct_phrases[key] for key in adjunct_order if key in selected_adjuncts]
    if rendered_adjuncts:
        blocks.append(
            template["block_templates"]["adjuncts"].format(
                adjunct_list=join_el(rendered_adjuncts)
            )
        )

    return template["join_policy"]["block_separator"].join(blocks)


def main() -> None:
    template = load_yaml(TEMPLATE_PATH)
    fixtures = load_yaml(FIXTURES_PATH)
    evidence = load_yaml(EVIDENCE_PATH)
    options = load_yaml(OPTION_CATALOG_PATH)
    registry = load_yaml(REGISTRY_PATH)
    route_requirements = load_yaml(ROUTE_REQUIREMENTS_PATH)

    assert template["version"] == "knee_oa_template_contract_v1"
    assert template["language"] == "el"
    assert template["scope"] == {
        "profile_id": "knee",
        "route_id": "knee_osteoarthritis",
        "product_slice": "knee_oa_only",
    }
    assert template["runtime_authorized"] is False
    assert template["patient_data_persistence"] is False
    assert template["llm_required"] is False

    assert template["product_overlay"]["allowed_fields"] == {
        "stiffness_symptom": "boolean",
        "weakness_symptom_or_context": "boolean",
    }
    assert template["product_overlay"]["persistence"] == "ephemeral_only"
    assert template["copy_readiness"]["laterality_allowed"] == ["right", "left", "bilateral"]
    assert template["copy_readiness"]["require_formal_diagnosis_assertion_yes"] is True
    assert set(template["laterality_phrases"]) == {"right", "left", "bilateral"}
    assert template["block_templates"]["indication"].startswith(
        "Παραπομπή για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση λόγω οστεοαρθρίτιδας"
    )

    knee_route = registry["profiles"]["knee"]["routes"]["knee_osteoarthritis"]
    assert "formal_diagnosis" in knee_route["wording_modes"]
    formal_policy = route_requirements["wording_mode_requirements"]["formal_diagnosis"]
    assert formal_policy["formal_assertion_policy"] == "required_yes_unless_route_override_defines_context_based_assertion"
    assert formal_policy["validation_error_if_not_yes"] == "formal_diagnosis_assertion_required"

    evidence_items = evidence["items"]
    rehab_ids = set(options["common_rehab_direction_ids"])
    adjunct_ids = set(options["adjunct_ids"])

    for item_id in template["active_plan"]["order"]:
        assert item_id in rehab_ids, f"unknown CU-1 rehab id: {item_id}"
        assert item_id in evidence_items, f"Step-3 plan item missing frozen Step-2 evidence item: {item_id}"

    for item_id in template["adjuncts"]["order"]:
        assert item_id in adjunct_ids, f"unknown CU-1 adjunct id: {item_id}"
        assert item_id in evidence_items, f"Step-3 adjunct missing frozen Step-2 evidence item: {item_id}"

    assert evidence["default_plan"]["selected"] == [
        "therapeutic_exercise",
        "progressive_strengthening",
        "education_and_self_management",
    ]
    assert template["active_plan"]["implicit_core_item"] == evidence["default_plan"]["implicit"][0]
    assert template["adjuncts"]["dry_needling_selectable"] is False
    assert evidence_items["dry_needling"]["product_selectable"] is False
    assert template["weight_management"]["selectable_in_step3"] is False
    assert evidence_items["weight_management"]["product_selectable"] is False

    routed = template["clinical_picture"]["routed_to_function_instead_of_clinical_picture"]
    assert routed == {
        "walking_limitation": "walking_tolerance",
        "stairs_limitation": "stairs",
        "sit_to_stand_limitation": "sit_to_stand",
        "sport_or_exercise_limitation": "sport_gym",
    }
    assert template["functional_task_retraining"]["task_phrases"]["stairs"] == "τις σκάλες"

    hard = set(template["hard_invariants"])
    required = {
        "diagnosis_must_be_clinician_asserted_before_copy",
        "suggestion_is_not_selection",
        "selection_is_required_for_plan_phrase",
        "stiffness_is_not_rom_restriction",
        "generic_weakness_is_not_objective_weakness",
        "functional_limitation_is_not_automatic_rehab_selection",
        "evidence_cue_is_not_referral_prose",
        "adjunct_is_not_core_rehabilitation",
        "output_projection_is_not_state_mutation",
        "manual_edit_is_not_structured_writeback",
        "missing_is_not_negative",
        "no_exact_exercise_dose_invention",
        "no_machine_id_leak",
        "routine_text_generation_is_deterministic",
    }
    assert required <= hard

    fixture_list = fixtures["fixtures"]
    assert len(fixture_list) >= 12
    seen: set[str] = set()
    for fixture in fixture_list:
        fixture_id = fixture["id"]
        assert fixture_id not in seen
        seen.add(fixture_id)
        fixture_input = fixture["input"]
        assert fixture_input.get("formal_assertion_state") == "yes", (
            f"{fixture_id} must be an explicitly clinician-asserted copy-ready OA fixture"
        )
        rendered = render(template, fixture_input)
        expected = fixture["expected_text"]
        assert rendered == expected, f"{fixture_id}\nEXPECTED: {expected}\nACTUAL:   {rendered}"
        assert "_" not in rendered, f"machine id leak in {fixture_id}"
        assert "Οι οδηγίες διαφέρουν" not in rendered, f"evidence UI text leaked into referral: {fixture_id}"
        for forbidden in fixture.get("forbidden_substrings", []):
            assert forbidden not in rendered, f"forbidden substring {forbidden!r} in {fixture_id}"

    # Named semantic oracles for the product boundary.
    assert "suggestion_without_selection_does_not_render" in seen
    assert "quadriceps_specific_suppresses_generic_and_refines_plan" in seen
    assert "combined_rom_and_selected_mobility" in seen
    assert "mixed_guideline_adjunct_does_not_leak_evidence_text" in seen
    assert "bilateral_grammar" in seen
    assert "no_visible_plan_items_uses_implicit_fallback" in seen

    print(
        "Knee-OA template contract PASS: "
        f"{len(fixture_list)} clinician-asserted deterministic fixtures, "
        f"{len(template['active_plan']['order'])} rehab phrases, "
        f"{len(template['adjuncts']['order'])} adjunct phrases"
    )


if __name__ == "__main__":
    main()
