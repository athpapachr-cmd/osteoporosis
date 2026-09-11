from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
PRODUCT = ROOT / "clinic_utilities" / "physio_referral_product"
TEMPLATE_PATH = PRODUCT / "contracts" / "knee_oa_template_contract_v1.yaml"
FIXTURES_PATH = PRODUCT / "contracts" / "knee_oa_template_fixtures_v1.yaml"
EDGE_FIXTURES_PATH = PRODUCT / "contracts" / "knee_oa_template_edge_fixtures_v1.yaml"
EVIDENCE_PATH = PRODUCT / "contracts" / "knee_oa_evidence_contract_v1.yaml"
OPTION_CATALOG_PATH = ROOT / "clinic_utilities" / "contracts" / "cu1_option_catalog_v1.yaml"
REGISTRY_PATH = ROOT / "clinic_utilities" / "contracts" / "cu1_registry_v1.yaml"
ROUTE_REQUIREMENTS_PATH = ROOT / "clinic_utilities" / "contracts" / "cu1_route_requirements_v1.yaml"
UI_SCOPE_PATH = ROOT / "clinic_utilities" / "contracts" / "cu1_ui_relevance_scope_v1.yaml"
LANGUAGE_PATH = ROOT / "clinic_utilities" / "contracts" / "cu1_referral_language_el_v1.yaml"
RULE_CATALOG_PATH = ROOT / "clinic_utilities" / "contracts" / "cu1_rule_catalog_v1.yaml"


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


def validate_product_scope(template: dict[str, Any], state: dict[str, Any]) -> None:
    scope = template["product_supported_input_scope"]
    category_map = {
        "findings": "findings",
        "functional_impairments": "functional_impairments",
        "rehab_directions": "rehab_directions",
        "adjunct_options": "adjuncts",
        "goals": "goals",
    }
    for state_key, scope_key in category_map.items():
        allowed = set(scope[scope_key])
        for item_id in state.get(state_key) or []:
            if item_id not in allowed:
                raise ValueError(f"unsupported_product_selection:{state_key if state_key != 'adjunct_options' else 'adjuncts'}:{item_id}")


def render_restrictions(
    template: dict[str, Any], language: dict[str, Any], restrictions: list[dict[str, Any]]
) -> str:
    restriction_labels = language["restrictions"]
    context_values = language.get("context_values", {})
    rendered: list[str] = []
    for restriction in restrictions:
        restriction_id = restriction["restriction_id"]
        if restriction_id not in restriction_labels:
            raise ValueError(f"unsupported_product_restriction:{restriction_id}")
        label = restriction_labels[restriction_id]
        value = restriction.get("state_or_value")
        if isinstance(value, str) and value:
            display_value = context_values.get(value, value)
            rendered.append(f"{label}: {display_value}")
        else:
            rendered.append(label)
    if not rendered:
        return ""
    return template["block_templates"]["restrictions"].format(
        restriction_list=join_el(rendered)
    )


def render(template: dict[str, Any], state: dict[str, Any], language: dict[str, Any]) -> str:
    validate_product_scope(template, state)

    laterality = state["laterality"]
    laterality_phrase = template["laterality_phrases"][laterality]
    blocks: list[str] = [
        template["block_templates"]["indication"].format(laterality_phrase=laterality_phrase)
    ]

    findings = set(state.get("findings") or [])
    phenotype = state.get("phenotype") or {}

    clinical_keys: set[str] = {
        key
        for key in findings
        if key not in template["clinical_picture"]["routed_to_function_instead_of_clinical_picture"]
    }
    if phenotype.get("stiffness_symptom"):
        clinical_keys.add("stiffness_symptom")
    if phenotype.get("weakness_symptom_or_context"):
        clinical_keys.add("weakness_symptom_or_context")

    pain_rule = template["clinical_picture"]["precedence"]["pain"]
    if pain_rule["suppress_generic_when_any_specific"] and any(
        key in clinical_keys for key in pain_rule["specific"]
    ):
        clinical_keys.discard(pain_rule["generic"])

    weakness_priority = template["clinical_picture"]["precedence"]["weakness"]["priority"]
    selected_weakness = next((key for key in weakness_priority if key in clinical_keys), None)
    for key in weakness_priority:
        clinical_keys.discard(key)
    if selected_weakness:
        clinical_keys.add(selected_weakness)

    swelling_rule = template["clinical_picture"]["precedence"]["swelling"]
    if (
        swelling_rule["suppress_generic_when_specific"]
        and swelling_rule["specific"] in clinical_keys
    ):
        clinical_keys.discard(swelling_rule["generic"])

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
            selected_tasks = [key for key in task_spec["task_order"] if key in function_keys]
            if selected_tasks:
                phrase = task_spec["phrase_with_tasks"].format(
                    task_list=join_el([task_spec["task_phrases"][key] for key in selected_tasks])
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
        goal_template = (
            "nonredundant_goal_emphasis_one"
            if len(rendered_goals) == 1
            else "nonredundant_goal_emphasis_many"
        )
        blocks.append(
            template["block_templates"][goal_template].format(
                goal_list=join_el(rendered_goals)
            )
        )

    selected_adjuncts = set(state.get("adjunct_options") or [])
    rendered_adjuncts = [
        template["adjuncts"]["phrases"][key]
        for key in template["adjuncts"]["order"]
        if key in selected_adjuncts
    ]
    if rendered_adjuncts:
        blocks.append(
            template["block_templates"]["adjuncts"].format(
                adjunct_list=join_el(rendered_adjuncts)
            )
        )

    restriction_block = render_restrictions(
        template, language, state.get("explicit_restrictions") or []
    )
    if restriction_block:
        blocks.append(restriction_block)

    note = state.get("clinician_free_text_optional")
    if isinstance(note, str) and note.strip():
        normalized_note = " ".join(note.split()).rstrip(". ")
        blocks.append(
            template["block_templates"]["clinician_note"].format(
                clinician_note=normalized_note
            )
        )

    return template["join_policy"]["block_separator"].join(blocks)


def main() -> None:
    template = load_yaml(TEMPLATE_PATH)
    fixtures = load_yaml(FIXTURES_PATH)
    edge_fixtures = load_yaml(EDGE_FIXTURES_PATH)
    evidence = load_yaml(EVIDENCE_PATH)
    options = load_yaml(OPTION_CATALOG_PATH)
    registry = load_yaml(REGISTRY_PATH)
    route_requirements = load_yaml(ROUTE_REQUIREMENTS_PATH)
    ui_scope = load_yaml(UI_SCOPE_PATH)
    language = load_yaml(LANGUAGE_PATH)
    rule_catalog = load_yaml(RULE_CATALOG_PATH)

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
    assert template["copy_readiness"]["product_scope_valid_required"] is True
    assert set(template["laterality_phrases"]) == {"right", "left", "bilateral"}
    assert template["block_templates"]["indication"].startswith(
        "Παραπομπή για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση λόγω οστεοαρθρίτιδας"
    )

    knee_route = registry["profiles"]["knee"]["routes"]["knee_osteoarthritis"]
    assert "formal_diagnosis" in knee_route["wording_modes"]
    formal_policy = route_requirements["wording_mode_requirements"]["formal_diagnosis"]
    assert formal_policy["formal_assertion_policy"] == "required_yes_unless_route_override_defines_context_based_assertion"
    assert formal_policy["validation_error_if_not_yes"] == "formal_diagnosis_assertion_required"

    supported = template["product_supported_input_scope"]
    assert supported["unsupported_selected_policy"] == "block_product_projection"
    valid_findings = set(options["common_findings"]) | set(options["profile_findings"]["knee"])
    valid_functions = set(options["common_functional_impairments"])
    valid_rehab = set(options["common_rehab_direction_ids"])
    valid_adjuncts = set(options["adjunct_ids"])
    valid_goals = set(options["common_goal_ids"])
    assert set(supported["findings"]) <= valid_findings
    assert set(supported["functional_impairments"]) <= valid_functions
    assert set(supported["rehab_directions"]) <= valid_rehab
    assert set(supported["adjuncts"]) <= valid_adjuncts
    assert set(supported["goals"]) <= valid_goals

    knee_ui = ui_scope["profiles"]["knee"]
    assert set(supported["findings"]) <= set(knee_ui["findings"])
    assert set(supported["functional_impairments"]) <= set(knee_ui["functional_impairments"])
    assert set(supported["rehab_directions"]) - {"walking_aid_assessment_and_training"} <= set(knee_ui["rehab_directions"])
    assert set(supported["adjuncts"]) <= set(knee_ui["adjuncts"])

    evidence_items = evidence["items"]
    for item_id in template["active_plan"]["order"]:
        assert item_id in supported["rehab_directions"]
        assert item_id in evidence_items, f"Step-3 plan item missing frozen Step-2 evidence item: {item_id}"
    for item_id in template["adjuncts"]["order"]:
        assert item_id in supported["adjuncts"]
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

    true_locking = template["integration_findings"]["true_locking_or_major_mechanical_rom_block"]
    assert true_locking["current_cu1_finding_exists"] is True
    assert true_locking["current_rule_catalog_auto_safety_trigger_from_finding"] is False
    assert true_locking["product_selectable_in_step3"] is False
    assert "true_locking_or_major_mechanical_rom_block" not in supported["findings"]
    assert "true_locking_or_major_mechanical_rom_block" not in yaml.safe_dump(
        rule_catalog["rules"], sort_keys=True
    )

    hard = set(template["hard_invariants"])
    required = {
        "diagnosis_must_be_clinician_asserted_before_copy",
        "every_selected_product_item_must_render_or_block",
        "true_locking_not_product_exposed_without_safety_mapping",
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

    render_fixtures = list(fixtures["fixtures"]) + list(edge_fixtures["render_fixtures"])
    assert len(render_fixtures) >= 15
    seen: set[str] = set()
    for fixture in render_fixtures:
        fixture_id = fixture["id"]
        assert fixture_id not in seen
        seen.add(fixture_id)
        fixture_input = fixture["input"]
        assert fixture_input.get("formal_assertion_state") == "yes", (
            f"{fixture_id} must be an explicitly clinician-asserted copy-ready OA fixture"
        )
        rendered = render(template, fixture_input, language)
        expected = fixture["expected_text"]
        assert rendered == expected, f"{fixture_id}\nEXPECTED: {expected}\nACTUAL:   {rendered}"
        assert "_" not in rendered, f"machine id leak in {fixture_id}"
        assert "Οι οδηγίες διαφέρουν" not in rendered, f"evidence UI text leaked into referral: {fixture_id}"
        for forbidden in fixture.get("forbidden_substrings", []):
            assert forbidden not in rendered, f"forbidden substring {forbidden!r} in {fixture_id}"

    for fixture in edge_fixtures["blocking_fixtures"]:
        fixture_id = fixture["id"]
        try:
            render(template, fixture["input"], language)
        except ValueError as exc:
            assert str(exc) == fixture["expected_error"], (fixture_id, str(exc))
        else:
            raise AssertionError(f"{fixture_id} should block product projection")

    named_oracles = {
        "suggestion_without_selection_does_not_render",
        "quadriceps_specific_suppresses_generic_and_refines_plan",
        "combined_rom_and_selected_mobility",
        "mixed_guideline_adjunct_does_not_leak_evidence_text",
        "bilateral_grammar",
        "no_visible_plan_items_uses_implicit_fallback",
        "power_user_findings_and_functions_preserved",
        "nonredundant_goal_plural_preserved",
        "restriction_and_clinician_note_preserved",
    }
    assert named_oracles <= seen

    print(
        "Knee-OA template contract PASS: "
        f"{len(render_fixtures)} clinician-asserted deterministic render fixtures, "
        f"{len(edge_fixtures['blocking_fixtures'])} fail-closed scope fixtures"
    )


if __name__ == "__main__":
    main()
