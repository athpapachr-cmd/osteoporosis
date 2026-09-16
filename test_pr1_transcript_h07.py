from __future__ import annotations

import json
from pathlib import Path

from clinical_excellence.modules.osteoporosis.transcript_profile import provider_profile


CASES = Path("evals/transcript_v1/cases.json")


def _cases_by_id():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    return cases, {item["id"]: item for item in cases}


def test_h07_provider_profile_freezes_exact_value_sets_and_case_sensitive_codes():
    profile = provider_profile()
    required_fragments = (
        "vertebral | hip | distal_radius | proximal_humerus | pelvis | other",
        "none | alendronate | risedronate | ibandronate_oral | zoledronate | ibandronate_iv | denosumab | teriparatide | romosozumab | raloxifene | hormone_therapy | other",
        "planned | active | completed | stopped | holiday | unknown",
        "done | due | overdue | missed | planned | not_applicable",
        "start | continue | stop | switch | defer | no_drug_treatment | complete_course | consolidate | refer | uncertain",
        "accepted | declined | undecided",
        "lab | DXA | administration | followup_visit | referral | VFA_or_imaging | adherence_check | exercise_or_falls | nutrition | other",
        "Use exact code=DXA",
        "Use code=followup_visit",
    )
    for fragment in required_fragments:
        assert fragment in profile


def test_h07_provider_profile_preserves_semantic_ownership_and_hard_safety_rules():
    profile = provider_profile()
    required_fragments = (
        "decision.selected_agent with semantic_type=option_discussed",
        "decision.selected_agent with semantic_type=clinician_recommendation",
        "decision.selected_agent with semantic_type=final_decision",
        "must NEVER create administration.agent, administration.status, administration.actual_date",
        "uncertain, use semantic_type=uncertain_needs_review and do not emit administration.status",
        "unrelated non-osteoporosis narrative",
        "speaker=third_party",
        "polarity=negative",
        "Do not emit positive treatment.status",
    )
    for fragment in required_fragments:
        assert fragment in profile


def _has_exact_rule(rules, expected):
    return any(rule == expected for rule in rules)


def test_h07_only_narrow_source_supported_fixture_permissions_are_added():
    cases, by_id = _cases_by_id()
    assert len(cases) == 22

    followup_rule = {
        "semantic_type": "followup_task",
        "concept_key": "followup.task_type",
        "source_assertion": {"speaker": "clinician"},
        "value": {"kind": "code", "code": "followup_visit"},
        "mapping": {"status": "mapped", "target_path": "step4.tasks[].type", "proposed_value": "followup_visit"},
    }
    for case_id in ("followup_vague", "followup_exact"):
        assert _has_exact_rule(by_id[case_id].get("allowed_assertions", []), followup_rule)

    frax_rule = {
        "semantic_type": "objective_result",
        "concept_key": "frax.tool_name",
        "source_assertion": {"speaker": "clinician"},
        "value": {"kind": "code", "code": "frax"},
        "mapping": {"status": "mapped", "target_path": "risk_assessment.tool_name", "proposed_value": "frax"},
    }
    for case_id in ("frax_original_adjusted", "out_of_range_runtime_values"):
        assert _has_exact_rule(by_id[case_id].get("allowed_assertions", []), frax_rule)

    # No wildcard concept-only permission is allowed for these H-07 additions.
    for case_id in ("followup_vague", "followup_exact", "frax_original_adjusted", "out_of_range_runtime_values"):
        for rule in by_id[case_id].get("allowed_assertions", []):
            if rule.get("concept_key") in {"followup.task_type", "frax.tool_name"}:
                assert "value" in rule
                assert "semantic_type" in rule
                assert "source_assertion" in rule


def test_h07_hard_a_cases_remain_fail_closed_in_fixtures():
    _, by_id = _cases_by_id()

    prescription = by_id["prescription_not_administration"]
    assert {"administration.agent", "administration.actual_date", "administration.status"}.issubset(
        set(prescription["forbidden_concepts"])
    )

    speaker = by_id["speaker_ambiguity"]
    assert "administration.status" in speaker["forbidden_concepts"]

    unrelated = by_id["unrelated_general_clinical_text"]
    assert not any(
        rule.get("concept_key") == "followup.task_type"
        for rule in unrelated.get("allowed_assertions", [])
    )

    third_party = by_id["third_party_treatment_history"]["required_assertions"]
    assert any(rule["source_assertion"] == {"speaker": "third_party", "polarity": "positive"} for rule in third_party)
    assert any(rule["source_assertion"] == {"speaker": "patient", "polarity": "negative"} for rule in third_party)

    negated = by_id["negated_treatment_exposure"]["required_assertions"][0]
    assert negated["source_assertion"] == {"speaker": "patient", "polarity": "negative"}
    assert negated["value"] == {"kind": "code", "code": "denosumab"}
