from __future__ import annotations

import json
from pathlib import Path

from clinical_excellence.modules.osteoporosis.transcript_profile import provider_profile


CASES = Path("evals/transcript_v1/cases.json")


def _cases_by_id():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    return cases, {item["id"]: item for item in cases}


def _has_exact_rule(rules, expected):
    return any(rule == expected for rule in rules)


def test_h08_profile_canonicalizes_prescription_recommendation_as_decision_not_treatment_episode():
    profile = provider_profile()
    assert "A prescription/recommendation alone is a clinician_recommendation using decision.selected_agent" in profile
    assert "Do NOT use treatment.agent solely from prescription/recommendation wording" in profile
    assert "may use a clinician_recommendation treatment.agent" not in profile


def test_h08_three_source_supported_extras_are_narrowly_allowed():
    cases, by_id = _cases_by_id()
    assert len(cases) == 22

    vfa_rule = {
        "semantic_type": "objective_result",
        "concept_key": "vfa.modality",
        "source_assertion": {"speaker": "clinician"},
        "value": {"kind": "code", "code": "VFA"},
        "mapping": {"status": "mapped", "target_path": "step3.vfa.modality", "proposed_value": "VFA"},
    }
    assert _has_exact_rule(by_id["negative_history_vs_negative_investigation"].get("allowed_assertions", []), vfa_rule)

    narrative_rule = {
        "semantic_type": "clinician_interpretation",
        "concept_key": "clinical.unmapped_narrative",
        "source_assertion": {"speaker": "clinician"},
        "value": {"kind": "text"},
        "mapping": {"status": "unmapped", "reason_code": "NO_CURRENT_RUNTIME_TARGET"},
    }
    assert _has_exact_rule(by_id["referral_not_completed_result"].get("allowed_assertions", []), narrative_rule)

    task_rule = {
        "semantic_type": "followup_task",
        "concept_key": "followup.task_type",
        "source_assertion": {"speaker": "clinician"},
        "value": {"kind": "code", "code": "administration"},
        "mapping": {"status": "mapped", "target_path": "step4.tasks[].type", "proposed_value": "administration"},
    }
    assert _has_exact_rule(by_id["planned_not_done_administration"].get("allowed_assertions", []), task_rule)


def test_h08_prescription_fixture_requires_non_authoritative_recommendation_and_blocks_treatment_or_administration_truth():
    _, by_id = _cases_by_id()
    case = by_id["prescription_not_administration"]

    assert case["expect_concepts"] == ["decision.selected_agent"]
    assert case["expect_semantics"] == ["clinician_recommendation"]
    assert case["required_assertions"] == [
        {
            "semantic_type": "clinician_recommendation",
            "concept_key": "decision.selected_agent",
            "source_assertion": {"speaker": "clinician"},
            "value": {"kind": "code", "code": "denosumab"},
            "mapping": {
                "status": "ambiguous",
                "reason_code": "FINAL_DECISION_REQUIRED",
                "target_path": "step4.decision.selected_agent",
                "proposed_value": "denosumab",
            },
        }
    ]
    assert {
        "treatment.agent",
        "administration.agent",
        "administration.actual_date",
        "administration.status",
    }.issubset(set(case["forbidden_concepts"]))


def test_h08_default_deny_remains_exact_for_new_permissions():
    _, by_id = _cases_by_id()
    targets = {
        "negative_history_vs_negative_investigation": "vfa.modality",
        "referral_not_completed_result": "clinical.unmapped_narrative",
        "planned_not_done_administration": "followup.task_type",
    }
    for case_id, concept in targets.items():
        matches = [
            rule for rule in by_id[case_id].get("allowed_assertions", [])
            if rule.get("concept_key") == concept
        ]
        assert len(matches) == 1
        rule = matches[0]
        assert "semantic_type" in rule
        assert "source_assertion" in rule
        assert "value" in rule
        assert "mapping" in rule
