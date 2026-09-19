from __future__ import annotations

import json
from pathlib import Path

from clinical_excellence.modules.osteoporosis.transcript_profile import provider_profile


CASES = Path("evals/transcript_v1/cases.json")


def _by_id():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    return cases, {item["id"]: item for item in cases}


def _has_exact_rule(rules, expected):
    return any(rule == expected for rule in rules)


def test_h11_profile_distinguishes_reviewed_vfa_from_current_performance():
    profile = provider_profile()
    assert 'references an existing VFA result' in profile
    assert 'use vfa.action code=already_available_reviewed' in profile
    assert 'Use code=performed only when performance itself is explicitly source-stated' in profile


def test_h11_vfa_case_allows_only_source_supported_review_action():
    cases, by_id = _by_id()
    assert len(cases) == 22
    case = by_id["negative_history_vs_negative_investigation"]

    reviewed_rule = {
        "semantic_type": "objective_result",
        "concept_key": "vfa.action",
        "source_assertion": {"speaker": "clinician"},
        "value": {"kind": "code", "code": "already_available_reviewed"},
        "mapping": {
            "status": "mapped",
            "target_path": "step3.vfa.action",
            "proposed_value": "already_available_reviewed",
        },
        "evidence_contains": "Στη VFA",
    }
    assert _has_exact_rule(case.get("allowed_assertions", []), reviewed_rule)

    performed_rule = {
        "semantic_type": "objective_result",
        "concept_key": "vfa.action",
        "value": {"kind": "code", "code": "performed"},
    }
    assert _has_exact_rule(case.get("forbidden_assertions", []), performed_rule)


def test_h11_profile_treats_no_administration_mentioned_as_absence_of_documentation():
    profile = provider_profile()
    assert 'is absence-of-documentation' in profile
    assert 'not proof that administration occurred' in profile
    assert 'not proof of a negative administration event' in profile
    assert 'clinician_interpretation + clinical.unmapped_narrative' in profile


def test_h11_prescription_case_allows_only_source_bound_nonruntime_absence_narrative():
    _, by_id = _by_id()
    case = by_id["prescription_not_administration"]

    narrative_rule = {
        "semantic_type": "clinician_interpretation",
        "concept_key": "clinical.unmapped_narrative",
        "source_assertion": {"speaker": "clinician"},
        "value": {"kind": "text"},
        "mapping": {"status": "unmapped", "reason_code": "NO_CURRENT_RUNTIME_TARGET"},
        "evidence_contains": "Δεν αναφέρεται ότι έγινε χορήγηση",
    }
    assert _has_exact_rule(case.get("allowed_assertions", []), narrative_rule)

    assert {
        "treatment.agent",
        "administration.agent",
        "administration.actual_date",
        "administration.status",
    }.issubset(set(case["forbidden_concepts"]))


def test_h11_permissions_remain_exact_not_wildcard():
    _, by_id = _by_id()
    checks = [
        ("negative_history_vs_negative_investigation", "vfa.action"),
        ("prescription_not_administration", "clinical.unmapped_narrative"),
    ]
    for case_id, concept in checks:
        rules = [r for r in by_id[case_id].get("allowed_assertions", []) if r.get("concept_key") == concept]
        assert len(rules) == 1
        rule = rules[0]
        assert "semantic_type" in rule
        assert "source_assertion" in rule
        assert "value" in rule
        assert "mapping" in rule
        assert rule.get("evidence_contains")
