from __future__ import annotations

import json
from pathlib import Path

from clinical_excellence.core.transcript_service import extract_candidates
from evals.transcript_v1.run_provider_eval import _component_matches_rule, _evaluate_case
from test_pr1_transcript_h10 import StaticProvider, _request


CASE = next(
    item for item in json.loads(Path("evals/transcript_v1/cases.json").read_text(encoding="utf-8"))
    if item["id"] == "frax_original_adjusted"
)
RULE = next(
    rule for rule in CASE["allowed_assertions"]
    if rule.get("concept_key") == "clinical.unmapped_narrative"
)


def _result(narrative: str):
    source = {"speaker": "clinician", "polarity": "positive", "temporality": "current", "certainty": "explicit"}
    candidates = [
        {
            "semantic_type": "objective_result",
            "components": [{"concept_key": concept, "value": {"kind": "number", "value": value}}],
            "source_assertion": source,
            "evidence_snippet": evidence,
            "confidence": "high",
        }
        for concept, value, evidence in (
            ("frax.mof_percent", 18, "MOF 18%"),
            ("frax.hip_percent", 4, "hip 4%"),
        )
    ]
    candidates.append({
        "semantic_type": "clinician_interpretation",
        "components": [{"concept_key": "clinical.unmapped_narrative", "value": {"kind": "text", "text": narrative}}],
        "source_assertion": source,
        "evidence_snippet": "Με κλινική προσαρμογή εκτιμώ τον κίνδυνο υψηλότερο",
        "confidence": "high",
    })
    return extract_candidates(_request(CASE["transcript"]), StaticProvider({"candidates": candidates, "warnings": []}))


def test_h13_swapped_authentic_frax_values_fail_full_promotion_oracle():
    result = _result("Με κλινική προσαρμογή εκτιμώ τον κίνδυνο υψηλότερο: hip 18%, MOF 4%")
    narrative = result.candidates[-1]
    component = narrative.components[0]
    assert not _component_matches_rule(narrative, component, RULE, CASE["transcript"])
    failures = _evaluate_case(CASE, result)
    assert "unexpected_assertion_clinical.unmapped_narrative" in failures
    assert "narrative_value_relation_mismatch" in failures


def test_h13_source_grounded_frax_paraphrase_and_reordered_pairs_pass():
    for narrative in (
        "εκτιμώ τον κίνδυνο υψηλότερο",
        "κίνδυνος αυξημένος: MOF 18%, hip 4%",
        "κίνδυνος αυξημένος: 4% hip, 18% MOF",
        "κίνδυνος αυξημένος: 4 hip, 18 MOF",
        "κίνδυνος αυξημένος: hip 4%",
    ):
        assert _evaluate_case(CASE, _result(narrative)) == []


def test_h13_unpaired_or_mixed_frax_percentages_fail_closed():
    for narrative in (
        "κίνδυνος αυξημένος: 18%",
        "κίνδυνος αυξημένος: MOF 18%, hip 18%",
        "κίνδυνος αυξημένος: hip 18, MOF 4",
        "κίνδυνος αυξημένος: MOF hip 18% 4%",
    ):
        assert "narrative_value_relation_mismatch" in _evaluate_case(CASE, _result(narrative))


def test_h13_binding_requires_the_source_relationship():
    narrative = _result("κίνδυνος αυξημένος: MOF 18%, hip 4%").candidates[-1]
    altered_source = CASE["transcript"].replace("MOF 18% και hip 4%", "MOF 4% και hip 18%")
    assert not _component_matches_rule(narrative, narrative.components[0], RULE, altered_source)
    conflicting_source = CASE["transcript"] + " MOF 4%."
    assert not _component_matches_rule(narrative, narrative.components[0], RULE, conflicting_source)
