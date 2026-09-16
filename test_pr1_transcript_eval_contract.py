import json
from pathlib import Path

from clinical_excellence.core.transcript_contracts import ProviderCandidateV1
from clinical_excellence.modules.osteoporosis.transcript_target_guard import map_candidate


def test_eval_fixture_is_synthetic_and_covers_required_cases():
    cases = json.loads(Path("evals/transcript_v1/cases.json").read_text(encoding="utf-8"))
    assert len(cases) >= 10
    ids = {item["id"] for item in cases}
    assert {
        "fracture_relative_time",
        "explicit_negative_smoking",
        "dxa_objective",
        "labs_objective",
        "options_one_final",
        "preference_only",
        "followup_vague",
        "garbled_speech",
        "frax_original_adjusted",
        "speaker_ambiguity",
    }.issubset(ids)
    joined = json.dumps(cases, ensure_ascii=False).lower()
    for forbidden in ("gesy id", "@gmail.com", "+357 9"):
        assert forbidden not in joined


def test_provider_eval_runner_is_fail_closed_and_does_not_print_transcript_content():
    runner = Path("evals/transcript_v1/run_provider_eval.py").read_text(encoding="utf-8")
    assert "provider-eval BLOCKED" in runner
    assert 'status["phi_provider_approved"]' in runner
    assert "item['transcript']" not in runner
    assert "result.candidates" in runner


def _candidate(concept_key, value):
    return ProviderCandidateV1.model_validate({
        "semantic_type": "patient_history_fact",
        "components": [{"concept_key": concept_key, "value": value}],
        "source_assertion": {
            "speaker": "clinician",
            "polarity": "positive",
            "temporality": "current",
            "certainty": "explicit",
        },
        "evidence_snippet": "",
        "confidence": "high",
    })


def test_step4_status_mapping_rejects_provider_codes_outside_runtime_enums():
    valid_episode = map_candidate(_candidate("treatment.status", {"kind": "code", "code": "active"}))[0]
    assert valid_episode.status == "mapped"
    assert valid_episode.proposed_value == "active"

    invalid_episode = map_candidate(_candidate("treatment.status", {"kind": "code", "code": "paused_forever"}))[0]
    assert invalid_episode.status == "ambiguous"
    assert invalid_episode.reason_code == "UNSUPPORTED_TREATMENT_STATUS"

    valid_admin = map_candidate(_candidate("administration.status", {"kind": "code", "code": "overdue"}))[0]
    assert valid_admin.status == "mapped"
    assert valid_admin.proposed_value == "overdue"

    invalid_admin = map_candidate(_candidate("administration.status", {"kind": "text", "text": "administered maybe"}))[0]
    assert invalid_admin.status == "ambiguous"
    assert invalid_admin.reason_code == "UNSUPPORTED_ADMINISTRATION_STATUS"


def test_treatment_duration_requires_numeric_runtime_range():
    valid = map_candidate(_candidate("treatment.duration_years", {"kind": "number", "value": 4.5}))[0]
    assert valid.status == "mapped"
    assert valid.proposed_value == 4.5

    wrong_type = map_candidate(_candidate("treatment.duration_years", {"kind": "code", "code": "four years"}))[0]
    assert wrong_type.status == "ambiguous"
    assert wrong_type.reason_code == "TYPE_MISMATCH"

    out_of_range = map_candidate(_candidate("treatment.duration_years", {"kind": "number", "value": 51}))[0]
    assert out_of_range.status == "ambiguous"
    assert out_of_range.reason_code == "OUT_OF_RUNTIME_RANGE"


def test_fixed_runtime_code_targets_are_whitelisted_locally():
    assert map_candidate(_candidate("fracture.site", {"kind": "code", "code": "hip"}))[0].status == "mapped"
    bad_site = map_candidate(_candidate("fracture.site", {"kind": "code", "code": "ankle"}))[0]
    assert bad_site.status == "ambiguous"
    assert bad_site.reason_code == "UNSUPPORTED_FRACTURE_SITE"

    assert map_candidate(_candidate("frax.tool_name", {"kind": "code", "code": "fraxplus"}))[0].status == "mapped"
    bad_tool = map_candidate(_candidate("frax.tool_name", {"kind": "code", "code": "magic_score"}))[0]
    assert bad_tool.status == "ambiguous"
    assert bad_tool.reason_code == "UNSUPPORTED_FRAX_TOOL"

    assert map_candidate(_candidate("risk.resulting_category", {"kind": "code", "code": "very_high"}))[0].status == "mapped"
    bad_risk = map_candidate(_candidate("risk.resulting_category", {"kind": "code", "code": "extreme"}))[0]
    assert bad_risk.status == "ambiguous"
    assert bad_risk.reason_code == "UNSUPPORTED_RISK_CATEGORY"


def test_vfa_runtime_enums_are_validated_and_boolean_indication_is_normalized():
    indicated = map_candidate(_candidate("vfa.indicated", {"kind": "boolean", "value": True}))[0]
    assert indicated.status == "mapped"
    assert indicated.proposed_value == "yes"

    uncertain = map_candidate(_candidate("vfa.indicated", {"kind": "code", "code": "uncertain"}))[0]
    assert uncertain.status == "mapped"
    assert uncertain.proposed_value == "uncertain"

    bad_action = map_candidate(_candidate("vfa.action", {"kind": "code", "code": "probably_later"}))[0]
    assert bad_action.status == "ambiguous"
    assert bad_action.reason_code == "UNSUPPORTED_VFA_ACTION"

    modality = map_candidate(_candidate("vfa.modality", {"kind": "code", "code": "MRI"}))[0]
    assert modality.status == "mapped"

    bad_modality = map_candidate(_candidate("vfa.modality", {"kind": "code", "code": "ultrasound"}))[0]
    assert bad_modality.status == "ambiguous"
    assert bad_modality.reason_code == "UNSUPPORTED_VFA_MODALITY"
