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
