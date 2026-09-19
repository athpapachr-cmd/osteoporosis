from __future__ import annotations

from clinical_excellence.core.transcript_contracts import ProviderTranscriptExtractionV1
from clinical_excellence.modules.osteoporosis.transcript_target_guard import map_candidate


def _candidate(
    concept: str,
    value: dict,
    *,
    semantic_type: str = "patient_history_fact",
    speaker: str = "clinician",
    polarity: str = "positive",
    temporality: str = "current",
):
    parsed = ProviderTranscriptExtractionV1.model_validate(
        {
            "candidates": [
                {
                    "semantic_type": semantic_type,
                    "components": [{"concept_key": concept, "value": value}],
                    "source_assertion": {
                        "speaker": speaker,
                        "polarity": polarity,
                        "temporality": temporality,
                        "certainty": "explicit",
                    },
                    "evidence_snippet": "",
                    "confidence": "high",
                }
            ],
            "warnings": [],
        }
    )
    return parsed.candidates[0]


def _multi_candidate(components: list[dict], *, semantic_type: str, temporality: str = "current"):
    parsed = ProviderTranscriptExtractionV1.model_validate(
        {
            "candidates": [
                {
                    "semantic_type": semantic_type,
                    "components": components,
                    "source_assertion": {
                        "speaker": "clinician",
                        "polarity": "positive",
                        "temporality": temporality,
                        "certainty": "explicit",
                    },
                    "evidence_snippet": "",
                    "confidence": "high",
                }
            ],
            "warnings": [],
        }
    )
    return parsed.candidates[0]


def test_h09_recommendation_or_option_cannot_create_treatment_episode_runtime_truth():
    recommendation = map_candidate(
        _candidate(
            "treatment.agent",
            {"kind": "code", "code": "denosumab"},
            semantic_type="clinician_recommendation",
        )
    )[0]
    assert recommendation.status == "ambiguous"
    assert recommendation.reason_code == "SEMANTIC_TYPE_NOT_ALLOWED_FOR_TREATMENT_EPISODE"
    assert recommendation.target_path == "step4.treatment_episodes[].agent"

    option = map_candidate(
        _candidate(
            "treatment.status",
            {"kind": "code", "code": "planned"},
            semantic_type="option_discussed",
        )
    )[0]
    assert option.status == "ambiguous"
    assert option.reason_code == "SEMANTIC_TYPE_NOT_ALLOWED_FOR_TREATMENT_EPISODE"


def test_h09_recommendation_or_decision_cannot_create_administration_runtime_truth():
    recommendation = map_candidate(
        _candidate(
            "administration.status",
            {"kind": "code", "code": "done"},
            semantic_type="clinician_recommendation",
        )
    )[0]
    assert recommendation.status == "ambiguous"
    assert recommendation.reason_code == "SEMANTIC_TYPE_NOT_ALLOWED_FOR_ADMINISTRATION_EVENT"

    final_decision = map_candidate(
        _candidate(
            "administration.agent",
            {"kind": "code", "code": "denosumab"},
            semantic_type="final_decision",
        )
    )[0]
    assert final_decision.status == "ambiguous"
    assert final_decision.reason_code == "SEMANTIC_TYPE_NOT_ALLOWED_FOR_ADMINISTRATION_EVENT"


def test_h09_negated_fracture_treatment_and_administration_presence_fail_closed():
    examples = [
        (
            "fracture.site",
            {"kind": "code", "code": "hip"},
            "patient_history_fact",
        ),
        (
            "treatment.agent",
            {"kind": "code", "code": "denosumab"},
            "patient_history_fact",
        ),
        (
            "administration.actual_date",
            {
                "kind": "date",
                "normalized": "2026-09-01",
                "precision": "day",
                "date_text": "01/09/2026",
            },
            "objective_result",
        ),
    ]

    for concept, value, semantic_type in examples:
        mapping = map_candidate(
            _candidate(
                concept,
                value,
                semantic_type=semantic_type,
                polarity="negative",
                temporality="past",
            )
        )[0]
        assert mapping.status == "ambiguous"
        assert mapping.reason_code == "NEGATED_ASSERTION_NOT_POSITIVE_RUNTIME_VALUE"


def test_h09_valid_actual_treatment_history_remains_mapped():
    mapping = map_candidate(
        _candidate(
            "treatment.agent",
            {"kind": "code", "code": "alendronate"},
            semantic_type="patient_history_fact",
            speaker="patient",
            temporality="past",
        )
    )[0]
    assert mapping.status == "mapped"
    assert mapping.target_path == "step4.treatment_episodes[].agent"
    assert mapping.proposed_value == "alendronate"


def test_h09_valid_actual_and_scheduled_administration_events_remain_mapped():
    actual = map_candidate(
        _candidate(
            "administration.actual_date",
            {
                "kind": "date",
                "normalized": "2026-09-01",
                "precision": "day",
                "date_text": "01/09/2026",
            },
            semantic_type="objective_result",
            temporality="past",
        )
    )[0]
    assert actual.status == "mapped"
    assert actual.target_path == "step4.administrations[].actual_date"
    assert actual.proposed_value == "2026-09-01"

    scheduled = _multi_candidate(
        [
            {
                "concept_key": "administration.agent",
                "value": {"kind": "code", "code": "denosumab"},
            },
            {
                "concept_key": "administration.scheduled_date",
                "value": {
                    "kind": "date",
                    "normalized": "2026-10-20",
                    "precision": "day",
                    "date_text": "20/10/2026",
                },
            },
            {
                "concept_key": "administration.status",
                "value": {"kind": "code", "code": "planned"},
            },
        ],
        semantic_type="followup_task",
        temporality="future",
    )
    mappings = {item.component_keys[0]: item for item in map_candidate(scheduled)}
    assert mappings["administration.agent"].status == "mapped"
    assert mappings["administration.scheduled_date"].status == "mapped"
    assert mappings["administration.status"].status == "mapped"
    assert mappings["administration.status"].proposed_value == "planned"
