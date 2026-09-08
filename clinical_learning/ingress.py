from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from .contracts import ContractIssue, LearningContractError, normalize_import_preview, validate_challenge_payload
from .learning_loop import build_learning_loop_plan, resource_candidates_from_source
from .privacy import find_forbidden_structured_fields, scan_persistable_strings


FOUNDATION_ALIASES = {
    "bone-remodeling-physiology": "ost.foundation.bone_remodeling",
    "bone_remodeling": "ost.foundation.bone_remodeling",
    "secondary-osteoporosis": "ost.foundation.secondary_osteoporosis",
    "secondary_osteoporosis": "ost.foundation.secondary_osteoporosis",
    "antiresorptive-pharmacology": "ost.foundation.antiresorptive_pharmacology",
    "antiresorptive_pharmacology": "ost.foundation.antiresorptive_pharmacology",
    "anabolic-pharmacology": "ost.foundation.anabolic_pharmacology",
    "anabolic_pharmacology": "ost.foundation.anabolic_pharmacology",
    "sequencing": "ost.foundation.sequencing",
    "denosumab-rebound": "ost.foundation.denosumab_rebound",
    "denosumab_rebound": "ost.foundation.denosumab_rebound",
    "btms": "ost.foundation.btms",
    "safety-contraindications-transitions": "ost.foundation.treatment_safety_transitions",
    "treatment-safety-transitions": "ost.foundation.treatment_safety_transitions",
    "treatment_safety_transitions": "ost.foundation.treatment_safety_transitions",
    "evidence-appraisal-directness-guideline-disagreement": "ost.foundation.evidence_appraisal",
    "evidence-appraisal": "ost.foundation.evidence_appraisal",
    "evidence_appraisal": "ost.foundation.evidence_appraisal",
    "dxa-lsc": "ost.foundation.dxa_lsc",
    "vfa": "ost.foundation.vfa",
    "fracture-risk-frax": "ost.foundation.fracture_risk_frax",
    "giop": "ost.foundation.giop",
    "communication-sdm-continuity": "ost.foundation.communication_sdm_continuity",
}

FACT_SCOPE_MAP = {
    "synthetic_initial_case_fact": "synthetic_case_fact",
    "synthetic_progressive_disclosure_fact": "synthetic_case_fact",
    "clinician_hypothesis": "clinician_hypothesis",
    "educational_inference": "ai_inference",
    "ai_inference": "ai_inference",
    "counterfactual_teaching_point": "counterfactual_teaching_point",
}

GAP_CLASS_MAP = {
    "knowledge": "knowledge",
    "knowledge_and_mechanism": "knowledge",
    "reasoning": "reasoning",
    "decision_safety": "reasoning",
    "evidence_appraisal": "reasoning",
    "execution": "execution",
    "workflow": "execution",
    "communication": "communication_system",
    "communication_system": "communication_system",
}


@dataclass(frozen=True)
class AdaptedLearningEpisode:
    source_event_id: str
    source_format: str
    normalized_challenge: dict[str, Any]
    loop_plan: dict[str, Any]
    resources: list[dict[str, Any]]
    warnings: list[str]


def _uuid(source_key: str, kind: str, *local_parts: object) -> str:
    local = "|".join(str(part) for part in local_parts)
    return str(uuid5(NAMESPACE_URL, f"clinical-learning-ingress|{source_key}|{kind}|{local}"))


def _parse_datetime(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc).isoformat()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return f"{text}T12:00:00+00:00"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc).isoformat()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def _foundation_nodes(values: Any, warnings: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values or []:
        text = str(raw).strip()
        node = text if text.startswith("ost.foundation.") else FOUNDATION_ALIASES.get(text.casefold())
        if not node:
            warnings.append(f"unknown_foundation_alias:{text}")
            continue
        if node not in out:
            out.append(node)
    return out


def _gap_class(raw: Any) -> str:
    text = str(raw or "").strip().casefold().replace("-", "_")
    return GAP_CLASS_MAP.get(text, "knowledge")


def _evidence_type(item: dict[str, Any]) -> str:
    text = " ".join(str(item.get(key) or "") for key in ("source_type", "evidence_level", "citation")).casefold()
    if "systematic" in text or "meta-analysis" in text or "meta analysis" in text:
        return "systematic_review_meta_analysis"
    if "guideline" in text:
        return "guideline"
    if "consensus" in text or "position statement" in text:
        return "consensus_position_statement"
    if "random" in text or " rct" in f" {text}":
        return "randomized_trial"
    if "cohort" in text or "observational" in text or "target-trial" in text or "target trial" in text:
        return "cohort"
    if "case-control" in text or "case control" in text:
        return "case_control"
    if "diagnostic" in text:
        return "diagnostic_accuracy"
    if "narrative" in text:
        return "narrative_review"
    if "mechanistic" in text or "translational" in text:
        return "mechanistic_or_translational"
    return "other"


def _pmid(item: dict[str, Any]) -> str | None:
    citation = str(item.get("citation") or "")
    match = re.search(r"(?i)PMID\s*:?\s*(\d{6,10})", citation)
    if match:
        return match.group(1)
    url = str(item.get("url") or "")
    match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{6,10})", url)
    return match.group(1) if match else None


def _rich_to_canonical(raw: dict[str, Any], source_event_id: str) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    source_key = str(raw.get("challenge_id") or source_event_id)
    session = raw.get("session") if isinstance(raw.get("session"), dict) else {}
    source_challenge_id = str(raw.get("challenge_id") or source_event_id)
    challenge_id = _uuid(source_key, "challenge", source_challenge_id)
    nodes = _foundation_nodes(raw.get("foundation_node_ids"), warnings)

    fact_map: dict[str, str] = {}
    facts: list[dict[str, Any]] = []
    for idx, item in enumerate(raw.get("fact_ledger") or []):
        if not isinstance(item, dict):
            continue
        local_id = str(item.get("fact_id") or f"fact-{idx + 1}")
        fact_id = _uuid(source_key, "fact", local_id)
        fact_map[local_id] = fact_id
        stage = str(item.get("stage") or "imported")
        fact_class = str(item.get("fact_class") or "").casefold()
        introduced_via = "initial_case"
        if stage.startswith("disclosure") or "progressive" in fact_class:
            introduced_via = "progressive_disclosure"
        elif "response" in stage or fact_class == "clinician_hypothesis":
            introduced_via = "clinician_entry"
        elif "mentor" in stage or fact_class in {"educational_inference", "ai_inference"}:
            introduced_via = "evidence_review"
        facts.append(
            {
                "fact_id": fact_id,
                "statement": str(item.get("content") or item.get("statement") or "Imported learning fact").strip(),
                "fact_scope": FACT_SCOPE_MAP.get(fact_class, "ai_inference"),
                "introduced_via": introduced_via,
                "authoritative_for_patient": False,
                "source": "rich_challenge_export_v1",
                "certainty": None,
                "introduced_at_stage": stage,
                "status": "active",
                "supersedes_fact_id": None,
            }
        )

    response_map: list[str] = []
    responses: list[dict[str, Any]] = []
    for idx, item in enumerate(raw.get("clinician_reasoning_responses") or []):
        if not isinstance(item, dict):
            continue
        local_stage = str(item.get("stage") or "followup")
        stage = "initial" if "initial" in local_stage else "final" if "final" in local_stage else "followup"
        summary = item.get("text")
        if not summary:
            values = item.get("summary") or []
            if isinstance(values, list):
                summary = "\n".join(f"- {str(value).strip()}" for value in values if str(value).strip())
                warnings.append("source_reasoning_was_summary_not_verbatim")
        response_id = _uuid(source_key, "response", idx + 1)
        response_map.append(response_id)
        responses.append(
            {
                "response_id": response_id,
                "stage": stage,
                "text": str(summary or "Imported clinician reasoning summary").strip(),
                "confidence_percent": item.get("confidence") if isinstance(item.get("confidence"), int) else None,
                "created_at": None,
            }
        )

    disclosures: list[dict[str, Any]] = []
    for idx, item in enumerate(raw.get("progressive_disclosures") or []):
        if not isinstance(item, dict):
            continue
        sequence = int(item.get("sequence") or idx + 1)
        disclosure_fact_ids: list[str] = []
        for raw_fact in raw.get("fact_ledger") or []:
            if not isinstance(raw_fact, dict):
                continue
            if str(raw_fact.get("stage") or "") == f"disclosure_{sequence}":
                local_id = str(raw_fact.get("fact_id") or "")
                if local_id in fact_map:
                    disclosure_fact_ids.append(fact_map[local_id])
        narrative_parts = [
            str(item.get("consequential_new_datum") or "").strip(),
            str(item.get("why_it_matters") or "").strip(),
        ]
        disclosures.append(
            {
                "disclosure_id": _uuid(source_key, "disclosure", sequence),
                "sequence": sequence,
                "label": f"Disclosure {sequence}",
                "narrative": "\n\n".join(part for part in narrative_parts if part) or None,
                "fact_ids": disclosure_fact_ids,
                "released_after_response_id": response_map[sequence - 1] if sequence - 1 < len(response_map) else None,
            }
        )

    references: list[dict[str, Any]] = []
    reference_by_url: dict[str, str] = {}
    for idx, item in enumerate(raw.get("evidence") or []):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip() or None
        reference_id = _uuid(source_key, "reference", idx + 1)
        if url:
            reference_by_url[url] = reference_id
        references.append(
            {
                "reference_id": reference_id,
                "title": str(item.get("citation") or item.get("title") or f"Imported reference {idx + 1}").strip(),
                "evidence_type": _evidence_type(item),
                "framework_or_guideline": str(item.get("source_type") or "").strip() or None,
                "pmid": _pmid(item),
                "doi": str(item.get("doi") or "").strip() or None,
                "url": url,
                "relation": "contextualizes",
                "verification_state": "unverified",
                "verification_note": None,
            }
        )

    observations: list[dict[str, Any]] = []
    mentor = raw.get("mentor_observations") if isinstance(raw.get("mentor_observations"), dict) else {}

    def add_observation(category: str, statement: str, local_id: object, gaps: list[str] | None = None) -> None:
        text = statement.strip()
        if not text:
            return
        observations.append(
            {
                "observation_id": _uuid(source_key, "observation", category, local_id),
                "category": category,
                "statement": text,
                "importance": "moderate",
                "linked_fact_ids": [],
                "linked_reference_ids": [],
                "gap_classes": list(dict.fromkeys(gaps or [])),
                "clinician_disposition": "pending",
                "clinician_modified_statement": None,
                "disposition_note": None,
            }
        )

    for idx, text in enumerate(mentor.get("strengths") or []):
        add_observation("strength", str(text), idx)
    for idx, item in enumerate(mentor.get("clear_errors") or []):
        statement = (
            f"{item.get('issue', '')} Correction: {item.get('correction', '')}"
            if isinstance(item, dict)
            else str(item)
        )
        add_observation("clear_error", statement, idx, ["knowledge"])
    for idx, item in enumerate(mentor.get("defensible_disagreements") or []):
        statement = (
            f"{item.get('issue', '')} Assessment: {item.get('assessment', '')}"
            if isinstance(item, dict)
            else str(item)
        )
        add_observation("defensible_disagreement", statement, idx, ["reasoning"])
    for category, key, default_gap in (
        ("evidence_gap", "evidence_gaps", "reasoning"),
        ("blind_spot", "blind_spots", "knowledge"),
        ("reasoning_pattern", "reasoning_patterns", "reasoning"),
    ):
        for idx, text in enumerate(mentor.get(key) or []):
            add_observation(category, str(text), idx, [default_gap] if category != "reasoning_pattern" else [])
    if mentor.get("clinical_insight"):
        add_observation("clinical_insight", str(mentor["clinical_insight"]), 0)

    canonical_gap_classes: list[str] = []
    for idx, item in enumerate(raw.get("gap_classes") or []):
        if isinstance(item, dict):
            gap_class = _gap_class(item.get("class"))
            statement = str(item.get("gap") or "").strip()
        else:
            gap_class = _gap_class(item)
            statement = str(item).strip()
        if gap_class not in canonical_gap_classes:
            canonical_gap_classes.append(gap_class)
        if statement:
            add_observation("missed_opportunity", f"Needs reinforcement: {statement}", f"gap-{idx}", [gap_class])

    actions: list[dict[str, Any]] = []
    for idx, item in enumerate(raw.get("learning_actions") or []):
        if not isinstance(item, dict):
            continue
        local_id = str(item.get("action_id") or f"action-{idx + 1}")
        source_type = str(item.get("source_type") or "").casefold()
        action_type = "targeted_reading"
        if "practice" in source_type or "expert" in source_type:
            action_type = "deliberate_practice"
        title = str(item.get("action") or item.get("title") or "Imported learning action").strip()
        url = str(item.get("url") or "").strip()
        reference_ids = [reference_by_url[url]] if url and url in reference_by_url else []
        actions.append(
            {
                "action_id": _uuid(source_key, "action", local_id),
                "action_type": action_type,
                "title": title,
                "rationale": str(item.get("reason") or "").strip() or None,
                "foundation_node_ids": [],
                "reference_ids": reference_ids,
                "due_on": None,
                "status": "planned",
                "completed_at": None,
            }
        )

    initial_case = raw.get("initial_case") if isinstance(raw.get("initial_case"), dict) else {}
    initial_parts: list[str] = []
    if isinstance(initial_case, dict):
        if initial_case.get("prompt"):
            initial_parts.append(str(initial_case["prompt"]).strip())
        if isinstance(initial_case.get("facts"), list):
            initial_parts.extend(f"- {str(value).strip()}" for value in initial_case["facts"] if str(value).strip())
    else:
        initial_parts.append(str(raw.get("initial_case") or "").strip())

    final_decision = None
    for item in reversed(raw.get("clinician_reasoning_responses") or []):
        if isinstance(item, dict) and item.get("final_decision"):
            final_decision = str(item["final_decision"]).strip()
            break

    payload = {
        "challenge_id": challenge_id,
        "schema_version": "clinical_learning_challenge_v1",
        "revision": int(raw.get("revision") or 1),
        "supersedes_revision": raw.get("supersedes_revision"),
        "module": "osteoporosis",
        "created_at": _parse_datetime(session.get("completed_on") or session.get("started_on") or raw.get("created_at")),
        "title": str(session.get("title") or raw.get("title") or "Imported Clinical Challenge").strip(),
        "challenge_mode": "synthetic",
        "topics": [str(value).strip() for value in raw.get("topic_tags") or raw.get("topics") or ["osteoporosis"] if str(value).strip()],
        "foundation_node_ids": nodes,
        "difficulty_label": str(session.get("format") or raw.get("difficulty_label") or "").strip() or None,
        "initial_case": "\n".join(initial_parts).strip() or "Synthetic imported teaching case.",
        "fact_ledger": facts,
        "progressive_disclosures": disclosures,
        "reasoning_responses": responses or [
            {
                "response_id": _uuid(source_key, "response", "fallback"),
                "stage": "initial",
                "text": "No clinician reasoning response was supplied in the external export.",
                "confidence_percent": None,
                "created_at": None,
            }
        ],
        "final_clinician_decision": final_decision,
        "observations": observations,
        "references": references,
        "gap_classes": canonical_gap_classes,
        "learning_actions": actions,
        "next_challenge_topic": None,
        "spaced_repetition_due": None,
        "linked_signal_ids": [],
        "record_review_state": "imported_pending_review",
        "reviewed_at": None,
        "privacy": {
            "contains_direct_identifiers": False,
            "deidentification_attested": True,
            "source_case_deidentified": None,
        },
    }
    validated = validate_challenge_payload(payload)
    normalized = normalize_import_preview(validated)
    normalized["privacy"]["deidentification_attested"] = False
    return normalized, warnings


def adapt_learning_episode(
    raw_episode: Any,
    *,
    source_event_id: str | None = None,
    supplied_loop_plan: dict[str, Any] | None = None,
    supplied_resources: list[dict[str, Any]] | None = None,
) -> AdaptedLearningEpisode:
    if not isinstance(raw_episode, dict):
        raise LearningContractError([ContractIssue("learning_episode_object_required", "episode")])
    structured_findings = find_forbidden_structured_fields(raw_episode)
    if structured_findings:
        raise LearningContractError(
            [ContractIssue(item.code, item.path) for item in structured_findings]
        )

    derived_source_event_id = source_event_id
    if derived_source_event_id is None:
        source_key = str(raw_episode.get("challenge_id") or raw_episode.get("title") or repr(sorted(raw_episode.keys())))
        derived_source_event_id = str(uuid5(NAMESPACE_URL, f"clinical-learning-source-event|{source_key}"))
    try:
        UUID(str(derived_source_event_id))
    except ValueError:
        raise LearningContractError([ContractIssue("source_event_id_must_be_uuid", "source_event_id")]) from None

    if raw_episode.get("schema_version") == "clinical_learning_challenge_v1":
        source_format = "canonical_challenge_v1"
        candidate = copy.deepcopy(raw_episode)
        candidate.setdefault("privacy", {})["contains_direct_identifiers"] = False
        # External pending imports do not self-certify clinician review. We set
        # attestation true only for the no-write canonical validator, then reset
        # the normalized pending candidate to false before any persistence.
        candidate["privacy"]["deidentification_attested"] = True
        challenge = validate_challenge_payload(candidate)
        normalized = normalize_import_preview(challenge)
        normalized["privacy"]["deidentification_attested"] = False
        warnings: list[str] = []
    elif raw_episode.get("schema_type") == "ClinicalLearningChallengeV1" and str(raw_episode.get("schema_version")) in {"1.0", "1"}:
        source_format = "rich_challenge_export_v1"
        normalized, warnings = _rich_to_canonical(raw_episode, str(derived_source_event_id))
    else:
        raise LearningContractError([ContractIssue("unsupported_learning_episode_format", "episode")])

    loop_plan = copy.deepcopy(supplied_loop_plan) if supplied_loop_plan else build_learning_loop_plan(
        normalized,
        source_payload=raw_episode if source_format == "rich_challenge_export_v1" else None,
    )
    resources = resource_candidates_from_source(
        normalized,
        source_payload=raw_episode if source_format == "rich_challenge_export_v1" else None,
        supplied_resources=supplied_resources,
    )
    persistable = {
        "normalized_challenge": normalized,
        "loop_plan": loop_plan,
        "resources": resources,
        "warnings": warnings,
    }
    findings = scan_persistable_strings(persistable)
    if findings:
        raise LearningContractError([ContractIssue(item.code, item.path) for item in findings])

    return AdaptedLearningEpisode(
        source_event_id=str(derived_source_event_id),
        source_format=source_format,
        normalized_challenge=normalized,
        loop_plan=loop_plan,
        resources=resources,
        warnings=list(dict.fromkeys(warnings)),
    )
