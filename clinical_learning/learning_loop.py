from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid5


LOOP_NAMESPACE = UUID("6c5c73aa-fc9f-4fd5-8d40-7c7d0d57d1b4")
DEFAULT_SPACING_DAYS = (3, 7, 14, 30)


def stable_learning_uuid(*parts: object) -> str:
    material = "|".join(str(part) for part in parts)
    return str(uuid5(LOOP_NAMESPACE, material))


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _gap_class(value: str | None) -> str:
    folded = str(value or "").casefold()
    if "communication" in folded or "system" in folded:
        return "communication_system"
    if "execution" in folded or "workflow" in folded or "safety" in folded:
        return "execution"
    if "reason" in folded or "decision" in folded or "evidence" in folded:
        return "reasoning"
    return "knowledge"


def _nodes_for_text(text: str, available_nodes: list[str]) -> list[str]:
    folded = text.casefold()
    rules: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
        (("ckd", "renal", "pth", "phosphate", "hyperparathy"), ("ost.foundation.secondary_osteoporosis", "ost.foundation.treatment_safety_transitions")),
        (("denosumab", "rebound"), ("ost.foundation.denosumab_rebound", "ost.foundation.antiresorptive_pharmacology", "ost.foundation.treatment_safety_transitions")),
        (("bisphosph", "zoled"), ("ost.foundation.antiresorptive_pharmacology", "ost.foundation.treatment_safety_transitions")),
        (("teriparat", "romosoz", "anabolic"), ("ost.foundation.anabolic_pharmacology", "ost.foundation.treatment_safety_transitions")),
        (("evidence", "indirect", "subgroup", "random", "guideline", "directness"), ("ost.foundation.evidence_appraisal",)),
        (("ctx", "turnover", "bone-specific", "alkaline phosphatase", "btm"), ("ost.foundation.btms", "ost.foundation.bone_remodeling")),
        (("sequenc", "transition", "exit", "consolid"), ("ost.foundation.sequencing", "ost.foundation.treatment_safety_transitions")),
        (("dxa", "lsc", "t-score"), ("ost.foundation.dxa_lsc",)),
        (("vfa", "vertebral"), ("ost.foundation.vfa",)),
        (("frax", "absolute risk", "fracture risk"), ("ost.foundation.fracture_risk_frax",)),
        (("secondary osteoporosis",), ("ost.foundation.secondary_osteoporosis",)),
    ]
    chosen: list[str] = []
    for keywords, nodes in rules:
        if any(keyword in folded for keyword in keywords):
            for node in nodes:
                if node in available_nodes and node not in chosen:
                    chosen.append(node)
    if not chosen:
        chosen = list(available_nodes[:2])
    return chosen


def _source_gap_objectives(
    *,
    challenge_id: str,
    challenge_payload: dict[str, Any],
    source_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    available_nodes = list(challenge_payload.get("foundation_node_ids") or [])
    objectives: list[dict[str, Any]] = []

    if source_payload:
        for idx, item in enumerate(source_payload.get("gap_classes") or []):
            if not isinstance(item, dict):
                continue
            title = str(item.get("gap") or "").strip()
            if not title:
                continue
            gap_class = _gap_class(str(item.get("class") or ""))
            objectives.append(
                {
                    "objective_id": stable_learning_uuid(challenge_id, "objective", "source-gap", idx, title),
                    "title": title,
                    "rationale": "Derived from the reviewed Challenge gap analysis.",
                    "foundation_node_ids": _nodes_for_text(title, available_nodes),
                    "gap_classes": [gap_class],
                    "source_observation_ids": [],
                }
            )

    if objectives:
        return objectives

    target_categories = {"clear_error", "evidence_gap", "blind_spot", "missed_opportunity", "uncertainty"}
    for idx, observation in enumerate(challenge_payload.get("observations") or []):
        if observation.get("category") not in target_categories:
            continue
        statement = str(observation.get("statement") or "").strip()
        if not statement:
            continue
        gaps = list(observation.get("gap_classes") or []) or ["reasoning"]
        objectives.append(
            {
                "objective_id": stable_learning_uuid(challenge_id, "objective", observation.get("observation_id") or idx),
                "title": statement,
                "rationale": "Targets a material Challenge observation requiring reinforcement.",
                "foundation_node_ids": _nodes_for_text(statement, available_nodes),
                "gap_classes": list(dict.fromkeys(gaps)),
                "source_observation_ids": [str(observation["observation_id"])] if observation.get("observation_id") else [],
            }
        )

    if not objectives and available_nodes:
        objectives.append(
            {
                "objective_id": stable_learning_uuid(challenge_id, "objective", "foundation-transfer"),
                "title": "Re-apply the Challenge's core reasoning in a new clinical context.",
                "rationale": "Provides transfer evidence rather than relying on recognition of the original case.",
                "foundation_node_ids": available_nodes[:2],
                "gap_classes": ["reasoning"],
                "source_observation_ids": [],
            }
        )
    return objectives


def _bridge_targets(
    challenge_id: str,
    available_nodes: list[str],
    objectives: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    node_set = set(available_nodes)
    patterns: list[tuple[tuple[str, ...], str]] = [
        (
            (
                "ost.foundation.secondary_osteoporosis",
                "ost.foundation.antiresorptive_pharmacology",
                "ost.foundation.treatment_safety_transitions",
            ),
            "Integrate secondary/renal bone-disease reasoning with antiresorptive selection and treatment-safety constraints.",
        ),
        (
            ("ost.foundation.evidence_appraisal", "ost.foundation.antiresorptive_pharmacology"),
            "Bridge evidence directness/calibration with pharmacologic decision-making instead of treating evidence appraisal as an isolated skill.",
        ),
        (
            ("ost.foundation.evidence_appraisal", "ost.foundation.anabolic_pharmacology"),
            "Bridge evidence directness/calibration with anabolic-treatment reasoning and extrapolation limits.",
        ),
        (
            ("ost.foundation.bone_remodeling", "ost.foundation.denosumab_rebound"),
            "Connect remodeling physiology to denosumab rebound/transition reasoning in a novel case.",
        ),
        (
            ("ost.foundation.btms", "ost.foundation.denosumab_rebound"),
            "Connect bone-turnover-marker interpretation with denosumab transition decisions.",
        ),
        (
            ("ost.foundation.sequencing", "ost.foundation.treatment_safety_transitions"),
            "Integrate sequencing strategy with treatment-safety and transition constraints.",
        ),
    ]
    bridges: list[dict[str, Any]] = []
    observation_ids = [
        obs_id
        for objective in objectives
        for obs_id in objective.get("source_observation_ids") or []
    ]
    for idx, (nodes, rationale) in enumerate(patterns):
        if all(node in node_set for node in nodes):
            bridges.append(
                {
                    "bridge_id": stable_learning_uuid(challenge_id, "bridge", idx, *nodes),
                    "foundation_node_ids": list(nodes),
                    "rationale": rationale,
                    "source_observation_ids": list(dict.fromkeys(observation_ids)),
                    "source_action_ids": [],
                    "state": "planned",
                }
            )
        if len(bridges) >= 3:
            break
    if not bridges and len(available_nodes) >= 2 and objectives:
        nodes = available_nodes[:2]
        bridges.append(
            {
                "bridge_id": stable_learning_uuid(challenge_id, "bridge", "fallback", *nodes),
                "foundation_node_ids": nodes,
                "rationale": "Demonstrate joint application of two relevant Foundation concepts in a novel transfer case.",
                "source_observation_ids": list(dict.fromkeys(observation_ids)),
                "source_action_ids": [],
                "state": "planned",
            }
        )
    return bridges


def build_learning_loop_plan(
    challenge_payload: dict[str, Any],
    *,
    source_payload: dict[str, Any] | None = None,
    anchor_date: date | None = None,
) -> dict[str, Any]:
    challenge_id = str(challenge_payload["challenge_id"])
    anchor = anchor_date or _today()
    objectives = _source_gap_objectives(
        challenge_id=challenge_id,
        challenge_payload=challenge_payload,
        source_payload=source_payload,
    )
    available_nodes = list(challenge_payload.get("foundation_node_ids") or [])
    bridges = _bridge_targets(challenge_id, available_nodes, objectives)
    source_prompt = ""
    if source_payload and isinstance(source_payload.get("spaced_repetition"), dict):
        source_prompt = str(source_payload["spaced_repetition"].get("prompt") or "").strip()

    objective_titles = [str(item.get("title") or "").strip() for item in objectives if item.get("title")]
    first_objective = objective_titles[0] if objective_titles else "the key reasoning principle from the Challenge"
    bridge_rationale = bridges[0]["rationale"] if bridges else "Integrate the relevant Foundation concepts in one decision."

    prompts = [
        source_prompt
        or f"Χωρίς να κοιτάξεις το προηγούμενο Challenge, εξήγησε τον βασικό κανόνα και το σκεπτικό για: {first_objective}",
        f"Δώσε δύο κοντινές κλινικές καταστάσεις όπου η απόφαση διαφέρει για το εξής θέμα και εξήγησε ποιο στοιχείο αλλάζει το reasoning: {first_objective}",
        f"Εφάρμοσε σε νέο υποθετικό περιστατικό τα εξής learning targets και αιτιολόγησε την απόφαση: {'; '.join(objective_titles[:2]) or first_objective}",
        f"Bridge transfer: {bridge_rationale} Χρησιμοποίησε νέο περιστατικό και εξήγησε ρητά πώς το ένα concept αλλάζει την εφαρμογή του άλλου.",
    ]
    kinds = ("retrieval", "discrimination", "transfer", "bridge_transfer")
    cycle_id = stable_learning_uuid(challenge_id, "cycle", challenge_payload.get("revision", 1))
    occurrences: list[dict[str, Any]] = []
    for idx, (offset, kind, prompt) in enumerate(zip(DEFAULT_SPACING_DAYS, kinds, prompts), start=1):
        occurrences.append(
            {
                "occurrence_id": stable_learning_uuid(cycle_id, "occurrence", idx),
                "sequence": idx,
                "kind": kind,
                "due_on": (anchor + timedelta(days=offset)).isoformat(),
                "prompt": prompt,
                "target_objective_ids": [item["objective_id"] for item in objectives],
                "target_bridge_ids": [item["bridge_id"] for item in bridges] if kind == "bridge_transfer" else [],
                "target_foundation_node_ids": available_nodes,
                "expected_points": objective_titles,
                "status": "planned",
            }
        )

    return {
        "cycle_id": cycle_id,
        "objectives": objectives,
        "bridge_targets": bridges,
        "consolidation_occurrences": occurrences,
        "default_spacing_days": list(DEFAULT_SPACING_DAYS),
    }


def resource_candidates_from_source(
    challenge_payload: dict[str, Any],
    *,
    source_payload: dict[str, Any] | None = None,
    supplied_resources: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    checked_at = datetime.now(timezone.utc).isoformat()
    challenge_id = str(challenge_payload["challenge_id"])
    available_nodes = list(challenge_payload.get("foundation_node_ids") or [])
    out: list[dict[str, Any]] = []

    for idx, item in enumerate(supplied_resources or []):
        if not isinstance(item, dict) or not item.get("title") or not item.get("url"):
            continue
        resource = dict(item)
        resource.setdefault("recommendation_id", stable_learning_uuid(challenge_id, "resource", "supplied", idx, resource.get("url")))
        resource.setdefault("kind", "other")
        resource.setdefault("provider", "External learning source")
        resource.setdefault("rationale", "Fresh resource supplied for a Challenge learning target.")
        resource.setdefault("foundation_node_ids", [])
        resource.setdefault("gap_classes", [])
        resource.setdefault("checked_at", checked_at)
        resource.setdefault("access_state", "unknown")
        resource.setdefault("status", "active")
        out.append(resource)

    if out or not source_payload:
        return out

    for idx, action in enumerate(source_payload.get("learning_actions") or []):
        if not isinstance(action, dict) or not action.get("url"):
            continue
        source_type = str(action.get("source_type") or "").casefold()
        kind = "guideline" if "guideline" in source_type or "consensus" in source_type else "article"
        title = str(action.get("action") or "Recommended learning resource").strip()
        out.append(
            {
                "recommendation_id": stable_learning_uuid(challenge_id, "resource", idx, action.get("url")),
                "kind": kind,
                "title": title,
                "provider": "Source Challenge recommendation",
                "url": str(action["url"]),
                "rationale": str(action.get("reason") or "Targets a Challenge learning need."),
                "foundation_node_ids": _nodes_for_text(title + " " + str(action.get("reason") or ""), available_nodes),
                "gap_classes": [],
                "checked_at": checked_at,
                "access_state": "unknown",
                "status": "active",
            }
        )
    return out
