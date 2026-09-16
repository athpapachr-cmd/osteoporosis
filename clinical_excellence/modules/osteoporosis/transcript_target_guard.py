from __future__ import annotations

from clinical_excellence.core.transcript_contracts import (
    BooleanValueV1,
    CodeValueV1,
    IntegerValueV1,
    NumberValueV1,
    ProviderCandidateV1,
    QuantityValueV1,
    TargetMappingV1,
)

from .transcript_targets import map_candidate as _base_map_candidate

EPISODE_STATUSES = {"planned", "active", "completed", "stopped", "holiday", "unknown"}
ADMINISTRATION_STATUSES = {"done", "due", "overdue", "missed", "planned", "not_applicable"}
FRACTURE_SITES = {"vertebral", "hip", "distal_radius", "proximal_humerus", "pelvis", "other"}
FRAX_TOOLS = {"frax", "fraxplus", "other"}
RISK_CATEGORIES = {"low", "intermediate", "high", "very_high", "uncertain", "not_applicable"}
VFA_INDICATION_CODES = {"yes", "no", "uncertain"}
VFA_ACTIONS = {"performed", "already_available_reviewed", "arranged", "reasoned_not_done", "missed", "not_applicable"}
VFA_MODALITIES = {"VFA", "spine_xray", "CT", "MRI", "other"}

RUNTIME_RANGES: dict[str, tuple[float, float]] = {
    "anthropometrics.weight": (20.0, 300.0),
    "anthropometrics.current_height": (100.0, 220.0),
    "frax.mof_percent": (0.0, 100.0),
    "frax.hip_percent": (0.0, 100.0),
    "dxa.spine_bmd": (0.1, 3.0),
    "dxa.total_hip_bmd": (0.1, 3.0),
    "dxa.femoral_neck_bmd": (0.1, 3.0),
    "dxa.spine_t_score": (-8.0, 5.0),
    "dxa.total_hip_t_score": (-8.0, 5.0),
    "dxa.femoral_neck_t_score": (-8.0, 5.0),
    "risk.falls_last_12_months": (0.0, 50.0),
    "risk.cfs_score": (1.0, 9.0),
    "treatment.duration_years": (0.0, 50.0),
}


def _component(candidate: ProviderCandidateV1, key: str):
    return next((item for item in candidate.components if item.concept_key == key), None)


def _ambiguous(mapping: TargetMappingV1, reason: str) -> TargetMappingV1:
    return TargetMappingV1(
        component_keys=mapping.component_keys,
        target_path=mapping.target_path,
        status="ambiguous",
        reason_code=reason,
        proposed_value=mapping.proposed_value,
    )


def _mapped_value(mapping: TargetMappingV1, value) -> TargetMappingV1:
    return TargetMappingV1(
        component_keys=mapping.component_keys,
        target_path=mapping.target_path,
        status="mapped",
        reason_code=mapping.reason_code,
        proposed_value=value,
    )


def _guard_code(mapping: TargetMappingV1, component, allowed: set[str], reason: str) -> TargetMappingV1:
    if not isinstance(component.value, CodeValueV1) or component.value.code not in allowed:
        return _ambiguous(mapping, reason)
    return mapping


def _numeric_value(component) -> float | None:
    if isinstance(component.value, (NumberValueV1, IntegerValueV1, QuantityValueV1)):
        return float(component.value.value)
    return None


def _guard_range(mapping: TargetMappingV1, component, key: str) -> TargetMappingV1:
    value = _numeric_value(component)
    if value is None:
        return _ambiguous(mapping, "TYPE_MISMATCH")
    lower, upper = RUNTIME_RANGES[key]
    if value < lower or value > upper:
        return _ambiguous(mapping, "OUT_OF_RUNTIME_RANGE")
    return mapping


def map_candidate(candidate: ProviderCandidateV1) -> list[TargetMappingV1]:
    """Apply exact runtime enum/type/range guards after the base deterministic mapper.

    Provider semantic output is untrusted. A value is not considered mapped merely
    because its concept key has a runtime destination; it must also fit the exact
    value and semantic contract of that destination.
    """

    mappings = _base_map_candidate(candidate)
    guarded: list[TargetMappingV1] = []

    for mapping in mappings:
        if mapping.status != "mapped" or len(mapping.component_keys) != 1:
            guarded.append(mapping)
            continue

        key = mapping.component_keys[0]
        component = _component(candidate, key)
        if component is None:
            guarded.append(_ambiguous(mapping, "MISSING_COMPONENT"))
            continue

        if key == "fracture.site":
            guarded.append(_guard_code(mapping, component, FRACTURE_SITES, "UNSUPPORTED_FRACTURE_SITE"))
            continue

        if key == "frax.tool_name":
            guarded.append(_guard_code(mapping, component, FRAX_TOOLS, "UNSUPPORTED_FRAX_TOOL"))
            continue

        if key == "risk.resulting_category":
            guarded.append(_guard_code(mapping, component, RISK_CATEGORIES, "UNSUPPORTED_RISK_CATEGORY"))
            continue

        if key == "vfa.indicated":
            if isinstance(component.value, BooleanValueV1):
                guarded.append(_mapped_value(mapping, "yes" if component.value.value else "no"))
            elif isinstance(component.value, CodeValueV1) and component.value.code in VFA_INDICATION_CODES:
                guarded.append(mapping)
            else:
                guarded.append(_ambiguous(mapping, "UNSUPPORTED_VFA_INDICATION"))
            continue

        if key == "vfa.action":
            guarded.append(_guard_code(mapping, component, VFA_ACTIONS, "UNSUPPORTED_VFA_ACTION"))
            continue

        if key == "vfa.modality":
            guarded.append(_guard_code(mapping, component, VFA_MODALITIES, "UNSUPPORTED_VFA_MODALITY"))
            continue

        if key == "treatment.status":
            guarded.append(_guard_code(mapping, component, EPISODE_STATUSES, "UNSUPPORTED_TREATMENT_STATUS"))
            continue

        if key == "administration.status":
            guarded.append(_guard_code(mapping, component, ADMINISTRATION_STATUSES, "UNSUPPORTED_ADMINISTRATION_STATUS"))
            continue

        if key in {"frax.mof_percent", "frax.hip_percent"}:
            if candidate.semantic_type != "objective_result":
                guarded.append(_ambiguous(mapping, "OBJECTIVE_RESULT_REQUIRED"))
                continue
            if not isinstance(component.value, NumberValueV1):
                guarded.append(_ambiguous(mapping, "TYPE_MISMATCH"))
                continue
            guarded.append(_guard_range(mapping, component, key))
            continue

        if key in {"risk.falls_last_12_months", "risk.cfs_score"}:
            if not isinstance(component.value, IntegerValueV1):
                guarded.append(_ambiguous(mapping, "TYPE_MISMATCH"))
                continue
            guarded.append(_guard_range(mapping, component, key))
            continue

        if key in {
            "anthropometrics.weight",
            "anthropometrics.current_height",
            "dxa.spine_bmd",
            "dxa.total_hip_bmd",
            "dxa.femoral_neck_bmd",
            "dxa.spine_t_score",
            "dxa.total_hip_t_score",
            "dxa.femoral_neck_t_score",
        }:
            guarded.append(_guard_range(mapping, component, key))
            continue

        if key == "treatment.duration_years":
            if not isinstance(component.value, (NumberValueV1, IntegerValueV1)):
                guarded.append(_ambiguous(mapping, "TYPE_MISMATCH"))
                continue
            guarded.append(_guard_range(mapping, component, key))
            continue

        guarded.append(mapping)

    return guarded


__all__ = [
    "map_candidate",
    "EPISODE_STATUSES",
    "ADMINISTRATION_STATUSES",
    "FRACTURE_SITES",
    "FRAX_TOOLS",
    "RISK_CATEGORIES",
    "VFA_INDICATION_CODES",
    "VFA_ACTIONS",
    "VFA_MODALITIES",
    "RUNTIME_RANGES",
]
