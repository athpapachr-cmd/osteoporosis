from __future__ import annotations

from clinical_excellence.core.transcript_contracts import (
    BooleanValueV1,
    CodeValueV1,
    IntegerValueV1,
    NumberValueV1,
    ProviderCandidateV1,
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


def map_candidate(candidate: ProviderCandidateV1) -> list[TargetMappingV1]:
    """Apply exact runtime enum/type guards after the base deterministic mapper.

    Provider semantic output is untrusted. A value is not considered mapped merely
    because its concept key has a runtime destination; it must also fit the exact
    value contract of that destination.
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

        if key == "treatment.duration_years":
            if not isinstance(component.value, (NumberValueV1, IntegerValueV1)):
                guarded.append(_ambiguous(mapping, "TYPE_MISMATCH"))
                continue
            value = float(component.value.value)
            if value < 0 or value > 50:
                guarded.append(_ambiguous(mapping, "OUT_OF_RUNTIME_RANGE"))
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
]
