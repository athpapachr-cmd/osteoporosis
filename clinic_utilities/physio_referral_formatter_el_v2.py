from __future__ import annotations

import copy
from typing import Any, Dict, Mapping

from clinic_utilities.physio_referral_formatter_el import CU1GreekReferralFormatter as _BaseGreekFormatter
from clinic_utilities.physio_referral_runtime import CU1ContractBundle, CU1ContractError, _normalize_whitespace


class CU1GreekReferralFormatter(_BaseGreekFormatter):
    """Formatter v2 using explicit route labels and a deterministic language overlay."""

    def __init__(self, bundle: CU1ContractBundle):
        super().__init__(bundle)

        base_language = copy.deepcopy(bundle.artifacts.get("referral_language_el"))
        corrections = bundle.artifacts.get("referral_language_el_corrections")
        if not isinstance(base_language, dict) or not isinstance(corrections, Mapping):
            raise CU1ContractError("CU-1 Greek referral language composition artifacts are missing")

        for section, values in corrections.items():
            if section in {"version", "language", "status"}:
                continue
            if not isinstance(values, Mapping):
                continue
            target = base_language.setdefault(str(section), {})
            if not isinstance(target, dict):
                raise CU1ContractError(f"Invalid Greek language section: {section}")
            for key, value in values.items():
                target[str(key)] = copy.deepcopy(value)
        self.language = base_language

        route_artifact = bundle.artifacts.get("route_labels_el")
        routes = route_artifact.get("routes") if isinstance(route_artifact, Mapping) else None
        if not isinstance(routes, Mapping):
            raise CU1ContractError("CU-1 explicit Greek route-label artifact is missing")
        self.explicit_route_labels: Mapping[str, Any] = routes

    def _route_label(self, profile_id: str, route_id: str) -> str:
        profile = self.explicit_route_labels.get(profile_id)
        label = profile.get(route_id) if isinstance(profile, Mapping) else None
        if not isinstance(label, str) or not label.strip():
            raise CU1ContractError(f"Missing explicit Greek route label for {profile_id}.{route_id}")
        label = _normalize_whitespace(label)
        if not self._is_greek_clinician_phrase(label):
            raise CU1ContractError(f"Invalid Greek route label for {profile_id}.{route_id}: {label}")
        return label

    def contract_route_labels(self) -> Dict[str, Dict[str, str]]:
        result: Dict[str, Dict[str, str]] = {}
        profiles = self.bundle.registry.get("profiles", {})
        if not isinstance(profiles, Mapping):
            raise CU1ContractError("CU-1 route registry is missing")
        for profile_id, profile_spec in profiles.items():
            if not isinstance(profile_spec, Mapping):
                continue
            result[str(profile_id)] = {}
            for route_id in (profile_spec.get("routes") or {}):
                result[str(profile_id)][str(route_id)] = self._route_label(str(profile_id), str(route_id))
        return result


__all__ = ["CU1GreekReferralFormatter"]
