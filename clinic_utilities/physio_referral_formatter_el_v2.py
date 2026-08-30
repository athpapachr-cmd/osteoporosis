from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Tuple

from clinic_utilities.physio_clinical_composition import CU1ClinicalContextComposer
from clinic_utilities.physio_referral_formatter_el import CU1GreekReferralFormatter as _BaseGreekFormatter
from clinic_utilities.physio_referral_runtime import CU1ContractBundle, CU1ContractError, _normalize_whitespace
from clinic_utilities.physio_rich_referral import CU1RichReferralRenderer


class CU1GreekReferralFormatter(_BaseGreekFormatter):
    """Greek formatter v2 with shared data-driven rich-referral rendering when configured."""

    def __init__(self, bundle: CU1ContractBundle):
        super().__init__(bundle)
        base_language = copy.deepcopy(bundle.artifacts.get("referral_language_el"))
        corrections = bundle.artifacts.get("referral_language_el_corrections")
        if not isinstance(base_language, dict) or not isinstance(corrections, Mapping):
            raise CU1ContractError("CU-1 Greek referral language composition artifacts are missing")
        for section, values in corrections.items():
            if section in {"version", "language", "status"} or not isinstance(values, Mapping):
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
        self.rich_renderer = CU1RichReferralRenderer(bundle.root)
        self.clinical_composer = CU1ClinicalContextComposer(bundle, self.rich_renderer)

    def _route_label(self, profile_id: str, route_id: str) -> str:
        profile = self.explicit_route_labels.get(profile_id)
        label = profile.get(route_id) if isinstance(profile, Mapping) else None
        if not isinstance(label, str) or not label.strip():
            raise CU1ContractError(f"Missing explicit Greek route label for {profile_id}.{route_id}")
        label = _normalize_whitespace(label)
        if not self._is_greek_clinician_phrase(label):
            raise CU1ContractError(f"Invalid Greek route label for {profile_id}.{route_id}: {label}")
        return label

    def _problem_label(self, problem: Mapping[str, Any], *, include_subtype: bool) -> str:
        profile_id = str(problem.get("profile_id") or "")
        route_id = str(problem.get("route_id") or "")
        wording_mode = str(problem.get("wording_mode") or "")
        route_spec = self.rich_renderer.routes.get(route_id)
        label: Optional[str] = None
        if isinstance(route_spec, Mapping):
            profile_ids = route_spec.get("profile_ids") or []
            labels = route_spec.get("problem_label_el_by_wording_mode")
            if (not profile_ids or profile_id in profile_ids) and isinstance(labels, Mapping):
                candidate = labels.get(wording_mode)
                if isinstance(candidate, str) and candidate.strip():
                    candidate = _normalize_whitespace(candidate)
                    if not self._is_greek_clinician_phrase(candidate):
                        raise CU1ContractError(
                            f"Invalid wording-aware Greek problem label for {profile_id}.{route_id}.{wording_mode}: {candidate}"
                        )
                    label = candidate

        if not label:
            return super()._problem_label(problem, include_subtype=include_subtype)

        laterality = self._optional_label("laterality", str(problem.get("laterality") or ""))
        if laterality:
            label = f"{label} ({laterality})"
        subtype = problem.get("subtype_id_optional")
        if include_subtype and subtype:
            subtype_label = self._optional_label("route_detail_labels", str(subtype))
            if subtype_label:
                label += f" — {subtype_label}"
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

    def _format_short(self, draft: Mapping[str, Any]) -> str:
        route = self._rich_route_identity(draft)
        if route:
            supported = self.rich_renderer.supports(
                profile_id=route[0], route_id=route[1], subtype_id=route[2], context=route[3]
            )
            if supported:
                return self.rich_renderer.render_short(
                    profile_id=route[0],
                    route_id=route[1],
                    subtype_id=route[2],
                    clinical_context=self._rich_clinical_context(draft, detailed=False, route=route),
                    context=route[3],
                )
            self._assert_context_gated_route_may_not_fallback(route)
        return super()._format_short(draft)

    def _format_detailed(self, draft: Mapping[str, Any]) -> str:
        route = self._rich_route_identity(draft)
        if route:
            supported = self.rich_renderer.supports(
                profile_id=route[0], route_id=route[1], subtype_id=route[2], context=route[3]
            )
            if supported:
                return self.rich_renderer.render_detailed(
                    profile_id=route[0],
                    route_id=route[1],
                    subtype_id=route[2],
                    clinical_context=self._rich_clinical_context(draft, detailed=True, route=route),
                    context=route[3],
                )
            self._assert_context_gated_route_may_not_fallback(route)
        return super()._format_detailed(draft)

    def _assert_context_gated_route_may_not_fallback(
        self,
        route: Tuple[str, str, Optional[str], Mapping[str, Any]],
    ) -> None:
        state = self.rich_renderer.rollout_state(profile_id=route[0], route_id=route[1])
        if state == "context_gated":
            raise CU1ContractError(
                f"Context-gated referral {route[0]}.{route[1]} has no applicable reviewed rich variant; "
                "legacy formatter fallback is forbidden"
            )

    @staticmethod
    def _rich_route_identity(draft: Mapping[str, Any]) -> Optional[Tuple[str, str, Optional[str], Mapping[str, Any]]]:
        problem = draft.get("primary_problem")
        if not isinstance(problem, Mapping):
            return None
        profile_id = problem.get("profile_id")
        route_id = problem.get("route_id")
        subtype_id = problem.get("subtype_id_optional")
        if not isinstance(profile_id, str) or not profile_id or not isinstance(route_id, str) or not route_id:
            return None
        raw_context = problem.get("context")
        context: Dict[str, Any] = copy.deepcopy(dict(raw_context)) if isinstance(raw_context, Mapping) else {}
        wording_mode = problem.get("wording_mode")
        if isinstance(wording_mode, str) and wording_mode:
            context["__wording_mode"] = wording_mode
        return profile_id, route_id, subtype_id if isinstance(subtype_id, str) and subtype_id else None, context

    def _rich_clinical_context(
        self,
        draft: Mapping[str, Any],
        *,
        detailed: bool,
        route: Tuple[str, str, Optional[str], Mapping[str, Any]],
    ) -> List[str]:
        problem = draft.get("primary_problem", {}) if isinstance(draft.get("primary_problem"), Mapping) else {}
        problem_label = self._problem_label(problem, include_subtype=detailed)

        clinical = self.clinical_composer.compose(
            draft,
            detailed=detailed,
            route=route,
            fallback_problem_label=problem_label,
        )

        work_context = self._explicit_work_or_sport_context(draft)
        restrictions = self._restriction_labels(draft, detailed=detailed)
        precautions = self._optional_selection_labels(draft, "precautions", "precautions")

        if work_context:
            clinical.append(work_context.rstrip("."))
        if restrictions or precautions:
            clinical.append(f"Περιορισμοί/προφυλάξεις: {self._join_greek(restrictions + precautions)}")
        if detailed:
            for sentence in self._detailed_context_sentences(problem):
                clean = sentence.strip().rstrip(".")
                if clean:
                    clinical.append(clean)
        return clinical

    @staticmethod
    def _explicit_work_or_sport_context(draft: Mapping[str, Any]) -> Optional[str]:
        patient_context = draft.get("patient_context")
        if not isinstance(patient_context, Mapping):
            return None
        value = patient_context.get("sport_or_work_demand_optional")
        if not isinstance(value, str) or not value.strip() or "_" in value:
            return None
        value = _normalize_whitespace(value)
        return value if len(value) <= 120 else value[:117].rstrip() + "…"


__all__ = ["CU1GreekReferralFormatter"]
