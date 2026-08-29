from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from clinic_utilities.physio_referral_runtime import CU1ContractError, _repo_root


_CONTENT_PATH = "clinic_utilities/contracts/cu1_rich_referral_content_el_v1.yaml"
_MIGRATION_PATH = "clinic_utilities/contracts/cu1_rich_referral_migration_matrix_v1.yaml"
_RICH_READY = "rich_ready"


def _load_yaml_mapping(root: Path, relative_path: str, *, label: str, require_runtime_authorized: bool = True) -> Dict[str, Any]:
    path = root / relative_path
    if not path.exists():
        raise CU1ContractError(f"Missing CU-1 {label} artifact: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - contract boundary
        raise CU1ContractError(f"Unable to parse CU-1 {label} artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise CU1ContractError(f"CU-1 {label} must be a mapping")
    if require_runtime_authorized and payload.get("runtime_authorized") is not True:
        raise CU1ContractError(f"CU-1 {label} is not runtime-authorized")
    return payload


def _load_content(root: Path) -> Dict[str, Any]:
    payload = _load_yaml_mapping(root, _CONTENT_PATH, label="rich-referral content")
    embedded = payload.get("routes") or {}
    if not isinstance(embedded, Mapping):
        raise CU1ContractError("CU-1 rich-referral route map is invalid")
    routes: Dict[str, Any] = copy.deepcopy(dict(embedded))
    shards = payload.get("route_shards") or []
    if not isinstance(shards, list):
        raise CU1ContractError("CU-1 rich-referral route_shards must be a list")
    for relative_path in shards:
        if not isinstance(relative_path, str) or not relative_path:
            raise CU1ContractError("CU-1 rich-referral shard path is invalid")
        shard = _load_yaml_mapping(root, relative_path, label="rich-referral shard")
        shard_routes = shard.get("routes") or {}
        if not isinstance(shard_routes, Mapping):
            raise CU1ContractError(f"CU-1 rich-referral shard route map is invalid: {relative_path}")
        for route_id, spec in shard_routes.items():
            route_key = str(route_id)
            if route_key in routes:
                raise CU1ContractError(f"Duplicate rich-referral route across content shards: {route_key}")
            if not isinstance(spec, Mapping):
                raise CU1ContractError(f"Invalid rich-referral route spec in {relative_path}: {route_key}")
            routes[route_key] = copy.deepcopy(dict(spec))
    payload = copy.deepcopy(payload)
    payload["routes"] = routes
    return payload


def _load_migration(root: Path) -> Dict[str, Any]:
    payload = _load_yaml_mapping(root, _MIGRATION_PATH, label="rich-referral migration matrix")
    if not isinstance(payload.get("profiles"), Mapping):
        raise CU1ContractError("CU-1 rich-referral migration profile map is missing")
    return payload


def _clean_phrase(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _clean_lines(values: Any) -> List[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [phrase for phrase in (_clean_phrase(item) for item in values) if phrase]


def _sentence(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value.endswith((".", ";", ":", "!", "?")):
        return value
    return value + "."


def _sentences(values: Iterable[str]) -> str:
    return " ".join(_sentence(value) for value in values if value.strip())


class CU1RichReferralRenderer:
    """Shared deterministic renderer for reviewed route-specific rich-referral content.

    Clinical treatment prose lives in reviewed structured content rather than route-specific Python.
    The rollout matrix is a second fail-closed gate: merely adding prose to a content shard cannot make
    a pending/evidence-limited/protocol-owned route eligible for rich rendering. Context-gated routes
    remain unsupported until their exact runtime context is explicitly resolved by a reviewed seam.
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = root or _repo_root()
        self.content = _load_content(self.root)
        self.migration = _load_migration(self.root)
        self.routes: Mapping[str, Any] = self.content["routes"]
        self.rollout_profiles: Mapping[str, Any] = self.migration["profiles"]
        self.max_chars = int(self.content.get("max_referral_chars") or 2000)
        self.standard_detailed_target_chars = int(self.content.get("standard_detailed_target_chars") or 1850)

    def rollout_entry(self, *, profile_id: str, route_id: str) -> Optional[Mapping[str, Any]]:
        profile = self.rollout_profiles.get(profile_id)
        if not isinstance(profile, Mapping):
            return None
        entry = profile.get(route_id)
        return entry if isinstance(entry, Mapping) else None

    def rollout_state(self, *, profile_id: str, route_id: str) -> Optional[str]:
        entry = self.rollout_entry(profile_id=profile_id, route_id=route_id)
        state = entry.get("state") if isinstance(entry, Mapping) else None
        return str(state) if isinstance(state, str) and state else None

    def supports(self, *, profile_id: str, route_id: str, subtype_id: Optional[str] = None) -> bool:
        if self.rollout_state(profile_id=profile_id, route_id=route_id) != _RICH_READY:
            return False
        spec = self.routes.get(route_id)
        if not isinstance(spec, Mapping):
            return False
        profile_ids = spec.get("profile_ids") or []
        if profile_ids and profile_id not in profile_ids:
            return False
        subtype_ids = spec.get("subtype_ids_optional") or []
        if subtype_ids:
            return bool(subtype_id and subtype_id in subtype_ids)
        return True

    def route_spec(self, *, profile_id: str, route_id: str, subtype_id: Optional[str] = None) -> Mapping[str, Any]:
        state = self.rollout_state(profile_id=profile_id, route_id=route_id)
        if state != _RICH_READY:
            raise CU1ContractError(
                f"Rich-referral route is not rollout-ready for {profile_id}.{route_id}: {state or 'unclassified'}"
            )
        if not self.supports(profile_id=profile_id, route_id=route_id, subtype_id=subtype_id):
            raise CU1ContractError(f"No applicable rich-referral content for {profile_id}.{route_id}.{subtype_id or '-'}")
        spec = self.routes.get(route_id)
        assert isinstance(spec, Mapping)
        return spec

    def evidence_profile_ids(self, *, profile_id: str, route_id: str, subtype_id: Optional[str] = None) -> List[str]:
        spec = self.route_spec(profile_id=profile_id, route_id=route_id, subtype_id=subtype_id)
        return [str(item) for item in (spec.get("evidence_profile_ids") or [])]

    def render_short(
        self,
        *,
        profile_id: str,
        route_id: str,
        subtype_id: Optional[str],
        clinical_context: Sequence[str],
    ) -> str:
        spec = self.route_spec(profile_id=profile_id, route_id=route_id, subtype_id=subtype_id)
        context = [_sentence(_clean_phrase(item)) for item in clinical_context if _clean_phrase(item)]
        flow = _clean_lines(spec.get("short_flow_el"))
        if not flow:
            raise CU1ContractError(f"Rich-referral short flow is missing for {route_id}")
        text = " ".join(context + [_sentence(item) for item in flow]).strip()
        return self._enforce_limit(text, mode="short", route_id=route_id)

    def render_detailed(
        self,
        *,
        profile_id: str,
        route_id: str,
        subtype_id: Optional[str],
        clinical_context: Sequence[str],
    ) -> str:
        spec = self.route_spec(profile_id=profile_id, route_id=route_id, subtype_id=subtype_id)
        context = [_sentence(_clean_phrase(item)) for item in clinical_context if _clean_phrase(item)]
        sections: List[str] = []
        if context:
            sections.append("ΚΛΙΝΙΚΗ ΕΙΚΟΝΑ\n" + " ".join(context))

        stages = spec.get("stages") or []
        if not isinstance(stages, list) or not stages:
            raise CU1ContractError(f"Rich-referral stages are missing for {route_id}")
        for stage in stages:
            if not isinstance(stage, Mapping):
                raise CU1ContractError(f"Invalid rich-referral stage for {route_id}")
            label = _clean_phrase(stage.get("label_el"))
            goals = _clean_lines(stage.get("goals_el"))
            directions = _clean_lines(stage.get("intervention_directions_el"))
            progress = _clean_lines(stage.get("progress_markers_el"))
            if not label or not goals or not directions or not progress:
                raise CU1ContractError(f"Incomplete rich-referral stage for {route_id}: {stage.get('stage_id')}")
            progress_label = "Πρόοδος/ολοκλήρωση" if "functional_return" in str(stage.get("stage_id") or "") else "Πρόοδος"
            stage_lines = [
                label,
                "Στόχοι: " + ", ".join(goals) + ".",
                "Κατευθύνσεις: " + _sentences(directions),
                f"{progress_label}: " + "; ".join(progress) + ".",
            ]
            sections.append("\n".join(stage_lines))

        tail_parts = []
        adjunct = _clean_phrase(spec.get("adjunct_boundary_el"))
        reassessment = _clean_phrase(spec.get("reassessment_el"))
        if adjunct:
            tail_parts.append(_sentence(adjunct))
        if reassessment:
            tail_parts.append(_sentence(reassessment))
        if tail_parts:
            sections.append(" ".join(tail_parts))

        text = "\n\n".join(sections).strip()
        return self._enforce_limit(text, mode="detailed", route_id=route_id)

    def _enforce_limit(self, text: str, *, mode: str, route_id: str) -> str:
        if len(text) <= self.max_chars:
            return text
        raise CU1ContractError(
            f"Rich-referral {mode} output for {route_id} exceeds {self.max_chars} characters ({len(text)})"
        )

    def contract_route_specs(self) -> Dict[str, Dict[str, Any]]:
        return {
            str(route_id): copy.deepcopy(dict(spec))
            for route_id, spec in self.routes.items()
            if isinstance(spec, Mapping)
        }

    def contract_rollout_entries(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for profile_id, routes in self.rollout_profiles.items():
            if not isinstance(routes, Mapping):
                continue
            result[str(profile_id)] = {
                str(route_id): copy.deepcopy(dict(entry))
                for route_id, entry in routes.items()
                if isinstance(entry, Mapping)
            }
        return result


__all__ = ["CU1RichReferralRenderer"]
