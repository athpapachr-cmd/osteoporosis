from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

from clinic_utilities.physio_referral_runtime import CU1ContractError, _repo_root


_VIEW_CONFIG = "clinic_utilities/contracts/cu1_clinician_evidence_view_v1.yaml"
_COLLECTIONS = ("sources", "claims", "route_history_prompts", "route_evidence_profiles", "rehabilitation_sequences")
_KNOWN_TRANCHE3_BAD_TITLE = "    title: British Elbow and Shoulder Society patient care pathway: Frozen shoulder"
_KNOWN_TRANCHE3_FIXED_TITLE = '    title: "British Elbow and Shoulder Society patient care pathway: Frozen shoulder"'


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise CU1ContractError(f"Missing CU-1 evidence artifact: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        # The reviewed tranche-3 promotion artifact predates runtime parsing and contains one known
        # semantically harmless unquoted colon in a source title. Repair only that exact legacy scalar
        # for the read-only clinician evidence projection; any other YAML defect must still fail closed.
        if path.name == "cu1_evidence_tranche3_promotion_v1.yaml" and _KNOWN_TRANCHE3_BAD_TITLE in text:
            repaired = text.replace(_KNOWN_TRANCHE3_BAD_TITLE, _KNOWN_TRANCHE3_FIXED_TITLE, 1)
            try:
                payload = yaml.safe_load(repaired)
            except yaml.YAMLError as repair_exc:  # pragma: no cover - defensive compatibility boundary
                raise CU1ContractError(f"Unable to parse CU-1 evidence artifact: {path}") from repair_exc
        else:  # pragma: no cover - defensive evidence boundary
            raise CU1ContractError(f"Unable to parse CU-1 evidence artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise CU1ContractError(f"CU-1 evidence artifact must be a mapping: {path}")
    return payload


def _merge_fields(target: Dict[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if key.endswith("_reason") or key in {"amendment_reason", "reason"}:
            continue
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            nested = dict(target[key])
            _merge_fields(nested, value)
            target[key] = nested
        else:
            target[key] = copy.deepcopy(value)


def _route_matches_claim(claim: Mapping[str, Any], route_id: str, subtype_id: Optional[str]) -> bool:
    route_ids = claim.get("applicable_route_ids") or []
    if route_id not in route_ids:
        return False
    subtype_ids = claim.get("applicable_subtype_ids_optional") or []
    if subtype_ids:
        return bool(subtype_id and subtype_id in subtype_ids)
    return True


def _route_matches_profile(profile: Mapping[str, Any], route_id: str, subtype_id: Optional[str]) -> bool:
    if profile.get("route_id") != route_id:
        return False
    subtype_ids = profile.get("subtype_ids_optional") or []
    if subtype_ids:
        return bool(subtype_id and subtype_id in subtype_ids)
    return True


def _human_source(source: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source_type": source.get("source_type"),
        "title": source.get("title"),
        "authors_or_organization": source.get("authors_or_organization"),
        "year_or_version": source.get("year_or_version"),
        "reference": source.get("reference"),
        "doi": source.get("doi_optional"),
        "url": source.get("url_optional"),
        "population_scope": source.get("population_scope_optional"),
        "setting_scope": source.get("setting_scope_optional"),
        "reviewed_on": source.get("reviewed_on"),
        "next_review_due": source.get("next_review_due"),
        "freshness_state": source.get("freshness_state"),
        "status": source.get("status"),
    }


def _human_claim(claim: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    titles: List[str] = []
    for evidence_id in claim.get("evidence_ids") or []:
        source = sources.get(str(evidence_id))
        if isinstance(source, Mapping) and source.get("title"):
            titles.append(str(source["title"]))
    return {
        "domain": claim.get("domain"),
        "claim_summary": claim.get("claim_summary"),
        "recommendation_direction": claim.get("recommendation_direction"),
        "output_scope": claim.get("output_scope"),
        "strength": claim.get("strength_optional"),
        "certainty": claim.get("certainty_optional"),
        "applicability_conditions": copy.deepcopy(claim.get("applicability_conditions_optional") or []),
        "source_titles": titles,
        "has_conflict": bool(claim.get("conflicts_with_claim_ids_optional")),
    }


class CU1ClinicianEvidenceResolver:
    """Read-only clinician evidence projection.

    The source evidence corpus remains design authority and is not promoted here to autonomous
    treatment-selection authority. This resolver only exposes human-readable route-relevant evidence
    metadata for the clinician evidence panel.
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = root or _repo_root()
        self.view_config = _load_yaml(self.root / _VIEW_CONFIG)
        if self.view_config.get("runtime_authorized") is not True:
            raise CU1ContractError("Clinician evidence view is not runtime-authorized")
        manifest_rel = self.view_config.get("source_evidence_manifest")
        coverage_rel = self.view_config.get("source_coverage_matrix")
        if not isinstance(manifest_rel, str) or not isinstance(coverage_rel, str):
            raise CU1ContractError("Clinician evidence view source paths are invalid")
        self.manifest = _load_yaml(self.root / manifest_rel)
        self.coverage = _load_yaml(self.root / coverage_rel)
        self.sources: Dict[str, Dict[str, Any]] = {}
        self.claims: Dict[str, Dict[str, Any]] = {}
        self.route_evidence_profiles: Dict[str, Dict[str, Any]] = {}
        self.rehabilitation_sequences: Dict[str, Dict[str, Any]] = {}
        self.route_history_prompts: Dict[str, Dict[str, Any]] = {}
        self._load_active_logical_view()

    def _load_active_logical_view(self) -> None:
        collections: Dict[str, Dict[str, Any]] = {name: {} for name in _COLLECTIONS}
        for shard in self.manifest.get("shards") or []:
            if not isinstance(shard, Mapping) or shard.get("state") != "active_design_authority":
                continue
            path = shard.get("path")
            if not isinstance(path, str):
                continue
            payload = _load_yaml(self.root / path)
            for collection in _COLLECTIONS:
                values = payload.get(collection) or {}
                if not isinstance(values, Mapping):
                    continue
                for object_id, obj in values.items():
                    if not isinstance(obj, Mapping):
                        continue
                    materialized = copy.deepcopy(dict(obj))
                    canonical_field = {
                        "sources": "evidence_id",
                        "claims": "claim_id",
                        "route_history_prompts": "prompt_id",
                        "route_evidence_profiles": "route_evidence_profile_id",
                        "rehabilitation_sequences": "sequence_id",
                    }[collection]
                    materialized.setdefault(canonical_field, str(object_id))
                    collections[collection][str(object_id)] = materialized

            projection_path = shard.get("promotion_projection")
            if isinstance(projection_path, str):
                self._apply_projection(collections, _load_yaml(self.root / projection_path))
            overlay_path = shard.get("promotion_overlay")
            if isinstance(overlay_path, str):
                self._apply_projection(collections, _load_yaml(self.root / overlay_path))

        for amendment_path in self.manifest.get("logical_amendments") or []:
            if isinstance(amendment_path, str):
                self._apply_amendment(collections, _load_yaml(self.root / amendment_path))

        self.sources = collections["sources"]
        self.claims = collections["claims"]
        self.route_history_prompts = collections["route_history_prompts"]
        self.route_evidence_profiles = collections["route_evidence_profiles"]
        self.rehabilitation_sequences = collections["rehabilitation_sequences"]

    @staticmethod
    def _apply_projection(collections: Dict[str, Dict[str, Any]], projection: Mapping[str, Any]) -> None:
        for source_id, patch in (projection.get("source_metadata_corrections") or {}).items():
            if source_id in collections["sources"] and isinstance(patch, Mapping):
                _merge_fields(collections["sources"][source_id], patch)
        for source_id, source in (projection.get("additional_sources") or {}).items():
            if isinstance(source, Mapping):
                materialized = copy.deepcopy(dict(source))
                materialized.setdefault("evidence_id", str(source_id))
                collections["sources"][str(source_id)] = materialized

        drops: Iterable[Any] = (
            projection.get("claims_to_drop_before_logical_merge")
            or projection.get("claims_to_drop")
            or []
        )
        for claim_id in drops:
            collections["claims"].pop(str(claim_id), None)
        for claim_id, claim in (projection.get("replacement_claims") or {}).items():
            if isinstance(claim, Mapping):
                materialized = copy.deepcopy(dict(claim))
                materialized.setdefault("claim_id", str(claim_id))
                collections["claims"][str(claim_id)] = materialized

        for profile_id, patch in (projection.get("profile_field_overrides") or {}).items():
            if profile_id in collections["route_evidence_profiles"] and isinstance(patch, Mapping):
                _merge_fields(collections["route_evidence_profiles"][profile_id], patch)
        for profile_id, patch in (projection.get("route_evidence_profile_overrides") or {}).items():
            if profile_id in collections["route_evidence_profiles"] and isinstance(patch, Mapping):
                _merge_fields(collections["route_evidence_profiles"][profile_id], patch)

    @staticmethod
    def _apply_amendment(collections: Dict[str, Dict[str, Any]], amendment: Mapping[str, Any]) -> None:
        mapping = {
            "source_amendments": "sources",
            "claim_amendments": "claims",
            "route_evidence_profile_amendments": "route_evidence_profiles",
            "rehabilitation_sequence_amendments": "rehabilitation_sequences",
        }
        for patch_key, collection in mapping.items():
            for object_id, patch in (amendment.get(patch_key) or {}).items():
                if object_id in collections[collection] and isinstance(patch, Mapping):
                    _merge_fields(collections[collection][object_id], patch)
        for claim_id in amendment.get("suppressed_claim_ids") or []:
            collections["claims"].pop(str(claim_id), None)

    def route_summary(self, *, profile_id: str, route_id: str, subtype_id: Optional[str] = None) -> Dict[str, Any]:
        profiles = [
            profile for profile in self.route_evidence_profiles.values()
            if _route_matches_profile(profile, route_id, subtype_id)
        ]
        claims = [
            claim for claim in self.claims.values()
            if _route_matches_claim(claim, route_id, subtype_id)
        ]

        source_ids = set()
        for profile in profiles:
            source_ids.update(str(item) for item in (profile.get("primary_source_ids") or []))
        for claim in claims:
            source_ids.update(str(item) for item in (claim.get("evidence_ids") or []))

        sources = []
        for source_id in sorted(source_ids):
            source = self.sources.get(source_id)
            if not isinstance(source, Mapping):
                continue
            if source.get("status") == "superseded" or source.get("freshness_state") == "superseded":
                continue
            sources.append(_human_source(source))

        gaps: List[str] = []
        for profile in profiles:
            gaps.extend(str(item) for item in (profile.get("evidence_gaps") or []))

        coverage_info = self._coverage_info(profile_id, route_id)
        for key in ("blocker", "reason"):
            value = coverage_info.get(key)
            if isinstance(value, str) and value and value not in gaps:
                gaps.append(value)

        conflicts = [
            _human_claim(claim, self.sources)
            for claim in claims
            if claim.get("conflicts_with_claim_ids_optional")
        ]
        human_claims = [_human_claim(claim, self.sources) for claim in claims]
        human_claims.sort(key=lambda item: (
            {"core_rehabilitation": 0, "rehab_phase": 1, "progression_criteria": 2, "adjunct": 3, "safety": 4}.get(str(item.get("domain")), 9),
            {"referral_core": 0, "clinician_ui_only": 1, "therapist_execution_detail": 2}.get(str(item.get("output_scope")), 9),
            str(item.get("claim_summary") or ""),
        ))

        return {
            "profile_id": profile_id,
            "route_id": route_id,
            "subtype_id": subtype_id,
            "coverage_status": coverage_info.get("status"),
            "sequence_status": coverage_info.get("sequence_status"),
            "evidence_state": coverage_info.get("evidence_state"),
            "has_applicable_profile": bool(profiles),
            "profile_count": len(profiles),
            "sources": sources,
            "claims": human_claims,
            "evidence_gaps": list(dict.fromkeys(gaps)),
            "conflicts": conflicts,
            "reviewed_on": max((str(profile.get("last_reviewed_on")) for profile in profiles if profile.get("last_reviewed_on")), default=None),
            "next_review_due": min((str(profile.get("next_review_due")) for profile in profiles if profile.get("next_review_due")), default=None),
        }

    def _coverage_info(self, profile_id: str, route_id: str) -> Dict[str, Any]:
        profiles = self.coverage.get("profiles") or {}
        if not isinstance(profiles, Mapping):
            return {}
        profile = profiles.get(profile_id)
        if not isinstance(profile, Mapping):
            return {}
        route = profile.get(route_id)
        return copy.deepcopy(dict(route)) if isinstance(route, Mapping) else {}


_RESOLVER: Optional[CU1ClinicianEvidenceResolver] = None


def get_cu1_evidence_resolver() -> CU1ClinicianEvidenceResolver:
    global _RESOLVER
    if _RESOLVER is None:
        _RESOLVER = CU1ClinicianEvidenceResolver()
    return _RESOLVER


__all__ = ["CU1ClinicianEvidenceResolver", "get_cu1_evidence_resolver"]
