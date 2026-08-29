from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Set

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from clinic_utilities.physio_evidence_runtime import (
    CU1ClinicianEvidenceResolver,
    _claim_matches_profiles,
    _human_claim,
    _human_source,
    _route_matches_claim,
    get_cu1_evidence_resolver,
)
from clinic_utilities.physio_referral_runtime import CU1ContractError, _require_clinical_key, get_cu1_bundle
from clinic_utilities.physio_rich_referral import CU1RichReferralRenderer


class CU1EvidenceRequest(BaseModel):
    profile_id: str
    route_id: str
    subtype_id_optional: Optional[str] = None
    wording_mode_optional: Optional[str] = None
    context_optional: Dict[str, Any] = Field(default_factory=dict)


def _selection_exists(profile_id: str, route_id: str, subtype_id: Optional[str]) -> bool:
    bundle = get_cu1_bundle()
    profile = (bundle.registry.get("profiles") or {}).get(profile_id)
    if not isinstance(profile, Mapping):
        return False
    route = (profile.get("routes") or {}).get(route_id)
    if not isinstance(route, Mapping):
        return False
    if subtype_id:
        allowed = (bundle.registry.get("subtypes") or {}).get(route_id, [])
        if subtype_id not in (allowed or []):
            return False
    return True


def _summary_for_exact_profiles(
    resolver: CU1ClinicianEvidenceResolver,
    *,
    profile_id: str,
    route_id: str,
    subtype_id: Optional[str],
    evidence_profile_ids: Set[str],
) -> Dict[str, Any]:
    base = resolver.route_summary(profile_id=profile_id, route_id=route_id, subtype_id=subtype_id)
    profiles = [
        resolver.route_evidence_profiles[item]
        for item in sorted(evidence_profile_ids)
        if item in resolver.route_evidence_profiles
        and resolver.route_evidence_profiles[item].get("route_id") == route_id
    ]
    resolved_ids = {
        str(profile.get("route_evidence_profile_id"))
        for profile in profiles
        if profile.get("route_evidence_profile_id")
    }
    if resolved_ids != evidence_profile_ids:
        raise CU1ContractError(
            f"Rich-referral evidence profile resolution mismatch for {profile_id}.{route_id}: "
            f"requested={sorted(evidence_profile_ids)} resolved={sorted(resolved_ids)}"
        )

    claims = [
        claim
        for claim in resolver.claims.values()
        if _route_matches_claim(claim, route_id, subtype_id)
        and _claim_matches_profiles(claim, resolved_ids)
    ]
    source_ids = set()
    for profile in profiles:
        source_ids.update(str(item) for item in (profile.get("primary_source_ids") or []))
    for claim in claims:
        source_ids.update(str(item) for item in (claim.get("evidence_ids") or []))

    sources = []
    for source_id in sorted(source_ids):
        source = resolver.sources.get(source_id)
        if not isinstance(source, Mapping):
            continue
        if source.get("status") == "superseded" or source.get("freshness_state") == "superseded":
            continue
        sources.append(_human_source(source))

    human_claims = [_human_claim(claim, resolver.sources) for claim in claims]
    human_claims.sort(key=lambda item: (
        {"core_rehabilitation": 0, "rehab_phase": 1, "progression_criteria": 2, "adjunct": 3, "safety": 4}.get(str(item.get("domain")), 9),
        {"referral_core": 0, "clinician_ui_only": 1, "therapist_execution_detail": 2}.get(str(item.get("output_scope")), 9),
        str(item.get("claim_summary") or ""),
    ))
    gaps = []
    for profile in profiles:
        gaps.extend(str(item) for item in (profile.get("evidence_gaps") or []))

    base.update({
        "selection_state": "resolved_context_profile",
        "has_applicable_profile": bool(profiles),
        "profile_count": len(profiles),
        "sources": sources,
        "claims": human_claims,
        "evidence_gaps": list(dict.fromkeys(gaps)),
        "conflicts": [
            _human_claim(claim, resolver.sources)
            for claim in claims
            if claim.get("conflicts_with_claim_ids_optional")
        ],
        "reviewed_on": max(
            (str(profile.get("last_reviewed_on")) for profile in profiles if profile.get("last_reviewed_on")),
            default=None,
        ),
        "next_review_due": min(
            (str(profile.get("next_review_due")) for profile in profiles if profile.get("next_review_due")),
            default=None,
        ),
    })
    return base


def contextual_evidence_summary(
    resolver: CU1ClinicianEvidenceResolver,
    renderer: CU1RichReferralRenderer,
    *,
    profile_id: str,
    route_id: str,
    subtype_id: Optional[str] = None,
    wording_mode: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    state = renderer.rollout_state(profile_id=profile_id, route_id=route_id)
    if state != "context_gated":
        return resolver.route_summary(profile_id=profile_id, route_id=route_id, subtype_id=subtype_id)

    rich_context: Dict[str, Any] = dict(context or {})
    if wording_mode:
        rich_context["__wording_mode"] = wording_mode
    if renderer.supports(
        profile_id=profile_id,
        route_id=route_id,
        subtype_id=subtype_id,
        context=rich_context,
    ):
        profile_ids = set(
            renderer.evidence_profile_ids(
                profile_id=profile_id,
                route_id=route_id,
                subtype_id=subtype_id,
                context=rich_context,
            )
        )
        return _summary_for_exact_profiles(
            resolver,
            profile_id=profile_id,
            route_id=route_id,
            subtype_id=subtype_id,
            evidence_profile_ids=profile_ids,
        )

    base = resolver.route_summary(profile_id=profile_id, route_id=route_id, subtype_id=subtype_id)
    gaps = list(base.get("evidence_gaps") or [])
    gaps.insert(0, "select_route_context_to_resolve_evidence")
    base.update({
        "selection_state": "context_required_for_evidence",
        "has_applicable_profile": False,
        "profile_count": 0,
        "sources": [],
        "claims": [],
        "conflicts": [],
        "evidence_gaps": list(dict.fromkeys(gaps)),
        "reviewed_on": None,
        "next_review_due": None,
    })
    return base


def build_cu1_physio_evidence_router() -> APIRouter:
    router = APIRouter(prefix="/clinical/clinic-utilities/physio-referral", tags=["cu1-physio-evidence"])
    protected = [Depends(_require_clinical_key)]

    @router.post("/api/evidence", dependencies=protected)
    def cu1_evidence(req: CU1EvidenceRequest) -> Dict[str, Any]:
        if not _selection_exists(req.profile_id, req.route_id, req.subtype_id_optional):
            raise HTTPException(status_code=422, detail="Invalid CU-1 route/subtype selection")
        try:
            return contextual_evidence_summary(
                get_cu1_evidence_resolver(),
                CU1RichReferralRenderer(),
                profile_id=req.profile_id,
                route_id=req.route_id,
                subtype_id=req.subtype_id_optional,
                wording_mode=req.wording_mode_optional,
                context=req.context_optional,
            )
        except CU1ContractError as exc:
            raise HTTPException(status_code=500, detail=f"CU-1 evidence contract error: {exc}") from exc

    return router


__all__ = [
    "build_cu1_physio_evidence_router",
    "CU1EvidenceRequest",
    "contextual_evidence_summary",
]
