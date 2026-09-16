from __future__ import annotations

import re
import uuid

from clinical_excellence.modules.registry import UnsupportedModuleError, get_transcript_module

from .transcript_contracts import (
    TranscriptCandidateV1, TranscriptCandidatesResponseV1, TranscriptExtractRequestV1,
    TranscriptResponseMetaV1, WarningCode,
)
from .transcript_provider import TranscriptProvider


def _norm_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _unique(items):
    return list(dict.fromkeys(items))


def extract_candidates(request: TranscriptExtractRequestV1, provider: TranscriptProvider) -> TranscriptCandidatesResponseV1:
    adapter = get_transcript_module(request.module)
    provider_result = provider.extract(request, adapter.provider_profile())
    top_warnings: list[WarningCode] = list(provider_result.warnings)
    candidates: list[TranscriptCandidateV1] = []
    transcript_norm = _norm_ws(request.transcript)

    for raw in provider_result.candidates:
        warnings: list[WarningCode] = []
        if raw.confidence == "low":
            warnings.append("LOW_SOURCE_CLARITY")
        snippet = raw.evidence_snippet.strip()
        if snippet and _norm_ws(snippet) not in transcript_norm:
            warnings.append("EVIDENCE_NOT_VERIFIABLE")
        mappings = adapter.map_candidate(raw)
        if not mappings or all(item.status == "unmapped" for item in mappings):
            warnings.append("UNMAPPED_CANDIDATE")
        if any(item.status == "ambiguous" for item in mappings):
            warnings.append("AMBIGUOUS_TARGET")
        warnings = _unique(warnings)
        top_warnings.extend(warnings)
        candidates.append(TranscriptCandidateV1(
            candidate_id=str(uuid.uuid4()),
            semantic_type=raw.semantic_type,
            components=raw.components,
            source_assertion=raw.source_assertion,
            evidence_snippet=snippet,
            confidence=raw.confidence,
            target_mappings=mappings,
            warnings=warnings,
            requires_clinician_review=True,
            status="proposed",
        ))

    return TranscriptCandidatesResponseV1(
        request_id=str(uuid.uuid4()),
        module=request.module,
        encounter_phase=request.encounter_phase,
        language=request.language,
        candidates=candidates,
        warnings=_unique(top_warnings),
        meta=TranscriptResponseMetaV1(candidate_count=len(candidates)),
    )


__all__ = ["extract_candidates", "UnsupportedModuleError"]
