from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from clinical_excellence.core.transcript_contracts import ProviderCandidateV1, TargetMappingV1


@dataclass(frozen=True)
class TranscriptModuleAdapter:
    provider_profile: Callable[[], str]
    map_candidate: Callable[[ProviderCandidateV1], list[TargetMappingV1]]


class UnsupportedModuleError(LookupError):
    pass


def get_transcript_module(module: str) -> TranscriptModuleAdapter:
    if module != "osteoporosis":
        raise UnsupportedModuleError(module)
    from .osteoporosis.transcript_profile import provider_profile
    from .osteoporosis.transcript_targets import map_candidate
    return TranscriptModuleAdapter(provider_profile=provider_profile, map_candidate=map_candidate)
