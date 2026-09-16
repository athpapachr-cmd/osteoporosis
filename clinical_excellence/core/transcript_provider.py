from __future__ import annotations

from typing import Protocol

from .transcript_contracts import ProviderTranscriptExtractionV1, TranscriptExtractRequestV1


class TranscriptProviderError(RuntimeError):
    pass


class ProviderNotConfigured(TranscriptProviderError):
    pass


class ProviderUnavailable(TranscriptProviderError):
    pass


class ProviderRefusal(TranscriptProviderError):
    pass


class ProviderInvalidOutput(TranscriptProviderError):
    pass


class TranscriptProvider(Protocol):
    def extract(self, request: TranscriptExtractRequestV1, provider_profile: str) -> ProviderTranscriptExtractionV1: ...
