from __future__ import annotations

import json
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

from clinical_auth import require_clinical_key
from clinical_excellence.modules.registry import UnsupportedModuleError

from .providers.openai_transcript import OpenAITranscriptProvider
from .transcript_contracts import TranscriptExtractRequestV1
from .transcript_provider import ProviderInvalidOutput, ProviderNotConfigured, ProviderRefusal, ProviderUnavailable, TranscriptProvider
from .transcript_service import extract_candidates

MAX_BODY_BYTES = 512 * 1024
MAX_TRANSCRIPT_CHARS = 120_000


def _error(status: int, code: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code})


async def _read_bounded_json(request: Request) -> dict:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                raise _error(413, "REQUEST_TOO_LARGE")
        except ValueError:
            pass
    total = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            raise _error(413, "REQUEST_TOO_LARGE")
        chunks.append(chunk)
    try:
        decoded = b"".join(chunks).decode("utf-8")
        payload = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _error(422, "INVALID_REQUEST")
    if not isinstance(payload, dict):
        raise _error(422, "INVALID_REQUEST")
    transcript = payload.get("transcript")
    if not isinstance(transcript, str):
        raise _error(422, "INVALID_REQUEST")
    transcript = transcript.strip()
    if not transcript:
        raise _error(422, "EMPTY_TRANSCRIPT")
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        raise _error(413, "TRANSCRIPT_TOO_LARGE")
    payload["transcript"] = transcript
    return payload


def build_transcript_router(provider_factory: Callable[[], TranscriptProvider] | None = None) -> APIRouter:
    provider_factory = provider_factory or OpenAITranscriptProvider
    router = APIRouter(prefix="/clinical/transcript", tags=["clinical-transcript-pr1"], dependencies=[Depends(require_clinical_key)])

    @router.post("/extract")
    async def extract(request: Request):
        payload = await _read_bounded_json(request)
        try:
            parsed = TranscriptExtractRequestV1.model_validate(payload)
        except ValidationError:
            raise _error(422, "INVALID_REQUEST")
        try:
            result = extract_candidates(parsed, provider_factory())
        except UnsupportedModuleError:
            raise _error(422, "UNSUPPORTED_MODULE")
        except ProviderNotConfigured:
            raise _error(503, "PROVIDER_NOT_CONFIGURED")
        except ProviderUnavailable:
            raise _error(503, "PROVIDER_UNAVAILABLE")
        except ProviderRefusal:
            raise _error(422, "PROVIDER_REFUSAL")
        except ProviderInvalidOutput:
            raise _error(502, "PROVIDER_INVALID_OUTPUT")
        except HTTPException:
            raise
        except Exception:
            raise _error(500, "INTERNAL_PROCESSING_ERROR")
        return result.model_dump(mode="json")

    return router


__all__ = ["build_transcript_router", "MAX_BODY_BYTES", "MAX_TRANSCRIPT_CHARS"]
