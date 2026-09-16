from __future__ import annotations
import json
import os
from pathlib import Path

from clinical_excellence.core.providers.openai_transcript import OpenAITranscriptProvider, provider_status
from clinical_excellence.core.transcript_contracts import TranscriptExtractRequestV1
from clinical_excellence.core.transcript_service import extract_candidates

CASES=Path(__file__).with_name("cases.json")

def main() -> int:
    status=provider_status()
    if not (status["enabled"] and status["api_key_configured"] and status["phi_provider_approved"]):
        print("provider-eval BLOCKED: transcript provider/privacy gate not configured")
        return 2
    cases=json.loads(CASES.read_text(encoding="utf-8"))
    provider=OpenAITranscriptProvider()
    failed=0
    for item in cases:
        request=TranscriptExtractRequestV1.model_validate({"schema_version":"clinical_transcript_extract_request_v1","source_type":"heidi_transcript","module":"osteoporosis","encounter_phase":"during_visit","language":item["language"],"transcript":item["transcript"],"context":{"encounter_archetype":None}})
        result=extract_candidates(request,provider)
        semantics={c.semantic_type for c in result.candidates}
        concepts={component.concept_key for c in result.candidates for component in c.components}
        ok=set(item["expect_semantics"]).issubset(semantics) and set(item["expect_concepts"]).issubset(concepts)
        print(f"{item['id']}: {'PASS' if ok else 'FAIL'} candidates={len(result.candidates)}")
        failed += 0 if ok else 1
    print(f"provider-eval summary: total={len(cases)} failed={failed}")
    return 1 if failed else 0

if __name__=="__main__":
    raise SystemExit(main())
