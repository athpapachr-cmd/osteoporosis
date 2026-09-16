# SLICE_PLAN_CURRENT.md — PR-1 Heidi-first Transcript Intake + Candidate Extraction v1

> **STATUS:** ACTIVE / DESIGN VERIFIED / IMPLEMENTATION AUTHORIZED.
> **Activated:** 2026-09-16 Asia/Nicosia.
> **Slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Bootstrap main:** `805c4fe4e723dacafc351ccf695ef01b66600079`.
> **Design ancestry:** corrected archived PR-1 v3 on `docs/pr1-replan-v3-clinic-utilities` (`8515dba581a631e28d5bfbfa81e302f6123576b5`).
> **Writer:** one bounded PR-1 implementation writer; operational owner is `CURRENT_OPERATIONAL.md`.

## 1. Objective

Add a reusable Clinical Excellence Core capability that accepts a pasted Heidi transcript and returns structured, **non-authoritative** clinical candidates for immediate clinician review, using Osteoporosis Module 01 as the first deterministic mapping profile.

```text
PASTE HEIDI TRANSCRIPT
→ protected Core endpoint
→ ephemeral processing
→ strict semantic candidates
→ deterministic Module-01 target mapping
→ transient grouped preview
→ NO authoritative write
```

The raw transcript is not patient truth, is not persisted by PR-1, and must not silently mutate the encounter.

## 2. Final verification disposition

The corrected archived v3 design was rechecked against current `main` before activation.

### Runtime seam verification

Current persisted/browser paths still support the v3 mapping design:

- `encounter_archetype`;
- `anthropometrics.weight_kg`, `anthropometrics.current_height_cm`;
- `fracture_history.events[]` with `site`, `month`, `low_trauma`, `occurred_on_treatment`, `vertebral_level`;
- `risk_context.glucocorticoids`, dose/duration, falls and frailty fields;
- `risk_assessment` formal FRAX/context fields;
- `step3.dxa.*`, `step3.vfa.*`, `step3.labs.*`;
- `step4.treatment_episodes[]`, `step4.administrations[]`, `step4.decision.*`, `step4.tasks[]`.

The current Patient Registry still sends the complete active encounter object as protected encounter `payload`, so the mapper remains anchored to **actual runtime paths**, not documentation-only schema names.

### Provider/API verification

Current official OpenAI sources still support the v3 provider pattern:

- Responses API + Python `responses.parse(..., text_format=<Pydantic model>)` structured parsing;
- GPT-5.6 availability through Responses API;
- SDK automatic retries remain enabled by default for selected failures unless explicitly set to `max_retries=0`;
- bounded client timeout remains configurable.

No material REPLAN trigger was found. Exact provider configuration remains implementation-time configurable and test/eval guarded.

## 3. Hard scope

PR-1 includes:

- protected `POST /clinical/transcript/extract`;
- request body ceiling **512 KiB**;
- transcript stripped, non-empty, max **120,000 Unicode characters**;
- controlled JSON parse + sanitized validation boundary;
- Core semantic candidate/component/source-assertion contracts;
- provider-neutral extraction protocol;
- isolated OpenAI adapter with `store=False`, no tools, `max_retries=0`, bounded timeout and strict structured output;
- generic module registry;
- Osteoporosis Module-01 concept profile and deterministic runtime-target mapper;
- mapped / ambiguous / unmapped candidate states;
- bounded evidence-snippet verification against the supplied transcript;
- ephemeral clinician-readable browser preview;
- deterministic tests plus synthetic/de-identified provider eval harness.

PR-1 excludes:

- candidate Accept/Edit/Reject into authoritative record;
- any patient/encounter/lab/task mutation;
- transcript or candidate persistence;
- PR-2 inline population;
- Practice Review coaching;
- KPI or audit changes;
- treatment recommendation;
- pilot activation;
- unrelated product mutations.

## 4. Request contract

```json
{
  "schema_version": "clinical_transcript_extract_request_v1",
  "source_type": "heidi_transcript",
  "module": "osteoporosis",
  "encounter_phase": "during_visit",
  "language": "el",
  "transcript": "<string>",
  "context": {
    "encounter_archetype": null
  }
}
```

Unknown fields are rejected. No patient ID, encounter ID, name, DOB or authoritative encounter payload is required merely to extract candidates.

## 5. Success contract

```json
{
  "schema_version": "clinical_transcript_candidates_v1",
  "request_id": "<uuid>",
  "source_type": "heidi_transcript",
  "module": "osteoporosis",
  "encounter_phase": "during_visit",
  "language": "el",
  "candidates": [],
  "warnings": [],
  "meta": {
    "processing_mode": "ephemeral_preview",
    "candidate_count": 0,
    "raw_persisted": false,
    "candidates_persisted": false,
    "authoritative_write": false
  }
}
```

Warnings are coded server-owned values, including `LOW_SOURCE_CLARITY`, `UNMAPPED_CANDIDATE`, `AMBIGUOUS_TARGET`, `EVIDENCE_NOT_VERIFIABLE`, `PARTIAL_EXTRACTION`.

## 6. Sanitized error boundary

Sensitive request errors must never echo transcript content, Pydantic input payloads, candidate values, provider prompts/output or raw provider exception/response bodies.

Required failure classes:

| Condition | HTTP | Public code |
|---|---:|---|
| auth failure | 401 | `UNAUTHORIZED` |
| provider not enabled/configured | 503 | `PROVIDER_NOT_CONFIGURED` |
| malformed/invalid request | 422 | `INVALID_REQUEST` |
| empty transcript | 422 | `EMPTY_TRANSCRIPT` |
| body >512 KiB | 413 | `REQUEST_TOO_LARGE` |
| transcript >120k chars | 413 | `TRANSCRIPT_TOO_LARGE` |
| unsupported module | 422 | `UNSUPPORTED_MODULE` |
| timeout/rate-limit/upstream unavailable | 503 | `PROVIDER_UNAVAILABLE` |
| refusal | 422 | `PROVIDER_REFUSAL` |
| invalid structured provider output | 502 | `PROVIDER_INVALID_OUTPUT` |
| internal mapper failure | 500 | `INTERNAL_PROCESSING_ERROR` |

## 7. Candidate semantic contract

One candidate is one semantic assertion with one or more related `components[]`.

Core semantic types:

```text
patient_history_fact
objective_result
clinician_interpretation
option_discussed
clinician_recommendation
patient_preference
final_decision
patient_accepted
patient_declined
patient_undecided
followup_task
uncertain_needs_review
```

Source assertion preserves:

```text
speaker: patient | clinician | third_party | unclear
polarity: positive | negative | not_applicable | unclear
temporality: current | past | planned | future | relative | unclear
normalized_date: exact supported day/month/year only or null
date_precision: day | month | year | null
date_text: original vague wording when needed
certainty: explicit | probable | uncertain
```

Hard distinctions:

```text
OPTION DISCUSSED != CLINICIAN RECOMMENDATION != FINAL DECISION
PATIENT PREFERENCE != PATIENT ACCEPTED
PATIENT/HISTORY FACT != OBJECTIVE INVESTIGATION RESULT
OBJECTIVE RESULT != CLINICIAN INTERPRETATION
VAGUE TIME != INVENTED EXACT DATE
```

Every candidate is server-forced `requires_clinician_review=true` and `status=proposed`.

## 8. Deterministic Module-01 mapping boundary

The provider may emit concept semantics and components only. It must never emit or choose application target paths.

```text
provider semantic assertion
→ local validation
→ deterministic osteoporosis mapper
→ target_mappings[]
```

Initial mapped families:

- encounter archetype;
- weight/current height;
- fracture event composite;
- smoking/alcohol/RA;
- glucocorticoid context, dose and duration;
- falls/frailty context;
- formal FRAX tool/model/FN-BMD/original MOF/hip results;
- stated risk category;
- DXA date/BMD/T-scores;
- VFA indication/action/modality/objective vertebral-fracture result;
- supported Ca/phosphate/Vit-D/PTH/CTX/P1NP values;
- treatment episode composite;
- actual/scheduled/next-due administration dates;
- final treatment decision type + selected agent;
- patient accepted/declined/undecided;
- follow-up task type, exact due date or vague timeframe text.

Explicit ambiguous/unmapped examples:

- checkbox-like explicit negative where current runtime cannot distinguish false from untouched/default;
- adjusted/contextual FRAX-like values that could overwrite original formal FRAX;
- route where current Step-4 has no authoritative field;
- option-discussed / recommendation / free preference content where current runtime has no lossless target;
- unsupported units;
- clinically meaningful narrative that has no current target.

Unmapped candidates remain visible. PR-1 does not expand the encounter schema just to make extraction convenient.

## 9. Evidence snippet

A short evidence snippet may be returned for immediate review, target max ~320 characters. The Core verifies that it exists in the submitted transcript after whitespace normalization. If verification fails, the quote is not rewritten; candidate gets `EVIDENCE_NOT_VERIFIABLE`.

Evidence snippets are never logged or persisted.

## 10. Privacy and provider gate

```text
raw transcript → no DB
raw transcript → no encounter payload
raw transcript → no localStorage/sessionStorage
raw transcript → no logs
provider response/candidates → no server persistence
candidate preview → transient JS/DOM only
```

`store=False` does not itself establish Zero Data Retention.

Therefore identifiable transcript use remains **blocked** behind a transcript-specific configuration/privacy gate. Deterministic CI, provider eval and engineering smoke use synthetic/de-identified transcripts only until that separate gate is explicitly closed.

## 11. Browser lifecycle

Transcript/candidate state is isolated from `currentCase`.

```text
open panel → empty
paste → textarea DOM only
submit → in-memory request only
success → clear textarea, retain candidate preview transiently
close/discard → clear textarea + candidates
pagehide/logout/navigation → explicit clear
pageshow/BFCache → defensive reset
failure → textarea may remain only for explicit immediate retry
```

UI must state clearly:

> AI-extracted candidate ≠ clinician-confirmed clinical record.

No red/green audit-performance styling.

## 12. Preferred implementation seams

```text
clinical_excellence/
  core/
    transcript_contracts.py
    transcript_provider.py
    transcript_service.py
    transcript_router.py
    providers/openai_transcript.py
  modules/
    registry.py
    osteoporosis/transcript_profile.py
    osteoporosis/transcript_targets.py
```

Minimal existing-owner changes:

- `clinical_auth.py`: reusable protected dependency without broad auth refactor;
- `main.py`: mount transcript router;
- `static/baseline-audit/index.html` / `app.js` only as needed to load isolated capture assets;
- new `transcript-capture.js` + CSS;
- dependency change only if current tested SDK contract requires it.

Do **not** modify `app-core.js`, `step3.js`, `step4.js`, `clinical_data.py`, `clinical_data_ext.py` or existing Step schemas merely for convenience.

## 13. Acceptance gate

Deterministic CI must cover at minimum:

- body/character limits and unknown-field rejection;
- sanitized errors cannot echo sentinel PHI;
- candidate/component strict validation;
- date precision/relative date behavior;
- speaker/polarity/temporality/certainty;
- evidence substring verification;
- server-forced review/proposed state;
- provider target-path injection impossible;
- module registry dispatch;
- actual osteoporosis mapped/ambiguous/unmapped behavior;
- adjusted risk cannot overwrite original FRAX;
- unsupported units remain unmapped/ambiguous;
- protected endpoint auth;
- fake provider called exactly once;
- no patient/encounter/lab mutation;
- no transcript/candidate content in logs;
- OpenAI adapter contract: structured output, `store=False`, no tools, `max_retries=0`, bounded timeout;
- browser has no transcript/candidate persistence and clears on close/pagehide/pageshow.

Synthetic provider eval must include representative Greek cases covering fracture history, explicit negatives, exact/vague timing, DXA/labs, history-vs-investigation, multiple options with one final decision, patient preference, follow-up timing, garbled speech, original-vs-adjusted risk and speaker ambiguity.

## 14. Definition of Done for implementation candidate

PR-1 implementation candidate is complete when:

- protected transcript UI + endpoint exist;
- Core/provider/module boundaries are implemented;
- strict structured extraction + local validation active;
- deterministic current-runtime mapper works;
- preview is transient/non-authoritative;
- no authoritative write path exists;
- transcript/candidates/content are not persisted/logged;
- deterministic tests pass;
- selected provider/model passes synthetic-only eval gate;
- transcript-specific identifiable-data gate remains fail-closed unless separately authorized;
- exact tested head and workflow evidence are checkpointed in canonicals.

## 15. Release boundary

Implementation authority is granted for the bounded slice above. Merge, production deployment, enabling identifiable transcript processing and real-clinic use remain separate lifecycle decisions requiring exact-head evidence and explicit Product Owner authority.
