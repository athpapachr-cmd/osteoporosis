# SLICE_PLAN_CURRENT.md — PR-1 Heidi-first Transcript Intake + Candidate Extraction v1

> **STATUS:** IMPLEMENTED / DETERMINISTIC-TESTED / HARDENED MINIMUM EVAL SUITE COMPLETE / LIVE SYNTHETIC PROVIDER-EVAL HOLD — NOT RELEASE READY.
> **Activated:** 2026-09-16 Asia/Nicosia.
> **Slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Activation main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Activation PR:** #115 — MERGED.
> **Runtime branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact tested runtime head:** `a79d68915bde230a53bb7b5fd31a4104a491b058`.
> **Latest deterministic/inherited gate:** `35084122094` — SUCCESS.
> **Focused PR-1 tests:** 26 PASS.
> **Synthetic provider minimum suite:** 13 synthetic/de-identified scenarios with structured promotion assertions.
> **Synthetic provider-eval probe:** `35056606836` — HOLD reconfirmed; no Actions `OPENAI_API_KEY`, no provider call.
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

The corrected archived v3 design was rechecked against current runtime before activation.

### Runtime seam verification

Current persisted/browser paths support the v3 mapping design:

- `encounter_archetype`;
- `anthropometrics.weight_kg`, `anthropometrics.current_height_cm`;
- `fracture_history.events[]` with `site`, `month`, `low_trauma`, `occurred_on_treatment`, `vertebral_level`;
- `risk_context.glucocorticoids`, dose/duration, falls and frailty fields;
- `risk_assessment` formal FRAX/context fields;
- `step3.dxa.*`, `step3.vfa.*`, `step3.labs.*`;
- `step4.treatment_episodes[]`, `step4.administrations[]`, `step4.decision.*`, `step4.tasks[]`.

The Patient Registry still sends the complete active encounter object as protected encounter `payload`, so the mapper is anchored to **actual runtime paths**, not documentation-only schema names.

### Provider/API verification

Current official OpenAI SDK sources support the selected adapter pattern:

- Responses API + Python `responses.parse(..., text_format=<Pydantic model>)` structured parsing;
- GPT-5.6 availability through Responses API;
- SDK automatic retries remain enabled by default for selected failures unless explicitly set to `max_retries=0`;
- bounded client timeout remains configurable;
- Responses structured parsing runs the returned text through local Pydantic parsing, so structured validation failures are correctly classified separately from transport/upstream failures.

No material REPLAN trigger was found.

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

Provider-supplied values are locally validated against exact current runtime contracts. Fixed enums for fracture site, FRAX tool/risk category, VFA indication/action/modality, treatment episode status and administration status must match current UI-supported values before they can be marked `mapped`. Treatment duration must be numeric and within the runtime 0–50-year range.

Explicit ambiguous/unmapped examples:

- checkbox-like explicit negative where current runtime cannot distinguish false from untouched/default;
- adjusted/contextual FRAX-like values that could overwrite original formal FRAX;
- route where current Step-4 has no authoritative field;
- option-discussed / recommendation / free preference content where current runtime has no lossless target;
- unsupported units or fixed-enum values;
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

UI states clearly:

> AI-extracted candidate ≠ clinician-confirmed clinical record.

No red/green audit-performance styling is added.

## 12. Implemented seams

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
    osteoporosis/transcript_target_guard.py
```

Minimal existing-owner changes:

- `clinical_auth.py`: reusable protected dependency without broad auth refactor;
- `main.py`: transcript router mount;
- `static/baseline-audit/app.js`: isolated transcript asset load;
- new `static/baseline-audit/transcript-capture.js`.

Not modified:

- `app-core.js`;
- `step3.js`;
- `step4.js`;
- `clinical_data.py`;
- `clinical_data_ext.py`.

## 13. Deterministic acceptance evidence

Exact tested runtime head:

```text
a79d68915bde230a53bb7b5fd31a4104a491b058
```

Latest deterministic/inherited GitHub gate:

```text
35084122094 — SUCCESS
```

The gate passed:

- Python + browser syntax;
- **26 focused PR-1 privacy/contract/mapping/UI/eval-contract tests**;
- body/character limits and unknown-field rejection;
- sanitized errors with no sentinel PHI echo/logging;
- candidate/component strict validation;
- date precision/relative-date fail-closed behavior;
- impossible normalized calendar dates rejected;
- speaker/polarity/temporality/certainty contracts;
- evidence substring verification;
- server-forced review/proposed state;
- provider target-path injection impossible;
- module registry dispatch;
- actual osteoporosis mapped/ambiguous/unmapped behavior;
- adjusted risk cannot overwrite original FRAX;
- unsupported units and fixed runtime enums fail closed;
- protected endpoint auth + cookie-session auth;
- one provider call per extraction;
- no transcript/candidate browser persistence;
- OpenAI adapter structured output, `store=False`, no tools, `max_retries=0`, bounded timeout;
- structured provider validation failure → `PROVIDER_INVALID_OUTPUT` rather than false provider-unavailable classification;
- provider-eval case matching now enforces required/forbidden semantic assertions, source semantics, values, deterministic mappings, invented-date prohibitions, semantic counts and ephemeral/non-authoritative response meta;
- inherited protected-clinical regressions — 6 PASS;
- inherited Medical Report regressions — 24 PASS;
- inherited workspace navigation regression — PASS;
- bounded PR-1 scope guard — PASS.

The synthetic/de-identified provider minimum suite now contains **13** representative scenarios:

1. positive fracture history with vague relative timing;
2. explicit negative smoking history;
3. exact DXA objective result;
4. laboratory objective results;
5. multiple options + one recommendation + patient acceptance + exactly one final decision;
6. patient preference without accidental decision/recommendation;
7. vague follow-up timeframe without fabricated due date;
8. garbled/uncertain speech without guessed administration truth;
9. original formal FRAX versus clinician-adjusted interpretation;
10. ambiguous speaker/treatment history;
11. explicit negative history versus negative objective investigation with reciprocal collapse checks;
12. exact follow-up date when the source supports an exact day;
13. unrelated general musculoskeletal text without osteoporosis-target hallucination.

This completes the frozen-v3 minimum **suite definition**. It is not a substitute for selected-model execution.

## 14. Live synthetic provider-eval HOLD

Temporary run `35056606836` attempted to execute the selected GPT-5.6 provider eval through GitHub Actions. The Actions environment had no usable `OPENAI_API_KEY`, so the harness exited deliberately with:

```text
PR1_SYNTHETIC_PROVIDER_EVAL_HOLD
No production configuration was changed and no transcript was sent.
```

A safe rerun on 2026-09-16 reconfirmed the same missing Actions credential and again made no provider call.

This is neither PASS nor FAIL for the model/provider behavior. No live provider call occurred. No transcript content was transmitted. The temporary workflow was removed from the branch after the HOLD was checkpointed.

Do not use the Medical Report production credential path, change Render configuration, expose/copy a secret, or substitute a different execution path merely to manufacture equivalent evidence.

## 15. Definition of Done status

Satisfied:

- protected transcript UI + endpoint exist;
- Core/provider/module boundaries implemented;
- strict structured extraction + local validation active;
- deterministic current-runtime mapper works and fixed runtime enums fail closed;
- normalized exact dates are calendar-valid and vague timing remains non-exact;
- preview is transient/non-authoritative;
- no authoritative write path exists;
- transcript/candidates/content are not persisted/logged;
- deterministic + inherited tests pass;
- hardened 13-case synthetic/de-identified promotion suite is defined and deterministically validated;
- transcript-specific identifiable-data gate remains fail-closed;
- exact tested runtime head and workflow evidence are checkpointed.

**Not yet satisfied:**

- selected provider/model passes the hardened 13-case synthetic-only live eval gate.

Therefore:

```text
IMPLEMENTED YES
DETERMINISTIC-TESTED YES
INHERITED REGRESSIONS PASS
MINIMUM SYNTHETIC SUITE DEFINED YES
LIVE SYNTHETIC PROVIDER EVAL HOLD
RELEASE READY NO
RUNTIME RELEASE PR NO
DEPLOY NO
IDENTIFIABLE TRANSCRIPT USE NO
```

## 16. Release boundary

PR-1 is not release-ready while the provider-eval HOLD remains. Do not open the runtime release PR, merge/deploy PR-1, enable identifiable transcript processing, begin PR-2, or start real-patient use from this state.

The next permitted material transition is resolution of the synthetic provider-eval HOLD through a safe credential path that does not expose/copy a secret and does not mutate production configuration, followed immediately by a durable canonical checkpoint of PASS/FAIL.
