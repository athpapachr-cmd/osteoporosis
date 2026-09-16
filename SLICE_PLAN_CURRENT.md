# SLICE_PLAN_CURRENT.md — PR-1 Heidi-first Transcript Intake + Candidate Extraction v1

> **STATUS:** H-01..H-06 DETERMINISTIC CLOSURE PROVEN / SECOND LIVE 22-CASE SYNTHETIC QUALIFICATION AUTHORIZED — NOT RELEASE READY.
> **Activated:** 2026-09-16 Asia/Nicosia.
> **Replanned:** independent READ-ONLY review; then live Structured Outputs schema evidence.
> **Slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Activation main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Runtime branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact H-06 deterministic runtime/eval head:** `d1470ce1c10cd69f6f9e4fe72096527157f084e9`.
> **Deterministic/inherited gate:** `35139716235` — SUCCESS.
> **Synthetic qualification suite:** 22 synthetic/de-identified cases.
> **First live provider run:** `35138694138` — schema failure before case 1; credential boundary proven.
> **Writer:** one bounded PR-1 implementation writer; operational owner is `CURRENT_OPERATIONAL.md`.

## 1. Objective

Add a reusable Clinical Excellence Core capability that accepts a pasted Heidi transcript and returns structured, **non-authoritative** clinical candidates for clinician review, using Osteoporosis Module 01 as the first deterministic mapping profile.

```text
PASTE HEIDI TRANSCRIPT
→ protected Core endpoint
→ ephemeral processing
→ strict semantic candidates
→ deterministic Module-01 target mapping
→ transient preview
→ NO authoritative write
```

PR-1 remains extraction/preview only. PR-2 owns later provisional in-card acceptance/edit/reject behavior.

## 2. Preserved safety and privacy contracts

- raw transcript is ephemeral and non-authoritative;
- no transcript/candidate persistence in DB, encounter payload, browser storage or logs;
- provider emits semantic assertions only and cannot choose runtime/storage paths;
- deterministic Module-01 code owns runtime mapping;
- candidates are always `proposed` and `requires_clinician_review=true`;
- preserve speaker/source, polarity, temporality, certainty and semantic type;
- preserve patient/history fact vs objective result vs interpretation;
- preserve option vs recommendation vs preference vs acceptance vs final decision;
- vague/relative timing must not become an invented exact date;
- no authoritative patient/encounter/lab/task write exists in PR-1;
- identifiable transcript use remains blocked behind a distinct privacy/provider approval boundary.

## 3. Independent-review findings H-01 through H-04 — CLOSED

### H-01 — unexpected assertion default deny

Every provider component must be covered by an explicit required/allowed fixture rule. Unexpected assertions fail promotion qualification even if all required facts are also present.

### H-02 — deterministic component/event identity

Duplicate `concept_key` values inside one provider candidate are invalid. Repeated real-world events use separate candidates, and eval grouping prevents cross-event value/mapping pairing.

### H-03 — synthetic vs identifiable-PHI authorization

Clinical/default purpose requires identifiable-PHI approval. Synthetic evaluation requires the dedicated synthetic gate + credential and does not set or impersonate PHI approval.

### H-04 — exact runtime guard surface

The deterministic mapper enforces current runtime enums, units, semantics/source protections and verified numeric ranges for weight, height, FRAX percentages, DXA BMD/T-score, falls, CFS and treatment duration.

## 4. Promotion suite

The frozen suite contains 22 synthetic/de-identified cases. In addition to the original semantic/date/result distinctions it includes:

- repeated fracture/event grouping;
- embedded prompt/instruction text treated as untrusted transcript content;
- referral/request ≠ completed investigation/result;
- prescription/recommendation ≠ medication actually taken/administered;
- self-correction;
- third-party history without patient attribution;
- out-of-range numeric extraction with fail-closed runtime mapping;
- planned administration ≠ completed administration;
- explicit negated treatment exposure;
- low-confidence failure for clean explicit cases unless explicitly permitted.

The oracle remains default-deny and response metadata must remain ephemeral/non-authoritative.

## 5. Synthetic-only credential boundary — PROVEN

The bounded workflow:

```text
.github/workflows/pr1-transcript-live-provider-eval.yml
```

uses:

```text
CLINICAL_TRANSCRIPT_AI_ENABLED=true
CLINICAL_TRANSCRIPT_SYNTHETIC_EVAL_ENABLED=true
CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED=false
CLINICAL_TRANSCRIPT_AI_MODEL=gpt-5.6
OPENAI_API_KEY from GitHub Actions secret store
```

Run `35138694138` proved the credential was available and masked, PHI approval remained false and the 22 fixture IDs were valid/unique. No production configuration changed.

## 6. First live request — H-06 discovery

The first provider call in run `35138694138` failed before case 1 with HTTP 400 `invalid_json_schema` because the discriminated Pydantic `CandidateValueV1` generated a `oneOf` schema at the provider response `value` property. The adapter also collapsed that deterministic request/schema failure into `ProviderUnavailable`.

This was adapter-contract evidence, not semantic model evidence.

## 7. H-06 — Structured Outputs schema compatibility — CLOSED deterministically

Exact remediation head:

```text
d1470ce1c10cd69f6f9e4fe72096527157f084e9
```

Exact verification:

```text
35139716235 — SUCCESS
52 focused PR-1 tests PASS
6 inherited protected-clinical tests PASS
24 inherited Medical Report tests PASS
workspace navigation PASS
bounded scope PASS
```

### Provider schema remediation

`CandidateValueV1` remains a strict union of the seven value models but is no longer represented by a Pydantic discriminator. Every variant still carries a mutually exclusive literal `kind`:

```text
text
code
number
integer
boolean
quantity
date
```

This preserves the local typed contract while removing the generated `oneOf` keyword that the live provider rejected. The generated `ProviderTranscriptExtractionV1` schema is deterministically checked to contain no `oneOf` keyword.

The response path remains strict Structured Outputs through:

```text
responses.parse(..., text_format=ProviderTranscriptExtractionV1)
```

No permissive JSON mode or free-form text fallback is allowed.

### Local validation preserved

Deterministic H-06 tests prove:

- all seven value payloads validate into the intended concrete Pydantic value class;
- kind/shape mismatches fail;
- impossible exact calendar dates fail;
- duplicate concept keys still fail;
- existing vague-date, H-01..H-04, mapping and fixture-contract protections remain green.

### Error classification corrected

Deterministic provider HTTP 400/request-contract failures, including `invalid_json_schema`, now fail as `ProviderInvalidOutput` rather than retryable `ProviderUnavailable`. Timeout/connection/rate-limit/upstream failures retain the availability classification.

## 8. Exact live qualification gate now authorized

After durable canonical verification of H-06, the next deliberate material action is a second synthetic-only selected-model qualification:

```text
exact H-06-compatible branch state
+ secure Actions OPENAI_API_KEY
+ synthetic eval gate true
+ PHI approval false
→ deliberately retrigger bounded workflow
→ run all 22 cases through gpt-5.6
→ require failed=0 under default-deny oracle
→ checkpoint exact live evidence
→ fresh independent READ-ONLY evidence review
```

If the API rejects another schema construct or any case fails, that exact evidence becomes the next checkpoint before further code/prompt/fixture mutation.

## 9. H-05 retained production-release blocker

The FastAPI route is async while the OpenAI adapter is synchronous and the current production command uses a single uvicorn worker. Before production enablement, provider execution must not block the event loop for the full provider timeout.

H-05 does not block isolated CLI synthetic qualification, but it must close before release/deploy. Stronger executable browser lifecycle/BFCache/logout cleanup evidence also remains production-release debt.

## 10. Definition of Done status

Satisfied deterministically:

- protected transcript UI + endpoint;
- strict request/response/provider contracts locally;
- provider target-path isolation;
- transient/non-authoritative preview;
- no authoritative write path;
- transcript/candidate non-persistence;
- H-01..H-06 deterministic remediation;
- expanded 22-case promotion suite;
- safe synthetic credential path;
- exact H-06 SHA/run evidence.

Still unsatisfied:

- successful live semantic execution of all 22 cases through selected GPT-5.6 provider path;
- independent review of live provider evidence;
- H-05 production async/blocking remediation;
- stronger executable browser lifecycle evidence before production release.

Therefore:

```text
SAFE CREDENTIAL PATH YES
H-01..H-06 DETERMINISTIC CLOSURE YES
LIVE SEMANTIC QUALIFICATION PASS NO
RELEASE READY NO
RUNTIME RELEASE PR NO
DEPLOY NO
IDENTIFIABLE TRANSCRIPT USE NO
PR-2 / REAL PILOT NO
```

## 11. Release boundary

No runtime release PR, merge/deploy, identifiable transcript use, PR-2 or real-patient pilot may occur until the live 22-case provider qualification passes, its evidence is independently reviewed, and separate production-release blockers are closed.
