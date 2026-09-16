# SLICE_PLAN_CURRENT.md — PR-1 Heidi-first Transcript Intake + Candidate Extraction v1

> **STATUS:** LIVE SYNTHETIC PROVIDER EVAL ATTEMPTED / H-06 OPENAI STRUCTURED-SCHEMA REPLAN — NOT RELEASE READY.
> **Activated:** 2026-09-16 Asia/Nicosia.
> **Replanned:** 2026-09-16 after independent READ-ONLY review; replanned again after first live provider schema rejection.
> **Slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Activation main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Runtime branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact deterministic runtime/eval head:** `0bb339986d5fc47fba67da941127c29f868f54b5`.
> **Deterministic/inherited gate:** `35091549349` — SUCCESS.
> **Synthetic qualification suite:** 22 synthetic/de-identified cases.
> **Synthetic-only wrapper head:** `e32d6e2006c6181e2a2f88ad596f307d98419753`.
> **First live provider run:** `35138694138` — schema failure before case 1.
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

## 3. H-01 through H-04 — CLOSED deterministically

The prior independent review findings remain closed on exact runtime/eval head `0bb339986d5fc47fba67da941127c29f868f54b5` with workflow `35091549349` SUCCESS:

- **H-01:** default-deny unexpected assertion protection;
- **H-02:** unique candidate component identity, matcher locality and repeated-event grouping;
- **H-03:** separate `synthetic_eval` authorization that does not require identifiable-PHI approval;
- **H-04:** exact reviewed runtime ranges plus formal-result/source semantic guards.

The promotion suite remains 22 synthetic/de-identified cases and includes repeated events, prompt-injection text, referral/request versus completed result, prescription versus administration, self-correction, third-party source, out-of-range numerics, planned versus completed administration and explicit negated exposure.

## 4. First live selected-model qualification attempt

A bounded synthetic-only workflow was introduced at:

```text
.github/workflows/pr1-transcript-live-provider-eval.yml
```

Exact wrapper commit:

```text
e32d6e2006c6181e2a2f88ad596f307d98419753
```

Run `35138694138` established the intended execution boundary:

```text
OPENAI_API_KEY available from Actions secret store
CLINICAL_TRANSCRIPT_AI_ENABLED=true
CLINICAL_TRANSCRIPT_SYNTHETIC_EVAL_ENABLED=true
CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED=false
model=gpt-5.6
purpose=synthetic_eval
fixture count=22
fixture IDs unique
```

The secret value remained masked. No production configuration was changed and identifiable-PHI approval remained false.

The provider call then failed before case 1 with:

```text
HTTP 400
code=invalid_json_schema
param=text.format.schema
response_format=ProviderTranscriptExtractionV1
context=('properties', 'value')
reason='oneOf' is not permitted
```

Therefore the credential prerequisite is closed, but live semantic qualification has **not** yet occurred.

## 5. H-06 — Structured Outputs schema compatibility

### Finding

`CandidateValueV1` is currently represented as a Pydantic discriminated union over:

```text
TextValueV1
CodeValueV1
NumberValueV1
IntegerValueV1
BooleanValueV1
QuantityValueV1
DateValueV1
```

The generated response schema uses `oneOf` for the candidate `value` property. The live OpenAI Structured Outputs request rejected that schema shape. The selected GPT-5.6 model supports Structured Outputs, but strict response schemas are subject to the provider-supported JSON-Schema subset; the live API is authoritative for this exact contract.

### Required remediation

The provider-facing typed schema must be changed so that:

1. the JSON schema sent through `responses.parse` does not contain unsupported `oneOf` constructs;
2. local Pydantic validation still preserves the exact typed value variants and their `kind` literals;
3. date validators, quantity units, duplicate concept-key rejection and all existing strict `extra=forbid` behavior remain intact;
4. no fallback to permissive free-form JSON/text is introduced;
5. the model still returns the same logical provider contract consumed by deterministic mapping and the 22-case oracle.

A plain/non-discriminated typed union is acceptable only if its generated schema is supported by the live Structured Outputs path and local tests prove unambiguous validation by the `kind` literal.

### Error classification remediation

The live run also proved that the adapter currently collapses an HTTP 400 `invalid_json_schema` into `ProviderUnavailable`.

That is incorrect because the condition is deterministic and non-transient. The adapter must distinguish provider-contract/request-schema failure from transport/rate-limit/timeout/upstream unavailability. Within the existing sanitized public boundary, this must fail closed as a deterministic provider-contract error rather than invite retries as an availability incident.

### Deterministic acceptance for H-06

Before another live provider run, the exact remediation head must prove:

- generated provider response schema contains no unsupported `oneOf` path;
- all seven typed candidate value variants still validate correctly;
- invalid/mismatched value shapes still fail Pydantic validation;
- duplicate concept keys still fail;
- impossible dates and vague-date protections still fail closed;
- `invalid_json_schema`-style adapter failures are not classified as `ProviderUnavailable`;
- existing H-01..H-04 tests remain green;
- all 22 fixture-contract tests remain green;
- inherited protected-clinical, Clinical Documents, navigation and bounded-scope gates remain green.

The exact SHA + workflow run must be checkpointed in `CURRENT_OPERATIONAL.md` before another deliberate live qualification.

## 6. Live qualification gate after H-06

Only after H-06 deterministic closure:

```text
secure Actions OPENAI_API_KEY
+ synthetic eval gate true
+ PHI approval false
+ OpenAI-compatible strict typed schema
→ deliberately retrigger bounded synthetic-only workflow
→ run all 22 cases through gpt-5.6
→ require failed=0 under default-deny oracle
→ checkpoint exact live evidence
→ fresh independent READ-ONLY evidence review
```

The failed run `35138694138` is adapter/schema evidence only and must not be counted as provider semantic-performance evidence.

## 7. H-05 retained production-release blocker

The FastAPI route is async while the OpenAI adapter is synchronous and the current production command uses a single uvicorn worker. Before production enablement, provider execution must not block the event loop for the full provider timeout.

H-05 does not block isolated CLI synthetic qualification, but it must close before release/deploy. Stronger executable browser lifecycle/BFCache/logout cleanup evidence also remains production-release debt.

## 8. Definition of Done status

Satisfied deterministically:

- protected transcript UI + endpoint;
- strict request/response/provider contracts locally;
- provider target-path isolation;
- transient/non-authoritative preview;
- no authoritative write path;
- transcript/candidate non-persistence;
- H-01..H-04 remediation;
- expanded 22-case promotion suite;
- safe synthetic credential path proven in GitHub Actions.

Still unsatisfied:

- H-06 provider-facing Structured Outputs schema compatibility;
- successful live execution of all 22 cases;
- independent review of live provider evidence;
- H-05 production async/blocking remediation;
- stronger executable browser lifecycle evidence before production release.

Therefore:

```text
SAFE CREDENTIAL PATH YES
LIVE PROVIDER REQUEST REACHED API YES
LIVE SEMANTIC CASE EVALUATION STARTED NO
H-06 OPEN
RELEASE READY NO
RUNTIME RELEASE PR NO
DEPLOY NO
IDENTIFIABLE TRANSCRIPT USE NO
PR-2 / REAL PILOT NO
```

## 9. Release boundary

The next permitted runtime mutation is the bounded H-06 provider-schema/error-classification remediation defined above. No second live provider run, release PR, merge/deploy, identifiable transcript use, PR-2 or real-patient pilot may occur until H-06 deterministic evidence is durably checkpointed.
