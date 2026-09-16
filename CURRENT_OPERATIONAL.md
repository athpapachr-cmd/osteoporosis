# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — LIVE SYNTHETIC PROVIDER EVAL ATTEMPTED / OPENAI STRUCTURED-SCHEMA HOLD — NOT RELEASE READY.
> **Updated:** 2026-09-16 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** this bounded PR-1 implementation lifecycle only.
> **Runtime implementation branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact deterministic runtime/eval head:** `0bb339986d5fc47fba67da941127c29f868f54b5`.
> **Deterministic/inherited gate:** `35091549349` — SUCCESS.
> **Expanded synthetic qualification suite:** 22 synthetic/de-identified cases.
> **Synthetic-only workflow wrapper head:** `e32d6e2006c6181e2a2f88ad596f307d98419753`.
> **First live provider run:** `35138694138` — FAILURE before case 1 due provider schema rejection.
> **Credential boundary:** CLOSED for synthetic evaluation; Actions secret was available and masked, PHI approval remained false.
> **Medical Report V1.1:** CLOSED; do not reopen without separate authority.

## Product-owner authority

On 2026-09-16 the Product Owner explicitly authorized bounded PR-1 implementation and subsequently confirmed that a repository Actions `OPENAI_API_KEY` had been added. That authorizes the already-designed synthetic-only qualification path, not production/PHI use.

This does **not** authorize runtime release PR merge/deploy, identifiable transcript processing, PR-2 authoritative writes, real-patient pilot use or unrelated product mutation.

## Independent-review remediation already closed deterministically

The prior independent READ-ONLY review identified H-01 through H-04. Those remain closed on exact runtime/eval head `0bb339986d5fc47fba67da941127c29f868f54b5` with gate `35091549349` SUCCESS:

- **H-01:** default-deny unexpected/hallucinated assertion oracle;
- **H-02:** unique candidate component identity + same-event grouping protection;
- **H-03:** synthetic-eval authorization separated from identifiable-PHI approval;
- **H-04:** reviewed runtime range/semantic/source guards.

The adversarial/repeated-event promotion suite remains frozen at 22 synthetic/de-identified cases.

## Synthetic-only workflow wrapper — credential boundary CLOSED

A bounded workflow was introduced at:

```text
.github/workflows/pr1-transcript-live-provider-eval.yml
```

Exact wrapper commit:

```text
e32d6e2006c6181e2a2f88ad596f307d98419753
```

Workflow run `35138694138` proved:

- `OPENAI_API_KEY` was available to the Actions job from the secure secret store and was masked in logs;
- `CLINICAL_TRANSCRIPT_AI_ENABLED=true`;
- `CLINICAL_TRANSCRIPT_SYNTHETIC_EVAL_ENABLED=true`;
- `CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED=false`;
- selected model `gpt-5.6`;
- provider purpose `synthetic_eval`;
- frozen fixture count = 22 and IDs unique;
- no production configuration was changed;
- no identifiable transcript/PHI authorization was enabled.

Therefore the prior **safe credential HOLD is closed**.

## Live provider qualification attempt — NEW MATERIAL BLOCKER

The first selected-model request was made, but the API rejected the structured-output schema before any synthetic case could be evaluated.

Observed API failure:

```text
HTTP 400
code: invalid_json_schema
param: text.format.schema
message: Invalid schema for response_format 'ProviderTranscriptExtractionV1':
         In context=('properties', 'value'), 'oneOf' is not permitted.
```

The current `CandidateValueV1` is a Pydantic discriminated union. Its generated schema uses `oneOf` for the `value` property. The selected OpenAI Structured Outputs path accepts only a supported JSON-Schema subset, and the live API rejected this schema shape.

This is **not** a model semantic-performance FAIL. Case 1 never ran. It is a live-discovered provider-adapter/schema-contract failure.

The run also exposed a secondary classification defect: this deterministic HTTP 400 schema rejection was caught by the adapter as `ProviderUnavailable`, which incorrectly describes a local/provider-contract incompatibility as transient upstream unavailability.

## H-06 — OpenAI Structured Outputs schema compatibility

**OPEN / blocks live promotion evaluation.**

Required remediation is bounded to the provider contract:

1. replace the unsupported generated `oneOf` value-union shape with an OpenAI-supported Structured Outputs schema shape while preserving strict local typed validation of text/code/number/integer/boolean/quantity/date values;
2. retain duplicate-concept rejection, exact date validation and all H-01..H-04 protections;
3. add deterministic schema-compatibility coverage so the provider response schema cannot silently reintroduce unsupported `oneOf` constructs;
4. classify deterministic `invalid_json_schema` / response-format request failures as a non-transient provider-contract failure rather than `ProviderUnavailable`;
5. run focused + inherited deterministic CI on the exact remediation head;
6. checkpoint that evidence before another deliberate live provider run.

Only after that checkpoint may the synthetic-only workflow be deliberately retriggered for the full 22-case GPT-5.6 qualification.

## Preserved PR-1 invariants

- raw transcript remains ephemeral and non-authoritative;
- no transcript/candidate DB, encounter, browser-storage or log persistence;
- provider emits semantic assertions, never application/storage paths;
- deterministic Module-01 code owns runtime mapping;
- candidates remain `proposed` and require clinician review;
- speaker/source, polarity, temporality, certainty and semantic distinctions are preserved;
- vague/relative timing cannot become an invented exact date;
- no authoritative patient/encounter/lab/task write exists in PR-1;
- identifiable transcript use remains blocked behind its separate privacy/provider approval gate.

## Explicitly blocked

Until H-06 is remediated, deterministically verified, checkpointed and the live 22-case qualification subsequently passes:

- no runtime release PR;
- no merge/deploy of PR-1;
- no identifiable transcript processing;
- no PR-2 or real-patient pilot;
- no claim that run `35138694138` evaluated provider semantic quality;
- no bypass to JSON mode or weaker unvalidated free-form output merely to make the provider call succeed.

## Separate production-release blockers/debt

**H-05 remains OPEN:** the async clinical route still invokes a synchronous provider client in the current single-worker web process. It does not block isolated CLI synthetic qualification, but it must be remediated and verified before production enablement/deploy.

Stronger executable browser lifecycle/BFCache/logout evidence also remains production-release debt.

## Exact next action

Remediate H-06 on the bounded PR-1 branch: make the typed provider response schema compatible with the supported Structured Outputs JSON-Schema subset without weakening local validation; correct deterministic schema-request error classification; add focused regression coverage; run full deterministic/inherited PR-1 CI; and checkpoint the exact remediation SHA/run before triggering another live 22-case provider evaluation.
