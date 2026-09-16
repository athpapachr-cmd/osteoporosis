# SLICE_PLAN_CURRENT.md — PR-1 Heidi-first Transcript Intake + Candidate Extraction v1

> **STATUS:** INDEPENDENT-REVIEW REPLAN / CODE+EVAL HARDENING HOLD — NOT READY FOR LIVE PROMOTION EVAL OR RELEASE.
> **Activated:** 2026-09-16 Asia/Nicosia.
> **Replanned:** 2026-09-16 Asia/Nicosia after independent READ-ONLY review.
> **Slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Activation main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Runtime branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Last pre-replan tested runtime head:** `a79d68915bde230a53bb7b5fd31a4104a491b058`.
> **Last pre-replan deterministic gate:** `35084122094` — SUCCESS.
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

## 2. Independent-review disposition and REPLAN scope

The original implementation remains architecturally viable, but independent review found four High promotion blockers that invalidate the previous `credential-only HOLD` state:

- **H-01:** provider-eval oracle lacks generic default-deny protection against unsupported extra clinical assertions;
- **H-02:** duplicate `concept_key` values can break deterministic component identity and allow matcher cross-wiring;
- **H-03:** synthetic qualification is incorrectly coupled to the identifiable-PHI approval gate;
- **H-04:** deterministic mapping does not yet enforce all actual runtime numeric ranges and formal-result semantic requirements.

A separate **H-05** remains a production-release blocker: synchronous provider execution currently runs inside a single-worker async web process. H-05 does not block isolated command-line synthetic provider qualification, but it must close before production enablement/deploy.

This replan does not broaden PR-1 into authoritative write, PR-2, pilot activation or unrelated product work.

## 3. Preserved contracts and privacy invariants

### Request contract

```json
{
  "schema_version": "clinical_transcript_extract_request_v1",
  "source_type": "heidi_transcript",
  "module": "osteoporosis",
  "encounter_phase": "during_visit",
  "language": "el",
  "transcript": "<string>",
  "context": {"encounter_archetype": null}
}
```

Unknown fields are rejected. Request body ceiling remains 512 KiB; transcript ceiling remains 120,000 Unicode characters.

### Success contract

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

### Hard invariants

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

## 4. H-01 remediation — promotion evaluator must default-deny unexpected assertions

The provider qualification oracle must no longer pass a case merely because required facts exist and selected forbidden facts do not.

### Required behavior

Each promotion fixture defines its expected/allowed assertion surface. For every returned candidate component, the evaluator must prove that the component is covered by either:

- a `required_assertions` rule with an explicit `concept_key`; or
- an explicit `allowed_assertions` rule.

Coverage matching must use the same candidate identity and may constrain:

```text
semantic_type
source_assertion
concept_key
value
mapping
```

Any returned component that is not covered by the fixture allowlist fails the case with a coded `unexpected_assertion`/equivalent failure. Promotion fixtures must not use a permissive wildcard for clinically material mapped concepts.

The evaluator must continue to reject:

- forbidden assertions/concepts;
- invented exact dates;
- semantic count violations;
- unverifiable evidence warnings;
- authoritative/non-ephemeral response metadata.

A deterministic test must prove that an otherwise-correct case fails if an unrelated hallucinated mapped treatment/fact is added.

## 5. H-02 remediation — component identity must be unique and matcher-local

### Provider contract

Within one `ProviderCandidateV1`, `components[].concept_key` values must be unique. Duplicate concept keys are invalid structured provider output and fail closed before mapping.

Repeated real-world events remain supported by using **separate candidates**, not duplicate same-key components inside one candidate.

### Mapper/guard behavior

The hardened mapper may assume candidate-local concept-key uniqueness only after schema validation. It must never resolve a mapping against a different same-key component.

### Eval matcher behavior

When matching an expected concept, value and mapping, the evaluator must bind them to the same candidate/component identity. A value from one component/candidate must not be combined with a mapping from another to satisfy one rule.

Deterministic tests must cover:

- duplicate key in one provider candidate → schema rejection;
- repeated same concept across separate candidates/events → allowed and independently evaluated;
- matcher cannot cross-wire value and mapping identities.

## 6. H-03 remediation — synthetic qualification and identifiable-PHI approval are separate gates

The default clinical provider path remains fail-closed for identifiable transcript use.

### Clinical runtime purpose

The protected clinical route continues to require:

```text
CLINICAL_TRANSCRIPT_AI_ENABLED=true
OPENAI_API_KEY present
CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED=true
```

No production/clinical caller gets a bypass.

### Synthetic-eval purpose

The provider adapter gains an explicit synthetic-evaluation purpose/mode available only to the engineering eval harness. Synthetic mode requires:

```text
CLINICAL_TRANSCRIPT_AI_ENABLED=true
OPENAI_API_KEY present
CLINICAL_TRANSCRIPT_SYNTHETIC_EVAL_ENABLED=true
```

and must **not** require `CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED=true`.

The synthetic runner uses only repository fixtures marked/validated as synthetic/de-identified. The identifiable-PHI flag remains false/independent during qualification.

Deterministic tests must prove both directions:

- clinical/default provider remains blocked without PHI approval;
- synthetic-eval provider can be considered configured with the synthetic gate + credential while PHI approval remains false.

## 7. H-04 remediation — exact runtime target validation

Provider output is untrusted. A concept key with a valid target path is not `mapped` unless its value and semantic type satisfy the actual current runtime contract.

Verified runtime ranges to enforce locally:

```text
anthropometrics.weight              20..300 kg
anthropometrics.current_height      100..220 cm
frax.mof_percent                    0..100
frax.hip_percent                    0..100
dxa.*_bmd                           0.1..3 g/cm²
dxa.*_t_score                       -8..5
risk.falls_last_12_months           integer 0..50
risk.cfs_score                      integer 1..9
treatment.duration_years            0..50 years
```

Existing fixed enums remain guarded for fracture site, FRAX tool, risk category, VFA indication/action/modality, treatment status and administration status.

### Formal-result semantic protection

Original formal FRAX percentage fields may be mapped only from `semantic_type=objective_result`. A clinician interpretation using the same formal concept key must fail closed rather than overwrite original formal FRAX truth.

DXA BMD/T-score and supported laboratory values continue to require `objective_result` as already designed.

Out-of-range, wrong-type, unsupported-unit or wrong-semantic values become `ambiguous` or `unmapped`; they are never silently coerced into authoritative-looking runtime values.

## 8. Promotion suite expansion

The existing 13 scenarios remain useful but are no longer sufficient promotion evidence by themselves.

Before live qualification, the synthetic/de-identified suite must include the existing scenarios plus explicit cases for at least:

1. repeated fracture/event identity with repeated concept keys across separate candidates;
2. embedded transcript instruction/prompt-injection text ignored as untrusted material;
3. referral/request for an investigation ≠ completed investigation/result;
4. prescription/recommendation ≠ medication actually taken/administered;
5. self-correction of timing/content (`June... no, May`) preserving the corrected assertion without inventing a second fact;
6. third-party treatment history not attributed to the patient;
7. out-of-range numeric transcription that extracts but fails deterministic runtime mapping;
8. planned administration ≠ completed administration;
9. explicit negated treatment exposure without creating positive administration/treatment truth.

Clean, explicit cases should also reject low-confidence output unless the fixture explicitly represents ambiguity/uncertainty.

Every live promotion case must use the H-01 default-deny unexpected-assertion check.

## 9. H-05 retained production-release blocker

The FastAPI route is async while the OpenAI adapter is synchronous and the current production command uses a single uvicorn worker. Before production enablement, provider execution must not be allowed to block the event loop for the full provider timeout.

Acceptable implementation must preserve the same sanitized error/timeout contract and should use a bounded thread/off-loop mechanism or an async provider path. This remediation is **not required before isolated CLI synthetic provider qualification**, but it is required before release/deploy.

## 10. Deterministic acceptance gate after H-01..H-04

The exact new runtime head must pass:

- Python/browser syntax;
- all existing PR-1 privacy/contract/mapping/UI tests;
- new duplicate-concept schema tests;
- new exact numeric-range and formal-FRAX semantic tests;
- new eval default-deny unexpected-assertion tests;
- new synthetic-vs-PHI authorization tests;
- expanded fixture-contract validation;
- inherited protected-clinical regressions;
- inherited Medical Report regressions;
- inherited workspace/navigation regression;
- bounded PR-1 scope guard.

The resulting exact SHA + workflow run must be checkpointed in `CURRENT_OPERATIONAL.md` before any live provider execution.

## 11. Live selected-model qualification gate

Only after H-01 through H-04 close deterministically:

```text
safe non-production credential path exists
+ CLINICAL_TRANSCRIPT_SYNTHETIC_EVAL_ENABLED=true
+ identifiable PHI approval remains independent
→ run expanded synthetic/de-identified suite through selected OpenAI adapter/model
→ require zero failed cases under default-deny promotion oracle
→ checkpoint exact provider/model/eval evidence
→ independent evidence review
```

A run of the old 13-case suite under the old oracle is exploratory only and cannot be used as promotion evidence.

## 12. Definition of Done status

Currently satisfied from the pre-replan implementation:

- protected transcript UI + endpoint;
- strict request/response contracts;
- provider target-path isolation;
- transient/non-authoritative preview;
- no authoritative write path;
- transcript/candidate non-persistence;
- calendar-valid normalized exact dates and vague-date fail-closed behavior;
- prior deterministic/inherited CI baseline.

Currently unsatisfied:

- H-01 generic unexpected-assertion default-deny evaluator;
- H-02 candidate concept-key identity contract + matcher-local proof;
- H-03 separate synthetic-eval authorization boundary;
- H-04 complete actual-runtime range/semantic validation;
- expanded promotion suite deterministic validation;
- selected provider/model live qualification;
- H-05 production async/blocking remediation;
- stronger executable browser lifecycle evidence before production release.

Therefore:

```text
IMPLEMENTATION BASELINE EXISTS
REPLAN ACTIVE
LIVE PROMOTION EVAL READY NO
RELEASE READY NO
RUNTIME RELEASE PR NO
DEPLOY NO
IDENTIFIABLE TRANSCRIPT USE NO
PR-2 / REAL PILOT NO
```

## 13. Release boundary

The next permitted runtime mutation is the bounded H-01/H-02/H-03/H-04 remediation defined above. No live provider promotion run, release PR, merge/deploy, identifiable transcript use, PR-2 or real-patient pilot may occur until the corresponding prior gate is durably checkpointed.
