# SLICE_PLAN_CURRENT.md — PR-1 Heidi-first Transcript Intake + Candidate Extraction v1

> **STATUS:** DETERMINISTIC PROMOTION HARNESS READY / LIVE SYNTHETIC PROVIDER EVAL CREDENTIAL HOLD — NOT RELEASE READY.
> **Activated:** 2026-09-16 Asia/Nicosia.
> **Replanned:** 2026-09-16 Asia/Nicosia after independent READ-ONLY review.
> **Evidence reconciled:** 2026-09-16 after H-01..H-04 remediation + expanded promotion suite.
> **Slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Activation main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Runtime branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact tested runtime/eval head:** `0bb339986d5fc47fba67da941127c29f868f54b5`.
> **Deterministic/inherited gate:** `35091549349` — SUCCESS.
> **Synthetic qualification suite:** 22 synthetic/de-identified cases.
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

## 2. Independent-review disposition and remediation status

The independent review found no Critical privacy breach or authoritative-write escape, but it identified four High live-promotion blockers:

- **H-01:** generic false-PASS risk from unexpected/hallucinated extra clinical assertions;
- **H-02:** duplicate `concept_key` identity and matcher cross-wiring defect;
- **H-03:** synthetic qualification coupled to identifiable-PHI approval;
- **H-04:** incomplete deterministic enforcement of actual runtime ranges/semantic contracts.

All four are now **closed deterministically** on exact runtime/eval head `0bb339986d5fc47fba67da941127c29f868f54b5` with gate `35091549349` SUCCESS.

The requested repeated-event/adversarial suite expansion is also complete deterministically at 22 cases.

A separate **H-05** remains a production-release blocker: synchronous provider execution in the current single-worker async web process. H-05 does not block isolated command-line synthetic provider qualification, but it must close before production enablement/deploy.

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

## 4. H-01 closure — default-deny promotion oracle

Every returned provider component must be covered by an explicit fixture authorization rule:

- `required_assertions` with explicit `concept_key`; or
- `allowed_assertions`; or
- an explicitly required same-candidate event-group rule.

Matching can constrain:

```text
semantic_type
source_assertion
concept_key
value
mapping
```

Unexpected components fail the case with coded `unexpected_assertion_<concept>` output. The oracle also rejects forbidden assertions/concepts, invented exact dates, semantic-count violations, unverifiable evidence and non-ephemeral/authoritative response metadata.

Deterministic tests prove an otherwise-correct case fails when a clinically material unsupported extra assertion is added.

## 5. H-02 closure — component identity and repeated events

Within one `ProviderCandidateV1`, `components[].concept_key` values must be unique. Duplicate keys are invalid structured provider output and fail before mapping.

Distinct real-world events use **separate candidates**. The eval matcher binds expected value/mapping to the same candidate/component identity and supports `required_candidate_groups`, which require related facts such as fracture site + fracture date to co-exist in the same event candidate.

Deterministic tests prove:

- duplicate key in one candidate is rejected;
- repeated same concept across separate event candidates is allowed;
- cross-paired repeated-event facts fail event-group qualification.

## 6. H-03 closure — synthetic and identifiable-PHI gates are separate

### Clinical runtime purpose

The protected clinical/default provider route still requires:

```text
CLINICAL_TRANSCRIPT_AI_ENABLED=true
OPENAI_API_KEY present
CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED=true
```

No production/clinical caller gets a synthetic bypass.

### Synthetic-eval purpose

The engineering evaluation purpose requires:

```text
CLINICAL_TRANSCRIPT_AI_ENABLED=true
OPENAI_API_KEY present
CLINICAL_TRANSCRIPT_SYNTHETIC_EVAL_ENABLED=true
```

and does **not** require `CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED=true`.

The live eval runner uses this explicit synthetic purpose. Deterministic tests prove clinical use remains blocked while synthetic qualification can be configured independently.

## 7. H-04 closure — exact runtime target validation

Provider output is untrusted. A concept key with a known target path is not `mapped` unless its value and semantic/source contract fit the actual current runtime target.

Enforced runtime ranges:

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

Original formal FRAX percentage fields map only from `semantic_type=objective_result`. Third-party facts cannot become patient-card truth, and explicit negative treatment/administration exposure cannot become a positive runtime episode. Out-of-range, wrong-type, unsupported-unit or wrong-semantic values fail closed as ambiguous/unmapped rather than being silently coerced.

## 8. Expanded 22-case promotion suite

The original 13 synthetic/de-identified scenarios remain, and the promotion suite now adds explicit cases for:

1. repeated fracture/event identity with repeated concept keys across separate candidates;
2. embedded transcript instruction/prompt-injection text ignored as untrusted material;
3. referral/request for an investigation ≠ completed investigation/result;
4. prescription/recommendation ≠ medication actually taken/administered;
5. self-correction preserving the corrected assertion without keeping the superseded value as current truth;
6. third-party treatment history not attributed to the patient;
7. out-of-range numeric transcription that extracts the stated value but fails deterministic runtime mapping;
8. planned administration ≠ completed administration;
9. explicit negated treatment exposure without creating positive administration/treatment truth.

The evaluator additionally fails clean explicit cases on low-confidence output unless the fixture explicitly allows ambiguity/uncertainty.

Every promotion case is evaluated under the H-01 default-deny assertion surface.

## 9. Deterministic acceptance evidence

GitHub Actions run `35091549349` executed against exact runtime/eval head `0bb339986d5fc47fba67da941127c29f868f54b5` and completed SUCCESS for:

- Python/browser syntax;
- all focused PR-1 privacy/contract/mapping/eval tests;
- duplicate-concept contract tests;
- exact numeric-range/formal-FRAX semantic tests;
- generic default-deny unexpected-assertion tests;
- synthetic-vs-PHI authorization tests;
- repeated-event grouping tests;
- low-confidence qualification behavior;
- expanded fixture-contract validation;
- inherited protected-clinical regressions;
- inherited Clinical Documents regressions;
- inherited workspace/navigation regression;
- bounded PR-1 scope guard.

No live provider/model call was used for this deterministic evidence.

## 10. Live selected-model qualification gate

The code/eval prerequisites are now satisfied. The remaining execution prerequisite is a safe non-production credential path.

```text
safe non-production credential in execution secret store
+ CLINICAL_TRANSCRIPT_AI_ENABLED=true
+ CLINICAL_TRANSCRIPT_SYNTHETIC_EVAL_ENABLED=true
+ identifiable PHI approval remains false/independent
→ run 22-case synthetic/de-identified suite through selected OpenAI adapter/model
→ require zero failed cases under default-deny promotion oracle
→ checkpoint exact SHA/run/provider/model evidence
→ independent evidence review
```

Do not place the credential in chat, source, workflow YAML, logs or public repository content.

A bounded synthetic-only workflow wrapper may be introduced only after the secure credential prerequisite exists. It must consume repository fixtures only and emit coded qualification results rather than provider/transcript payloads.

## 11. H-05 retained production-release blocker

The FastAPI route is async while the OpenAI adapter is synchronous and the current production command uses a single uvicorn worker. Before production enablement, provider execution must not block the event loop for the full provider timeout.

A later remediation must preserve the sanitized error/timeout contract and use a bounded off-loop/thread mechanism or async provider path. H-05 is **not required before isolated CLI synthetic qualification**, but it is required before release/deploy.

Stronger executable browser lifecycle/BFCache/logout cleanup evidence is also retained as production-release debt.

## 12. Definition of Done status

Satisfied deterministically:

- protected transcript UI + endpoint;
- strict request/response/provider schemas;
- provider target-path isolation;
- transient/non-authoritative preview;
- no authoritative write path;
- transcript/candidate non-persistence;
- calendar-valid normalized exact dates and vague-date fail-closed behavior;
- H-01 generic unexpected-assertion default-deny evaluator;
- H-02 unique component identity and same-event grouping protection;
- H-03 separate synthetic-eval authorization boundary;
- H-04 reviewed runtime range/semantic/source validation;
- expanded 22-case promotion suite deterministic validation;
- inherited regression + scope-gate success on exact runtime/eval SHA.

Still unsatisfied:

- selected provider/model live 22-case qualification through a safe non-production credential path;
- independent review of that live qualification evidence;
- H-05 production async/blocking remediation;
- stronger executable browser lifecycle evidence before production release.

Therefore:

```text
DETERMINISTIC PROMOTION HARNESS READY YES
LIVE PROMOTION EVAL EXECUTED NO
LIVE PROMOTION EVAL EXECUTION HOLD SAFE CREDENTIAL
RELEASE READY NO
RUNTIME RELEASE PR NO
DEPLOY NO
IDENTIFIABLE TRANSCRIPT USE NO
PR-2 / REAL PILOT NO
```

## 13. Release boundary

The next permitted material transition is the bounded synthetic-only live provider qualification wrapper **after** a safe non-production credential exists in the execution secret store. No release PR, merge/deploy, identifiable transcript use, PR-2 or real-patient pilot may occur before live qualification evidence is checkpointed and independently reviewed.
