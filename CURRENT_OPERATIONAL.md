# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — DETERMINISTIC PROMOTION HARNESS READY / LIVE SYNTHETIC PROVIDER EVAL CREDENTIAL HOLD — NOT RELEASE READY.
> **Updated:** 2026-09-16 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** this bounded PR-1 implementation lifecycle only.
> **Runtime implementation branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact tested runtime/eval head:** `0bb339986d5fc47fba67da941127c29f868f54b5`.
> **Deterministic/inherited gate:** `35091549349` — SUCCESS.
> **Expanded synthetic qualification suite:** 22 synthetic/de-identified cases.
> **Prior live provider probe:** `35056606836` — no Actions credential; no provider call.
> **Medical Report V1.1:** CLOSED; do not reopen without separate authority.

## Product-owner authority

On 2026-09-16 the Product Owner explicitly authorized bounded PR-1 implementation. That authority covers code, deterministic tests/evals and implementation-candidate preparation inside this slice.

It does **not** authorize runtime release PR merge/deploy, identifiable transcript processing, PR-2 authoritative writes, real-patient pilot use or unrelated product mutation.

## Independent-review replan — closure status

A separate fresh-bootstrap independent READ-ONLY review found no Critical privacy breach or authoritative-write escape but identified four High promotion blockers plus a separate production-release blocker.

The previous `credential-only HOLD` was correctly superseded while those code/eval findings were unresolved. The remediation and expanded deterministic qualification harness are now complete and proven on exact runtime/eval head `0bb339986d5fc47fba67da941127c29f868f54b5`.

### H-01 — generic unexpected/hallucinated-extra false PASS

**CLOSED deterministically.**

The promotion evaluator is default-deny at returned-component level. Every provider component must be covered by an explicit required/allowed assertion rule. Unexpected clinically material extras fail the case with coded `unexpected_assertion_<concept>` output. The oracle also checks non-ephemeral metadata, forbidden assertions/concepts, invented exact dates, semantic counts and evidence-verification warnings.

### H-02 — duplicate concept identity / matcher cross-wiring

**CLOSED deterministically.**

`ProviderCandidateV1` rejects duplicate `components[].concept_key` values inside one candidate. Repeated real-world events use separate candidates. Eval matching binds semantic/source/value/mapping to the same candidate/component identity, and repeated-event fixtures add same-candidate grouping checks so individually correct facts cannot pass when paired to the wrong event.

### H-03 — synthetic qualification coupled to identifiable-PHI approval

**CLOSED deterministically.**

Clinical/default provider purpose remains fail-closed behind:

```text
CLINICAL_TRANSCRIPT_AI_ENABLED=true
OPENAI_API_KEY present
CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED=true
```

Synthetic qualification uses a separate engineering-only purpose requiring:

```text
CLINICAL_TRANSCRIPT_AI_ENABLED=true
OPENAI_API_KEY present
CLINICAL_TRANSCRIPT_SYNTHETIC_EVAL_ENABLED=true
```

and does not require identifiable-PHI approval. Deterministic tests prove the two gates remain distinct.

### H-04 — incomplete runtime-target range/semantic guards

**CLOSED for the reviewed runtime contracts.**

The deterministic guard enforces the current runtime ranges and types for weight, height, FRAX percentages, DXA BMD/T-score, falls/CFS and treatment duration. Original formal FRAX percentages require `objective_result`. Third-party patient-card facts fail closed as `THIRD_PARTY_SOURCE_NOT_PATIENT`, and negated treatment/administration exposure cannot become a positive runtime episode.

## Expanded promotion suite — deterministic evidence

The former 13-case suite has been expanded to **22 synthetic/de-identified cases**. In addition to the previous fracture, negation, DXA, labs, options/final-decision, preference, follow-up, garbled speech, FRAX original-vs-adjusted, speaker ambiguity and unrelated-clinical-text coverage, the suite now explicitly covers:

1. repeated fracture-event identity/grouping with repeated concept keys across separate candidates;
2. embedded transcript prompt/instruction text as untrusted material;
3. referral/request ≠ completed investigation/result;
4. prescription/recommendation ≠ medication taken/administered;
5. explicit self-correction without preserving the superseded value as current truth;
6. third-party treatment history without patient attribution;
7. out-of-range numeric transcription with deterministic fail-closed mapping;
8. planned administration ≠ completed administration;
9. explicit negated treatment exposure;
10. clean-case low-confidence output as a qualification failure unless ambiguity is explicitly allowed.

GitHub Actions run `35091549349` executed on exact head `0bb339986d5fc47fba67da941127c29f868f54b5` and completed SUCCESS across:

- Python syntax;
- browser syntax;
- focused PR-1 privacy/contract/mapping/eval tests;
- inherited protected-clinical regressions;
- inherited Clinical Documents regressions;
- inherited workspace navigation regression;
- bounded PR-1 scope verification.

No live provider/model call was used for this evidence.

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

## Current lifecycle state

The independent review's H-01/H-02/H-03/H-04 promotion blockers and its requested adversarial/repeated-event suite expansion are now closed deterministically.

Therefore the project may return to the external prerequisite for a **decision-bearing live synthetic provider evaluation**, but only through a safe non-production credential path.

This is now accurately a:

```text
DETERMINISTIC PROMOTION HARNESS READY
→ SAFE NON-PRODUCTION CREDENTIAL HOLD
→ LIVE 22-CASE SELECTED-MODEL PROVIDER EVAL
→ INDEPENDENT EVIDENCE REVIEW
→ SEPARATE RELEASE/PRIVACY DECISIONS
```

## Safe credential boundary

Current connected GitHub tooling does not expose repository Actions secret management, and no production configuration should be mutated or copied into chat merely to manufacture qualification evidence.

Do **not** paste an API key into conversation text, source code, workflow YAML, logs or public repository content.

When a dedicated non-production credential is safely available to the repository's synthetic-eval execution environment, a bounded synthetic-only workflow wrapper may be introduced. It must:

- use the repository synthetic fixtures only;
- set `CLINICAL_TRANSCRIPT_AI_ENABLED=true`;
- set `CLINICAL_TRANSCRIPT_SYNTHETIC_EVAL_ENABLED=true`;
- leave `CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED` false/unset;
- obtain the API credential only from the secure execution secret store;
- execute `evals/transcript_v1/run_provider_eval.py` against the exact selected branch/head;
- emit only coded case results/summary, never transcript/provider payload contents;
- require zero failed cases before any promotion claim.

## Explicitly blocked until live qualification evidence exists

- no runtime release PR;
- no merge/deploy of PR-1;
- no identifiable transcript processing;
- no PR-2 or real-patient pilot;
- no production credential/config mutation to manufacture evidence;
- no claim that deterministic CI alone qualifies the selected live provider/model.

## Separate production-release blockers/debt

**H-05 remains OPEN:** the async clinical route still invokes a synchronous provider client in the current single-worker web process. It does not block isolated command-line synthetic qualification, but it must be remediated and verified before production enablement/deploy.

Stronger executable browser lifecycle/BFCache/logout evidence also remains production-release debt.

## Exact next action

Preserve this HOLD until a dedicated safe non-production provider credential exists in an execution secret store without exposing the key. Then introduce/verify the bounded synthetic-only workflow wrapper, run the **22-case** suite through the selected OpenAI adapter/model, require zero failed cases under the default-deny oracle, checkpoint exact SHA/run/provider/model evidence, and send that evidence for a fresh independent READ-ONLY review.
