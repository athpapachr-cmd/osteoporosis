# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — H-01..H-04 DETERMINISTIC CLOSURE PROVEN / PROMOTION-SUITE EXPANSION ACTIVE — NOT YET LIVE-EVAL OR RELEASE READY.
> **Updated:** 2026-09-16 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** this bounded PR-1 implementation lifecycle only.
> **Runtime implementation branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact tested H-01..H-04 runtime head:** `66e575257f6e8ed7f2f08aebad5386e1d9826447`.
> **Deterministic/inherited gate:** `35090766419` — SUCCESS.
> **Prior live provider probe:** `35056606836` — no Actions credential; no provider call.
> **Medical Report V1.1:** CLOSED; do not reopen without separate authority.

## Product-owner authority

On 2026-09-16 the Product Owner explicitly authorized bounded PR-1 implementation. That authority covers code, deterministic tests/evals and implementation-candidate preparation inside this slice.

It does **not** authorize runtime release PR merge/deploy, identifiable transcript processing, PR-2 authoritative writes, real-patient pilot use or unrelated product mutation.

## Independent review disposition

The independent READ-ONLY review found no Critical privacy breach or authoritative-write escape but identified four High promotion blockers plus one separate production-release blocker.

The canonical replan replaced the previous `credential-only HOLD` with a code/eval hardening lifecycle. The first remediation transition is now complete and deterministically proven on exact runtime head `66e575257f6e8ed7f2f08aebad5386e1d9826447`.

## H-01 through H-04 closure evidence

### H-01 — generic unexpected/hallucinated-extra false PASS

**Closed deterministically.**

The promotion evaluator is now default-deny at the returned-component level. Every provider component must be covered by an explicit `required_assertions` rule with a concept key or an explicit `allowed_assertions` rule. An otherwise-correct output with an unanticipated extra assertion now fails with a coded `unexpected_assertion_<concept>` failure without requiring fixture authors to predict that hallucination in advance.

### H-02 — duplicate `concept_key` identity defect

**Closed deterministically.**

`ProviderCandidateV1` now rejects duplicate `components[].concept_key` values within one candidate. Repeated real-world events remain representable as separate candidates. Eval matching binds value and mapping checks to the same candidate/component identity rather than searching same-key values and mappings independently.

### H-03 — synthetic qualification coupled to identifiable-PHI approval

**Closed deterministically.**

The clinical/default provider purpose still requires:

```text
CLINICAL_TRANSCRIPT_AI_ENABLED=true
OPENAI_API_KEY present
CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED=true
```

A separate `synthetic_eval` purpose now requires:

```text
CLINICAL_TRANSCRIPT_AI_ENABLED=true
OPENAI_API_KEY present
CLINICAL_TRANSCRIPT_SYNTHETIC_EVAL_ENABLED=true
```

and does not require the identifiable-PHI approval flag. The eval runner uses only the explicit synthetic purpose. Deterministic tests prove that clinical use remains blocked while synthetic qualification can be configured independently.

### H-04 — incomplete runtime-target range/semantic guards

**Closed for the reviewed runtime contracts.**

The hardened mapper now enforces exact runtime ranges for:

```text
weight                 20..300 kg
current height          100..220 cm
FRAX MOF / hip          0..100%
DXA BMD                 0.1..3 g/cm²
DXA T-score             -8..5
falls / 12 months       integer 0..50
CFS                      integer 1..9
treatment duration      0..50 years
```

Original formal FRAX percentage fields additionally require `semantic_type=objective_result`; clinician interpretation cannot map into those original formal-result fields.

## Exact deterministic evidence

GitHub Actions run `35090766419` executed against exact head `66e575257f6e8ed7f2f08aebad5386e1d9826447` and completed SUCCESS across:

- Python syntax;
- browser syntax;
- focused PR-1 privacy/contract/mapping tests, including new H-01..H-04 tests;
- inherited protected-clinical regressions;
- inherited Clinical Documents regressions;
- inherited workspace navigation regression;
- bounded PR-1 scope verification.

No live provider/model call was used for this closure evidence.

## Preserved PR-1 invariants

- raw transcript remains ephemeral and non-authoritative;
- no transcript/candidate DB, encounter, browser-storage or log persistence;
- provider emits semantic assertions, never application/storage paths;
- deterministic Module-01 code owns runtime mapping;
- candidates remain `proposed` and require clinician review;
- vague/relative timing cannot become an invented exact date;
- no authoritative patient/encounter/lab/task write exists in PR-1;
- identifiable transcript use remains blocked behind its separate privacy/provider approval gate.

## Remaining promotion work before live selected-model evaluation

The old 13-case suite is still not sufficient promotion evidence. The next permitted material transition is expansion/hardening of the synthetic/de-identified qualification suite to cover at least:

1. repeated-event identity/grouping with repeated concept keys across separate candidates;
2. embedded prompt/instruction text as untrusted transcript material;
3. referral/request ≠ completed investigation/result;
4. prescription/recommendation ≠ medication taken/administered;
5. self-correction;
6. third-party history without patient attribution;
7. out-of-range numeric transcription with deterministic fail-closed mapping;
8. planned administration ≠ completed administration;
9. explicit negated treatment exposure.

Clean explicit cases should also fail promotion qualification on low-confidence output unless ambiguity is part of the fixture contract.

After that suite transition, run deterministic/inherited CI on the exact new head and checkpoint the evidence before any live provider execution.

## Explicitly blocked

- no live provider promotion run yet;
- no runtime release PR;
- no merge/deploy of PR-1;
- no identifiable transcript processing;
- no PR-2 or real-patient pilot;
- no production credential/config mutation to manufacture evidence.

## Separate production-release blocker

**H-05 remains open:** the async clinical route still invokes a synchronous provider client in the current single-worker web process. This does not block isolated command-line synthetic qualification, but it must be remediated and verified before production enablement/deploy. Stronger executable browser lifecycle evidence also remains release debt.

## Exact next action

Expand the synthetic qualification suite and deterministic coverage under the frozen `SLICE_PLAN_CURRENT.md` contract, then execute the full deterministic/inherited gate on the exact new runtime/eval head. Only after that checkpoint may the project return to the safe non-production credential prerequisite for live selected-model provider qualification.
