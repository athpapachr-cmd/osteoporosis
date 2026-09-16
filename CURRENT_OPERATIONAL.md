# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — INDEPENDENT REVIEW REPLAN / CODE+EVAL HARDENING HOLD — NOT READY FOR LIVE PROMOTION EVAL OR RELEASE.
> **Updated:** 2026-09-16 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** this bounded PR-1 implementation lifecycle only.
> **Runtime implementation branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Branch head before this checkpoint:** `d1c3f0371e46b4a17c16654ab0b5127c8dc1c7af`.
> **Last exact tested runtime head before independent review:** `a79d68915bde230a53bb7b5fd31a4104a491b058`.
> **Last deterministic/inherited gate:** `35084122094` — SUCCESS on that older runtime head.
> **Focused PR-1 tests before replan:** 26 PASS.
> **Prior synthetic suite:** 13 synthetic/de-identified scenarios.
> **Prior live provider probe:** `35056606836` — no Actions credential; no provider call.
> **Medical Report V1.1:** CLOSED; do not reopen without separate authority.

## Product-owner authority

On 2026-09-16 the Product Owner explicitly authorized bounded PR-1 implementation. That authority covers code, deterministic tests/evals and implementation-candidate preparation inside this slice.

It does **not** authorize runtime release PR merge/deploy, identifiable transcript processing, PR-2 authoritative writes, real-patient pilot use or unrelated product mutation.

## Independent READ-ONLY review — material REPLAN trigger

A separate fresh-bootstrap independent review evaluated the current branch and hardened 13-case suite without mutation. It found no Critical privacy breach or authoritative-write escape, but it invalidated the previous operational statement that the only remaining blocker was an external credential prerequisite.

The review identified four High findings that must close before a live synthetic provider run can be treated as promotion evidence:

1. **H-01 — generic unexpected/hallucinated-extra false-PASS path.** The eval oracle can accept required facts while allowing additional clinically material unsupported assertions unless those extras are explicitly forbidden case-by-case.
2. **H-02 — duplicate `concept_key` identity defect.** Provider candidates do not currently require component-key uniqueness; the guard resolves by first matching component and the eval matcher can mix same-key value/mapping evidence.
3. **H-03 — synthetic eval coupled to identifiable-PHI approval.** The provider/eval path currently requires `CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED` even for synthetic/de-identified qualification, conflating two authorization boundaries.
4. **H-04 — incomplete deterministic runtime-target guards.** Current mapping does not fully enforce actual runtime numeric ranges and semantic-type requirements for several mapped targets, including weight/height, FRAX percentages, DXA BMD/T-score, falls/CFS and original formal FRAX semantics.

The review also identified:

- **H-05 — synchronous provider call in the single-worker async web process.** This is a production-release blocker but not an isolated command-line synthetic provider-eval blocker.
- Medium coverage debt for repeated-event identity/grouping, adversarial semantic distinctions, low-confidence qualification behavior and executable browser lifecycle evidence.

Therefore the previous label:

```text
credential-only HOLD
```

is superseded by:

```text
CODE/EVAL HARDENING HOLD
→ deterministic evidence on exact new runtime head
→ safe synthetic-only credential path
→ live expanded provider evaluation
→ independent evidence review
→ separate release/privacy decision
```

## Preserved PR-1 invariants

The following remain mandatory during replan/hardening:

- raw transcript is ephemeral and non-authoritative;
- no transcript/candidate DB, encounter, localStorage/sessionStorage/indexedDB or log persistence;
- provider emits semantic assertions, never application/storage paths;
- deterministic Module-01 code owns target mapping;
- preserve negation, temporality, speaker/source, certainty and semantic distinctions;
- vague/relative timing must not become an invented exact date;
- candidates remain `proposed` and require clinician review;
- no PR-1 authoritative patient/encounter/lab/task write;
- identifiable transcript use remains blocked behind a separate privacy/provider approval gate;
- synthetic/de-identified provider qualification must not require pretending that identifiable-PHI approval is already granted.

## Current proven evidence that remains valid

The older exact runtime head `a79d68915bde230a53bb7b5fd31a4104a491b058` still has deterministic gate `35084122094` SUCCESS and demonstrates the pre-review implementation baseline, including strict structured contracts, calendar-valid dates, provider-path isolation, transient browser preview, no authoritative write and the then-current mapper/eval safeguards.

That evidence does **not** close H-01 through H-04 and must not be represented as promotion-quality live-eval readiness.

## Exact next action

Before any runtime mutation, reconcile `SLICE_PLAN_CURRENT.md` to this independent-review disposition and freeze the remediation contract for H-01/H-02/H-03/H-04 plus the expanded synthetic qualification suite.

After that canonical design checkpoint is durable, the authorized implementation sequence is:

```text
H-01 evaluator default-deny unexpected-assertion protection
→ H-02 unique concept identity + matcher hardening
→ H-03 synthetic-only provider authorization separated from identifiable-PHI approval
→ H-04 exact runtime numeric/semantic guards
→ expand deterministic/eval coverage for repeated events + adversarial semantic cases + range failures
→ deterministic/inherited CI on exact new runtime head
→ canonical checkpoint
```

Only after those blockers are closed may a safe non-production credential path be attached to a synthetic-only execution wrapper and a live selected-model provider evaluation be considered promotion evidence.

## Explicitly blocked until the hardening gate closes

- no runtime release PR;
- no merge/deploy of PR-1;
- no identifiable transcript processing;
- no PR-2 or real-patient pilot;
- no production credential/config mutation to manufacture evidence;
- no use of the Medical Report credential path as a substitute;
- no claim that the existing 13-case suite is currently sufficient promotion evidence.

## Production-release debt retained separately

H-05 remains open after live-eval readiness unless separately fixed and verified: synchronous provider execution must not be allowed to block the single async server worker for the full provider timeout. Browser lifecycle evidence also needs stronger executable coverage before production release.
