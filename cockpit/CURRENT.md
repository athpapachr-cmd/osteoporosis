# Cockpit Home CURRENT

> **STATUS:** COCKPIT HOME V1 RELEASE COMPLETE / PR #118 MERGED / RENDER LIVE / ROOT SMOKE VERIFIED.
> **Workstream:** Clinical Excellence Cockpit Home v1.
> **Branch:** feat/cockpit-home-v1-2026-09-27.
> **Base main:** 88ad125f0a25a471b0151eeb26e68b8b8a93c84f.
> **Exact tested head:** 7744e2142b54ba0b7c8b92c87e4e511cffffc26b.
> **Root writer lock:** unchanged; PR-1 Heidi-first transcript capture remains the repo-wide CURRENT_OPERATIONAL owner.
> **Overlap:** none; this slice is limited to global Cockpit navigation/home presentation and Osteoporosis sidebar cleanup.

## Product-owner decisions

- / must open the global Cockpit Home, not the Osteoporosis module.
- Osteoporosis is Module 01 / proving ground, not the whole Cockpit.
- Osteoporosis sidebar must contain only Module-01 navigation.
- Global Clinic Utilities belong on Cockpit Home.
- The duplicate physiotherapy referral entry must collapse to one global tool entry.
- The top-level Heidi AI navigation item must be removed from Osteoporosis; Heidi remains an encounter-capture capability inside the Module-01 workflow and future reusable Core.
- The separate Reception/calls dashboard remains owned by ortho-reception-backend-v2; Cockpit Home links to it rather than duplicating its implementation.
- Calendar/Cal.com reason ingestion is a separate follow-up integration slice because its source-of-truth boundary spans repositories.

## Home v1 information architecture

Cockpit Home
- Today
  - Clinical Calendar
- Clinical Modules
  - Osteoporosis · Module 01 · Active
- Learning & Improvement
  - Clinical Learning Hub
- Clinic Utilities
  - Παραπεμπτικό Φυσιοθεραπείας
  - Αναρρωτική άδεια
  - Ιατρικές εκθέσεις
  - Ραδιοκύματα
- Reception
  - Call / Reception dashboard (external owner)

Module 02+ appear as future placeholders only; no fake runtime is implied.

## Safety / scope

- no patient/clinical data model change;
- no Cal.com/Setmore/Digital Secretary mutation in this slice;
- no PR-1 transcript runtime mutation;
- no duplicate copy of the Reception dashboard;
- no new browser patient persistence;
- root Clinical Auth boundary remains unchanged.

## Verification

Cockpit Home gate:

```text
run: 36273987676
head: 7744e2142b54ba0b7c8b92c87e4e511cffffc26b
result: SUCCESS
```

Verified:

- Python syntax PASS;
- Cockpit browser JavaScript syntax PASS;
- 4 deterministic Cockpit Home tests PASS;
- Osteoporosis G4 workspace/navigation regression PASS;
- diff hygiene PASS.

Behavior proven:

- / enters /static/cockpit/;
- global Cockpit Home contains Modules, Calendar, Learning, Clinic Utilities and Reception;
- Clinic Utilities are absent from the Osteoporosis sidebar;
- top-level Heidi AI is absent from the Osteoporosis sidebar;
- one global physiotherapy referral entry exists;
- Home calendar summary uses aggregate counts only and does not render patient identity fields.

## Release PR

PR: #118
URL: https://github.com/athpapachr-cmd/osteoporosis/pull/118
base: main
base_sha: 88ad125f0a25a471b0151eeb26e68b8b8a93c84f
head: feat/cockpit-home-v1-2026-09-27
head_sha at PR creation: f0f252dfaef9c08146c636e5d10fbf7ede97d553
draft: NO
merged: NO
deploy: NO

## Release completion

```text
PR: #118
merge method: squash
merge commit: b7b8779d943eb1d8db1f8966a81796bc69647c3b
merged: YES
Render service: osteoporosis / srv-d5qfk31r0fns73di596g
deploy: dep-das41frncjis73e6q47g
deploy trigger: new_commit
deploy status: live
manual redeploy: NO
```

Final PR exact head `ba2de644f4fd16e70d12f04831b28f5fd42730ec` passed all 16 checks, including Cockpit Home, Canonical Impact, G3/G2/G1, Clinical Learning L1/L1B/L1C, Clinical Documents, CU-1 and Physio inherited browser gates.

Production smoke from Render logs:

```text
GET /                 -> 307 Temporary Redirect
GET /static/cockpit/  -> 200 OK
Application startup   -> complete
Clinical storage      -> PostgreSQL online
```

Released navigation ownership:

- global Clinic Utilities live only on Cockpit Home;
- Osteoporosis remains Module 01 and no longer owns the general utilities group;
- top-level Heidi AI navigation is absent from the Osteoporosis sidebar;
- Heidi capture/exposure content remains inside the Module-01 encounter workflow;
- exactly one global physiotherapy referral entry is exposed on Home;
- Reception remains a linked separately-owned system.

## Current release state

```text
IMPLEMENTED: YES
TESTED: YES
PR #118: MERGED
DEPLOYED: YES / LIVE
ROOT -> COCKPIT HOME: VERIFIED
MODULE-01 SIDEBAR CLEANUP: VERIFIED
CAL.COM REASON BRIDGE: NOT YET IMPLEMENTED
```

## Exact next action

HOLD this completed Home release. The next separate integration slice is Cal.com visit-reason normalization into the Clinical Calendar; do not reopen Cockpit Home unless post-use evidence shows a presentation/navigation defect.


---

# Clinical Calendar reason bridge — bounded follow-up slice

> **STATUS:** CONSUMER SNAPSHOT IMPLEMENTED / FOCUSED CI PASS / DRAFT PR #119 OPEN.
> **Workstream:** Cal.com visit-reason normalization into the Clinical Calendar.
> **Branch:** `feat/calendar-reason-snapshot-v1-2026-09-27`.
> **Base main:** `a0cf912b0fcb5caf54f4a48f8f8a698908fbb0a4`.
> **Product-owner resume:** 2026-09-27 — continue the exact follow-up slice named above after recovery from the interrupted PR1 Βελτιώσεις Eval conversation.
> **Root writer lock:** unchanged; PR-1 Heidi-first transcript capture remains the repo-wide CURRENT_OPERATIONAL owner.
> **Overlap:** none; this branch is limited to the Clinical Calendar integration contract and its tests/docs. It does not mutate PR-1 transcript runtime.

## Accepted boundary

- Cal.com remains the appointment-source truth for the feed.
- Reception/Digital Secretary remains the external integration owner; no Cal API credential is added to this repository.
- Clinical Calendar remains the normalized clinical consumer.
- This slice is read-only with respect to bookings: it must not create, cancel, reschedule or authorize appointments.
- Reception reason-triage / availability / booking semantics are explicitly out of scope.
- The existing Clinical Calendar classifier is reused; no second semantic reason owner is introduced.
- Snapshot reconciliation must remove stale source rows inside the declared source/time window so cancellations/reschedules do not remain as phantom appointments.
- Raw provider payloads are not persisted; only the existing normalized appointment fields are stored.

## Current implementation order

1. **DONE IN BRANCH:** bounded server-to-server snapshot ingest contract, shared import path, exact source/window validation, stale-row reconciliation, contract YAML, and focused deterministic tests.
2. **DONE:** focused consumer CI passed on implementation head `5d6fb963f33491c43fc856fb4473a42fce76e1de`.
3. **NOW:** hold PR #119 as the proven consumer boundary while preparing the separately-owned Reception producer contract; do not merge #119 yet.
4. Then: add the Reception-side producer against this proven consumer contract, reusing existing Cal.com read/normalization work rather than adding another provider reader.
3. Then: cross-repo review/evidence.
4. Merge/deploy/smoke remain separately gated and are **not** authorized by this checkpoint.

## Recovery checkpoint

```text
RECOVERY CHECKPOINT ID: COCKPIT-CALENDAR-REASON-BRIDGE-20260927-A
STATUS: IN_PROGRESS
IMPLEMENTATION COMMITS:
- 20e5687fc4283318966fca7e680b3e62eaede43f — snapshot reconciliation endpoint + shared import path
- 83da6339c8e4c4841e910a8e6135a91bf00380e5 — canonical source / exact window validation
- 426a4037687d83179d8df5639957f621a579cfa2 — focused snapshot tests
- 3bfba4c9d13b0a4a955ed3d470e80115c75c7d84 — current snapshot/reason integration contract
- c07daaf75359b071ec2f5b51d5864127965d122a — path-scoped Clinical Calendar CI gate
VERIFICATION:
- Clinical Calendar snapshot workflow run 36294743431 / job 108551384222: PASS
- Python syntax: PASS
- deterministic Clinical Calendar tests: 10 passed in 0.80s
- diff hygiene: PASS
- Canonical impact guard run 36294743389: PASS
DRAFT PR: #119 — Clinical Calendar: add source snapshot reconciliation
PR URL: https://github.com/athpapachr-cmd/osteoporosis/pull/119
EVIDENCE HEAD: 5d6fb963f33491c43fc856fb4473a42fce76e1de
LAST SAFE RESUME POINT: consumer snapshot boundary is implemented and executable-evidence PASS; PR #119 remains draft/unmerged; no Reception backend files changed
EXACT NEXT ACTION: preserve #119 as consumer boundary and design/canonicalize the separate Reception producer before any backend mutation
FORBIDDEN ON RESUME: PR-1 mutation, Reception reason/availability/booking behavior change, merge, deploy, provider/business mutation
```
