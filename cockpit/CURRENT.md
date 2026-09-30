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
CAL.COM REASON BRIDGE: CONSUMER MERGED / PRODUCER PENDING
```

## Exact next action

HOLD this completed Home release. The Clinical Calendar consumer side of the separate Cal.com visit-reason integration is now merged; the Reception producer remains separately gated. Do not reopen Cockpit Home unless post-use evidence shows a presentation/navigation defect.


---

# Clinical Calendar reason bridge — bounded follow-up slice

> **STATUS:** CONSUMER + PROVIDER-CORRECTED PRODUCER LIVE / FIRST SNAPSHOT DELIVERY EVIDENCE PENDING.
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

1. **DONE / MERGED / LIVE:** consumer snapshot endpoint and reconciliation contract via PR #119.
2. **DONE:** initial producer PR #152 merged after exact-head compatibility PASS.
3. **DONE / SOURCE-CORRECTED:** provider evidence showed no dedicated visit-reason `bookingFieldsResponses` field on current Limassol/Evrychou Cal.com event types. The Digital Secretary's own booking schema/create path and provider logs establish `metadata.notes` as the Secretary-created booking reason source.
4. **DONE / MERGED / LIVE:** producer correction PR #153 binds the clinical feed strictly to `metadata.notes`; missing notes stay blank and generic reason-like booking-field fallbacks remain forbidden.
5. **DONE / CONFIGURED:** consumer shared ingest authentication and producer snapshot URL/shared ingest key are live. No Cal.com custom-field configuration was added.
6. **NOW:** observe the first lawful existing scheduled sync and verify an actual Clinical Calendar snapshot delivery.
7. **SEPARATE AFTERWARD:** reconcile Render cron cadence drift. Actual Render schedule is daily `0 4 * * *`; repository manifest says hourly `0 * * * *`. Do not create a second cron.

## Recovery checkpoint

```text
RECOVERY CHECKPOINT ID: COCKPIT-CALENDAR-REASON-BRIDGE-20260927-A
STATUS: RUNTIME ACTIVE / FIRST DELIVERY EVIDENCE PENDING
IMPLEMENTATION COMMITS:
- 20e5687fc4283318966fca7e680b3e62eaede43f — snapshot reconciliation endpoint + shared import path
- 83da6339c8e4c4841e910a8e6135a91bf00380e5 — canonical source / exact window validation
- 426a4037687d83179d8df5639957f621a579cfa2 — focused snapshot tests
- 3bfba4c9d13b0a4a955ed3d470e80115c75c7d84 — current snapshot/reason integration contract
- c07daaf75359b071ec2f5b51d5864127965d122a — path-scoped Clinical Calendar CI gate
- c28ba8c5901e43801ab5998781b45521dd3495e5 — DST/cardinality acceptance correction
- cb0752d770d2afc2b6828c6fa4e7c48f25c12754 — deterministic compatibility boundary tests
- bbd5c7bca0fd109797fac5a03c15a51f0f53af21 — schema contract alignment / final reviewed consumer head
VERIFICATION:
- Clinical Calendar snapshot workflow run 36308943111 / job 108590995955: PASS
- Python syntax: PASS
- deterministic Clinical Calendar tests: 12 passed in 1.29s
- diff hygiene: PASS
- Canonical impact guard run 36308943000: PASS
PR #119: MERGED — Clinical Calendar: add source snapshot reconciliation
PR URL: https://github.com/athpapachr-cmd/osteoporosis/pull/119
EVIDENCE HEAD: bbd5c7bca0fd109797fac5a03c15a51f0f53af21
MERGE COMMIT: 08251120a1743a1ac629e38626d04747a8fda1e9
PRODUCER INITIAL MERGE: PR #152 / 4923cc83a1a57a844be5c713b83af125f7c86f61
PROVIDER-EVIDENCE CORRECTION: PR #153 / d99738940ed439fedbcadde100c3993177aa556d / workflow 36310831720 / 16 tests PASS
PROVIDER REASON SOURCE: metadata.notes for Secretary-created bookings; no current custom reason-field slug observed on event types 341357/341358
CONSUMER AUTH DEPLOY: dep-daseid97lnhs738rkd1g / LIVE
PRODUCER ACTIVATION DEPLOY: dep-dasejdgjo6nc73bf2keg / LIVE
POST-ACTIVATION MANUAL SYNC: 0
LAST SAFE RESUME POINT: consumer and corrected producer are live and mutually configured; first real snapshot delivery has not yet been observed and must not be inferred from configuration alone
EXACT NEXT ACTION: verify the first existing scheduled /admin/trigger-sync produces clinical_calendar configured=true / attempted=true / sent=true; then reconcile the separate cron cadence drift
FORBIDDEN ON RESUME: PR-1 mutation, Reception reason/availability/booking behavior change, second sync cron, guessed Cal.com reason field, phone/SIP or booking/business mutation
```


---

# Pending Surgery Queue — bounded Cockpit follow-up slice

> **STATUS:** V1 IMPLEMENTATION + FOCUSED CI PASS / PRE-MERGE.
> **Workstream:** global Clinical Excellence Cockpit / pending surgery coordination.
> **Branch:** `feat/cockpit-surgery-queue-v1-2026-09-30`.
> **Base main:** `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`.
> **Product-owner direction:** 2026-09-30 — add a pending surgery list to the dashboard with patient identity/contact data, procedure/laterality, manual ordering/sorting and surgery date.
> **Root writer lock:** unchanged; this workstream uses `cockpit/CURRENT.md` under AGENTS §4.2.

## V1 product contract

- Global Cockpit feature; not Osteoporosis-specific and not Reception booking logic.
- Reuse the existing protected Clinical Auth browser session and the existing PostgreSQL engine.
- Patient identity stays server-side; no localStorage/sessionStorage persistence.
- Pending list fields:
  - ονοματεπώνυμο;
  - ΑΔΤ / stable patient identity;
  - ημερομηνία γέννησης;
  - τύπος επέμβασης;
  - πλευρά;
  - τηλέφωνο;
  - ημερομηνία χειρουργείου.
- Manual persisted queue order supports move up/down.
- Table supports non-persistent sorting by displayed columns without silently changing the manual queue order.
- Marking a case completed removes it from the default pending list but preserves the record.
- Patient demographics reuse `clinical_patients`; surgery-specific state uses a separate protected surgery-queue table.
- Synthetic test data only; no identifiable patient data in repository or CI.

## Implementation checkpoint

- protected `clinical_surgery_queue.py` API with PostgreSQL persistence;
- reuses `clinical_patients` demographics by ΑΔΤ/stable patient identity;
- create/update/list/move/complete lifecycle;
- Cockpit table with sortable columns, inline surgery date, ↑/↓ manual order, edit and complete controls;
- no browser storage of patient identity;
- focused synthetic API/UI tests + dedicated CI workflow;
- no Reception/Cal.com/booking behavior change.

## Evidence

- PR #124: `Cockpit: add protected pending surgery queue`.
- exact implementation head before this checkpoint: `a4feb0aeec452677b2b07cc283bd5c16aa2e0ec4`.
- focused workflow `36714622069`: PASS.
- Python syntax: PASS.
- Cockpit JavaScript syntax: PASS.
- surgery queue API/UI tests: **10 passed in 0.64s**.
- diff hygiene: PASS.
- Canonical impact guard `36714622173`: PASS.
- inherited Cockpit Home, Clinical Documents P1/P2, Learning L1, CU-1, Physio integration and G3 regression gates: PASS.

## Exact next action

Fresh-verify PR #124/base/head after this canonical checkpoint, rerun affected checks if triggered, then merge only if the exact head remains green. Deployment/smoke follows as a separate checkpoint.
