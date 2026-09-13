# CURRENT_OPERATIONAL.md — Physiotherapy Referral Knee-OA V5 release

> **STATUS:** V5 MERGED / DEPLOYED — AUTHENTICATED PRODUCTION SMOKE PENDING EXECUTION.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice:** `CU-PHYSIO-KNEE-OA-V5-RELEASE-2026-09-13`.
> **Release PR:** `#101` — squash-merged.
> **Release runtime SHA:** `8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`.
> **Render deploy:** `dep-daj2398u01pc738ojvkg` — LIVE at the exact release runtime SHA.
> **Production jurisdiction profile:** `CY_GESY` via explicit server-side configuration.
> **Authenticated V5 smoke:** `34737351354` — QUEUED / NOT YET EXECUTED at this canonical draft.
> **Writer:** none.
> **Patient/referral persistence authority:** NONE.

## 1. Product Owner release authority

On 2026-09-13 the Product Owner explicitly instructed:

`RELEASE V5`

This authorized the previously reviewed and exact-head-tested fresh-main V5 integration to proceed through squash merge, Render auto-deploy, authenticated production verification and canonical release closeout.

It did **not** authorize a second diagnosis, evidence reclassification, new local-rule automation, patient persistence, analytics/billing, Greece/England localization or unrelated Medical Report mutation.

## 2. Merge and deployment — complete

PR #101 was squash-merged using exact expected head:

`4e4bd2ae40c606562a982b3e38f9f859b49986eb`

The resulting release runtime commit is:

`8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`

Render auto-deploy `dep-daj2398u01pc738ojvkg` checked out that exact commit and reached `live`. The previous deploy was deactivated automatically. No redundant manual deploy was triggered.

Render startup evidence shows:

- exact release commit checkout;
- application startup complete;
- Uvicorn serving on the expected port;
- PostgreSQL online storage still configured for the wider Clinical Excellence service;
- clinical authentication key configured;
- service marked live at the production URL.

## 3. Released V5 interaction

The production code now carries the reviewed V5 behavior:

1. first tap on inactive `Πόνος`, `Δυσκαμψία`, `Αδυναμία` selects the generic symptom without forcing a detail sheet;
2. second tap opens optional focused refinement;
3. `Λειτουργικότητα` remains a first-tap chooser;
4. weakness second-tap contains only:
   - `Μυϊκή αδυναμία στην εξέταση`
   - `Αδυναμία τετρακεφάλου στην εξέταση`;
5. `Ατροφία τετρακεφάλου` is not duplicated in that second-tap sheet and remains available at `Περισσότερα → Εξέταση` as an objective examination finding;
6. the weakness count reflects weakness refinement only and does not count separately selected atrophy;
7. bare ambiguous `Περιαρθρικά` remains absent from visible routine/advanced UI;
8. overlapping pain-location prose is reconciled;
9. richer referrals separate clinical picture/function from physiotherapy assessment/priorities;
10. mechanical `Επιπλέον στόχος:` wording is replaced by connected prose;
11. low-information referrals remain compact.

## 4. Pre-release exact-head evidence

Exact PR head `4e4bd2ae40c606562a982b3e38f9f859b49986eb` passed:

- V5 integration gate `34736919952` — SUCCESS;
- jurisdiction overlay v1 gate `34736920005` — SUCCESS;
- clinical-sheet v4 gate `34736920059` — SUCCESS;
- prototype gate `34736919984` — SUCCESS;
- Cockpit integration gate `34736920081` — SUCCESS;
- evidence design gate `34736919959` — SUCCESS;
- CU-1 focused tests `34736919945` — SUCCESS.

PR-head artifact: `10311660626`, digest `sha256:003311ab6d45297211d9c5e24bfb26768d26d6c435c9b9069e7fff5acf140277`.

Clinical Learning red checks were adjacent-owner scope-only after substantive tests/frozen-owner guards passed; their isolation contract was not weakened.

## 5. Preserved invariants

V5 does not change:

- international evidence states/source positions;
- `CY_GESY` local-position semantics or explicit-account activation;
- default rehabilitation selections;
- suggestion vs clinician selection;
- diagnosis/laterality authority;
- safety fail-closed behavior;
- deterministic referral ownership;
- manual-edit stale/reconciliation behavior;
- no patient/referral browser persistence;
- clinical guidance vs GeSY administrative/reimbursement separation.

No local-only/admin jurisdiction row becomes a clinical evidence item and Cyprus context does not silently rewrite referral prose.

## 6. Authenticated production verification — pending runner

Temporary ops branch:

`ops/physio-knee-oa-v5-live-smoke-2026-09-13`

Workflow branch commit:

`b46c948c38181336d23f90899cdd331dffe09f4c`

Authenticated smoke run:

`34737351354`

The workflow uses the protected GitHub Actions `CLINICAL_DATA_KEY` without printing it and sends only a generated UUID plus non-identifiable Knee-OA state. It verifies protected bootstrap/project behavior, active `CY_GESY`, international evidence integrity, unchanged defaults/referral semantics, V5 live static-asset routing of quadriceps atrophy and the no-browser-storage boundary.

At this draft state GitHub has queued the job but has not assigned a runner. A rerun of the previously successful authenticated jurisdiction smoke is also queued. Therefore there is **no observed application smoke failure**, but there is also **no authenticated V5 smoke PASS claim yet**.

## 7. Current truthful lifecycle

```text
IMPLEMENTED                   YES
FRESH-MAIN INTEGRATED         YES
EXACT-HEAD TESTED             YES
PR #101 MERGED                YES
RENDER DEPLOYED               YES
V5 AUTHENTICATED LIVE SMOKE   PENDING EXECUTION
PRODUCT OWNER DEVICE USE      NOT YET RECORDED
REAL CLINICAL PILOT           NO
COMMERCIAL VALIDATION         NO
SECOND DIAGNOSIS              NO
```

## 8. Parallel project state

V5 release does not mutate the Medical Report product or silently close any independent Medical Report validation gate. Separate product lifecycles remain separate.

## 9. Exact next action

Execute the queued authenticated production smoke. If it passes, promote V5 to `PRODUCTION-SMOKE-VERIFIED` and merge the docs-only canonical closeout. If it fails, diagnose the bounded production issue and do not claim full release verification.
