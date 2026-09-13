# Knee OA V5 — release closeout

> **STATUS:** MERGED / DEPLOYED; AUTHENTICATED PRODUCTION SMOKE PENDING EXECUTION.
> **Date:** 2026-09-13 Asia/Nicosia.
> **Diagnosis:** Knee Osteoarthritis only.
> **Release PR:** #101.
> **Release runtime SHA:** `8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`.
> **Render deploy:** `dep-daj2398u01pc738ojvkg` — LIVE at the exact release runtime SHA.
> **Production jurisdiction profile:** `CY_GESY` via explicit server-side configuration.
> **Authenticated V5 smoke:** run `34737351354` — QUEUED / NOT YET EXECUTED at the time of this draft.
> **Writer:** none.

## 1. Release authority and merge

The Product Owner explicitly authorized `RELEASE V5` on 2026-09-13.

PR #101 was squash-merged with an exact-head guard against reviewed PR head:

`4e4bd2ae40c606562a982b3e38f9f859b49986eb`

The resulting main/runtime release commit is:

`8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`

The release did not merge the stale historical V5 branch directly. The fresh-main integration preserved the already-released `CY_GESY` jurisdiction architecture and current protected Cockpit transport.

## 2. Exact PR-head evidence

Before release, exact PR head `4e4bd2ae...` passed the relevant product-owned gates:

- V5 integration gate `34736919952` — SUCCESS;
- jurisdiction overlay v1 gate `34736920005` — SUCCESS;
- clinical-sheet v4 gate `34736920059` — SUCCESS;
- prototype gate `34736919984` — SUCCESS;
- Cockpit integration gate `34736920081` — SUCCESS;
- evidence design gate `34736919959` — SUCCESS;
- CU-1 focused tests `34736919945` — SUCCESS.

V5 PR-head artifact: `10311660626`, digest `sha256:003311ab6d45297211d9c5e24bfb26768d26d6c435c9b9069e7fff5acf140277`.

Clinical Learning red checks were adjacent-owner scope-only after their substantive syntax/runtime/inherited/frozen-owner tests passed; owner isolation was not weakened.

## 3. Released V5 behavior

The released interaction contract is:

- first tap on inactive `Πόνος`, `Δυσκαμψία`, `Αδυναμία` selects the generic symptom without a forced popup;
- second tap opens optional focused refinement;
- `Λειτουργικότητα` retains its first-tap chooser;
- the weakness second-tap contains only `Μυϊκή αδυναμία στην εξέταση` and `Αδυναμία τετρακεφάλου στην εξέταση`;
- duplicated `Ατροφία τετρακεφάλου` is absent from the weakness second-tap and remains available through `Περισσότερα → Εξέταση` as an objective examination finding;
- weakness detail count does not count separately selected quadriceps atrophy;
- bare ambiguous `Περιαρθρικά` remains absent from visible routine/advanced UI;
- pain-location duplicate prose is reconciled;
- richer referrals separate clinical picture/function from physiotherapy assessment/priorities;
- mechanical `Επιπλέον στόχος:` wording is replaced by connected prose;
- low-information referral output remains compact.

## 4. Preserved clinical/product invariants

The release does not change:

- international evidence states or source positions;
- `CY_GESY` local-position semantics or explicit-account activation;
- default rehabilitation selections;
- suggestion vs clinician-selection semantics;
- diagnosis assertion/laterality authority;
- safety fail-closed behavior;
- manual-edit stale/reconciliation behavior;
- deterministic referral ownership;
- no-patient-persistence / no-browser-storage boundary;
- clinical guidance vs GeSY administrative/reimbursement separation.

No second diagnosis, Greece/England profile, new treatment selector, patient persistence, analytics/billing or autonomous literature-to-live behavior is part of V5.

## 5. Deployment

Render auto-deploy `dep-daj2398u01pc738ojvkg` was triggered by the squash-merge commit and reached `live` at exact commit `8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`.

Render logs show the service checked out that exact commit, completed application startup, retained PostgreSQL online storage for the wider Clinical Excellence service, retained configured clinical authentication, and made the service live. No additional manual deploy was triggered.

## 6. Authenticated production verification

A temporary non-merged ops workflow was created at branch:

`ops/physio-knee-oa-v5-live-smoke-2026-09-13`

Workflow branch commit:

`b46c948c38181336d23f90899cdd331dffe09f4c`

Run `34737351354` is intended to verify the live protected bootstrap/project boundary, `CY_GESY`, international evidence integrity, unchanged default selections/referral semantics, V5 atrophy routing in live static assets, and no browser-storage markers using only a generated UUID plus non-identifiable smoke state. The protected `CLINICAL_DATA_KEY` is referenced through GitHub Actions secrets and is never printed.

At the time this draft was prepared, GitHub had queued the job but had not assigned a runner. A rerun of the previously successful authenticated jurisdiction smoke was also queued, which supports classifying the delay as runner scheduling rather than an observed V5 application failure.

**This document must not be promoted to final `PRODUCTION-SMOKE-VERIFIED` status until an authenticated smoke actually executes successfully.**

## 7. Current truthful lifecycle

```text
IMPLEMENTED                  YES
FRESH-MAIN INTEGRATED        YES
EXACT-HEAD TESTED            YES
PR #101 MERGED               YES
RENDER DEPLOYED              YES
V5 AUTHENTICATED LIVE SMOKE  PENDING EXECUTION
PRODUCT OWNER DEVICE USE     NOT YET RECORDED
REAL CLINICAL PILOT          NO
COMMERCIAL VALIDATION        NO
SECOND DIAGNOSIS             NO
```
