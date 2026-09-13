# Knee OA V5 — release closeout

> **STATUS:** RELEASED / DEPLOYED / AUTHENTICATED PRODUCTION-SMOKE-VERIFIED.
> **Date:** 2026-09-13 Asia/Nicosia.
> **Diagnosis:** Knee Osteoarthritis only.
> **Release PR:** `#101` — squash-merged.
> **Release runtime SHA:** `8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`.
> **Render deploy:** `dep-daj2398u01pc738ojvkg` — LIVE at the exact release runtime SHA.
> **Production jurisdiction profile:** `CY_GESY` via explicit server-side configuration.
> **Authenticated V5 smoke:** `34737943402` — SUCCESS.
> **Smoke job:** `103672560340` — SUCCESS.
> **Temporary smoke PR:** `#103` — CLOSED UNMERGED.
> **Writer:** none.

## 1. Release authority and merge

The Product Owner explicitly authorized `RELEASE V5` on 2026-09-13.

PR #101 was squash-merged with an exact-head guard against reviewed PR head:

`4e4bd2ae40c606562a982b3e38f9f859b49986eb`

The resulting main/runtime release commit is:

`8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`

The release did not merge the stale historical V5 branch directly. The fresh-main integration preserved the already-released `CY_GESY` jurisdiction architecture and current protected Cockpit transport.

## 2. Exact PR-head evidence

Before release, exact PR head `4e4bd2ae...` passed:

- V5 integration gate `34736919952` — SUCCESS;
- jurisdiction overlay v1 gate `34736920005` — SUCCESS;
- clinical-sheet v4 gate `34736920059` — SUCCESS;
- prototype gate `34736919984` — SUCCESS;
- Cockpit integration gate `34736920081` — SUCCESS;
- evidence design gate `34736919959` — SUCCESS;
- CU-1 focused tests `34736919945` — SUCCESS.

V5 PR-head artifact: `10311660626`.

Artifact digest:

`sha256:003311ab6d45297211d9c5e24bfb26768d26d6c435c9b9069e7fff5acf140277`

Clinical Learning red checks were adjacent-owner scope-only after their substantive syntax/runtime/inherited/frozen-owner tests passed; owner isolation was not weakened.

## 3. Released V5 behavior

The released interaction contract is:

- first tap on inactive `Πόνος`, `Δυσκαμψία`, `Αδυναμία` selects the generic symptom without a forced popup;
- second tap opens optional focused refinement;
- `Λειτουργικότητα` retains its first-tap chooser;
- weakness second-tap contains only `Μυϊκή αδυναμία στην εξέταση` and `Αδυναμία τετρακεφάλου στην εξέταση`;
- duplicated `Ατροφία τετρακεφάλου` is absent from weakness second-tap;
- quadriceps atrophy remains available through `Περισσότερα → Εξέταση` as an objective examination finding;
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

Render auto-deploy `dep-daj2398u01pc738ojvkg` was triggered by the squash-merge commit and reached `live` at exact commit:

`8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`

Render logs confirmed:

- checkout of the exact V5 release commit;
- successful build/startup;
- application startup complete;
- Uvicorn serving on the expected production port;
- PostgreSQL online storage still configured for the wider Clinical Excellence service;
- clinical authentication configured;
- service marked live at the primary production URL.

No redundant manual deploy was triggered.

## 6. Authenticated production verification — PASS

The first branch-push smoke run `34737351354` was recognized by GitHub but remained queued without being assigned a runner. A rerun of the previously successful jurisdiction smoke was also queued.

Rather than weaken authentication, expose secrets or create an unauthenticated test endpoint, the same temporary V5 smoke workflow was allowed to execute through a same-repository draft pull-request trigger, where runners were available.

Temporary ops branch:

`ops/physio-knee-oa-v5-live-smoke-2026-09-13`

Temporary workflow head:

`784f8319339b33b1749275fcb309521da3ba95ea`

Temporary PR:

`#103` — CLOSED UNMERGED after successful verification.

Authenticated production smoke:

`34737943402` — **SUCCESS**

Job:

`103672560340` — **SUCCESS**

The smoke verified the actual live production service:

1. unauthenticated product bootstrap returned `401`;
2. authenticated bootstrap returned `200`;
3. bootstrap remained real production context with `synthetic_only == false`;
4. jurisdiction profile remained `CY_GESY` with `selection_source == explicit_account_configuration`;
5. authenticated deterministic Knee-OA projection returned `200` and a valid allowed gate;
6. international acupuncture state remained `guideline_conflict_or_mixed` while Cyprus local direction remained `against` and relation remained `local_position_within_international_conflict`;
7. international manual-therapy state remained mixed while Cyprus local direction remained `conditional_for`;
8. local-only `electrotherapy` did not become an international clinical evidence item;
9. default rehab directions remained exactly `therapeutic_exercise`, `progressive_strengthening`, `education_and_self_management`;
10. referral text preserved expected Knee-OA/laterality content without silent Cyprus/GeSY evidence wording;
11. live `product-clinical-sheet-v4.js` exposed the two approved weakness exam options and did **not** contain `Ατροφία τετρακεφάλου`;
12. live `product-more-v3.js` retained `data-v3-atrophy-quadriceps` and `Ατροφία τετρακεφάλου` under More/Exam;
13. tested live V5 assets contained no `localStorage` or `sessionStorage` patient/referral persistence markers.

The protected `CLINICAL_DATA_KEY` remained masked throughout the GitHub Actions log. Only a generated UUID plus non-identifiable smoke state was sent. No patient identifiers, real patient history or patient persistence were involved.

## 7. Final lifecycle

```text
IMPLEMENTED                   YES
FRESH-MAIN INTEGRATED         YES
EXACT-HEAD TESTED             YES
PR #101 MERGED                YES
RENDER DEPLOYED               YES
V5 AUTHENTICATED LIVE SMOKE   PASS
PRODUCTION-SMOKE-VERIFIED     YES
PRODUCT OWNER DEVICE USE      NOT YET RECORDED
REAL CLINICAL PILOT           NO
COMMERCIAL VALIDATION         NO
SECOND DIAGNOSIS              NO
```

## 8. Final boundary

The V5 release lifecycle is closed once this docs-only canonical closeout is merged.

No additional diagnosis, evidence change, patient persistence, analytics/billing or jurisdiction expansion is authorized by this closeout. Any next product work requires a fresh bounded decision.
