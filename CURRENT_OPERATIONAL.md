# CURRENT_OPERATIONAL.md — Physiotherapy Referral Knee-OA V5 release

> **STATUS:** V5 RELEASED / DEPLOYED / AUTHENTICATED PRODUCTION SMOKE VERIFIED.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice:** `CU-PHYSIO-KNEE-OA-V5-RELEASE-2026-09-13` — CLOSED.
> **Release PR:** `#101` — squash-merged.
> **Release runtime SHA:** `8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`.
> **Render deploy:** `dep-daj2398u01pc738ojvkg` — LIVE at the exact release runtime SHA.
> **Production jurisdiction profile:** `CY_GESY` via explicit server-side configuration.
> **Authenticated V5 smoke:** `34737943402` — SUCCESS.
> **Temporary smoke PR:** `#103` — CLOSED UNMERGED after successful verification.
> **Writer:** none.
> **Patient/referral persistence authority:** NONE.

## 1. Product Owner release authority

On 2026-09-13 the Product Owner explicitly instructed:

`RELEASE V5`

This authorized the reviewed fresh-main V5 integration to proceed through squash merge, Render deployment, authenticated production verification and canonical release closeout.

It did **not** authorize a second diagnosis, evidence reclassification, local-rule automation, patient persistence, analytics/billing, Greece/England localization or unrelated Medical Report mutation.

## 2. Merge and deployment — verified

PR #101 was squash-merged using exact expected head:

`4e4bd2ae40c606562a982b3e38f9f859b49986eb`

The resulting release runtime commit is:

`8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`

Render auto-deploy `dep-daj2398u01pc738ojvkg` checked out that exact commit and reached `live`. No redundant manual deploy was triggered.

Render evidence confirms:

- exact release commit checkout;
- application startup complete;
- Uvicorn serving on the expected port;
- wider Clinical Excellence storage remained PostgreSQL/online;
- clinical authentication remained configured;
- service marked live at the production URL.

## 3. Released V5 interaction

The production product now uses the reviewed V5 behavior:

1. first tap on inactive `Πόνος`, `Δυσκαμψία`, `Αδυναμία` selects the generic symptom without forcing a detail sheet;
2. second tap opens optional focused refinement;
3. `Λειτουργικότητα` remains a first-tap chooser;
4. weakness second-tap contains only:
   - `Μυϊκή αδυναμία στην εξέταση`
   - `Αδυναμία τετρακεφάλου στην εξέταση`;
5. `Ατροφία τετρακεφάλου` is absent from weakness second-tap and remains available at `Περισσότερα → Εξέταση` as an objective examination finding;
6. weakness count reflects weakness refinement only and does not count separately selected atrophy;
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

## 5. Authenticated production verification — PASS

The initial branch-push smoke run `34737351354` remained queued without a runner. A rerun of the previously successful jurisdiction smoke was likewise queued. No application failure occurred in those queued jobs.

To obtain executable release evidence without weakening credentials or exposing secrets, the same temporary smoke workflow was allowed to run through a same-repository draft pull-request trigger where runners were available.

Temporary ops PR:

`#103` — `TEMP: authenticated Knee OA V5 production smoke`

Temporary ops workflow head:

`784f8319339b33b1749275fcb309521da3ba95ea`

Authenticated production smoke:

`34737943402` — **SUCCESS**

Job:

`103672560340` — **SUCCESS**

The run verified against the live production service:

- unauthenticated product bootstrap remains rejected with `401`;
- authenticated bootstrap succeeds;
- `synthetic_only == false` and production deployment context are intact;
- jurisdiction profile remains `CY_GESY` with `explicit_account_configuration`;
- authenticated deterministic Knee-OA projection succeeds;
- export gate remains allowed for a valid non-identifiable state;
- international acupuncture state remains `guideline_conflict_or_mixed` while Cyprus direction remains `against`;
- international manual-therapy state remains mixed while Cyprus direction remains `conditional_for`;
- local-only `electrotherapy` is absent from international clinical evidence items;
- default rehabilitation selections remain exactly unchanged;
- referral prose contains the expected Knee-OA/laterality content without silent Cyprus/GeSY wording;
- live `product-clinical-sheet-v4.js` contains only the two explicit weakness exam options and **does not contain `Ατροφία τετρακεφάλου`**;
- live `product-more-v3.js` retains `data-v3-atrophy-quadriceps` and `Ατροφία τετρακεφάλου` under More/Exam;
- those live V5 assets contain no `localStorage` or `sessionStorage` patient/referral persistence markers.

The GitHub Actions secret was masked throughout the log. Only a generated UUID plus non-identifiable smoke state was sent. No patient identifiers or real patient history were used.

Temporary PR #103 was closed unmerged immediately after successful smoke verification.

## 6. Preserved invariants

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

## 7. Final lifecycle

```text
IMPLEMENTED                    YES
FRESH-MAIN INTEGRATED          YES
EXACT-HEAD TESTED              YES
PR #101 MERGED                 YES
RENDER DEPLOYED                YES
V5 AUTHENTICATED LIVE SMOKE    PASS
PRODUCTION-SMOKE-VERIFIED      YES
PRODUCT OWNER DEVICE USE       NOT YET RECORDED
REAL CLINICAL PILOT            NO
COMMERCIAL VALIDATION          NO
SECOND DIAGNOSIS               NO
```

## 8. Parallel project state

V5 release did not mutate the Medical Report product or silently close any independent Medical Report validation gate. Separate product lifecycles remain separate.

## 9. Next bounded product boundary

The V5 release slice is closed and the writer lock is free.

Product Owner real-device/use acceptance may now provide practical product evidence. Any later clinical/workflow pilot, commercial validation, second diagnosis or new jurisdiction requires a separate bounded decision rather than being inferred from this release.
