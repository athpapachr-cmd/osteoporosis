# Knee OA v1 — current commercial release record

> **STATUS:** V5 MERGED / DEPLOYED ON RELEASED KNEE-OA + CY_GESY FOUNDATION; AUTHENTICATED V5 LIVE SMOKE PENDING EXECUTION.
> **Diagnosis:** Knee Osteoarthritis only.
> **V5 release runtime:** `8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`.
> **Release PR:** `#101`.
> **Render deploy:** `dep-daj2398u01pc738ojvkg` — LIVE at exact V5 runtime SHA.
> **Production jurisdiction profile:** `CY_GESY` via explicit server-side configuration.
> **Prior authenticated CY_GESY production smoke:** `34703453615` — SUCCESS.
> **Authenticated V5 smoke:** `34737351354` — QUEUED / NOT YET EXECUTED at this reconciliation.

## Product surface now deployed

The production product includes:

- explicit Knee-OA diagnosis assertion + laterality;
- live deterministic Greek referral, no routine Generate button;
- V5 compact clinical-picture interaction;
- direct manual editing with stale-text reconciliation;
- reviewed active-rehab defaults;
- international evidence-aware suggestions and mixed-guideline disclosure;
- advanced examination only on demand;
- `★ Συχνά / Σχετικά τώρα / Όλα` scan-first advanced UI;
- Cyprus/GeSY jurisdiction overlay as separate progressive evidence context;
- explicit server-side `CY_GESY` activation, no patient-location inference;
- ephemeral patient/referral draft with no product-level patient persistence.

## V5 interaction now deployed

- inactive `Πόνος`, `Δυσκαμψία`, `Αδυναμία`: first tap selects generic symptom without forced popup;
- second tap opens optional detail;
- `Λειτουργικότητα`: first-tap chooser retained;
- weakness second-tap contains only explicit weakness examination options;
- duplicated `Ατροφία τετρακεφάλου` removed from weakness second-tap;
- quadriceps atrophy retained at `Περισσότερα → Εξέταση` as objective examination finding;
- weakness badge/count excludes separately selected atrophy;
- bare ambiguous `Περιαρθρικά` remains absent from visible routine/advanced UI;
- pain qualifier prose owns location specificity without legacy duplicate tails;
- richer referrals use a paragraph boundary before physiotherapy assessment/priorities;
- `Επιπλέον στόχος:` wording is replaced by connected prose;
- low-information output remains compact.

## Pre-release exact-head evidence

Exact reviewed PR head:

`4e4bd2ae40c606562a982b3e38f9f859b49986eb`

Successful gates:

```text
V5 integration                  34736919952  SUCCESS
CY_GESY jurisdiction            34736920005  SUCCESS
clinical-sheet v4               34736920059  SUCCESS
prototype                       34736919984  SUCCESS
protected Cockpit integration   34736920081  SUCCESS
evidence design                 34736919959  SUCCESS
CU-1 focused                    34736919945  SUCCESS
```

Final PR-head artifact `10311660626`, digest `sha256:003311ab6d45297211d9c5e24bfb26768d26d6c435c9b9069e7fff5acf140277`.

Clinical Learning red checks were scope-only adjacent-owner failures after substantive regressions/frozen-owner guards passed; they are not product regressions.

## Production architecture

The product remains deployed at:

`/clinical/clinic-utilities/physio-referral`

Architecture:

```text
real CU-1 validation/safety
+
deterministic Knee-OA projection/presentation
+
international evidence core
+
separate CY_GESY jurisdiction overlay
+
V5 interaction/prose layer
+
protected Clinical Excellence transport
```

No unauthenticated clinical endpoint, autonomous treatment decision, second diagnosis or patient-draft persistence is introduced by V5.

## Evidence / jurisdiction invariants

V5 does not alter:

- international evidence states/source positions;
- Cyprus local directions;
- explicit `CY_GESY` activation semantics;
- default rehab selection;
- suggestion vs selection behavior;
- referral safety gating;
- GeSY admin/reimbursement separation from clinical evidence.

Local context still cannot silently overwrite international evidence or routine referral prose.

## Merge and deployment evidence

PR #101 was squash-merged to:

`8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`

Render auto-deploy `dep-daj2398u01pc738ojvkg` checked out that exact commit and reached `live`. Startup logs confirm application startup complete and configured clinical authentication. No redundant manual deploy occurred.

## Authenticated V5 production verification

Temporary ops branch:

`ops/physio-knee-oa-v5-live-smoke-2026-09-13`

Workflow commit:

`b46c948c38181336d23f90899cdd331dffe09f4c`

Run:

`34737351354`

The smoke is designed to verify the protected production API, `CY_GESY`, international/local evidence separation, unchanged default selection/referral semantics, live V5 atrophy routing and no browser-storage markers using a generated UUID plus non-identifiable state. The credential remains in GitHub Actions secrets and is not printed.

At this reconciliation the run is queued awaiting a GitHub runner. The older successful authenticated jurisdiction smoke was also re-run and is likewise queued, supporting classification as runner scheduling rather than observed application failure.

## Current lifecycle

```text
V5 merged                                    YES
V5 deployed                                  YES
Render exact release runtime                 LIVE
V5 authenticated production smoke            PENDING EXECUTION
actual iPhone Safari / VoiceOver              NOT YET PROVEN
formal receiver validation                    DEFERRED / NON-BLOCKING
paid conversion / retention                   NOT YET PROVEN
real clinical pilot                           NOT YET PROVEN
second diagnosis                              NOT AUTHORIZED
```

A final `PRODUCTION-SMOKE-VERIFIED` closeout requires an actually successful authenticated smoke run.
