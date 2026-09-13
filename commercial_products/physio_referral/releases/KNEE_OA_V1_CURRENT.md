# Knee OA v1 — current commercial release record

> **STATUS:** V5 RELEASED / DEPLOYED / AUTHENTICATED PRODUCTION-SMOKE-VERIFIED ON KNEE-OA + CY_GESY FOUNDATION.
> **Diagnosis:** Knee Osteoarthritis only.
> **V5 release runtime:** `8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`.
> **Release PR:** `#101`.
> **Render deploy:** `dep-daj2398u01pc738ojvkg` — LIVE at exact V5 runtime SHA.
> **Production jurisdiction profile:** `CY_GESY` via explicit server-side configuration.
> **Authenticated V5 production smoke:** `34737943402` — SUCCESS.
> **Temporary smoke PR:** `#103` — CLOSED UNMERGED.

## Product surface released

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

## V5 interaction released

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

The product is deployed at:

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

Local context cannot silently overwrite international evidence or routine referral prose.

## Merge and deployment evidence

PR #101 was squash-merged to:

`8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`

Render auto-deploy `dep-daj2398u01pc738ojvkg` checked out that exact commit and reached `live`. Startup logs confirmed application startup complete and configured clinical authentication. No redundant manual deploy occurred.

## Authenticated V5 production verification — PASS

The original temporary branch-push smoke remained queued without receiving a runner. The identical authenticated smoke was therefore executed using a temporary same-repository draft PR trigger, preserving the same protected-secret and non-identifiable-data boundaries.

Temporary PR #103 remained unmerged and was closed after verification.

Run:

`34737943402` — **SUCCESS**

Job:

`103672560340` — **SUCCESS**

Live assertions passed for:

- unauthenticated bootstrap rejection;
- authenticated bootstrap/project success;
- explicit active `CY_GESY` profile;
- unchanged international acupuncture/manual-therapy evidence states with separate local directions;
- no local-only electrotherapy evidence item;
- unchanged default rehab selections;
- referral text free of silent Cyprus/GeSY evidence wording;
- live weakness second-tap asset containing the two approved weakness exam concepts and no `Ατροφία τετρακεφάλου`;
- live More/Exam asset retaining quadriceps atrophy;
- no `localStorage`/`sessionStorage` markers in the tested V5 live assets.

The credential was masked throughout the log. Only a generated UUID plus non-identifiable smoke state was sent.

## Current lifecycle

```text
V5 merged                                    YES
V5 deployed                                  YES
Render exact release runtime                 LIVE
V5 authenticated production smoke            PASS
production-smoke-verified                    YES
actual iPhone Safari / VoiceOver              NOT YET PROVEN
formal receiver validation                    DEFERRED / NON-BLOCKING
paid conversion / retention                   NOT YET PROVEN
real clinical pilot                           NOT YET PROVEN
second diagnosis                              NOT AUTHORIZED
```

The V5 release lifecycle is closed after canonical docs merge. Any next diagnosis/product expansion requires a fresh bounded decision.
