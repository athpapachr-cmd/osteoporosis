# SLICE_PLAN_CURRENT.md — Physiotherapy Referral Knee-OA V5 release

> **STATUS:** RELEASE COMPLETE / AUTHENTICATED PRODUCTION SMOKE PASS / SLICE CLOSED.
> **Activated:** 2026-09-13 Asia/Nicosia.
> **Closed:** 2026-09-13 Asia/Nicosia.
> **Slice:** `CU-PHYSIO-KNEE-OA-V5-RELEASE-2026-09-13`.
> **Release PR:** `#101`.
> **Release runtime SHA:** `8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`.
> **Render deploy:** `dep-daj2398u01pc738ojvkg` — LIVE at exact runtime SHA.
> **Authenticated V5 smoke:** `34737943402` — SUCCESS.
> **Temporary smoke PR:** `#103` — CLOSED UNMERGED.
> **Writer:** none.
> **Diagnosis vertical:** Knee Osteoarthritis only.

## 1. Objective — complete

Release the accepted fresh-main V5 integration while preserving the current Knee-OA / CU-1 / `CY_GESY` architecture and verify the protected production behavior.

Completed sequence:

```text
exact tested V5 PR head
→ squash merge with expected-head guard
→ Render auto-deploy
→ authenticated live smoke
→ docs-only canonical closeout
```

## 2. Released interaction contract

### Pain / Stiffness / Weakness

```text
first inactive tap
→ generic symptom selected
→ no forced detail sheet

second tap while selected
→ optional focused refinement sheet
```

### Function

`Λειτουργικότητα` remains a first-tap chooser.

### Weakness / quadriceps atrophy

Second-tap `Αδυναμία` contains only:

```text
Μυϊκή αδυναμία στην εξέταση
Αδυναμία τετρακεφάλου στην εξέταση
```

`Ατροφία τετρακεφάλου` is absent from that sheet and remains available through:

```text
Περισσότερα → Εξέταση → Ατροφία τετρακεφάλου
```

Weakness count reflects weakness-detail state only and does not count separately selected atrophy.

## 3. Semantic/copy contract

The released V5 contract preserves:

- generic weakness as symptom/context rather than objective finding;
- explicit examination authority for objective/quadriceps weakness;
- quadriceps atrophy as an objective examination finding;
- qualifier-owned pain location without redundant legacy tails;
- no visible bare `Περιαρθρικά`;
- paragraph separation between clinical picture/function and physiotherapy assessment/priorities in richer referrals;
- connected functional-goal prose instead of `Επιπλέον στόχος:`;
- compact low-information output;
- no invented physiotherapy dose/frequency/protocol.

## 4. Jurisdiction contract

Released architecture:

```text
international evidence core
+
JurisdictionOverlayV1
+
explicit CY_GESY production profile
+
V5 presentation/prose layer
```

Hard requirements verified:

```text
international evidence state unchanged
CY_GESY local context remains separate
jurisdiction never auto-selects treatment
jurisdiction never rewrites referral prose
GeSY admin/reimbursement never becomes clinical evidence
planned != active
```

## 5. Pre-release gates — satisfied

Exact reviewed PR head:

`4e4bd2ae40c606562a982b3e38f9f859b49986eb`

Successful gates:

- V5 integration `34736919952`;
- CY_GESY jurisdiction `34736920005`;
- clinical-sheet v4 `34736920059`;
- prototype `34736919984`;
- protected Cockpit integration `34736920081`;
- evidence design `34736919959`;
- CU-1 focused `34736919945`.

All completed `SUCCESS`.

Final V5 artifact: `10311660626`, digest `sha256:003311ab6d45297211d9c5e24bfb26768d26d6c435c9b9069e7fff5acf140277`.

Clinical Learning red checks were adjacent-owner scope-only after substantive tests/frozen-owner guards passed.

## 6. Merge / deploy — satisfied

PR #101 was squash-merged to:

`8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`

Render deploy:

`dep-daj2398u01pc738ojvkg`

reached `live` at that exact runtime commit. No manual duplicate deploy was triggered.

## 7. Authenticated production smoke — satisfied

The first branch-push smoke was queued without a runner, so the identical smoke was executed via a temporary same-repository draft PR trigger rather than weakening authentication or exposing secrets.

Temporary PR:

`#103` — closed unmerged after verification.

Executed smoke:

`34737943402` — **SUCCESS**

Job:

`103672560340` — **SUCCESS**

Verified live production behavior:

1. unauthenticated bootstrap rejected;
2. authenticated bootstrap/project operational;
3. production profile `CY_GESY` via `explicit_account_configuration`;
4. international acupuncture/manual-therapy evidence states unchanged while local Cyprus directions remain separate;
5. local-only electrotherapy absent from international clinical evidence items;
6. default rehab selections unchanged;
7. no Cyprus/GeSY evidence leakage into referral prose;
8. live weakness clinical-sheet asset exposes only the two approved weakness exam options and no quadriceps atrophy option;
9. live More/Exam asset retains quadriceps atrophy;
10. no `localStorage` / `sessionStorage` persistence marker in the tested live V5 assets;
11. only generated UUID + non-identifiable state sent; protected key masked throughout logs.

## 8. Explicit exclusions

No:

- second diagnosis;
- new treatment selector;
- evidence-state reclassification;
- new country selector;
- local-rule automation;
- patient persistence;
- analytics/billing/entitlements;
- Greece/England localization;
- Medical Report runtime/config mutation.

## 9. Exit gate — PASS

```text
MERGED                    YES
DEPLOYED                  YES
AUTHENTICATED SMOKE       PASS
PRODUCTION-SMOKE-VERIFIED YES
CANONICAL CLOSEOUT        READY TO MERGE
WRITER                     NONE
```

The release slice is closed after the docs-only closeout merge. Any next product work requires a new bounded decision.
