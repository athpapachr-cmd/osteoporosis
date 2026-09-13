# SLICE_PLAN_CURRENT.md — Physiotherapy Referral Knee-OA V5 release

> **STATUS:** RELEASE MERGED / DEPLOYED — AUTHENTICATED SMOKE PENDING EXECUTION.
> **Activated:** 2026-09-13 Asia/Nicosia.
> **Slice:** `CU-PHYSIO-KNEE-OA-V5-RELEASE-2026-09-13`.
> **Release PR:** `#101`.
> **Release runtime SHA:** `8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`.
> **Render deploy:** `dep-daj2398u01pc738ojvkg` — LIVE.
> **Authenticated V5 smoke:** run `34737351354` — QUEUED / NOT YET EXECUTED at this draft.
> **Writer:** none.
> **Diagnosis vertical:** Knee Osteoarthritis only.

## 1. Release objective

Release the accepted fresh-main V5 integration while preserving the current Knee-OA / CU-1 / `CY_GESY` architecture and verifying the resulting protected production behavior.

Release sequence:

```text
exact tested V5 PR head
→ squash merge with expected-head guard
→ Render auto-deploy
→ authenticated live smoke
→ docs-only canonical closeout
```

The first three steps through deployment are complete. Authenticated smoke has been submitted but has not yet received a GitHub runner.

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

`Ατροφία τετρακεφάλου` is not duplicated there. It remains available through:

```text
Περισσότερα → Εξέταση → Ατροφία τετρακεφάλου
```

Weakness count reflects weakness-detail state only.

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

V5 remains layered over the released architecture:

```text
international evidence core
+
JurisdictionOverlayV1
+
explicit CY_GESY production profile
+
V5 presentation/prose layer
```

Hard requirements remain:

```text
international evidence state unchanged
CY_GESY local context remains separate
jurisdiction never auto-selects treatment
jurisdiction never rewrites referral prose
GeSY admin/reimbursement never becomes clinical evidence
planned != active
```

## 5. Pre-release gates — satisfied

Exact PR head `4e4bd2ae40c606562a982b3e38f9f859b49986eb` passed all Physio-owned exact-head gates plus CU-1 focused tests before merge.

Key V5 gate: `34736919952` — SUCCESS.

Final V5 artifact: `10311660626`, digest `sha256:003311ab6d45297211d9c5e24bfb26768d26d6c435c9b9069e7fff5acf140277`.

## 6. Merge / deploy — satisfied

PR #101 was squash-merged to:

`8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`

Render deploy:

`dep-daj2398u01pc738ojvkg`

reached `live` at that exact runtime commit. No manual duplicate deploy was triggered.

## 7. Authenticated production-smoke contract

The temporary non-merged ops workflow must prove, on the live service:

1. unauthenticated product bootstrap remains rejected;
2. authenticated bootstrap/project remain operational;
3. production jurisdiction profile remains `CY_GESY` via explicit account configuration;
4. international acupuncture/manual-therapy evidence states remain unchanged while local directions remain separate;
5. local-only electrotherapy does not become an international clinical evidence item;
6. default rehabilitation selection remains unchanged;
7. referral prose contains no silent Cyprus/GeSY evidence leakage;
8. live clinical-sheet asset has only the two weakness exam options and no duplicated quadriceps-atropy option;
9. live More/Exam asset retains quadriceps atrophy;
10. no browser patient/referral persistence marker is introduced;
11. only generated UUID/non-identifiable smoke state is sent and the protected key is never printed.

Run `34737351354` is queued but has not yet executed.

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

## 9. Exit gate

Final release-closeout requires authenticated production smoke `SUCCESS`.

Current state:

```text
MERGED                    YES
DEPLOYED                  YES
AUTHENTICATED SMOKE       PENDING EXECUTION
CANONICAL FINAL CLOSEOUT  NOT YET MERGED
```

A queued job is not a PASS and is not an application failure.
