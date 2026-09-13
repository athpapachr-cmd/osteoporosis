# SLICE_PLAN_CURRENT.md — Physiotherapy Referral Knee-OA V5 integration patch

> **STATUS:** IMPLEMENTATION ACTIVE — PRODUCT OWNER AUTHORIZED.
> **Activated:** 2026-09-13 Asia/Nicosia.
> **Slice:** `CU-PHYSIO-KNEE-OA-V5-INTEGRATION-2026-09-13`.
> **Branch:** `feat/physio-knee-oa-v5-integration-2026-09-13`.
> **Bootstrap main:** `8aeb91ae37b83caaa188054128db98e04b638fd8`.
> **Original reviewed V5 candidate:** `a47357c602120d3678e8f2f23b99775e616c79e1`.
> **Independent review disposition:** `ACCEPT WITH REQUIRED CHANGES` / `PATCH V5 THEN MERGE`.
> **Writer:** `feat/physio-knee-oa-v5-integration-2026-09-13`.
> **Diagnosis vertical:** Knee Osteoarthritis only.

## 1. Objective

Integrate the already-reviewed/tested Knee-OA V5 workflow/prose refinement onto the current production ancestry instead of directly merging the stale historical V5 head.

The integrated target is:

```text
current main / current shared integrations
+ released CY_GESY jurisdiction overlay
+ bounded V5 workflow/prose deltas
+ Product Owner-approved atrophy simplification
```

No clinical/evidence redesign is part of this slice.

## 2. Historical candidate and current-main divergence

Original tested V5 evidence:

```text
head  a47357c602120d3678e8f2f23b99775e616c79e1
gate  34693751545 — SUCCESS
```

Its merge base is the older v4 closeout ancestry `90e4377fc92e76d767a4b911a0dcff523b50b71e`.

Current main at activation is `8aeb91ae37b83caaa188054128db98e04b638fd8` and includes later released/shared work, including jurisdiction-overlay runtime/tests and Clinical Documents work.

Direct merge/cherry-pick of the historical branch is forbidden.

## 3. Exact V5 interaction contract

### Pain / Stiffness / Weakness

Inactive first tap:

```text
select generic symptom
→ no forced detail sheet
```

Tap again while selected:

```text
open focused optional refinement sheet
```

The selected-state control must retain an understandable progressive-disclosure cue and accessible wording.

### Function

`Λειτουργικότητα` remains a first-tap chooser because an unqualified generic function state is not sufficiently meaningful.

### Weakness refinement after Product Owner simplification

Second-tap `Αδυναμία` refinement contains only:

```text
Μυϊκή αδυναμία στην εξέταση
Αδυναμία τετρακεφάλου στην εξέταση
```

`Ατροφία τετρακεφάλου` is removed from that sheet.

It remains available through:

```text
Περισσότερα
→ Εξέταση
→ Ατροφία τετρακεφάλου
```

This removes duplicate access while preserving the objective examination capability.

## 4. Semantic/copy contract

- Generic weakness remains symptom/context and does not assert objective weakness.
- Objective weakness and quadriceps weakness require explicit examination selections.
- Quadriceps atrophy remains an objective examination finding, not a weakness subtype.
- Bare ambiguous `Περιαρθρικά` must remain absent from the visible routine/advanced UI.
- Pain qualifier ownership must prevent redundant location prose.
- Rich referral separates clinical picture/functional impact from physiotherapy assessment/priorities with a paragraph boundary.
- Additional functional goals are rendered as connected natural prose rather than `Επιπλέον στόχος:`.
- No dose/frequency/protocol is invented.

## 5. Jurisdiction integration contract

V5 is layered over the already-released jurisdiction architecture.

Hard requirements:

```text
international evidence state unchanged
CY_GESY local context remains separate
jurisdiction never auto-selects treatment
jurisdiction never rewrites referral prose
GeSY admin/reimbursement never becomes clinical evidence
planned GeSY IT state remains planned
```

The integrated exact head must prove V5 and jurisdiction behavior together.

## 6. Implementation seams

Expected V5-owned seams from the reviewed candidate:

```text
clinic_utilities/physio_referral_product/knee_oa_presentation_v4.py
clinic_utilities/physio_referral_product/prototype/clinical_sheet_v4.js
clinic_utilities/physio_referral_product/prototype/more_v3.js
clinic_utilities/physio_referral_product/prototype/qualifier_overlay.py
clinic_utilities/physio_referral_product/prototype/qualifiers.js
static/clinic-utilities/physio-referral/product-clinical-sheet-v4.js
static/clinic-utilities/physio-referral/product-more-v3.js
static/clinic-utilities/physio-referral/product-qualifiers.js
V5-focused tests/workflow
```

Shared files modified after the historical V5 merge base, especially protected Cockpit tests/workflows/canonicals, must be merged against current main rather than replaced.

Generated Python cache artifacts are explicitly excluded.

## 7. Test contract

Required focused assertions:

1. first tap on inactive Pain/Stiffness/Weakness selects generic state without popup;
2. second tap opens optional detail;
3. Function first tap still opens chooser;
4. weakness second-tap has exactly the two explicit weakness examination concepts and no quadriceps atrophy option;
5. `Περισσότερα → Εξέταση` still exposes `Ατροφία τετρακεφάλου`;
6. generic weakness never becomes objective weakness;
7. pain composition avoids joint-line/pes-anserine duplication;
8. rich referral has the reviewed paragraph structure;
9. natural functional-goal prose replaces mechanical label;
10. low-information output remains proportionally compact;
11. evidence states/source positions/default selections/suggestions/safety/manual-edit semantics remain unchanged;
12. `CY_GESY` overlay remains active/separate in protected integration tests;
13. local-only/admin positions never become clinical evidence items;
14. no localStorage/sessionStorage patient/referral persistence is introduced;
15. inherited current-main physio/CU-1/evidence/browser regressions pass.

Non-blocking independent-review suggestions that are cheap and bounded may be added to tests, especially broader goal-prose coverage, provided they do not change product semantics.

## 8. Explicit exclusions

No:

- second diagnosis;
- new treatment selector;
- new jurisdiction/country selector;
- evidence-state reclassification;
- local-rule automation;
- receiver-validation gate;
- patient persistence;
- analytics/billing/entitlements;
- Medical Report runtime/config mutation;
- generated/cache artifacts.

## 9. REPLAN triggers

Stop and replan if integration requires any of:

- changing international evidence semantics;
- changing jurisdiction semantics to make V5 work;
- changing safety/diagnosis authority;
- introducing patient/referral persistence;
- overwriting a current-main shared integration with a stale historical file;
- broad redesign beyond the reviewed V5 interaction/prose scope.

## 10. Exit gate

Implementation-complete means one exact integrated head has passed:

- V5 focused tests;
- current jurisdiction-overlay tests;
- current protected Cockpit integration/browser tests;
- inherited v4/prototype/CU-1/evidence/safety coverage;
- scope/diff hygiene proving no unrelated owner mutation.

Merge/deploy/production smoke remain distinct lifecycle states and must be recorded as such.