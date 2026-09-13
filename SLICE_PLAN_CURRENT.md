# SLICE_PLAN_CURRENT.md — Physiotherapy Referral Knee-OA V5 integration patch

> **STATUS:** IMPLEMENTATION COMPLETE / EXACT-HEAD TESTED — RELEASE HOLD.
> **Activated:** 2026-09-13 Asia/Nicosia.
> **Slice:** `CU-PHYSIO-KNEE-OA-V5-INTEGRATION-2026-09-13`.
> **Branch:** `feat/physio-knee-oa-v5-integration-2026-09-13`.
> **Bootstrap main:** `8aeb91ae37b83caaa188054128db98e04b638fd8`.
> **Exact tested integrated head:** `9a7f360745710deacb0ff82f03249723bdfe87d6`.
> **Final gate:** `34736389860` — SUCCESS.
> **Artifact:** `10311108002` / `sha256:c51e3e29c4057b0e90773011d54648703eca5d15c66a9f64e612952f1dd324d6`.
> **Original reviewed V5 candidate:** `a47357c602120d3678e8f2f23b99775e616c79e1`.
> **Independent review disposition:** `ACCEPT WITH REQUIRED CHANGES` / `PATCH V5 THEN MERGE`.
> **Writer:** none — implementation slice closed.
> **Diagnosis vertical:** Knee Osteoarthritis only.

## 1. Objective — achieved

The reviewed/tested V5 workflow/prose refinement has been re-integrated onto current production ancestry instead of directly merging the stale historical V5 branch.

Integrated result:

```text
fresh main 8aeb91ae...
+ current shared integrations
+ released CY_GESY jurisdiction overlay
+ bounded reviewed V5 deltas
+ Product Owner-approved atrophy simplification
→ exact integrated head 9a7f3607...
```

No clinical/evidence redesign was required.

## 2. Review blocker — resolved

Historical V5 head `a47357c...` was tested on older v4 ancestry. Fresh-main integration was therefore required before release consideration.

The historical branch was not merged/cherry-picked wholesale. V5 source/test deltas were reconstructed selectively; current-main shared files were merged semantically; generated/cache artifacts were excluded.

Final gate `34736389860` proves V5 + current `CY_GESY` + current protected Cockpit behavior on one exact SHA.

## 3. Final interaction contract — implemented

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

### Weakness refinement

The second-tap sheet contains only:

```text
Μυϊκή αδυναμία στην εξέταση
Αδυναμία τετρακεφάλου στην εξέταση
```

`Ατροφία τετρακεφάλου` is not duplicated there.

It remains reachable through:

```text
Περισσότερα
→ Εξέταση
→ Ατροφία τετρακεφάλου
```

The weakness count reflects weakness-detail state only. Selecting atrophy through Examination does not create a hidden/phantom weakness-detail count.

## 4. Semantic/copy contract — implemented

- generic weakness remains symptom/context;
- objective weakness/quadriceps weakness require explicit examination selections;
- quadriceps atrophy remains an objective examination finding;
- bare ambiguous `Περιαρθρικά` remains absent from visible routine/advanced UI;
- pain qualifier ownership prevents redundant joint-line/pes-anserine prose;
- rich referral separates clinical picture/functional impact from physiotherapy assessment/priorities;
- functional goals use connected natural prose rather than `Επιπλέον στόχος:`;
- low-information output remains compact;
- no treatment dose/frequency/protocol is invented.

## 5. Jurisdiction integration contract — verified

The released architecture remains:

```text
international evidence core
+
separate CY_GESY jurisdiction overlay
+
V5 presentation/prose layer
```

Verified on the same integrated head:

- international evidence state unchanged;
- local Cyprus context remains separate;
- jurisdiction does not auto-select treatment;
- jurisdiction does not rewrite referral prose;
- GeSY admin/reimbursement rows do not become clinical evidence;
- local-only interventions do not become product controls;
- planned GeSY IT status remains planned.

## 6. Exact test evidence

Final run `34736389860` on `9a7f360745710deacb0ff82f03249723bdfe87d6` passed:

1. bounded-scope + no `.pyc`/`__pycache__` guard;
2. syntax;
3. V5 focused prose/server regressions;
4. inherited server/qualifier regressions;
5. current jurisdiction-overlay tests;
6. current protected Cockpit integration tests;
7. V5 first-tap/second-tap Chromium tests;
8. explicit weakness-vs-atrophy route regression;
9. inherited prototype/qualifier/usability/More browser tests;
10. protected Cockpit browser tests with active `CY_GESY`;
11. manual-edit fail-closed and no-storage boundaries;
12. adjacent-owner isolation smoke;
13. package closure.

Artifact: `10311108002`, digest `sha256:c51e3e29c4057b0e90773011d54648703eca5d15c66a9f64e612952f1dd324d6`.

Earlier failed integration runs are diagnostic history only and are not acceptance evidence.

## 7. Explicit exclusions retained

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

## 8. Exit gate — satisfied

The implementation exit gate is satisfied on exact head `9a7f3607...`.

The slice is now **implementation-complete and tested**.

This does not equal release.

## 9. Release hold

Truthful lifecycle:

```text
implementation        complete
tests                 pass
PR                     next
merge                  not authorized yet
Render deploy          not authorized yet
V5 production smoke    not performed
```

## 10. Exact next action

Open a bounded PR preserving Product Owner release HOLD, verify current-main ancestry/diff/review threads/exact-head gates, and wait for explicit release authority before merge/deploy.