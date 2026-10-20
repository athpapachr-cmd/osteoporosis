# S1-P6B-R1 — Delta + cumulative independent P6B review brief

> **TASK:** `S1-FRACTURE-FRAGILITY-P6B-R1-DELTA-CUMULATIVE-REVIEW`
> **MODE:** fresh independent READ-ONLY exact-head delta+cumulative review.
> **TARGET PR:** #123.
> **BLOCKED PREDECESSOR HEAD:** `f4fe36bcfbd0ca475768844387536ec71e2fd38d`.
> **CORRECTED TARGET HEAD:** `ba2f8635372f85cd94409fa66e08c3d8b428bcba`.
> **BASE MAIN:** `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`.
> **VERDICT:** PASS or BLOCK only.
> **STOP after verdict.**

## 1. Role

You are the fresh independent delta+cumulative P6B reviewer.

You are NOT:
- the original implementation author;
- the R1 correction author;
- the S1 coordinator;
- the programme coordinator;
- the OST-LIFECOURSE coordinator;
- a merge/deploy/smoke executor.

Do not fix code.

## 2. Fresh bootstrap

Fresh-verify `athpapachr-cmd/osteoporosis/main` and complete the six-canonical bootstrap in AGENTS order.

Then consume fully from programme PR #121 / branch `docs/ost-programme-lifecourse-bootstrap-2026-09-27`:

- `programme/OST-CLINICAL/S1-P6B-BLOCK-RESULT.md`
- `programme/OST-CLINICAL/S1-P6B-R1-RESIDUAL-CORRECTION-BRIEF.md`
- `programme/OST-CLINICAL/S1-R1-CORRECTION-PROGRAMME-RECONCILIATION.md`
- `programme/OST-CLINICAL/S1-IMPLEMENTATION-BRIEF.md`
- `programme/OST-CLINICAL/S1-PRECODE-PROGRAMME-RECONCILIATION.md`.

Also inspect `programme/OST-CLINICAL/CURRENT.md` on PR #123 branch.

## 3. Exact-target gate

Fresh-verify PR #123.

If predecessor or corrected head identities do not match exactly:

```text
PREDECESSOR = f4fe36bcfbd0ca475768844387536ec71e2fd38d
TARGET      = ba2f8635372f85cd94409fa66e08c3d8b428bcba
```

STOP and report TARGET DRIFT.

Do not review a later head.

## 4. Review scope

This is NOT a full redesign or fresh broad review.

Perform:

```text
R1 DELTA REVIEW
+
CUMULATIVE PRESERVATION CHECK OF D1–D6 / REGRESSIONS
```

## 5. R1 exact objective

Verify that a no-edit load → render → save cycle preserves the exact raw `fracture_history.events[].low_trauma` value, even when it is noncanonical.

Required cases include:

- `yes`;
- `no`;
- `uncertain`;
- empty;
- `unknown`;
- ` YES `;
- another unrecognised raw token.

### No-edit invariant

```text
NO EXPLICIT CLINICIAN EDIT
→ exact raw stored low_trauma survives unchanged
```

### Explicit-edit invariant

```text
EXPLICIT CLINICIAN EDIT
→ selected canonical value is persisted
```

Verify explicit changes to at least:
- `yes`;
- `no`.

## 6. Semantic-stability check

Verify that source preservation did not alter the accepted semantic interpretation:

- a raw representation deliberately normalized by current semantics to yes remains semantically yes before and after a no-edit save;
- unknown/unrecognised remains fail-closed;
- raw preservation itself does not promote unknown to positive fragility.

## 7. Actual integration-path requirement

Independently verify that the regression test exercises the relevant real app-core DOM/load/render/save path.

A helper-only test that bypasses the actual writer path is insufficient.

Verify that the correction does not simply avoid saving all low_trauma changes; explicit clinician edits must still persist.

## 8. Cumulative D1–D6 preservation

Reconfirm, at least through source inspection + inherited regression evidence, that:

- D1 remains closed;
- D2 remains closed;
- D3 remains closed;
- D4 remains closed;
- D5 remains closed;
- D6 remains closed.

The R1 delta must not reopen any of them.

## 9. Preserved-behavior check

Verify preservation of:
- G1 generic fracture behavior;
- generic vertebral/VFA handling;
- fracture-on-treatment / R07 independence;
- R05/R06/R08 fail-closed fragility semantics;
- denosumab/milestone/treatment logic;
- no synthetic event/UUID creation from legacy prior=true;
- stable event identity.

## 10. Delta/scope verification

Expected predecessor→target delta is exactly:

- `static/baseline-audit/app-core.js`;
- `test_s1_fracture_fragility_app_core.js`;
- `programme/OST-CLINICAL/CURRENT.md`.

BLOCK if the exact delta contains an unexplained material file or scope expansion.

Verify no:
- G2/G3 semantic code mutation in R1 delta;
- DB/schema/migration change;
- clinical_data.py change;
- patient-registry change;
- PR-1 change;
- Product Constitution/LifeCourse change;
- new semantic truth owner/store/helper architecture.

## 11. Independent evidence

Do not rely solely on author handback.

Independently inspect source/diff and independently execute or tool-verify the focused corrected regression and inherited gates.

Reported exact-target workflows to cross-check:

- G3 cumulative/S1 `36477599188` — SUCCESS;
- G2 `36477599178` — SUCCESS;
- G1 `36477599192` — SUCCESS;
- Canonical Impact guard `36477599182` — SUCCESS.

If local execution is unavailable, state the limitation explicitly and independently inspect exact workflow jobs/logs plus exact source/diff. Do not falsely claim local execution.

## 12. Verdict

PASS only if:
- R1 is actually closed;
- exact raw values survive no-edit save;
- explicit edits persist;
- normalized clinical semantics remain correct;
- D1–D6 remain closed;
- inherited behavior remains intact;
- exact delta stays bounded;
- exact-head evidence is adequate.

Otherwise BLOCK with exact residual finding(s).

## 13. Prohibited

- no code fixes;
- no branch/PR mutation;
- no canonical/current mutation;
- no merge/deploy/smoke;
- no successor task creation;
- no review-of-review.

## 14. Required handback

Return:

REVIEW SOURCE IDENTITY
→ EXACT PREDECESSOR / TARGET / BASE
→ DELTA SCOPE VERIFIED
→ R1 RAW-PRESERVATION CHECK
→ EXPLICIT-EDIT CHECK
→ SEMANTIC-STABILITY CHECK
→ D1–D6 CUMULATIVE PRESERVATION
→ PRESERVED-BEHAVIOR CHECK
→ INDEPENDENT TEST / CI EVIDENCE
→ RESIDUAL FINDINGS
→ VERDICT = PASS | BLOCK
→ REGISTRY SYNC
→ STOP