# S1-P6B-R1 — Correction programme reconciliation

> **STATUS:** R1 CORRECTION VERIFIED / DELTA+CUMULATIVE P6B READY.
> **Date:** 2026-09-29.
> **Existing implementation PR:** #123 — OPEN / DRAFT / NOT MERGED.
> **Blocked predecessor head:** `f4fe36bcfbd0ca475768844387536ec71e2fd38d`.
> **Corrected exact head:** `ba2f8635372f85cd94409fa66e08c3d8b428bcba`.
> **Base main:** `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`.
> **Root writer:** unchanged — OST-CAPTURE / PR-1.

## Fresh programme verification

Programme coordinator independently verified:

- current `main` remains `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`;
- PR #123 remains OPEN / DRAFT / NOT MERGED;
- corrected PR head is exactly `ba2f8635372f85cd94409fa66e08c3d8b428bcba`;
- corrected branch is behind main by `0`;
- merge-base remains exact base main;
- predecessor→corrected delta contains exactly three files;
- cumulative base→head file set remains the same bounded 10-file S1 surface.

## Exact R1 delta

From blocked predecessor `f4fe36bc...` to corrected head `ba2f863...`:

1. `static/baseline-audit/app-core.js`
2. `test_s1_fracture_fragility_app_core.js`
3. `programme/OST-CLINICAL/CURRENT.md`

No G2/G3 semantic file changed in the residual correction.
No DB/schema/migration/clinical_data/patient-registry/PR-1/Product Constitution/OST-LIFECOURSE file changed.

## Accepted R1 correction intent

The corrected app-core path now separates:

```text
raw stored low_trauma
from
rendered canonical select state
from
whether the clinician explicitly edited the control
```

Required behavior:

```text
NO EXPLICIT EDIT
→ preserve exact raw source value

EXPLICIT EDIT
→ persist selected canonical value
```

Clinical normalization remains independent from raw source preservation.

## Exact-head CI verification

Programme coordinator independently verified all relevant final-head workflows on `ba2f8635372f85cd94409fa66e08c3d8b428bcba`:

- Canonical impact guard `36477599182` — SUCCESS;
- G1 progressive guidance foundation `36477599192` — SUCCESS;
- G2 evidence guidance runtime `36477599178` — SUCCESS;
- G3 guidance salience longitudinal summary `36477599188` — SUCCESS.

The corrected S1 app-core regression executes within the G3 cumulative gate.

## Programme disposition

```text
D1–D6: PREVIOUSLY INDEPENDENTLY CLOSED
R1 CORRECTION: IMPLEMENTED / EXACT-HEAD CI GREEN
MERGE: NOT AUTHORIZED
DEPLOY: NOT AUTHORIZED
SMOKE: NOT AUTHORIZED
NEXT: FRESH DELTA+CUMULATIVE P6B
```

## Review strategy

Do not repeat a from-scratch broad review.

The next reviewer must:
- review the three-file R1 delta in detail;
- independently verify no-edit raw preservation and explicit-edit semantics;
- cumulatively verify that the previously closed D1–D6 and preserved behaviors remain intact;
- issue PASS or BLOCK and STOP.

No implementation, merge, deploy or smoke is authorized by this reconciliation.