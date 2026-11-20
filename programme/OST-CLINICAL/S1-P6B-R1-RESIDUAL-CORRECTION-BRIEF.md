# S1-P6B-R1 — Raw low_trauma preservation residual correction brief

> **TASK:** `S1-FRACTURE-FRAGILITY-P6B-R1-RAW-VALUE-PRESERVATION-CORRECTION`
> **MODE:** fresh separate bounded correction implementation.
> **EXISTING PR:** #123.
> **EXACT BLOCKED PREDECESSOR HEAD:** `f4fe36bcfbd0ca475768844387536ec71e2fd38d`.
> **BRANCH:** `fix/ost-clinical-s1-fracture-fragility-semantics-2026-09-27`.
> **DO NOT create a new implementation PR unless source state makes reuse impossible and programme coordinator explicitly reauthorizes it.**

## 1. Role

You are the fresh separate correction implementation author.

You are NOT:
- the original implementation author for review purposes;
- the independent P6B reviewer;
- the S1 coordinator;
- the programme coordinator;
- the OST-LIFECOURSE coordinator;
- a merge/deploy/smoke executor.

## 2. Fresh gate

Fresh-verify `athpapachr-cmd/osteoporosis/main` and perform the six-canonical bootstrap.

Fresh-verify PR #123.

If PR #123 head is not exactly:

`f4fe36bcfbd0ca475768844387536ec71e2fd38d`

STOP with TARGET DRIFT before mutation.

Then consume fully from programme PR #121 / branch `docs/ost-programme-lifecourse-bootstrap-2026-09-27`:

- `programme/OST-CLINICAL/S1-IMPLEMENTATION-BRIEF.md`
- `programme/OST-CLINICAL/S1-P6B-BLOCK-RESULT.md`
- `programme/OST-CLINICAL/S1-P6B-R1-RESIDUAL-CORRECTION-BRIEF.md`

and inspect `programme/OST-CLINICAL/CURRENT.md` on PR #123 branch.

## 3. What is already closed — preserve it

Do not reopen or redesign D1–D6.

The blocked review already verified:
- D1 PASS;
- D2 PASS;
- D3 PASS;
- D4 PASS;
- D5 PASS;
- D6 PASS;
- preserved G1/VFA/R07/denosumab/milestone behavior PASS;
- scope/authority discipline PASS;
- CI trigger ownership PASS.

The correction must preserve all of those closures.

## 4. Exact residual

Raw `fracture_history.events[].low_trauma` can be silently rewritten during no-edit load → render → save when the stored raw value is not exactly one of the select's canonical string values.

Examples include:
- `unknown`;
- ` YES `;
- mixed-case or whitespace variants;
- any other unrecognised legacy raw string.

Current semantic normalization may understand some of those values, but the select can render blank and ordinary save then overwrite the raw value with `""`.

## 5. Required correction contract

### A. No-edit preservation

For an existing event, if the clinician does not explicitly change the `low_trauma` control, a load → render → save cycle MUST preserve the exact stored raw `low_trauma` value.

Examples:

```text
"unknown" → "unknown"
" YES "   → " YES "
"Yes"     → "Yes"
"yes "    → "yes "
```

Do not canonicalize merely because the record was rendered or saved.

### B. Semantic interpretation remains normalized

Clinical derivation may continue to normalize raw values for interpretation.

For example a raw value that the existing accepted normalizer deliberately interprets as canonical yes may support the same derived semantics as before the save.

Preserving source bytes and deriving normalized meaning are distinct responsibilities.

### C. Explicit clinician edit

If the clinician explicitly changes the `low_trauma` selector, the newly chosen canonical value may replace the prior raw value.

Thus:

```text
NO EDIT → preserve exact raw source
EXPLICIT EDIT → persist the explicit canonical selection
```

### D. Unknown/unrecognised values remain fail-closed

Preservation of raw source does NOT make an unrecognised/unknown value positive fragility evidence.

The governing S1 semantic invariant remains unchanged.

### E. No invented facts

Do not:
- synthesize a new fracture event;
- synthesize `low_trauma=yes/no` from a raw unknown value;
- rewrite historical records globally;
- perform migration/backfill.

## 6. Preferred design property

The UI must distinguish:

```text
what was originally stored
from
what canonical option is currently displayed/selected
from
whether the clinician actually changed the field
```

The implementation mechanism is for the correction author to choose after source inspection, but it must be minimal and must not create a second fracture truth owner.

## 7. Minimum new regression coverage

Add focused coverage that exercises the real writer/load-render-save behavior for at least:

1. exact canonical `yes` → unchanged;
2. exact canonical `no` → unchanged;
3. exact canonical `uncertain` → unchanged;
4. exact empty → unchanged;
5. raw `unknown` + no edit → exact raw `unknown` preserved;
6. raw ` YES ` + no edit → exact raw ` YES ` preserved;
7. another unrecognised raw token + no edit → exact raw preserved;
8. raw noncanonical value + explicit user change to `yes` → canonical `yes` persisted;
9. raw noncanonical value + explicit user change to `no` → canonical `no` persisted;
10. semantic derived interpretation before vs after a no-edit save remains stable;
11. legacy prior=true + zero events remains zero events;
12. D1–D6 regression matrix remains green.

At least one test must exercise the complete relevant save path sufficiently to prove that render state cannot silently overwrite the raw value.

Do not satisfy this residual with a helper-only unit test that bypasses the actual DOM/save integration responsible for the defect.

## 8. Mutation boundary

Expected correction should be as small as possible.

Likely owner:
- `static/baseline-audit/app-core.js`;
- focused S1 app-core regression test;
- only other S1 files if source evidence proves necessary;
- `programme/OST-CLINICAL/CURRENT.md` checkpoint.

Do not alter G2/G3 semantics unless a regression demonstrates that the residual correction requires it.

Do not touch:
- DB/schema/migrations;
- `clinical_data.py`;
- patient-registry persistence architecture;
- PR-1;
- Product Constitution;
- OST-LIFECOURSE architecture;
- Q1–Q12.

## 9. Test and PR rules

Keep using existing PR #123 / branch.

Run:
- focused S1 app-core regression;
- relevant G1/G2/G3 suites;
- workflow/syntax checks affected by the delta;
- Canonical Impact guard.

Update `programme/OST-CLINICAL/CURRENT.md` to identify:
- blocked predecessor head;
- residual R1;
- correction commits/head;
- exact tests/CI;
- next action = fresh delta+cumulative independent P6B review.

## 10. Completion gate

STOP when:
- R1 correction is implemented;
- D1–D6 closures remain preserved;
- new raw-preservation tests pass;
- relevant inherited regressions pass;
- exact corrected head is known;
- PR #123 remains draft;
- workstream CURRENT is checkpointed;
- exact-head CI is green or any limitation is explicitly reported.

Do not merge/deploy/smoke.

## 11. Required handback

Return:

CORRECTION SOURCE IDENTITY
→ PREDECESSOR HEAD
→ CORRECTED HEAD
→ FILES CHANGED IN DELTA
→ EXACT R1 FIX
→ RAW-PRESERVATION TEST EVIDENCE
→ D1–D6 PRESERVATION EVIDENCE
→ PR #123 / CI STATUS
→ WORKSTREAM CURRENT STATUS
→ RESIDUAL RISKS
→ DELTA+CUMULATIVE P6B READY = YES | NO
→ STOP