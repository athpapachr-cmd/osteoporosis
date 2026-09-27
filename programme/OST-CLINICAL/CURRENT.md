# OST-CLINICAL CURRENT — S1 Fracture / Fragility Semantics

> **TASK:** `S1-FRACTURE-FRAGILITY-SEMANTICS-CORRECTION`
> **STATUS:** IMPLEMENTATION ACTIVE / BOUNDED S1 CANDIDATE
> **Date:** 2026-09-27 Asia/Nicosia.
> **Fresh base main:** `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`.
> **Implementation branch:** `fix/ost-clinical-s1-fracture-fragility-semantics-2026-09-27`.
> **Root operational owner:** unchanged — `CURRENT_OPERATIONAL.md` remains OST-CAPTURE / PR-1-owned.

## Governing semantic invariant

```text
FRACTURE EXISTS
!=
FRAGILITY FRACTURE

MISSING / UNKNOWN / UNCERTAIN TRAUMA MECHANISM
!=
POSITIVE FRAGILITY FACT

CONFIRMED EVENT-LEVEL FRAGILITY
⇔ normalized low_trauma === "yes"
```

## Bounded mutation scope

Expected S1 owners only:

- `static/baseline-audit/app-core.js`;
- `static/baseline-audit/osteoporosis-evidence-guidance-core.js`;
- `static/baseline-audit/osteoporosis-longitudinal-summary-core.js`;
- `static/baseline-audit/progressive-guidance-ui.js` only for contaminated G3 wording/state;
- focused S1 regressions and extensions to existing G2/G3 tests;
- minimum CI path/wiring required to guarantee S1 tests execute;
- this workstream checkpoint.

`static/baseline-audit/progressive-guidance-core.js` remains out of mutation scope unless a focused failing test proves an S1 defect.

## Required verification

Before implementation candidate completion:

- app writer/load-render regression: generic/uncertain/missing events do not write positive compatibility fragility and legacy prior=true + zero events creates no event/UUID;
- G2 yes/no/uncertain/missing current-fragility matrix;
- traumatic/uncertain/missing vertebral fragility fail-closed while generic VFA behavior remains;
- fracture-on-treatment R07 preserved independently of fragility-specific R05/R06;
- archetype alone cannot manufacture current fragility;
- stable fracture IDs remain deduplicated without rewriting raw records;
- G3 generic fracture count remains distinct from confirmed fragility;
- unrelated denosumab/milestone/G1/generic-fracture/finalization regressions remain green;
- CI path trigger covers `app-core.js`.

## Current state

Fresh bootstrap and source inspection complete. Confirmed S1 defects D1–D6 are present on current main. Implementation changes have not yet been applied on this branch.

## Draft PR

Not opened yet. Open only after bounded code/test correction and exact-head verification.

## Exact next action

Implement the accepted D1–D6 correction inside existing owners, add the focused regression matrix and minimum CI wiring, then run exact-head verification.

## Explicitly forbidden

- modifying root `CURRENT_OPERATIONAL.md`;
- PR-1 transcript implementation;
- `clinical_data.py`, database/schema/migrations or patient-registry storage transport;
- new fracture store/event bus/lifecourse engine;
- Product Constitution or OST-LIFECOURSE architecture changes;
- resolving Q1–Q12;
- independent post-code review by this author;
- merge, deploy or production smoke.
