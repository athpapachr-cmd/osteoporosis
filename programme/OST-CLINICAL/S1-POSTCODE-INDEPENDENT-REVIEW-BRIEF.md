# S1 Fracture / Fragility Semantics — Independent post-code review brief

> **TASK:** `S1-FRACTURE-FRAGILITY-SEMANTICS-CORRECTION-P6B`
> **MODE:** fresh independent READ-ONLY exact-head post-code review.
> **TARGET PR:** #123.
> **EXACT TARGET HEAD:** `f4fe36bcfbd0ca475768844387536ec71e2fd38d`.
> **VERDICT:** PASS or BLOCK only.
> **STOP after verdict.**

## 1. Role

You are the fresh independent reviewer.

You are NOT:
- the implementation author;
- the S1 coordinator;
- the OST-LIFECOURSE coordinator;
- the programme coordinator;
- a merge/deploy/smoke executor.

Do not fix code.

## 2. Fresh bootstrap

Fresh-verify `athpapachr-cmd/osteoporosis/main` and read the six active canonicals in AGENTS order.

Then consume fully from draft PR #121 / branch `docs/ost-programme-lifecourse-bootstrap-2026-09-27`:

- `programme/OST-LIFECOURSE/S1-FRACTURE-FRAGILITY-ROUTING-BRIEF.md`
- `programme/OST-CLINICAL/S1-PRECODE-PROGRAMME-RECONCILIATION.md`
- `programme/OST-CLINICAL/S1-IMPLEMENTATION-BRIEF.md`
- `programme/OST-CLINICAL/S1-IMPLEMENTATION-PROGRAMME-RECONCILIATION.md`
- accepted fracture ownership boundary from `programme/OST-LIFECOURSE/P1-PROGRAMME-RECONCILIATION.md`.

Then inspect draft PR #123 at exact head:

`f4fe36bcfbd0ca475768844387536ec71e2fd38d`

Do not review a later moving head without stopping and reporting target drift.

## 3. Governing invariant

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

## 4. Exact review objectives

Independently verify both implementation fidelity and regression safety.

### A. Writer / load-render path
Verify that:
- generic fracture existence no longer auto-writes positive prior fragility;
- generic events no longer overwrite compatibility last-fragility fields;
- load/render of legacy prior=true cannot synthesize a structured event or UUID;
- stable event IDs and raw mechanism values are preserved.

### B. G2 semantics
Verify that:
- current fragility requires actual confirmed event-level positive evidence;
- encounter archetype alone cannot manufacture mechanism truth;
- historical vertebral fragility count/recency requires `low_trauma === "yes"`;
- unknown/uncertain/missing fail closed;
- `fracture_on_treatment` remains independent;
- R07 remains preserved where intended;
- R05/R06 and R08 do not activate from non-confirmed fragility evidence.

### C. G3 semantics / presentation
Verify that:
- generic fracture count and confirmed fragility count are separated;
- stale `event.fragility` is not treated as canonical current mechanism evidence;
- generic/legacy fracture history is not worded as confirmed fragility;
- conflicting stable-ID mechanism snapshots fail closed rather than silently preferring positive.

### D. Preserved behavior
Verify no material regression in:
- G1 generic fracture behavior;
- VFA / generic vertebral-fracture handling;
- denosumab guidance;
- therapy milestones;
- treatment-history logic;
- unrelated finalization/wiring paths covered by the existing suites.

### E. Scope discipline
Verify:
- no schema/database/migration change;
- no `clinical_data.py` change;
- no patient-registry persistence change;
- no PR-1 mutation;
- no Product Constitution/LifeCourse architecture mutation;
- no new fracture store/event bus/parallel semantic owner;
- Q1–Q12 remain parked.

### F. CI / regression ownership
Verify that the added workflow wiring genuinely causes the focused S1 app-core regression to execute when relevant S1-owned files change, rather than merely existing as an untriggered test.

## 5. Required evidence

Freshly verify target identity and inspect the exact diff.

Independently execute or otherwise tool-verify, at minimum:
- focused S1 app-core writer/load-render regression;
- G2 evidence-guidance test suite including S1 cases;
- G3 longitudinal-summary test suite including S1 cases;
- G3 wiring test;
- existing G1 progressive-guidance regression;
- any syntax/check command required by the modified workflow.

Do not rely solely on the implementation author's reported green CI.

Record exact commands/results or exact workflow evidence you independently verified.

## 6. Known implementation CI to cross-check

Reported exact-head workflows:
- Canonical impact guard `36329166319` — SUCCESS;
- G1 `36329166357` — SUCCESS;
- G2 `36329166353` — SUCCESS;
- G3 `36329166349` — SUCCESS.

These are evidence to cross-check, not substitutes for independent review.

## 7. Backward-compatibility review

Explicitly inspect these cases:
- legacy prior=true + zero events;
- prior=true + low_trauma=no;
- prior=true + uncertain/missing;
- prior=false + low_trauma=yes;
- stale event.fragility without low_trauma;
- repeated stable event ID;
- conflicting same-ID low_trauma snapshots.

Confirm that raw historical data is preserved and no invented canonical fact is written.

## 8. Verdict rule

PASS only if:
- all six confirmed defect classes are actually closed within the accepted bounded contract;
- fail-closed semantics hold for unknown/uncertain/missing;
- preserved behavior remains intact;
- implementation stays inside existing owners;
- exact-head regression evidence is adequate;
- no material new defect or authority drift is introduced.

Otherwise BLOCK and state exact residual findings.

Do not issue implementation suggestions beyond what is needed to identify the blocking defect.

## 9. Prohibited actions

- no code fixes;
- no branch mutation;
- no PR metadata mutation;
- no canonical/current mutation;
- no merge/deploy/smoke;
- no successor task creation;
- no review-of-review.

## 10. Required handback

Return:

REVIEW SOURCE IDENTITY
→ EXACT PR / HEAD / BASE
→ DIFF SCOPE VERIFIED
→ INDEPENDENT TEST / CI EVIDENCE
→ D1–D6 CLOSURE CHECK
→ BACKWARD-COMPATIBILITY CHECK
→ PRESERVED-BEHAVIOR CHECK
→ SCOPE / AUTHORITY CHECK
→ RESIDUAL FINDINGS
→ VERDICT = PASS | BLOCK
→ REGISTRY SYNC
→ STOP