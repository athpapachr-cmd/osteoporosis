# OST-LIFECOURSE-P1 — Current trajectory truth / semantic-ownership reconciliation

> **MODE:** documentation/design only.
> **ROLE:** fresh separate OST-LIFECOURSE coordinator.
> **ROOT LOCK:** unchanged; PR-1 remains root active writer.
> **PARKED QUESTIONS:** Q1–Q12 remain unresolved.

## Objective

Reconcile the ownership semantics of longitudinal mechanisms that already exist, before proposing any new life-course architecture.

Do not choose the final event/state storage architecture.

## Inputs

Fresh bootstrap current main using AGENTS.md and the six active canonicals, then consume fully:

- programme/PRODUCT-CONSTITUTION-V0.2.md
- programme/WORKSTREAM-REGISTRY.md
- programme/OST-LIFECOURSE/PROJECT-INDEX.md
- programme/OST-LIFECOURSE/CURRENT.md
- programme/OST-LIFECOURSE/PHASE0-PROGRAMME-RECONCILIATION.md

Use the returned Phase-0 handback as evidence, but fresh-verify any source fact that drives a design conclusion.

## Exact scope

For each current representation below, determine whether it is:

- authoritative fact;
- derived projection;
- working cache;
- historical snapshot;
- presentation-only projection;
- clinical-rule source;
- or a combination that must be split/clarified.

Reconcile only these domains:

1. fracture_history / low_trauma semantics and current ownership boundaries;
2. DXA current facts versus longitudinal_review.dxa_history;
3. step3.labs versus clinical_lab_snapshots;
4. Step-4 treatment episodes, administrations and task continuity;
5. LongitudinalGuidanceProjectionV1;
6. EncounterContext contract versus executable runtime;
7. GuidanceRule / TherapyMilestone registries versus executable runtime.

## Required decisions

For every domain, state:

- current authoritative owner;
- current derived/projection owner;
- duplicate or ambiguous owner if present;
- what should be reused unchanged;
- what existing owner should be extended;
- what duplicate representation should eventually be retired or demoted;
- whether the issue is design-only, data-integrity, clinical-safety or presentation-only;
- dependency on another workstream;
- whether a parked question must eventually be activated.

## Special rule — S1

The current fracture/fragility runtime defect is separately routed.

P1 may define the desired ownership/semantic contract for fracture facts and fragility interpretation, but MUST NOT implement or absorb the runtime correction.

## Hard boundaries

- no code changes;
- no schema/database changes;
- no UI redesign;
- no new patient/event/timeline/treatment/obligation/evidence store;
- no cross-module runtime;
- no PR-1 mutation;
- no Product Constitution rewrite;
- no resolution of Q1–Q12;
- no merge/deploy/smoke.

## Required output

Return one bounded design handback:

CURRENT OWNERSHIP MAP
→ AUTHORITATIVE VS DERIVED CONTRACT
→ DUPLICATES / DRIFT
→ EXISTING OWNER TO EXTEND
→ RETIRE / DEMOTE CANDIDATES
→ SAFETY / DATA-INTEGRITY FLAGS
→ CROSS-WORKSTREAM DEPENDENCIES
→ PARKED-QUESTION DEPENDENCIES
→ RECOMMENDED NEXT BOUNDED ACTION
→ REGISTRY SYNC
→ STOP