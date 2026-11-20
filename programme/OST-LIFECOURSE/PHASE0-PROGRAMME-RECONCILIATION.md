# OST-LIFECOURSE Phase 0 — Programme reconciliation

> STATUS: ACCEPTED / PHASE 0 COMPLETE / SUCCESSOR DESIGN READY.
> Date: 2026-09-27.
> Role: programme-coordinator reconciliation only; no implementation authority.
> Fresh main at reconciliation: 2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b.

## 1. Programme disposition

Phase-0 result: ACCEPT.

Central decision: DO NOT BUILD A NEW LONGITUDINAL STACK.

The existing substrate already provides protected patient identity, protected encounters/labs, longitudinal projection, EncounterContext, VisitPlan, evidence-backed guidance and clinician decision capture.

The primary current problem is fragmented ownership, contract/runtime drift, incomplete semantics and presentation gaps — not absence of longitudinal infrastructure.

Therefore, absent later source-proven insufficiency, the programme forbids introducing a second patient store, event store/event bus, treatment table, timeline database, obligation engine, evidence engine or AI semantic stack.

## 2. Reuse direction accepted

Reuse or extend existing owners for patient identity/auth, encounter history, fracture events, treatment episodes, actual/scheduled administrations, longitudinal guidance projection, VisitPlan/GuidedCardState, clinician decision, conflict handling, Cockpit/module navigation, provider-agnostic capture and the existing evidence engine.

## 3. Existing-owner fragmentation requiring reconciliation

- labs: encounter step3.labs versus clinical_lab_snapshots;
- DXA: protected encounter facts versus longitudinal_review.dxa_history;
- treatment snapshots across encounters;
- follow-up task stable identity versus semantic-key reconciliation;
- EncounterContext contract versus executable runtime;
- GuidanceRule declarative registry versus executable G2 semantics;
- TherapyMilestone registry versus executable runtime;
- fracture semantics;
- recommendation / patient preference / final-decision reconstruction.

These are consolidation or extension problems, not evidence for replacement stores.

## 4. Genuine gaps accepted without choosing implementation

- shared cross-module factual event plus module-specific interpretation seam;
- goals / target-state representation;
- evidence-at-decision-time provenance;
- reusable typed module interpretation layer;
- generic decision/intervention to later outcome linkage without causal overclaim;
- future provider-agnostic Care Communication Layer;
- future practice-level cohort trajectory intelligence.

All remain subject to later bounded design and the parked Product Constitution questions.

## 5. UX / adapter gaps

The full Life Course Timeline, What Matters Today consolidation, richer multi-site DXA presentation, richer historical decision presentation and general→patient-aware→decision-aware Quick Tool surfaces must reuse the existing substrate rather than trigger new stores or rule engines.

## 6. Current safety/data-integrity finding S1

S1 is independently confirmed and elevated outside OST-LIFECOURSE implementation scope.

Current runtime has inconsistent fracture/fragility semantics:
- structured fracture events use low_trauma;
- collectFractureEventsFromDom() can set risk_context.prior_fragility_fracture=true merely because any structured fracture event exists;
- longitudinal summary can derive prior fragility from any fracture event and reads a non-current event.fragility field;
- current G2 interval-fragility handling requires low_trauma=yes, but historical vertebral-fragility handling counts anything not explicitly low_trauma=no.

Programme classification: CURRENT RUNTIME SAFETY / DATA-INTEGRITY DEFECT, existing owner.

Required routing: return to the existing Module-01 clinical/fracture semantics owner; preserve missing/uncertain != positive fragility fact; scope/test/review separately; OST-LIFECOURSE must not opportunistically fix it.

No code correction is authorised by this reconciliation artifact.

## 7. Other material integrity items accepted

- obligations/tasks lack adequate disposition continuity for refusal/reschedule/supersession;
- dental/special-context semantics are incomplete;
- cross-module fracture ingestion does not exist and must not be simulated by copying a fact and silently declaring fragility;
- DXA/lab duplicated truth must be reconciled before new longitudinal persistence;
- guidance contract/runtime drift creates future clinical-rule drift risk;
- task continuity by semantic tuple rather than stable identity can create duplicate unresolved obligations.

## 8. Twelve parked questions

All twelve Product Constitution questions remain PARKED / UNRESOLVED / NOT ACTIVATED.

## 9. Successor decision

Programme accepts the proposed successor:

OST-LIFECOURSE-P1 — CURRENT TRAJECTORY TRUTH / SEMANTIC-OWNERSHIP RECONCILIATION.

P1 must declare, for current representations only, which are authoritative fact, derived projection, working cache, historical snapshot, presentation-only projection or clinical-rule source.

P1 bounded scope:
- fracture-history / low-trauma ownership semantics only; runtime S1 correction is separate;
- DXA current facts versus longitudinal_review.dxa_history;
- step3.labs versus clinical_lab_snapshots;
- Step-4 treatment / administrations / task continuity;
- LongitudinalGuidanceProjectionV1;
- EncounterContext contract versus runtime;
- GuidanceRule / TherapyMilestone registries versus executable runtime.

P1 hard boundaries: documentation/design only; no persistence/schema/database/UI/runtime mutation; no event bus; no new timeline/treatment/obligation/evidence engine; no PR-1 mutation; no cross-module runtime; no resolution of Q1–Q12.

## 10. Parallel routing decision

Two lanes may proceed independently:

A. OST-LIFECOURSE-P1 — design-only current-owner reconciliation.
B. Module-01 fracture/fragility S1 — separately scoped current-runtime safety correction under the existing clinical owner.

P1 does not absorb the S1 correction, and S1 does not decide future life-course architecture.

## 11. Programme next action

1. checkpoint Phase-0 COMPLETE in OST-LIFECOURSE local state;
2. prepare a self-contained P1 design brief;
3. separately prepare/rout a bounded S1 safety-correction task to the existing Module-01 owner;
4. keep PR #121 draft until these organisational artifacts are visible and the Product Owner decides whether to merge the programme bootstrap.

No implementation, merge, deploy or parked-question activation is implied.