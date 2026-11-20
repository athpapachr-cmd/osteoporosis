# OST-LIFECOURSE — Project Index

> **Status:** ACTIVE DESIGN WORKSTREAM / Phase 0 inventory.
> **Product direction:** `../PRODUCT-CONSTITUTION-V0.2.md`
> **Operational sidecar:** `CURRENT.md`
> **Root writer lock:** remains `CURRENT_OPERATIONAL.md` and is not transferred by this project.

## Mission

Define the minimum reusable conceptual contract needed for the Cockpit to understand a patient's clinical care trajectory across time without creating a parallel data model or prematurely redesigning the UI.

The first responsibility is **reuse/gap discovery**, not invention.

## Phase 0 objective

Answer:

> Which parts of the intended longitudinal model already exist in current Core/Module-01 runtime and canonicals, which responsibilities are duplicated or fragmented, and which genuinely new concepts remain?

Phase 0 is read-only.

## Inventory domains

Inspect and map at minimum:

1. patient identity and protected encounter persistence;
2. encounter/history ownership;
3. fracture events and other longitudinal facts;
4. treatment episodes;
5. actual vs scheduled administrations;
6. laboratory/DXA/VFA longitudinal state;
7. `LongitudinalGuidanceProjectionV1`;
8. `EncounterContextV1`;
9. `VisitPlanV1`;
10. `GuidanceRuleV1`;
11. `GuidedCardStateV1`;
12. `TherapyMilestoneProfileV1`;
13. existing follow-up task/due-state mechanics;
14. clinician decision/rationale/patient-preference capture;
15. conflict/uncertainty semantics;
16. Practice Review / Learning seams that consume encounter truth;
17. existing global Cockpit/module ownership boundaries.

## Required Phase-0 output

Produce one inventory with columns/concepts equivalent to:

```text
Product-Constitution responsibility
→ existing owner/mechanism
→ evidence / exact path
→ reuse as-is / extend / consolidate / gap
→ ambiguity or duplicate owner
→ safety/data-integrity concern
→ downstream question
```

The handback must distinguish:

- proven current runtime;
- canonical design only;
- future concept;
- missing mechanism.

## Reuse-before-new-path gate

No new object, event bus, patient state store, obligation engine or timeline store may be proposed until the inventory demonstrates that the current owners cannot safely satisfy the responsibility.

## Synthetic pressure-test cases

Use these cases to test the conceptual boundaries, without implementing them:

### A. Denosumab continuation refusal
Next dose becomes due; patient declines. The old obligation must receive a disposition and the case becomes a new decision point rather than a forced overdue task.

### B. Dental procedure changes the decision context
A planned treatment milestone meets a new dental/extraction issue. The system must surface changed context without imposing a rigid treatment command.

### C. Cross-module fracture
A distal-radius or ankle fracture is recorded outside Osteoporosis. Osteoporosis becomes aware of the shared event but must perform its own module-specific fragility interpretation.

### D. Divergent skeletal-site response
Treatment is followed by improvement at one DXA site and deterioration or lack of improvement at another. The model must preserve measurements, interpretation and subsequent reasoning separately.

### E. Patient preference explains guideline divergence
Clinician recommends one route; patient declines and chooses another. Historical review must preserve who recommended/decided what and why.

### F. Quick Clinical Tool
"Is zoledronic acid safe?" must work generally, patient-aware, and decision-aware without creating a second evidence engine or automatically writing guidance into the patient record.

### G. Overlapping interventions
Pharmacotherapy, vitamin-D replacement, exercise and falls interventions may coexist; a single exclusive "epoch" state is insufficient.

## Explicitly parked

The 12 Product Constitution downstream design questions remain parked during Phase 0. Phase 0 may identify which future workstream owns them but must not resolve them opportunistically.

## Phase 0 completion gate

Phase 0 is complete when:

- current mechanisms are mapped with exact evidence;
- duplicate-mechanism risks are identified;
- real gaps are distinguished from presentation/UI gaps;
- no runtime mutation has occurred;
- a bounded recommendation is returned to the programme coordinator.

After that, the programme coordinator decides whether to:

- extend OST-LIFECOURSE into a specific design slice;
- activate independent review;
- route a finding to OST-CLINICAL / OST-PRODUCT / OST-SAFETY;
- or preserve the existing mechanism unchanged.
