# OST-LIFECOURSE Phase 0 — Read-only inventory brief

> **TASK TYPE:** fresh separate workstream-coordinator design inventory.
> **MODE:** READ-ONLY / REUSE-BEFORE-NEW-PATH / NO RUNTIME MUTATION.
> **Programme branch at issuance:** `docs/ost-programme-lifecourse-bootstrap-2026-09-27`.
> **Programme PR:** #121.
> **Product Constitution:** `programme/PRODUCT-CONSTITUTION-V0.2.md`.
> **Workstream charter:** `programme/OST-LIFECOURSE/PROJECT-INDEX.md`.
> **Workstream state:** `programme/OST-LIFECOURSE/CURRENT.md`.

## Role

You are the fresh **OST-LIFECOURSE workstream coordinator** for Phase 0.

You are not:

- the root PR-1 implementation writer;
- the programme-level coordinator;
- a runtime/schema implementation author;
- an independent commercial-product reviewer;
- an independent clinical-evidence reviewer;
- a merge/deploy executor.

Your job is to inventory current mechanisms and return a bounded handback.

## Fresh bootstrap

Fresh-verify `athpapachr-cmd/osteoporosis/main`.

Then read the six active canonicals in the order required by `AGENTS.md`:

1. `AGENTS.md`
2. `TODO.md`
3. `CLINICAL_EXCELLENCE_PLAN.md`
4. `SLICE_PLAN_CURRENT.md`
5. `CURRENT_OPERATIONAL.md`
6. `osteoporosis-change-log.md`

Do not assume the programme branch or prompt-issued main SHA is still current.

After the six canonicals, consume fully from PR #121 / branch `docs/ost-programme-lifecourse-bootstrap-2026-09-27`:

- `programme/PRODUCT-CONSTITUTION-V0.2.md`
- `programme/WORKSTREAM-REGISTRY.md`
- `programme/OST-LIFECOURSE/PROJECT-INDEX.md`
- `programme/OST-LIFECOURSE/CURRENT.md`

The six canonicals remain operational authority. The programme documents define the bounded design question; they do not override active runtime locks.

## Exact objective

Perform a **read-only reuse/gap inventory** answering:

> Which responsibilities required by the Product Constitution are already owned by current Clinical Excellence Core / Module-01 mechanisms, which are fragmented or duplicated, and which are genuine gaps?

Do **not** design the final replacement architecture yet.

## Required inventory domains

Inspect with exact evidence:

1. patient identity and protected persistence;
2. encounter history and current encounter ownership;
3. fracture events and other longitudinal facts;
4. treatment episodes;
5. actual versus scheduled administrations;
6. DXA/VFA/laboratory longitudinal representation;
7. `LongitudinalGuidanceProjectionV1`;
8. `EncounterContextV1`;
9. `VisitPlanV1`;
10. `GuidanceRuleV1`;
11. `GuidedCardStateV1`;
12. `TherapyMilestoneProfileV1`;
13. follow-up task / due-state / unresolved-item mechanisms;
14. clinician decision, rationale and patient-preference capture;
15. uncertainty/conflict/provenance semantics;
16. Practice Review / Clinical Learning consumers of encounter truth;
17. shared Core versus Osteoporosis-specific ownership;
18. current Cockpit/global-module navigation boundaries;
19. any existing mechanism relevant to future cross-module facts or communication obligations.

## Required classification

For every material Product Constitution responsibility classify it as one of:

```text
REUSE AS-IS
EXTEND EXISTING OWNER
CONSOLIDATE DUPLICATE / FRAGMENTED OWNERS
GENUINE GAP
PRESENTATION / UX GAP ONLY
INSUFFICIENT EVIDENCE
```

Each finding must include:

- exact current owner/mechanism;
- file/schema/runtime path or canonical evidence;
- whether it is implemented vs design-only;
- safety/data-integrity implications;
- dependency on another workstream;
- whether a parked question would eventually need activation.

## Reuse-before-new-path rule

Do not propose a new:

- patient state store;
- event store/event bus;
- obligation engine;
- treatment-epoch store;
- timeline database;
- cross-module broker;
- evidence engine;
- AI semantic stack;

unless the inventory demonstrates that the relevant current owner cannot safely satisfy the responsibility.

A gap in the current UI is not automatically a gap in underlying architecture.

## Synthetic pressure tests

Use, at minimum, these cases to test current ownership:

1. next denosumab dose due, patient refuses;
2. dental extraction changes treatment timing/context;
3. fracture recorded in another future module should become visible to Osteoporosis without automatic fragility diagnosis;
4. different DXA-site trajectories after treatment;
5. patient preference explains divergence from clinician recommendation/guideline pathway;
6. overlapping pharmacologic/exercise/vitamin-D/falls interventions;
7. general vs patient-aware vs decision-aware zoledronate Quick Tool;
8. historical decision reviewed against evidence-at-the-time versus current evidence.

Do not implement these cases.

## Twelve parked questions

The 12 questions in Product Constitution §29 remain **PARKED**.

You may state that a finding depends on one of them. You must not solve it during Phase 0.

## Required output

Produce one concise but evidence-rich handback with:

```text
REVIEW TARGET / SOURCE IDENTITY
CURRENT REUSABLE MECHANISMS
FRAGMENTED / DUPLICATED RESPONSIBILITIES
GENUINE GAPS
UX-ONLY GAPS
SAFETY / DATA-INTEGRITY FINDINGS
CROSS-WORKSTREAM DEPENDENCIES
PARKED-QUESTION DEPENDENCIES
RECOMMENDED NEXT BOUNDED DESIGN SLICE
STOP
```

Do not rank political/irrelevant matters; remain strictly within product architecture.

## Prohibited actions

- no code changes;
- no schema/database changes;
- no canonical mutation;
- no branch/PR creation for implementation;
- no PR-1 changes;
- no Product Constitution rewrite;
- no UI redesign;
- no Live Copilot implementation;
- no patient-app or Zadarma integration;
- no independent-review referral;
- no merge/deploy/smoke.

Return the Phase-0 handback to the programme coordinator and STOP.
