# OST-LIFECOURSE CURRENT

> **STATUS:** ACTIVATED / PHASE 0 READ-ONLY PRODUCT-ARCHITECTURE INVENTORY.
> **Workstream:** OST-LIFECOURSE — longitudinal clinical care trajectory.
> **Branch:** `docs/ost-programme-lifecourse-bootstrap-2026-09-27`.
> **Base main:** `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`.
> **Root writer lock:** unchanged. `CURRENT_OPERATIONAL.md` remains owned by the active PR-1 Heidi-first transcript-capture lifecycle.
> **Overlap:** none. This workstream is documentation/design only and must not mutate PR-1 runtime, active encounter schemas, database/persistence, production UI or existing clinical guidance rules.

## Product-owner authority

On 2026-09-27 the Product Owner approved the Product Constitution v0.2 direction and explicitly instructed the programme coordinator to proceed to the next organisational step.

The approved direction includes:

- one canonical factual patient reality across modules;
- longitudinal care trajectory spanning past, present and intended future;
- multiple provenance-preserving module interpretations of shared facts/events;
- one actual timeline, with planned/possible future states kept non-authoritative until they occur;
- treatment/intervention epochs, goals/target states, decisions, outcomes and advisory obligations;
- obligations require disposition, not obedience;
- general → patient-aware → decision-aware clinical tools using one clinical/evidence engine;
- provider-agnostic capture and future live-copilot compatibility;
- future patient communication through a provider-agnostic Care Communication Layer;
- cohort intelligence for audit/learning/research without automatic conversion into clinical rules.

## Phase 0 hard scope

Execute **read-only inventory and conceptual reconciliation only**:

1. map existing longitudinal/state concepts in current canonicals/runtime to the Product Constitution;
2. identify reusable owners/mechanisms before proposing any new path;
3. identify gaps, duplicate responsibilities and incompatible semantics;
4. distinguish shared patient fact/event ownership from module-specific interpretation;
5. produce a bounded inventory handback for programme-coordinator synthesis.

## Explicitly out of scope

- runtime/schema/database mutation;
- timeline UI design or implementation;
- cross-module event-bus implementation;
- Quick Tool implementation;
- Care Communication/Zadarma implementation;
- patient app implementation;
- PR-1/PR-2 mutation;
- Practice Review implementation;
- Live Copilot implementation;
- resolving the 12 parked downstream design questions;
- changing the six active canonical authorities.

## Exact next action

Fresh-bootstrap current `main`, then perform the OST-LIFECOURSE Phase-0 reuse-before-new-path inventory against:

- existing encounter/patient persistence;
- longitudinal guidance projection / EncounterContext;
- treatment episodes and administrations;
- VisitPlan / GuidanceRule / GuidedCardState;
- follow-up tasks and current due-state mechanics;
- Clinical Learning / Practice Review seams where relevant;
- cross-module/global Cockpit ownership already present.

Produce one read-only design inventory artifact and STOP for programme-coordinator reconciliation.

## State semantics

```text
PRODUCT CONSTITUTION: CONCEPTUALLY APPROVED
OST-LIFECOURSE: ACTIVATED
PHASE 0 INVENTORY: NOT YET EXECUTED
RUNTIME CHANGE: NO
SCHEMA CHANGE: NO
ROOT WRITER TRANSFER: NO
12 PARKED QUESTIONS: STILL PARKED
```


## Organisational bootstrap checkpoint

Created on this branch:

- `programme/PRODUCT-CONSTITUTION-V0.2.md`
- `programme/WORKSTREAM-REGISTRY.md`
- `programme/OST-LIFECOURSE/PROJECT-INDEX.md`
- `programme/OST-LIFECOURSE/CURRENT.md`

Checkpoint commits:

- `37487c4d6d92eb0a6bcd4b4df604563c5f3fe8a2` — activate OST-LIFECOURSE Phase 0 CURRENT;
- `5070e0a8cf4bc34d8bdf12f78c8e40a6c9f3a58c` — record Product Constitution v0.2;
- `1892aa0622301b341c53faff7b2ad0ae8ae5a64c` — add programme workstream registry;
- `14592904baed5054b49d7d70141d2828153c0639` — define Phase-0 project charter.

The branch remains documentation/design only.

### Exact next material action after this checkpoint

Open a **draft organisational PR** for review/visibility only. Do not merge it yet and do not execute the Phase-0 inventory until the programme coordinator has verified the PR identity and canonical-impact declaration.


## Draft PR checkpoint

```text
PR: #121
URL: https://github.com/athpapachr-cmd/osteoporosis/pull/121
STATE: OPEN / DRAFT
BASE: main
BASE SHA AT CREATION: 2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b
HEAD AT CREATION: aad50e75119ac566c4d20559d672b398d3f8117c
RELEASE AFFECTING: NO
ROOT CURRENT MUTATION: NO
RUNTIME MUTATION: NO
```

The PR is a governance/design checkpoint only. Do not merge or begin implementation from it automatically.

### Exact next action

Verify the updated PR head and governance/Canonical Impact checks. Then the programme coordinator may issue the bounded OST-LIFECOURSE Phase-0 **read-only inventory** task. That inventory must STOP with a handback; it must not mutate runtime or resolve the 12 parked questions.
