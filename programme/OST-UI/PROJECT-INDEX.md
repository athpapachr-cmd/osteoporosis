# PROJECT-INDEX.md — OST-UI / Osteoporosis Product Reconstruction

> **STATUS:** R1–R4 COMPLETE; SINGLE SYNTHESIS AND PROTOTYPE 1 CONTRACT RECORDED; RUNTIME NOT STARTED. The live workstream checkpoint is `CURRENT.md`.
> **Workstream:** `OST-UI`.
> **Scope:** Osteoporosis Module product reconstruction review only.
> **Authority model:** parallel sidecar workstream; the six root canonicals remain authoritative for repo-wide state.
> **Runtime mutation:** NOT AUTHORIZED.
> **UI redesign / prototype implementation:** NOT AUTHORIZED.

---

## 1. Mission

OST-UI exists to determine, from current product/runtime evidence, what the Osteoporosis Module actually is today and what product reconstruction is justified.

This phase is **not** a redesign exercise and does not assume either incrementalism or a total rebuild.

The programme must answer:

1. what the real product semantics are;
2. how closely the current product/runtime represents those semantics;
3. which structural/product/UI problems are observed rather than assumed;
4. what should survive;
5. what should be removed, relocated, adapted or rebuilt;
6. what interaction model is justified at the point of care;
7. which two vertical prototypes should later test the target product model;
8. which current mechanisms can safely be reused.

---

## 2. Governing product constitution

The Osteoporosis Module is:

> **the longitudinal clinical operating environment for a patient's osteoporosis care trajectory.**

The unit of the product is:

> **the patient and the continuous clinical care trajectory over time.**

The longitudinal state may include, where clinically supported by the authoritative owner:

- treatment epochs;
- medication starts, stops and switches;
- denosumab timing and delayed/missed treatment;
- fractures and fragility interpretation;
- DXA;
- laboratory results and bone-turnover markers;
- renal-function constraints;
- adverse events;
- clinical decisions;
- monitoring plans;
- unresolved questions;
- future obligations;
- outcomes.

Clinical pathways, forms, calculators, decision-support panels and AI interactions are **contextual projections of longitudinal state**. They are not independently the product.

Core longitudinal rule:

```text
PAST CLINICAL TRUTH
→ changes CURRENT INTERPRETATION
→ may create FUTURE OBLIGATIONS
→ next encounter resumes from accumulated patient state
```

A major interaction should help at least one of:

- understand the patient's trajectory;
- understand current state;
- act safely on current state;
- capture a clinically meaningful event or decision;
- manage a future obligation;
- inspect evidence/reasoning behind an interpretation.

---

## 3. Cockpit / module boundary

### Shared / global concerns

Candidate shared concerns include patient identity, encounter context, shared capture, transcripts, generic files/documents, navigation shell, reusable UI primitives, evidence/provenance primitives, global history where justified and cross-module routing where justified.

### Osteoporosis-specific concerns

Candidate module concerns include osteoporosis treatment state, fracture/fragility interpretation, osteoporosis risk state, treatment sequencing, bone-turnover monitoring, osteoporosis-specific future obligations, pathways and evidence interpretation.

Hard rule:

```text
ONE PATIENT TRUTH
→ MULTIPLE DOMAIN INTERPRETATIONS

ONE SEMANTIC CLINICAL STATE
→ MULTIPLE INTERFACES
```

OST-UI may identify an ownership or semantic gap. It must not repair clinical meaning in presentation code.

---

## 4. Governance boundary

OST-UI is an independent parallel review workstream.

It has **no authority** to mutate:

- root `CURRENT_OPERATIONAL.md`;
- root `SLICE_PLAN_CURRENT.md`;
- `TODO.md`;
- `CLINICAL_EXCELLENCE_PLAN.md`;
- root clinical runtime/schema/database code;
- OST-LIFECOURSE semantics;
- fracture/fragility clinical semantics;
- treatment clinical contracts;
- guideline content;
- PR-1/PR-2 transcript ownership;
- unrelated workstreams.

The only durable OST-UI control-plane files in Phase 1 are:

```text
programme/OST-UI/PROJECT-INDEX.md
programme/OST-UI/CURRENT.md
```

Review outputs are created only when the corresponding review is actually executed.

No separate global registry, new root canonical, review-of-review lane or prototype file is created by this bootstrap.

---

## 5. Review discipline

Phase 1 contains exactly:

```text
R1
→ R2
→ R3
→ R4
→ one synthesis
```

Each review owns a different decision question and must reuse prior inventory/evidence instead of re-performing it under another name.

Every finding must distinguish evidence status where relevant:

```text
OBSERVED
INFERRED
UNVERIFIED
```

A new review lane requires one of:

- an unresolved contradiction;
- an explicit acceptance gate;
- a genuinely different decision question.

“Would be good to check” is not sufficient.

---

# R1 — Current Product vs Product Constitution Audit

## Decision question

> **What is the current Osteoporosis product in practice, and how closely does it align with the longitudinal product constitution?**

## Required inputs

Fresh-read current `main` and inspect only the current product/runtime surfaces needed for the inventory, including:

- `static/cockpit/`;
- `static/baseline-audit/`;
- current Osteoporosis navigation/sidebar/entry points;
- patient/context handling;
- `clinical_data.py` / `clinical_data_ext.py` persistence boundaries;
- current G-1/G-2/G-3/G-4 guidance/summary mechanisms;
- current pages, cards, tools and panels;
- current state persistence/history representation;
- relevant current tests where they establish behavior;
- current global-versus-module ownership seams.

Use historical/canonical material only to explain current mechanisms, not to substitute for runtime inspection.

## Must answer

Inventory whether the current product behaves primarily as a collection of pages, document library, isolated tools, form system, decision workspace, longitudinal clinical environment or a mixture.

Inspect specifically:

- global vs Osteoporosis navigation;
- duplicate navigation;
- module contamination;
- information architecture;
- patient-first vs page-first behavior;
- presence/absence of longitudinal trajectory;
- whether history changes current state;
- representation of future obligations;
- durable vs ephemeral events;
- hidden useful capabilities;
- duplicate knowledge;
- dead/low-value surfaces;
- information density;
- fragmented state;
- reusable mechanisms/components.

## Out of scope

- target visual design;
- clinical-rule correction;
- deciding reconstruction scope in advance;
- prototype design/code.

## Output

`programme/OST-UI/R1-CURRENT-PRODUCT-CONSTITUTION-AUDIT.md`

R1 must finish the current-state inventory before proposing any redesign implication.

---

# R2 — Longitudinal Clinical Trajectory Review

## Decision question

> **Can the product represent and preserve the patient's osteoporosis story across time?**

## Required inputs

- completed R1 inventory;
- protected patient/encounter/lab persistence model;
- current longitudinal projection/summary mechanisms;
- current treatment/administration history representation;
- current guidance context derivation;
- authoritative existing clinical-semantic contracts only where needed;
- fresh authoritative owner state for any clinical meaning not owned by OST-UI.

## Representative journeys

At minimum:

1. new patient entering osteoporosis care;
2. longitudinal denosumab treatment;
3. delayed/missed denosumab dose;
4. denosumab discontinuation/transition;
5. fracture during treatment;
6. new fragility fracture changing future management;
7. glucocorticoid-associated osteoporosis;
8. CKD/renal constraints;
9. monitoring across multiple visits;
10. possible treatment failure requiring reassessment.

## Must distinguish

```text
EVENT
STATE
DERIVED STATE
DECISION
PLAN
FUTURE OBLIGATION
OUTCOME
UNCERTAINTY
```

For each journey determine what is durable patient truth, what is transient UI state, how treatment epochs/events/time relationships are represented, whether current state is derivable from history, whether obligations are created and visible later, how unknown/missing data behave and whether the next encounter resumes from prior truth.

## Out of scope

- inventing fracture/fragility meaning;
- inventing treatment-sequencing rules;
- UI redesign;
- implementation.

Where clinical meaning is missing or contested, record a dependency/referral rather than filling the gap.

## Output

`programme/OST-UI/R2-LONGITUDINAL-CLINICAL-TRAJECTORY-REVIEW.md`

---

# R3 — Point-of-Care Clinical Interaction Review

## Decision question

> **When a clinician must decide now, which interaction model reduces cognitive load and exposes only information that can change the decision?**

## Required inputs

- R1 current-surface inventory;
- R2 established longitudinal-state and missing-state boundaries;
- current guided-visit and evidence UI;
- representative current workflows;
- Physio interaction evidence only as a reusable pattern source;
- authoritative clinical rules from their actual owners.

## Representative journeys

At minimum:

1. currently receiving denosumab;
2. delayed denosumab dose;
3. discontinuation planning;
4. new fragility fracture;
5. fracture while already treated;
6. monitoring / possible treatment failure;
7. CKD constraint;
8. glucocorticoid-associated osteoporosis.

## Must inspect

- starting-point clarity;
- recognition of prior state;
- irrelevant information;
- missing decision-changing information;
- input order and branching;
- cognitive load and context switching;
- need to remember information outside the UI;
- current-state visibility;
- next-action visibility;
- dynamic reaction to inputs;
- upstream editability;
- downstream consistency;
- final output/rationale clarity;
- evidence access without clutter.

Candidate principles are hypotheses to test, not conclusions:

- problem-first entry;
- state-aware starting point;
- progressive disclosure;
- dynamic dependent choices;
- live decision output;
- persistent current-state summary;
- explicit missing-information state;
- editable upstream inputs;
- expandable evidence;
- visible uncertainty;
- visible future actions/monitoring.

The review must also determine how the UI should visibly distinguish:

```text
PATIENT DATA
HISTORICAL EVENT
CURRENT DERIVED STATE
GUIDELINE POSITION
SYSTEM RECOMMENDATION
ALTERNATIVE OPTION
UNCERTAINTY
WARNING
CONTRAINDICATION
HARD STOP
FUTURE OBLIGATION
EVIDENCE / PROVENANCE
```

## Out of scope

- defining new medical rules;
- treating Physio as the Osteoporosis product model;
- styling specification;
- prototype implementation.

## Output

`programme/OST-UI/R3-POINT-OF-CARE-INTERACTION-REVIEW.md`

---

# R4 — Shared Core / Module Architecture / Reuse Review

## Decision question

> **Which implementation architecture can support the longitudinal product model, dynamic clinical workflows and future Live Copilot without duplicated patient truth or duplicated clinical logic?**

## Required inputs

- R1 current implementation inventory;
- R2 longitudinal requirements/gaps;
- R3 interaction requirements;
- current Cockpit shell;
- protected patient/context/persistence model;
- current browser longitudinal/guidance/evidence mechanisms;
- active PR-1 boundaries as read-only architecture context;
- Physio projection/interaction mechanisms as reuse evidence;
- current shared/module ownership boundaries.

## Must inspect

- shared Cockpit shell;
- patient/context model;
- state/persistence architecture;
- workflow/state-machine suitability;
- event handling;
- treatment-history representation;
- future-obligation representation;
- evidence/rule separation;
- UI component reuse;
- output rendering;
- navigation/deep links;
- global vs module ownership;
- reusable Physio mechanisms;
- risks of copying Physio too literally;
- cross-module reuse;
- clinical logic embedded in presentation code;
- duplicate patient models;
- future transcript/live-capture compatibility.

Classify major mechanisms:

```text
KEEP
KEEP + ADAPT
MOVE TO SHARED CORE
KEEP MODULE-LOCAL
REPLACE
RETIRE
UNKNOWN UNTIL PROTOTYPE
```

## Out of scope

- runtime refactor;
- schema migration;
- implementation;
- clinical semantic changes.

## Output

`programme/OST-UI/R4-SHARED-CORE-MODULE-ARCHITECTURE-REUSE.md`

---

# Synthesis — Osteoporosis Product Reconstruction Decision

The synthesis occurs only after R1-R4 are complete.

It is **not** a fifth independent review and must not repeat all evidence gathering.

It reconciles findings, contradictions, confirmed unknowns, dependencies and preservation opportunities into:

`programme/OST-UI/OST-PRODUCT-RECONSTRUCTION-DECISION.md`

Required sections:

A. Product semantics confirmation  
B. Current-state diagnosis  
C. Preservation map  
D. Removal / relocation map  
E. Target product model  
F. Target interaction model  
G. Reconstruction scope  
H. Prototype recommendation

Section G must decide from evidence whether the justified scope is incremental improvement, substantial restructuring, near-total UI reconstruction, deeper product/data-model reconstruction or a combination.

No answer is preselected.

Section H selects exactly two initial vertical slices. Preferred candidates unless the evidence strongly contradicts them:

1. **Denosumab longitudinal management**
2. **New fragility fracture / treatment reassessment**

These are prototypes of the product model, not forms.

---

## 6. Prototype gate — future phase only

No prototype is implemented in Phase 1.

A later prototype should prove that the clinician can:

- identify relevant longitudinal context;
- understand current state/problem;
- see missing information;
- see how important inputs change state;
- identify next action;
- inspect rationale/evidence without clutter;
- revise upstream information without restart;
- see downstream consequences;
- understand future monitoring/obligations;
- avoid irrelevant information;
- resume later from preserved patient state.

Target experience:

> **the system knows where this patient is in their osteoporosis journey**

not:

> **I opened another smart form.**

---

## 7. Full reconstruction gate

A full UI/product rebuild is not justified merely because the current interface is unattractive, the sidebar is chaotic or a cleaner architecture exists.

A large rebuild may proceed only if:

1. R1-R4 demonstrate structural/product need;
2. synthesis establishes a coherent target product model;
3. synthesis establishes a target interaction model;
4. the two later prototypes demonstrate clear clinical-workflow benefit;
5. longitudinal state can be represented safely;
6. unresolved clinical-semantic dependencies do not invalidate the workflows;
7. reuse opportunities have been explicitly considered.

---

## 8. Dependency / ownership register

### DEP-01 — Root operational writer / PR-1

Current root `CURRENT_OPERATIONAL.md` owns the active PR-1 transcript implementation lifecycle. OST-UI is read-only with respect to that scope and does not modify root operational state.

### DEP-02 — Fracture / fragility semantics

Fracture/fragility clinical meaning is outside OST-UI authority. The task contract identifies the parallel lifecycle/clinical semantic owner as the dependency boundary. At each relevant review, fresh-resolve the current authoritative owner/artifact before using clinical semantics. If no authoritative answer exists, mark the point unresolved and issue a bounded referral; do not encode an answer in UI review.

Draft PR #121 currently contains parallel `programme/OST-LIFECOURSE/` and `programme/OST-CLINICAL/` work but is **unmerged** and therefore is coexistence context, not current-main authority.

### DEP-03 — Treatment / guideline semantics

Existing reviewed osteoporosis guidance/evidence contracts remain owned by their clinical/evidence owners. OST-UI may inspect how they are projected but may not change their medical meaning.

### DEP-04 — Shared patient/Core ownership

R4 may identify a need to move or generalize patient/context/state mechanisms, but changes to shared Core ownership require a later bounded owner decision/referral.

### DEP-05 — Global programme registry

Draft PR #121 also proposes a broader `programme/` registry. OST-UI does not modify or duplicate that global registry during this bootstrap. If it later becomes merged authority, reconcile navigation only; do not create a competing registry.

### DEP-06 — Live Clinical Copilot

Future Live Copilot is an architecture compatibility constraint only. It must use the same semantic patient truth/state/evidence/obligation layer rather than become a parallel clinical brain. No Live Copilot implementation is authorized here.

---

## 9. Execution order and anti-overlap rule

Exact Phase-1 order:

```text
R1 current reality / inventory
→ R2 longitudinal truth across time
→ R3 point-of-care interaction over that truth
→ R4 architecture/reuse required to support R1-R3
→ one synthesis
→ STOP / Product Owner decision
```

Why this order:

- R2 must not invent a product model before R1 establishes current reality.
- R3 should test interaction against the real longitudinal requirements established by R2.
- R4 should classify architecture against demonstrated product/interaction requirements, not aesthetic preference.
- synthesis reconciles; it does not collect a fifth evidence set.

No review automatically opens its successor. The coordinator checkpoints completion, confirms dependencies, and then starts only the next listed review.

---

## 10. Current-phase stop rule

The original bootstrap stop rule below governed the first R1 dispatch. It is superseded by the accepted R1–R4 artifacts, the single synthesis and the current Product Owner instruction. For present authority, use `CURRENT.md` and the bounded prototype contract. The continuing boundaries are:

- do not start prototype runtime outside its bounded contract and writer claim;
- do not merge/deploy;
- do not mutate root canonicals;
- do not mutate clinical semantics;
- do not create extra reviews;
- do not treat draft PR #121 as merged authority;
- do not let OST-UI take the root writer lock.

The authorized documentation sequence is the single synthesis followed by Prototype 1's bounded implementation contract. Neither document alone is a merge, deploy, clinical-rule or identifiable-transcript-use authority.
