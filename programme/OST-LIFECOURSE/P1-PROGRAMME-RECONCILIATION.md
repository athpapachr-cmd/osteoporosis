# OST-LIFECOURSE-P1 — Programme reconciliation result

> **STATUS:** P1 SEMANTIC-OWNERSHIP CORRECTION AFTER INDEPENDENT BLOCK / DELTA REVIEW READY.
> **Date:** 2026-09-27.
> **Scope:** documentation/design reconciliation only.
> **Root writer:** unchanged — OST-CAPTURE / PR-1.
> **Q1–Q12:** PARKED / UNRESOLVED.
> **S1:** separately routed; no runtime correction in P1.

## Accepted ownership contract

### Fracture / fragility
- `fracture_history.events[]` remains the structured factual owner.
- `low_trauma` is the explicit event-level fragility/low-trauma assertion.
- `risk_context.prior_fragility_fracture` is compatibility/summary, not independent factual authority.
- encounter archetype is visit intent/context, not fracture-trauma authority.
- stale `event.fragility` is non-canonical where current structured events use `low_trauma`.

### DXA
- encounter `step3.dxa` is a protected encounter-scoped factual snapshot.
- `longitudinal_review.dxa_history` contains manually authored historical DXA factual records within encounter payload state. A row may be the only recorded fact for its historical scan, or may duplicate an observation also recorded in `step3.dxa`.
- overlapping representations of the same scan require provenance-aware duplicate/conflict reconciliation. A manual row without a peer protected encounter observation remains factual recorded history with its existing provenance limitations.
- deriving longitudinal history from protected dated measurements is a possible future consolidation direction, not grounds to demote current manual facts without separate reconciliation and design.

### Labs
- `step3.labs` = encounter-scoped captured/reviewed snapshot.
- `clinical_lab_snapshots` = patient-level dated longitudinal laboratory owner.
- same observation should be linked by provenance/source encounter rather than treated as two facts.

### Step-4 treatment / administrations / tasks
- treatment episodes remain encounter snapshots; LGP derives longitudinal active-treatment state.
- scheduled/planned administration != actual administration != derived expected due.
- actual chronology derives only from reliable `actual_date` facts.
- task UUID identifies a row in the stored Step-4 representation; current executable cross-encounter unresolved-task reconciliation does not use that UUID for continuity.
- current reconciliation uses the semantic tuple `type | due_date | timeframe_text`. This is the current continuity mechanism, not a stable longitudinal obligation identity; changing due date or timeframe may leave the prior planned tuple unresolved while the changed task appears as a new tuple.

### LongitudinalGuidanceProjectionV1
- remains rebuildable read-only derived projection over completed/amended factual encounters.
- carries no patient-fact write authority.

### EncounterContext
- remains derived current-visit context.
- G-0 frozen contract owns intended semantics.
- G-1 builder owns only current executable truth.
- G-2 evidence context is a bounded rule-evaluation adapter, not a second patient-context authority.

### GuidanceRule / TherapyMilestone
- semantic authority chain is reviewed evidence → G-2 registries/manifests → deterministic executor → ephemeral guidance output.
- runtime JS executes the reviewed contract; it does not acquire independent clinical-authoring authority.

## Accepted duplicate/drift findings

1. fracture/fragility semantic collision;
2. DXA dual history;
3. laboratory dual-write lifecycle;
4. task identity drift;
5. EncounterContext contract/runtime drift;
6. duplicate context derivation risk;
7. declarative-rule/runtime duplication;
8. multiple longitudinal scanners.

## Accepted future demotion candidates

These are not deletion instructions:
- `risk_context.prior_fragility_fracture` → compatibility/derived summary;
- legacy `event.fragility` → non-canonical relative to `low_trauma`;
- `longitudinal_review.dxa_history` → possible later consolidation/demotion only after unique historical facts, provenance and duplicates have been reconciled in separately bounded design; no migration or final authority decided here;
- semantic tuple task continuity → current executable mechanism with unstable obligation semantics; any future replacement/consolidation remains an unresolved bounded design question;
- `osteoporosis_evidence_context_v1` → execution adapter, not patient-state truth;
- G-3 direct cross-encounter reconstruction → presentation logic until reconciled projection ownership is available;
- duplicated G-2 JS predicates/constants → executable implementation, not semantic authority.

## Safety/data-integrity disposition

- S1 remains separate from P1; it is merged on current `main` and its auto-deploy is live, with production smoke not run. P1 does not reopen or review S1.
- DI-2 DXA duplicate truth accepted.
- DI-3 lab snapshot orphan/staleness risk accepted.
- DI-4 task continuity/disposition gap accepted.
- DI-5 EncounterContext schema identity mismatch accepted.
- DI-6 rule contract/runtime drift risk accepted.

None of these findings authorises implementation from this artifact.

## Cross-workstream boundaries

- OST-CLINICAL owns S1 correction and osteoporosis evidence semantics.
- OST-CAPTURE / PR-1 remains root active writer.
- OST-PRODUCT may later present trajectory/timeline but must consume existing truth owners.
- future evidence/versioning remains separate.
- future Core promotion / cross-module event seam remains parked.

## Programme next action

Request one independent READ-ONLY delta review of the corrected P1 map covering:
- seven-domain ownership map;
- S1 separation;
- no smuggled new store/engine;
- authoritative vs derived boundaries;
- EncounterContext and rule/runtime conclusions against source;
- Q1–Q12 remaining parked.

The review returns PASS or BLOCK and STOP.

Do not start implementation, P2 architecture, schema work or migration automatically.
