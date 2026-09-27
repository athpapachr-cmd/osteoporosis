# OST-LIFECOURSE-P1 — Programme reconciliation result

> **STATUS:** P1 SEMANTIC-OWNERSHIP RECONCILIATION ACCEPTED / INDEPENDENT REVIEW READY.
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
- encounter `step3.dxa` is the protected factual encounter record.
- longitudinal history should be derived from protected dated measurements.
- `longitudinal_review.dxa_history` is a duplicate/manual history candidate for later demotion, not peer truth.

### Labs
- `step3.labs` = encounter-scoped captured/reviewed snapshot.
- `clinical_lab_snapshots` = patient-level dated longitudinal laboratory owner.
- same observation should be linked by provenance/source encounter rather than treated as two facts.

### Step-4 treatment / administrations / tasks
- treatment episodes remain encounter snapshots; LGP derives longitudinal active-treatment state.
- scheduled/planned administration != actual administration != derived expected due.
- actual chronology derives only from reliable `actual_date` facts.
- task UUID is the strongest current continuity identity; semantic tuple matching is fallback/legacy only.

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
- `longitudinal_review.dxa_history` → legacy/backfill-only candidate after separate migration/backfill decision;
- semantic tuple task identity → fallback only when stable ID unavailable;
- `osteoporosis_evidence_context_v1` → execution adapter, not patient-state truth;
- G-3 direct cross-encounter reconstruction → presentation logic until reconciled projection ownership is available;
- duplicated G-2 JS predicates/constants → executable implementation, not semantic authority.

## Safety/data-integrity disposition

- S1 remains separately routed and unimplemented.
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

Open exactly one fresh independent READ-ONLY P1 review covering:
- seven-domain ownership map;
- S1 separation;
- no smuggled new store/engine;
- authoritative vs derived boundaries;
- EncounterContext and rule/runtime conclusions against source;
- Q1–Q12 remaining parked.

The review returns PASS or BLOCK and STOP.

Do not start implementation, P2 architecture, schema work or migration automatically.