# M01 G-0 Dynamic Guidance Design Review v1

> **Review type:** exact pre-runtime architecture / runtime-seam review.
> **Module:** Osteoporosis Module 01.
> **Branch:** `design/module01-dynamic-guided-visit-replan-2026-08-30`.
> **Verified remote main at bootstrap:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Parent tested C1 ancestry:** `a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871`.
> **Classification:** `DESIGN-COMPLETE` with explicit downstream implementation/evidence gates.
> **Runtime implementation authorized by this review:** NO.

---

## 1. Review question

Can the current Osteoporosis runtime safely support the newly approved product model:

```text
longitudinal patient state
+ current visit intent
+ event / treatment / due / unresolved triggers
→ dynamic clinician-facing Visit Plan
→ Heidi-assisted provisional population
→ clinician verification
→ explicit decision / Close
```

without:

- replacing the current protected encounter store;
- creating one hard-coded screen for every treatment visit number;
- inventing treatment milestone rules;
- conflating Clinical Guidance with audit/performance coaching?

**Answer: YES at architecture/design level.**

---

## 2. Product-purpose review — PASS

The canonicals now consistently define the primary product outcome as:

1. improve the current clinical encounter while it is happening;
2. reduce duplicate/manual capture;
3. review whether what was said, reasoned and decided was appropriate;
4. improve future clinician performance longitudinally.

Audit remains an underlying measurement mechanism rather than the clinician-facing product identity.

The design explicitly separates:

```text
Clinical Guidance
!= Transcript-assisted Capture
!= Audit / Measurement
!= Clinical Practice Review
```

**Result: PASS.**

---

## 3. Current encounter-archetype seam — PASS, but insufficient alone

Current runtime already persists `encounter_archetype` and exposes these coarse visit intents:

```text
initial_assessment_new_or_uncertain_diagnosis
initial_assessment_known_osteoporosis_or_osteopenia
routine_followup_stable
treatment_start
treatment_continuation_or_due_monitoring
treatment_change_or_transition
post_fragility_fracture
fracture_on_treatment
adverse_effect_or_intolerance
treatment_completion_or_consolidation
other
```

`adaptive-applicability.js` already uses an archetype→domain map with:

```text
applicable
uncertain
not_applicable
```

This proves an existing presentation seam, but the current map is too coarse for the approved product because it lacks:

- `why now`;
- new-event override;
- prior unresolved items;
- agent-specific relevance;
- due/overdue state;
- treatment milestones;
- exact longitudinal treatment context.

The G-0 design extends rather than discards this foundation.

**Result: PASS.**

---

## 4. Core current-encounter data paths — PASS

Verified persisted/current paths required by the first guidance engine include:

```text
patient_relationship_status
encounter_archetype
fracture_history.events[]
step3.dxa
step3.vfa
step3.secondary
step3.labs
step3.function
step4.treatment_episodes[]
step4.administrations[]
step4.tasks[]
step4.decision
step4.transition
step4.close
applicability_review
```

Step 4 administration events currently preserve:

```text
agent
scheduled_date
actual_date
status
next_due_date
```

Treatment episodes preserve agent/status/start/end/duration/adherence/tolerance/fracture-on-episode/response context. This is sufficient to support a first deterministic context resolver, subject to the longitudinal projection rules below.

**Result: PASS.**

---

## 5. Longitudinal patient-state seam — PASS after G-0 correction

### Finding

There is currently **no separate patient-level canonical treatment timeline table**.

Clinical truth is stored primarily as per-encounter `payload_json`, with laboratory snapshots in a separate patient-level table.

### Critical verification

The protected endpoint:

```text
GET /clinical/patient/{patient_id}/encounters
```

returns the full `EncounterRecord`, including full encounter `payload`, for every patient encounter.

Therefore G-1 can construct longitudinal guidance context from existing protected data without an immediate database migration.

### Required correction frozen during review

Added:

```text
schemas/longitudinal_guidance_projection_v1.yaml
```

This defines a **read-only derived projection** rather than a second source of truth.

Key safeguards:

- completed/amended prior encounters are historical sources;
- a blank later snapshot does not erase prior authoritative history;
- planned/scheduled administration does not count as an actual dose;
- administration count is computed only from reliable unique actual events;
- count and elapsed exposure remain separate;
- conflicts remain conflicts rather than being silently resolved by “latest wins”;
- provisional transcript candidates do not alter longitudinal authority until clinician acceptance;
- no patient-level treatment table/database migration is required by default for G-1.

**Result: PASS after design correction.**

---

## 6. Repeated treatment / Prolia modelling — PASS at capability level

The product-owner examples demonstrate that different repeated-treatment visits need different content, including early course, later milestones, long-duration review and event-triggered visits.

The architecture deliberately does **not** encode:

```text
prolia_visit_1
prolia_visit_2
...
prolia_visit_10
```

Instead it supports:

```text
coarse treatment-continuation visit intent
+ active agent
+ actual administration history
+ reliable administration count
+ elapsed exposure
+ due/overdue state
+ monitoring-due state
+ evidence/policy milestone rules
+ event/safety overrides
```

This handles the product need without form explosion.

Important limitation retained:

> No exact rule for the 4th, 8th, 10th or any other denosumab administration becomes active merely because the architecture can represent it.

Exact milestone content requires reviewed evidence or an explicitly approved clinic policy with provenance.

**Result: PASS at architecture level; clinical milestone content remains a later evidence gate.**

---

## 7. Rule-priority model — PASS

Frozen order:

```text
critical safety / urgent event
→ unresolved prior critical item
→ treatment/agent-specific requirement
→ evidence-defined milestone/due item
→ archetype base flow
→ contextual item
```

Consequences:

- a new fracture can override a routine administration flow;
- an overdue time-critical therapy state cannot be hidden by a “routine follow-up” default;
- one card can retain several `why now` reasons;
- generic `not_applicable` cannot suppress a higher-priority current trigger;
- clinician judgment/override remains distinct from rule authority.

**Result: PASS.**

---

## 8. Machine contract — PASS

Normative entrypoint:

```text
schemas/dynamic_guided_visit_contract_manifest_v1.yaml
```

Normative contracts:

```text
schemas/dynamic_guided_visit_v1.yaml
schemas/longitudinal_guidance_projection_v1.yaml
```

Frozen design objects:

```text
EncounterContextV1
LongitudinalGuidanceProjectionV1
ProjectionConflictV1
GuidanceRuleV1
VisitPlanV1
GuidedCardStateV1
TherapyMilestoneProfileV1
GuidanceExposureV1
```

**Result: PASS.**

---

## 9. Heidi / transcript compatibility — PASS

The corrected archived PR-1 v3 design already maps extraction against actual persisted runtime paths rather than YAML wording alone and preserves:

- composite candidates;
- negation;
- temporality;
- speaker/source;
- uncertainty;
- option discussed vs recommendation vs final decision;
- provider-neutral semantics;
- deterministic Module 01 target mapping;
- ephemeral raw transcript handling;
- no PR-1 authoritative writes.

G-0 changes the intended UX/sequence, not these semantic safety principles.

New direction:

```text
candidate extraction
→ deterministic mapping
→ provisional value inside destination clinical card
→ Accept / Edit / Reject
→ authoritative write only after clinician review
```

This avoids rebuilding manual data entry around a detached candidate list.

**Result: PASS.**

---

## 10. Baseline methodology — PASS after explicit revision

Old methodology is superseded because it would pilot a workflow already known to be operationally unacceptable.

New sequence:

```text
finalization integrity deployed/smoked
→ dynamic guidance
→ transcript extraction
→ inline review/population
→ 5 system-assisted pilot encounters
→ one refinement/freeze
→ 30 scored system-assisted encounters
→ baseline lock
→ interventions / re-measurement
```

During the 30-case cohort:

- stable Clinical Guidance remains active;
- transcript-assisted capture remains active;
- routine KPI/performance feedback remains hidden;
- routine clinician-facing Practice Review remains hidden by default;
- safety-critical feedback is allowed;
- guidance exposure is captured where reliable;
- cohort is labelled `system-assisted baseline`.

This is methodologically transparent and aligned with the actual product objective.

**Result: PASS.**

---

## 11. Guidance-exposure / clinician-improvement model — PASS with non-causal limitation

`GuidanceExposureV1` can preserve whether an item was surfaced and, where event timing is reliable, whether relevant content was already present before the cue or was resolved afterward.

Potential longitudinal interpretation:

```text
prompt-dependent correct execution
→ repeated supported performance
→ increasingly pre-prompt/spontaneous correct behavior
```

This is useful to the product goal of improving the clinician, but the contract explicitly forbids causal learning claims when event sequence is not reliable.

**Result: PASS.**

---

## 12. Storage migration requirement — NO for G-1

The design review does not identify a need for a database schema migration merely to start G-1.

Preferred G-1 boundary:

```text
protected historical encounter payloads
+ current encounter state
→ ephemeral LongitudinalGuidanceProjectionV1
→ EncounterContextV1
→ deterministic rule evaluation
→ VisitPlanV1
```

A future patient-level canonical treatment timeline may still become useful, but it requires separate evidence that the derived projection is insufficient. It must not be introduced pre-emptively.

---

## 13. Remaining deliberate non-decisions

G-0 does **not** freeze:

- exact denosumab 4th/8th/10th-dose guidance;
- exact DXA/lab monitoring cadence;
- exact medication-specific safety questions;
- provider/model/API configuration for PR-1;
- detailed final visual design of cards;
- automatic treatment recommendations;
- a new patient-level treatment database;
- full Practice Review runtime.

These are not omissions from G-0; they are correctly deferred to evidence review or bounded implementation slices.

---

## 14. G-1 implementation boundary recommended by this review

The next runtime slice should be limited to **generic dynamic-guidance mechanics**, not all osteoporosis clinical content at once:

```text
1. build read-only LongitudinalGuidanceProjectionV1 from protected prior encounters;
2. build EncounterContextV1;
3. implement deterministic GuidanceRuleV1 evaluator / priority resolution;
4. produce VisitPlanV1 + GuidedCardStateV1;
5. render `why now` and current coarse archetype flow using synthetic/product-flow rules;
6. prove event override, unresolved-prior and treatment/due plumbing with synthetic fixtures;
7. do not yet invent agent-specific milestone content;
8. no transcript/provider work in G-1 unless separately widened by an approved slice.
```

After G-1 mechanics are stable, evidence-backed osteoporosis guidance profiles and PR-1/PR-2 can be added in their own bounded slices before the real pilot.

---

## 15. Final classification

```text
PRODUCT PURPOSE                            PASS
FOUR-FUNCTION SYSTEM BOUNDARY              PASS
CURRENT RUNTIME SOURCE PATHS               PASS
LONGITUDINAL SOURCE AVAILABILITY           PASS
LONGITUDINAL PROJECTION CONTRACT           PASS
ARCHETYPE + TRIGGER MODEL                  PASS
RULE PRIORITY                              PASS
REPEATED-THERAPY CAPABILITY                PASS
MACHINE CONTRACT MANIFEST                  PASS
HEIDI SEMANTIC COMPATIBILITY               PASS
SYSTEM-ASSISTED BASELINE METHODOLOGY        PASS
RUNTIME IMPLEMENTATION IN G-0              NO
CLINICAL MILESTONE CONTENT INVENTED         NO
DATABASE MIGRATION REQUIRED FOR G-1        NO
```

**G-0 classification: `DESIGN-COMPLETE`.**

This classification authorizes no runtime mutation by itself.
