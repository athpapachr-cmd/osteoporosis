# SLICE_PLAN_CURRENT.md — Dynamic Guided Visit + Heidi-First Pilot Replan v1

> **STATUS:** ACTIVE PRE-RUNTIME DESIGN / METHODOLOGY REPLAN.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** M01-G0-DYNAMIC-VISIT-v1.
> **Verified remote main:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Parent tested runtime ancestry:** `fix/module01-c1-authoritative-finish-2026-08-30` @ `a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871`.
> **Writer:** `design/module01-dynamic-guided-visit-replan-2026-08-30`.
> **Runtime writer:** NONE.
> **Runtime mutation:** NOT AUTHORIZED in this design slice.
> **Merge/deploy/preview:** NOT AUTHORIZED / NOT DONE.

---

# 1. Product-owner correction that triggered REPLAN

The product owner clarified three linked facts:

1. the current largely manual Baseline Audit capture can take roughly 12 minutes and is not a viable intended clinical workflow;
2. Heidi transcript should populate as much of the structured encounter as possible, with clinician review for omissions/uncertainty/conflicts;
3. the application was created to improve the **visit itself** and then improve the clinician longitudinally by reviewing whether what was said/reasoned/decided was appropriate.

The product owner also clarified that osteoporosis visits are highly dynamic. Examples include:

- first assessment;
- a later results/work-up review where management is decided;
- repeated denosumab administrations;
- different long-term treatment milestones;
- fracture or fracture-on-treatment;
- treatment transition/exit;
- adverse effects or other new events.

Therefore the prior closure order:

```text
manual 5-case pilot
→ later transcript extraction
→ much later adaptive consultation flow
```

is invalid for the intended product and is superseded.

---

# 2. Exact problem in the current runtime

The current runtime already contains useful foundations:

- coarse encounter archetypes in `static/baseline-audit/index.html`;
- an `adaptive-applicability.js` map that marks domains `applicable / uncertain / not_applicable` by archetype;
- longitudinal treatment episodes and administration events in Step 4;
- protected patient/encounter/lab persistence;
- the tested authoritative Finish correction in parent ancestry;
- archived corrected PR-1 v3 transcript extraction design.

But the current adaptive layer is too coarse.

It answers roughly:

> Is this domain usually applicable for this archetype?

It does **not** yet answer:

> Why is this card relevant now for this exact patient at this exact longitudinal treatment point?

Missing inputs include:

```text
active agent / treatment episode
actual administration history
administration count when reliable
elapsed treatment exposure
last actual dose / next due / delay state
monitoring due state
new fracture / adverse event / safety trigger
unresolved prior task/prerequisite
transcript uncertainty/conflict
```

---

# 3. Design objective

Freeze the minimum architecture needed so the clinician-facing product can behave as:

```text
LONGITUDINAL PATIENT STATE
+
TODAY'S VISIT INTENT
+
CURRENT EVENT / TREATMENT / DUE TRIGGERS
+
PRIOR UNRESOLVED ITEMS
        ↓
DYNAMIC VISIT PLAN
        ↓
ONLY RELEVANT CLINICAL CARDS
        ↓
HEIDI-ASSISTED PROVISIONAL POPULATION
        ↓
CLINICIAN REVIEW / RESOLUTION
        ↓
EXPLICIT DECISION + CLOSE
        ↓
POST-VISIT PRACTICE REVIEW / AUDIT LATER
```

The design must avoid both extremes:

```text
one giant static checklist for every encounter
and
one separate hard-coded form for every visit/dose number
```

---

# 4. Core object — `EncounterContextV1`

Minimum candidate contract:

```yaml
schema_version: encounter_context_v1
module: osteoporosis
patient_relationship: new_to_service | established_patient | unknown
encounter_archetype: <coarse visit intent>
active_treatment:
  agent: <normalized agent or null>
  episode_id: <optional>
  start_date: <optional exact date>
  elapsed_exposure_months: <optional derived>
  administration_count: <optional integer>
  last_actual_administration: <optional exact date>
  next_due_date: <optional exact date>
  due_state: not_applicable | not_due | due | overdue | uncertain
monitoring_due:
  labs: true | false | uncertain
  dxa: true | false | uncertain
  other: []
new_events:
  fracture: true | false | uncertain
  fracture_on_treatment: true | false | uncertain
  adverse_effect: true | false | uncertain
  other_safety: []
unresolved_prior_items: []
special_context_flags: []
```

Rules:

- derived values use authoritative treatment history only;
- missing dates remain missing/uncertain;
- administration count is not reconstructed from guessed cadence;
- elapsed exposure and administration count remain separate;
- a nominal appointment label does not prove an administration occurred.

---

# 5. Core object — `GuidanceRuleV1`

Minimum candidate contract:

```yaml
rule_id: string
module: osteoporosis
domain: string
card_id: string
rule_class: critical_safety | event_triggered | unresolved_prior | agent_specific | milestone_due | archetype_core | contextual
priority: integer
applies_if: <deterministic predicate over EncounterContextV1>
reason_code: string
human_reason: string
source:
  type: guideline | evidence | approved_clinic_policy | product_flow
  id: string | null
  version: string | null
  strength_or_certainty: string | null
reviewed_on: date | null
status: active | draft | superseded
```

Clinical guidance rules must be deterministic once the structured context is known.

---

# 6. Rule priority / conflict resolution

Frozen priority:

```text
critical safety / urgent event
→ unresolved prior critical item
→ treatment/agent-specific requirement
→ evidence-defined milestone/due item
→ archetype base flow
→ patient-specific contextual item
```

Rules:

1. higher-priority triggers cannot be hidden by a lower-priority archetype default;
2. a card may carry multiple reason codes;
3. generic `not_applicable` is overridden by a current higher-priority event/safety/due trigger;
4. clinician override remains possible where appropriate and should retain reason/provenance;
5. the engine surfaces checks/prerequisites but does not silently choose treatment.

---

# 7. Core object — `VisitPlanV1`

Candidate output:

```yaml
visit_plan_id: uuid
module: osteoporosis
encounter_archetype: string
ordered_cards:
  - card_id: string
    priority: integer
    reason_codes: []
    why_now: string
    state: required | due | contextual
critical_unresolved: []
close_requirements: []
rule_trace: []
```

A Visit Plan is ephemeral/derivable presentation state. It does not duplicate authoritative clinical facts.

---

# 8. Core object — `GuidedCardStateV1`

A card needs more than current `applicable/conditional/N/A` state.

Candidate state:

```yaml
card_id: string
visibility: surfaced | collapsed | hidden
priority: integer
reason_codes: []
why_now: string
prior_data_state: available | absent | stale_or_due | uncertain
capture_state: resolved | unresolved | partial
provisional_candidate_count: integer
conflict_count: integer
critical_unresolved: boolean
```

UX requirements:

- prior stable authoritative data may appear as summary/read-only rather than blank re-entry fields;
- due/current facts are prominent;
- provisional transcript values appear in place;
- unresolved/conflicting values are obvious;
- irrelevant cards do not dominate the visit;
- `why now` is available for dynamically surfaced content.

---

# 9. Archetype strategy — coarse intent, not dose-number explosion

Preserve current coarse intents as much as possible:

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

Potential addition to review during G-1 design/runtime:

```text
results_or_workup_review_with_management_decision
```

because this represents a materially different visit intent from either a full initial assessment or a stable routine follow-up.

Do not create:

```text
prolia_visit_1
prolia_visit_2
prolia_visit_3
...
prolia_visit_10
```

Instead use `treatment_continuation_or_due_monitoring` plus treatment timeline/milestone/event rules.

---

# 10. Repeated denosumab / repeated-therapy milestone model

The product-owner examples (“1–3”, “4th/8th”, “10th” Prolia visits) demonstrate a real need for longitudinal milestone-aware behavior, but do not themselves establish clinical rule authority.

Required model:

```text
base administration flow
+
agent-specific rules
+
actual administration count when reliable
+
elapsed exposure
+
monitoring-due rules
+
long-duration review rules
+
event overrides
```

A `TherapyMilestoneProfileV1` should allow triggers such as:

```text
every_administration
administration_count in {...}
elapsed_exposure >= X
monitoring_due == true
next_due_state == overdue
new_fracture == true
fracture_on_treatment == true
course_completion / transition state
```

Hard rules:

- exact clinical milestone content requires reviewed evidence or approved clinic-policy provenance;
- count and elapsed time remain separate;
- delays must not be hidden by a nominal dose number;
- no invented “10th dose rule” without defining its evidence/policy rationale.

---

# 11. Visit-flow examples to prove architecture

These are design fixtures, not final clinical protocols.

## Fixture A — first assessment

Expected plan shape:

```text
why today
→ fracture/risk characterization
→ DXA/VFA/imaging as relevant
→ secondary causes/labs
→ falls/function as relevant
→ risk synthesis
→ treatment decision if ready
→ communication/close
```

## Fixture B — second visit / results review / management decision

Prior full history exists.

Expected behavior:

- do not reopen all first-visit fields;
- surface pending labs/results/DXA/VFA;
- surface unresolved secondary-cause questions;
- surface risk synthesis;
- surface options/recommendation/preference/final decision;
- surface prerequisites/tasks/close.

## Fixture C — routine repeated denosumab administration without new issue

Expected behavior:

- concise interval-change/fracture check;
- current administration/timing state;
- only due monitoring/milestone items;
- agent-specific safety/tolerance items defined by reviewed rules;
- short Close;
- no full secondary-cause/risk reassessment without a trigger.

## Fixture D — same scheduled administration but new fracture

Expected behavior:

- fracture/event override supersedes routine administration flow;
- verify fracture and treatment exposure/timing;
- surface relevant reassessment/response/decision cards;
- transition/escalation reasoning as applicable;
- no routine-only abbreviated visit plan.

## Fixture E — delayed/missed time-critical administration

Expected behavior:

- timing/safety override prominently surfaced;
- exact actual administration history used;
- ordinary “routine continuation” flow cannot hide the delay state.

## Fixture F — long-duration treatment milestone

Expected behavior:

- milestone rules add the defined reassessment content to the otherwise routine flow;
- milestone is derived from reviewed timeline/count rules, not hard-coded screen identity.

---

# 12. Heidi-first capture architecture retained and repositioned

The corrected archived `PR1_TRANSCRIPT_INTAKE_V3.md` remains the technical starting point.

New product sequencing:

```text
PR-1 semantic extraction
→ deterministic target mapping
→ PR-2 inline provisional card population
→ clinician review
→ authoritative merge
```

This now occurs **before the five real pilot cases**.

---

# 13. Inline population contract

Mapped transcript candidates should appear in their destination clinical cards.

Candidate state:

```text
proposed
accepted
edited_and_accepted
rejected
conflict_needs_resolution
unmapped_needs_review
```

Rules:

- `proposed` values are visually populated but not authoritative;
- authoritative existing data are never silently overwritten;
- evidence snippet/confidence may be available on demand;
- one clinician action may accept a safe group only after category-specific safeguards exist;
- unmapped clinically meaningful assertions remain visible rather than disappearing;
- “not mentioned” never becomes a negative finding.

---

# 14. Guidance exposure / clinician internalization

Introduce `GuidanceExposureV1` where technically reliable:

```yaml
encounter_id: string
item_id: string
reason_code: string
was_surfaced: boolean
content_present_before_surface: yes | no | unknown
resolved_after_surface: yes | no | unknown
resolution_source: transcript | clinician_entry | prior_data | mixed | unknown
```

Purpose:

- measure system-supported execution without pretending it was unassisted;
- later assess whether correct behavior becomes increasingly pre-prompt/spontaneous;
- avoid withholding useful guidance merely to manufacture a baseline.

Do not infer causality if event timing cannot be established.

---

# 15. Revised pilot/baseline methodology

## 15.1 Five-case pilot

Pilot only after:

```text
authoritative Finish deployed/smoked
+
minimum dynamic Visit Plan
+
transcript extraction
+
inline clinician review/population
```

Pilot metrics:

- completion time;
- number of manual corrections/entries;
- clinically meaningful transcript omissions;
- false/incorrect candidates;
- ambiguous/conflict candidates;
- wrong card relevance;
- duplicate questioning;
- persistence/finalization problems;
- cognitive burden;
- safety/data-integrity issues.

After five consecutive eligible cases: one deliberate refinement, then freeze.

## 15.2 Thirty-case baseline

Frozen policy:

- Clinical Guidance active;
- transcript-assisted capture active;
- routine KPI score/performance feedback hidden;
- routine clinician-facing Practice Review hidden by default;
- safety-critical feedback allowed;
- label cohort **system-assisted baseline**;
- capture guidance exposure when reliable.

---

# 16. Scope of this design slice

IN SCOPE:

- canonical methodology rebase;
- dynamic Visit Plan object model;
- guidance reason/priority semantics;
- repeated-treatment milestone architecture;
- positioning of PR-1/PR-2 before pilot;
- system-assisted baseline semantics;
- synthetic design fixtures;
- machine-readable contract file(s).

OUT OF SCOPE:

- runtime code changes;
- actual clinical milestone rules for Prolia/other agents without evidence review;
- provider/model implementation;
- transcript endpoint runtime;
- Accept/Reject/Edit runtime;
- UI restyle;
- Practice Review runtime;
- merge/deploy of C1;
- physiotherapy/RF work.

---

# 17. REPLAN triggers

STOP and replan if:

- actual current persistence cannot supply the longitudinal context needed for Visit Plan evaluation;
- treatment history cannot reliably distinguish scheduled vs actual administration;
- a proposed rule needs undocumented clinical assumptions;
- coarse archetype + trigger layering cannot represent a common real encounter without special-case explosion;
- transcript inline population requires silent overwrite of existing authoritative data;
- guidance exposure cannot be represented without invasive surveillance or unreliable event claims;
- system-assisted baseline cannot be described transparently enough to preserve methodological honesty.

---

# 18. Acceptance of G-0 design

G-0 is design-complete when:

```text
product purpose reconciled in AGENTS                 YES
roadmap sequence reconciled                           YES
phase methodology reconciled                         YES
EncounterContextV1 frozen                            YES
GuidanceRuleV1 frozen                                YES
VisitPlanV1 frozen                                   YES
GuidedCardStateV1 frozen                             YES
rule priority frozen                                 YES
therapy milestone architecture frozen                YES
Heidi/PR-2 moved before pilot                        YES
system-assisted baseline semantics frozen            YES
design fixtures defined                              YES
machine contract added                               REQUIRED
runtime mutation                                     NO
```

---

# 19. Exact next action

```text
1. add machine-readable dynamic-guidance contract/schema;
2. run exact design-completeness review against actual current runtime paths;
3. update CURRENT_OPERATIONAL with PASS/BLOCK;
4. STOP this design slice;
5. if PASS and runtime work is separately authorized, create bounded G-1 implementation slice for context resolver + Visit Plan/rule engine only.
```

Do not merge/deploy the C1 runtime fix as part of this G-0 design slice unless a separate merge/deploy decision is recorded.
