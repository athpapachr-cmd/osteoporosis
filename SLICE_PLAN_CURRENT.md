# SLICE_PLAN_CURRENT.md — Dynamic Guided Visit + Heidi-First Pilot Replan v1

> **STATUS:** DESIGN-COMPLETE / PRE-RUNTIME STOP.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** M01-G0-DYNAMIC-VISIT-v1.
> **Verified remote main at bootstrap:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Parent tested runtime ancestry:** `fix/module01-c1-authoritative-finish-2026-08-30` @ `a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871`.
> **Design branch:** `design/module01-dynamic-guided-visit-replan-2026-08-30`.
> **Runtime writer:** NONE.
> **Runtime mutation:** NOT AUTHORIZED by G-0.
> **Merge/deploy/preview:** NOT AUTHORIZED / NOT DONE.

---

# 1. Product correction

The product owner clarified that Module 01 exists primarily to:

```text
improve the current visit
+
reduce duplicate/manual data entry
+
review whether what was said/reasoned/decided was appropriate
+
improve the clinician longitudinally
```

The current largely manual Steps 1–6 workflow is already known to impose unacceptable burden for intended routine use. Therefore it is not the product that should be tested in the five-case real pilot.

Osteoporosis encounters are intrinsically dynamic: first assessment, later results/work-up decision visit, routine treatment administration, treatment milestones, delayed therapy, fracture/fracture-on-treatment, adverse effects, transition/exit and other states require different emphasis.

---

# 2. Methodology correction

The former order is superseded:

```text
5 manual pilot cases
→ freeze
→ transcript extraction later
→ adaptive workflow later
```

Approved order:

```text
C1 finalization integrity merge/deploy/smoke
→ dynamic Clinical Guidance foundation
→ PR-1 Heidi transcript extraction
→ PR-2 inline provisional population / clinician review
→ guided clinical-card UX sufficient for real use
→ 5 consecutive real system-assisted pilot encounters
→ one deliberate refinement
→ freeze Guidance/Capture/KPI applicability contracts
→ minimum Quick Practice Review shadow capability
→ 30 consecutive unique scored system-assisted encounters
→ baseline lock
→ reviewed Signals/intervention
→ re-measure
→ Module 01 closure review
```

During the 30-case baseline:

- stable Clinical Guidance remains active;
- transcript-assisted capture remains active;
- routine KPI/performance feedback remains hidden;
- routine clinician-facing Practice Review remains hidden by default;
- safety-critical feedback remains allowed;
- guidance exposure is recorded where reliable;
- the cohort is labelled **system-assisted baseline**.

---

# 3. Four-function product boundary

```text
Clinical Guidance
!= Transcript-assisted Capture
!= Audit / Measurement
!= Clinical Practice Review
```

### Clinical Guidance
Helps conduct today's encounter: what is due, newly triggered, unresolved or needed before safe closure.

### Transcript-assisted Capture
Turns what was said into provisional structured data without duplicate manual entry.

### Audit
Measures whether applicable process/standards occurred and how performance changes.

### Practice Review
Reviews whether reasoning, communication and decisions were appropriate and converts repeated observations into improvement Signals.

---

# 4. Normative machine-contract entrypoint

```text
schemas/dynamic_guided_visit_contract_manifest_v1.yaml
```

It points to:

```text
schemas/dynamic_guided_visit_v1.yaml
schemas/longitudinal_guidance_projection_v1.yaml
```

Frozen objects:

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

Formal exact review:

```text
M01_G0_DYNAMIC_GUIDANCE_DESIGN_REVIEW_V1.md
classification: DESIGN-COMPLETE
```

---

# 5. Existing runtime foundations verified

The current protected/runtime system already supplies:

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

Current coarse archetypes remain useful as **visit intent**:

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

A candidate future intent `results_or_workup_review_with_management_decision` is preserved for later runtime review because a second visit reviewing results and making management decisions is materially different from both a full first assessment and stable routine follow-up. It is not yet a persisted runtime enum.

---

# 6. Why archetype alone is insufficient

Current `adaptive-applicability.js` can only express roughly:

```text
applicable
uncertain
not_applicable
```

The approved product needs relevance derived from:

```text
visit intent
+ longitudinal treatment state
+ actual administration history
+ elapsed exposure
+ reliable administration count
+ due/overdue state
+ monitoring due state
+ new fracture/adverse event/safety trigger
+ unresolved prior tasks/prerequisites
+ patient-specific context
+ transcript uncertainty/conflict
```

Every non-obvious surfaced card should be able to answer:

> **WHY NOW?**

---

# 7. Rule hierarchy

Frozen priority:

```text
critical safety / urgent event
→ unresolved prior critical item
→ treatment/agent-specific requirement
→ evidence-defined milestone/due item
→ archetype base flow
→ contextual item
```

Rules:

1. higher-priority safety/event triggers cannot be hidden by lower-priority routine defaults;
2. one card may retain multiple reason codes;
3. generic `not_applicable` cannot suppress a current higher-priority trigger;
4. clinician override remains possible where appropriate and retains reason/provenance;
5. the guidance engine structures checks/prerequisites but does not silently make the treatment decision.

---

# 8. Longitudinal projection correction found during review

Storage is currently encounter-snapshot based; there is no separate patient-level treatment timeline table.

However:

```text
GET /clinical/patient/{patient_id}/encounters
```

returns every protected historical encounter with full `payload`.

Therefore the first guidance runtime can derive a read-only longitudinal projection without a database migration.

`LongitudinalGuidanceProjectionV1` freezes these rules:

- completed/amended prior encounters are historical sources;
- blank later snapshots do not erase prior authoritative history;
- material conflicts remain explicit conflicts;
- scheduled/planned administration does not count as an actual administered dose;
- administration count is computed only from reliable unique actual events;
- exact `agent + actual_date` may identify a repeated representation of the same actual administration when a stable event ID is unavailable;
- administration count and elapsed exposure remain separate;
- missing doses are never reconstructed from expected cadence;
- unresolved prior tasks may surface into today's context;
- provisional transcript candidates do not alter longitudinal authority until clinician acceptance.

Preferred G-1 implementation is an **ephemeral derived projection**, not a new persistent patient-level treatment database.

---

# 9. Repeated Prolia / repeated-treatment design

The product-owner examples of early, later and long-duration Prolia visits demonstrate the need for milestone-aware behavior, but do not themselves establish clinical milestone rules.

Do not create:

```text
prolia_visit_1
prolia_visit_2
...
prolia_visit_10
```

Use:

```text
treatment_continuation_or_due_monitoring
+
active agent
+
actual administrations
+
reliable administration count
+
elapsed exposure
+
due/overdue state
+
monitoring state
+
reviewed TherapyMilestoneProfile rules
+
event/safety overrides
```

Hard rule:

> No exact “4th / 8th / 10th Prolia” clinical guidance is activated without reviewed evidence or an explicitly approved clinic-policy source explaining the milestone.

If treatment delays make count and elapsed time diverge, preserve both.

---

# 10. Heidi-first / inline review direction

The corrected archived PR-1 v3 semantics remain authoritative for extraction safety:

```text
raw transcript
→ ephemeral provider processing
→ structured semantic candidates
→ deterministic Module 01 target mapping
→ no authoritative PR-1 write
```

Preserve:

- negation;
- temporality;
- speaker/source;
- uncertainty;
- objective result vs interpretation;
- option discussed vs recommendation vs final decision;
- patient preference vs acceptance;
- exact vs vague dates;
- actual runtime target mapping;
- PHI-safe validation/logging.

PR-2 UX direction is now:

```text
mapped candidate
→ provisional value in destination clinical card
→ Accept / Edit / Reject
→ authoritative value only after clinician review
```

Do not rebuild duplicate data entry around a detached candidate list.

---

# 11. Guidance exposure / clinician improvement

Where technically reliable, `GuidanceExposureV1` may retain:

```text
item surfaced?
reason
content already present before cue? yes/no/unknown
resolved after cue? yes/no/unknown
resolution source: transcript / clinician / prior data / mixed / unknown
```

Purpose:

```text
system-supported correct execution
→ repeated supported performance
→ potentially increasingly pre-prompt/spontaneous correct behavior
```

This is descriptive context, not a punitive score and not proof of causal learning when event timing is uncertain.

---

# 12. Design fixtures frozen

The machine contract includes at least:

1. first assessment;
2. results/work-up review and management decision using prior data;
3. routine repeated denosumab administration without new issue;
4. scheduled administration with new fracture/fracture-on-treatment override;
5. delayed/missed time-critical administration;
6. long-duration milestone where additional content appears only if an active reviewed milestone profile matches.

Longitudinal projection fixtures include duplicate administration snapshots, scheduled-not-actual events, delayed count-vs-time divergence, later blank snapshots, conflicting dates and unresolved/completed prior tasks.

---

# 13. G-0 exact review result

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

**G-0 classification: DESIGN-COMPLETE.**

---

# 14. Explicit non-decisions / out of scope

G-0 does not freeze or implement:

- exact denosumab 4th/8th/10th-dose guidance;
- exact DXA/lab monitoring cadence;
- exact medication-specific safety questions;
- provider/model/API configuration for transcript extraction;
- final detailed visual styling;
- automatic treatment recommendations;
- new patient-level treatment persistence;
- Practice Review runtime;
- physiotherapy/RF work.

---

# 15. Recommended next runtime slice — G-1

If separately authorized after a fresh bootstrap, G-1 should be limited to generic dynamic-guidance mechanics:

```text
1. derive LongitudinalGuidanceProjectionV1 from protected historical encounters;
2. build EncounterContextV1;
3. implement deterministic GuidanceRuleV1 evaluation / priority resolution;
4. produce VisitPlanV1 + GuidedCardStateV1;
5. render `why now` and current coarse archetype flow;
6. prove event override, unresolved-prior and treatment/due plumbing with synthetic fixtures;
7. use only product-flow/synthetic rules needed to prove mechanics;
8. do not invent agent-specific clinical milestone content;
9. do not implement transcript/provider/PR-2 in G-1 unless a later approved slice explicitly includes them.
```

Agent-specific evidence-backed guidance profiles and PR-1/PR-2 then follow as bounded slices **before the five-case real pilot**.

---

# 16. Stop rule

G-0 is complete. Runtime mutation is not authorized by this design-complete state.

Next actions require fresh canonical bootstrap and a separate product-owner/runtime decision. C1 merge/deploy remains a separate release decision.
