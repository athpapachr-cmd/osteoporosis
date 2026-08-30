# CLINICAL_EXCELLENCE_PLAN.md — Clinical Excellence architecture v3

> **STATUS:** ACTIVE detailed phase plan.
> **PHASE:** dynamic guided consultation + transcript-assisted capture → Practice Review → measurement/improvement loop.
> **CANONICAL HOME:** `athpapachr-cmd/osteoporosis`.
> **MODULE 01:** Osteoporosis.
> **UPDATED:** 2026-08-30 Asia/Nicosia.

This document owns the **detailed phase architecture and stage sequence**. Operational branch/PR/deploy state belongs in `CURRENT_OPERATIONAL.md`; exact current-slice design belongs in `SLICE_PLAN_CURRENT.md`.

---

# 1. Product purpose

Build a reusable **Personal Clinical Excellence System** whose primary purpose is twofold:

1. improve the **current clinical encounter** while it is happening; and
2. improve the **clinician over time** by reconstructing and reviewing what was said, reasoned and decided and whether it was appropriate.

Documentation quality, structured capture and audit are enabling mechanisms, not the product's primary end state.

The system should improve:

- what is asked;
- what is noticed;
- whether the right issue is surfaced at the right visit;
- how evidence is interpreted;
- how clinical decisions are made;
- how uncertainty is handled;
- what is explained to the patient;
- whether preferences materially affect the plan;
- whether prerequisites and follow-up are closed;
- what is learned from each encounter;
- whether correct behavior becomes more spontaneous and less prompt-dependent over time.

Canonical learning-health loop:

```text
STANDARD
→ LEARN
→ TEST / MASTER
→ APPLY IN REAL PRACTICE
→ GUIDE / SUPPORT WHEN NEEDED
→ MEASURE
→ AUDIT
→ CLINICAL PRACTICE REVIEW
→ GAP OR STRENGTH
→ INTERVENE / REINFORCE
→ RE-MEASURE
→ SYSTEM LEARNS
```

Module 01 — Osteoporosis is the proving ground for a reusable Core Engine.

---

# 2. Four distinct system functions

The system must not collapse clinical support, capture, measurement and coaching into one black box.

## 2.1 Clinical Guidance

Purpose:

> Given this patient's longitudinal state and today's clinical purpose, what should be surfaced, checked, resolved or closed now?

Properties:

- available during the real encounter;
- dynamic by visit intent, treatment state and longitudinal triggers;
- shows relevance/`why now` rather than a generic checklist;
- may surface safety prerequisites and due/milestone items;
- reuses prior patient data rather than asking everything again;
- does not silently choose the clinician's treatment decision.

## 2.2 Transcript-assisted Capture

Purpose:

> Convert what was actually said into provisional structured data with minimal duplicate manual entry.

Properties:

- Heidi/raw transcript ephemeral by default;
- structured candidates;
- deterministic module mapping;
- provisional in-place population;
- clinician Accept/Reject/Edit before authoritative write;
- explicit conflicts and uncertainty.

## 2.3 Audit / Measurement

Purpose:

> Determine whether an applicable defined process/standard occurred and measure change over time.

Properties:

- deterministic where possible;
- explicit numerator/denominator/applicability;
- transparent missing/N/A handling;
- no black-box composite score;
- guidance exposure context retained where useful.

## 2.4 Clinical Practice Review

Purpose:

> Critically review whether the consultation, reasoning, communication and final decisions were appropriate and what should change next time.

Properties:

- encounter evidence/provenance;
- linked standard/evidence for material claims;
- confidence and clinical importance;
- strength/gap/safety/uncertainty direction;
- clinician disposition;
- longitudinal aggregation into Signals.

Hard distinction:

```text
LIVE GUIDANCE
!= DATA CAPTURE
!= AUDIT SCORE
!= POST-VISIT PRACTICE REVIEW
```

---

# 3. Reusable Core architecture

```text
PERSONAL CLINICAL EXCELLENCE SYSTEM
│
├── Standards / Competencies
├── Evidence / Guidelines
├── Clinical Guidance / Visit Planning
├── Clinical Practice / Encounters
├── Longitudinal Patient Context
├── Transcript-assisted Capture
├── Clinical Practice Review
├── Audit / Measurement
├── Signal Engine
├── Learning / Mastery / Calibration
├── Patient Voice
├── Safety
├── Benchmarking
├── Improvement Projects
├── CareTasks / Continuity
└── Clinical Excellence Home

MODULE 01 — OSTEOPOROSIS
└── osteoporosis standards, treatment/milestone profiles, schemas, workflows, evidence, cases and metrics
```

Core owns reusable mechanics. Module 01 owns osteoporosis-specific clinical content.

---

# 4. Existing Core objects retained

The following existing concepts remain valid:

- `Signal`;
- `ClinicalStandard`;
- `Competency`;
- `EvidenceItem`;
- `LearningItem`;
- `AssessmentAttempt`;
- `ClinicalEncounter`;
- `ClinicalDecision`;
- `AuditMetric`;
- `ImprovementProject`;
- `ActionTask` / later `CareTask`;
- `PracticeReview`;
- `PracticeObservation`.

They are extended, not replaced.

---

# 5. New Core object — `EncounterContextV1`

`EncounterContextV1` represents the minimum longitudinal state required to decide what should be shown today.

Candidate contract:

```text
encounter_id_optional
module
patient_relationship
encounter_archetype / visit_intent
current_problem_state
active_treatment_episode_optional
active_agent_optional
administration_history_summary
administration_count_optional
elapsed_treatment_exposure_optional
last_actual_administration_optional
next_due_date_optional
due_status_optional
monitoring_due_states
new_event_flags
safety_flags
unresolved_prior_items
special_context_flags
available_prior_data_summary
context_generated_at
```

Important distinctions:

```text
visit intent != disease status
visit intent != treatment agent
administration count != elapsed exposure
scheduled date != actual administration date
new event != routine milestone
prior unresolved item != new finding
```

The context is derived from authoritative longitudinal data plus explicit current-visit facts. It must not invent missing history.

---

# 6. New Core object — `VisitPlanV1`

`VisitPlanV1` is the ordered clinician-facing plan for the current encounter.

```text
visit_plan_id
module
encounter_context_version
ordered_card_ids
critical_item_ids
due_item_ids
contextual_item_ids
suppressed_item_ids
close_requirements
rule_evaluation_trace
created_at
```

A Visit Plan is not a treatment recommendation. It determines **what should be surfaced and in what order**, preserving clinician judgment for interpretation and treatment choice.

---

# 7. New Core object — `GuidanceRuleV1`

A `GuidanceRuleV1` determines when a clinical item/card should be surfaced.

Candidate fields:

```text
rule_id
module
domain
card_id / item_id
rule_class
priority
applies_if
suppresses_if_optional
reason_code
human_reason
source_framework_or_policy
source_version
strength_or_certainty_optional
reviewed_on
status
```

Rule classes:

```text
critical_safety
event_triggered
unresolved_prior
agent_specific
milestone_due
archetype_core
contextual
not_applicable
```

The system must be able to answer:

> Why am I seeing this now?

---

# 8. Rule-resolution hierarchy

When multiple rules apply, resolve in this order unless an explicitly reviewed domain rule requires otherwise:

```text
1. critical safety / urgent event override
2. unresolved prior critical item
3. treatment/agent-specific requirement
4. evidence-defined milestone or due item
5. archetype base flow
6. patient-specific contextual item
7. optional/background item
```

Rules:

- higher-priority safety/event content cannot be hidden by a lower-priority routine archetype;
- a card may be surfaced by multiple reasons and should retain all relevant reason codes;
- `not_applicable` from a generic archetype does not suppress a higher-priority current trigger;
- clinician override is permitted where clinically appropriate and should retain reason/provenance;
- rule evaluation must be deterministic for the same structured context.

---

# 9. Existing archetypes retained as coarse visit intent

The current runtime already has useful coarse archetypes:

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

These remain valuable but are insufficient alone.

The visible workflow must combine the archetype with longitudinal treatment/event state.

Additional useful intent may be represented explicitly if runtime design demonstrates a need, including a **results/work-up review with management decision** visit. Avoid proliferating archetypes merely to encode every possible treatment dose number.

---

# 10. Dynamic osteoporosis consultation examples

These examples describe **flow architecture**, not final evidence-authoritative rule content.

## 10.1 First assessment — new/uncertain diagnosis

Likely emphasis:

```text
why today / fracture characterization
→ relevant risk factors
→ DXA/VFA/imaging
→ secondary causes / labs
→ falls/function where relevant
→ risk synthesis
→ treatment discussion/decision if ready
→ communication + close
```

## 10.2 Initial-to-service known osteoporosis/osteopenia

Likely emphasis:

```text
verify prior diagnosis and fracture history
→ prior DXA/work-up validity
→ treatment history/exposure
→ current monitoring/response
→ unresolved gaps
→ current management decision
→ close
```

## 10.3 Results/work-up review and treatment decision

This is clinically different from a full first assessment even if it is the patient's second physical visit.

Likely emphasis:

```text
what was pending from prior visit
→ labs/DXA/VFA/results
→ secondary-cause questions resolved or still open
→ risk/problem synthesis
→ treatment options
→ clinician recommendation
→ patient preference / acceptance
→ prerequisites / administration plan / follow-up
```

The system should reuse the first visit's structured history rather than ask it all again.

## 10.4 Routine stable follow-up

Likely emphasis:

```text
interval fracture / meaningful change
→ treatment adherence/tolerance/timing
→ due monitoring only
→ response/current plan
→ unresolved tasks
→ close
```

Do not repeat full diagnostic work-up or full education without a current reason.

## 10.5 Treatment start

Likely emphasis:

```text
final decision rationale
→ medication-specific safety/prerequisites
→ patient acceptance
→ administration/use instructions
→ monitoring / next due state
→ close
```

## 10.6 Repeated administration / continuation

Base flow should be small and fast.

Possible trigger layers include:

```text
interval fracture/new event
agent-specific safety/tolerance/timing
actual administration date / due state
monitoring due today
milestone review due
unresolved prior task
```

The system should not automatically reopen the full osteoporosis assessment at every administration.

## 10.7 Fracture on treatment

This is an event override and should supersede a routine administration/follow-up flow.

Likely emphasis:

```text
fracture verification / mechanism / timing
→ exact treatment exposure/adherence/administration history
→ risk reassessment
→ relevant secondary causes / monitoring
→ treatment-response interpretation
→ escalation/change/transition reasoning
→ falls/function where relevant
→ close
```

## 10.8 Transition / exit / consolidation

Likely emphasis:

```text
why transition
→ exact prior last dose/end
→ sequencing safety
→ next agent / timing
→ prerequisites
→ tracking / next action
→ close
```

---

# 11. Therapy-specific milestone architecture

Repeated treatment must not generate one hard-coded form per ordinal visit.

Introduce `TherapyMilestoneProfileV1`:

```text
profile_id
agent
rule_source
reviewed_on
milestones[]
```

A milestone may trigger on one or more of:

```text
every_administration
administration_count
elapsed_months_or_years
next_due_or_overdue_state
monitoring_due_state
new_fracture_or_response_concern
course_completion_state
transition_state
```

For denosumab and other time-critical therapies:

- actual administration history is authoritative over nominal appointment labels;
- ordinal dose number may be retained when reliable;
- elapsed treatment exposure is retained separately;
- if delays mean count and elapsed time diverge, preserve both;
- exact milestone behavior requires reviewed evidence/approved clinic-policy provenance;
- do not encode “4th/8th/10th dose” behavior merely from conversational examples without defining why that milestone exists.

This architecture allows an evidence-reviewed rule later to say, for example, “surface periodic reassessment at this treatment milestone” without building an entirely separate visit screen.

---

# 12. New Core object — `GuidedCardStateV1`

Each clinician-facing card/item should carry state such as:

```text
card_id
visibility: surfaced / collapsed / hidden
priority
reason_codes[]
why_now
known_prior_data_state
current_capture_state
provisional_candidate_count
conflict_count
unresolved_count
critical_unresolved
clinician_override_optional
```

Card behavior:

- prior stable data may appear read-only/summary rather than as blank fields;
- due/current items are prominent;
- transcript candidates appear in place;
- ambiguous/conflicting items are easy to review;
- irrelevant content remains out of the main flow;
- one click may reveal contextual detail when needed;
- safety-critical unresolved items remain visible until resolved or explicitly dispositioned.

---

# 13. `Why now?` is part of the UX contract

Every dynamically surfaced non-obvious item should be able to explain its reason, for example:

```text
Always for this visit type
Due based on treatment timeline
New fracture changed the visit flow
Unresolved from the previous visit
Needed before today's treatment decision
Medication-specific safety item
Transcript contains conflicting/uncertain information
```

This is clinically safer and more usable than unexplained adaptive hiding/showing.

---

# 14. Transcript-assisted capture — Core flow

Canonical flow:

```text
PASTE / PROVIDE HEIDI TRANSCRIPT
        ↓
EPHEMERAL PROCESSING
        ↓
STRUCTURED EXTRACTION CANDIDATES
        ↓
DETERMINISTIC MODULE TARGET MAPPING
        ↓
PROVISIONAL IN-PLACE CARD POPULATION
        ↓
CLINICIAN REVIEW / EDIT / ACCEPT / REJECT
        ↓
ACCEPTED NORMALIZED ENCOUNTER DATA
        ↓
AUDIT + PRACTICE REVIEW
```

The raw transcript is not authoritative patient truth and is not persisted by default.

---

# 15. PR-1 v3 extraction architecture retained

The corrected archived PR-1 v3 design remains the starting technical authority for transcript extraction.

Core owns:

- protected transcript endpoint/transport;
- request size/validation;
- sanitized error boundary;
- semantic candidate envelope;
- composite components;
- speaker/polarity/temporality/certainty;
- provider abstraction;
- privacy/logging;
- module registry.

Module 01 owns:

- osteoporosis concept vocabulary;
- normalization rules;
- actual runtime target registry;
- mapped/ambiguous/unmapped decisions.

Critical rule:

```text
provider extracts semantic clinical assertions
→ deterministic Module 01 code chooses runtime targets
```

Provider output must never manufacture application storage paths.

---

# 16. Transcript semantic invariants

Preserve distinctions between:

```text
PATIENT/HISTORY FACT
OBJECTIVE RESULT
CLINICIAN INTERPRETATION
OPTION DISCUSSED
CLINICIAN RECOMMENDATION
PATIENT PREFERENCE
FINAL DECISION
PATIENT ACCEPTED / DECLINED / UNDECIDED
FOLLOW-UP TASK
UNCERTAIN / NEEDS REVIEW
```

Hard rules:

- preserve negation;
- preserve temporality;
- preserve speaker/source;
- preserve uncertainty;
- do not infer exact dates from vague timing;
- do not invent diagnosis/treatment/history/preferences;
- do not convert negative history into a normal investigation;
- do not collapse option/recommendation into final decision;
- do not overwrite original formal FRAX with adjusted/contextual risk;
- do not squeeze clinically meaningful unmapped concepts into unrelated fields.

---

# 17. Inline provisional population — PR-2 UX direction

The original “candidate preview” concept is retained semantically but changed in presentation.

Preferred workflow:

```text
candidate mapped to card/field
→ value appears in place with AI/provisional marker
→ evidence snippet/confidence available on demand
→ Accept / Edit / Reject
→ accepted value becomes authoritative
```

The clinician should not have to review a long detached candidate list and then manually re-enter the same facts in the form.

Candidate states:

```text
proposed
accepted
edited_and_accepted
rejected
conflict_needs_resolution
unmapped_needs_review
```

Existing authoritative longitudinal data are not overwritten silently.

---

# 18. Transcript coverage summary

After extraction, the system may show a compact operational summary such as:

```text
mapped provisional values: N
needs review/conflict: N
clinically meaningful unmapped: N
relevant current cards still unresolved: N
```

Do not interpret “not mentioned in transcript” as a negative clinical finding.

Coverage should be measured against **today's applicable Visit Plan**, not against every field in the entire osteoporosis schema.

---

# 19. Pre-existing data reuse

A core usability rule is:

```text
KNOWN STABLE LONGITUDINAL DATA
!= ASK AGAIN / TYPE AGAIN AT EVERY VISIT
```

The visit UI should distinguish:

- prior authoritative fact still relevant;
- interval update required;
- data due for remeasurement;
- prior fact invalidated/uncertain;
- new transcript candidate;
- current clinician confirmation.

Example: a stable established patient's full historical secondary-cause work-up should not be presented as blank compulsory fields at every routine administration unless a rule indicates reassessment is due or a new trigger exists.

---

# 20. Close architecture

Every encounter should end with a concise explicit close appropriate to that visit:

```text
Σήμερα αποφασίσαμε
Πριν ξεκινήσει / prerequisites
Εκκρεμεί
Ο ασθενής πρέπει να κάνει
Εμείς πρέπει να κάνουμε
Αποτελέσματα / επικοινωνία
Επόμενος επανέλεγχος / administration due
Unresolved critical item
```

Close content should normalize follow-up tasks rather than duplicate scattered free text.

A routine administration may have a very small Close state; a new treatment decision or fracture-on-treatment visit may have a much richer one.

---

# 21. Clinical Practice Review objects

`PracticeReviewV1` remains an AI-assisted but clinician-governed review of one encounter.

Candidate fields:

```text
review_id
encounter_id
module
review_mode
source_set
created_at
review_status
baseline_exposure_state
summary
strength_observation_ids
gap_observation_ids
safety_observation_ids
uncertainty_observation_ids
decision_reconstruction_id
communication_review_id
workflow_review_id
linked_signal_ids
clinician_overall_response
```

`PracticeObservationV1`:

```text
observation_id
review_id
dimension
direction
root_cause_class_optional
importance
confidence
statement
why_it_matters
encounter_provenance
linked_standard_ids
linked_evidence_ids
suggested_change
suggested_intervention_type
clinician_disposition
clinician_note_optional
linked_signal_id_optional
```

Important rule: model opinion is not established clinical truth merely because the model is confident.

---

# 22. Practice Review dimensions

At minimum:

1. clinical completeness for this **specific Visit Plan/archetype**, not a universal checklist;
2. clinical reasoning sequence;
3. decision quality;
4. risk interpretation;
5. safety;
6. communication scientific accuracy/clarity;
7. shared decision making/preferences;
8. consultation flow/efficiency/cognitive burden;
9. follow-up execution.

This directly addresses the product question:

> Was what was said, reasoned and finally decided appropriate for this patient at this point in the longitudinal pathway?

---

# 23. Decision Reconstruction — anti-hindsight architecture

Reconstruct:

```text
DATA AVAILABLE AT TIME T
↓
INTERPRETATION
↓
OPTIONS CONSIDERED
↓
NEW INFORMATION
↓
RECOMMENDATION
↓
PATIENT PREFERENCE
↓
FINAL DECISION
```

The review asks:

> Given only the information available at that point, what decision was reasonable?

Later results must not retrospectively make an earlier reasonable decision look irrational.

---

# 24. New Core object — `GuidanceExposureV1`

Because live Clinical Guidance is now part of the intended product, measurement should not require withholding useful guidance merely to create an “unassisted” baseline.

Where technically reliable, capture:

```text
encounter_id
card_or_item_id
rule_reason
was_guidance_surfaced
surfaced_at_optional
content_present_before_surface: yes / no / unknown
resolved_after_surface: yes / no / unknown
source_of_resolution: transcript / clinician_entry / prior_data / mixed / unknown
```

This is not a punitive metric.

Longitudinally it may help distinguish:

```text
system-dependent execution
→ repeated supported correct behavior
→ increasingly spontaneous/pre-prompt correct behavior
```

A later reduction in prompt dependence may become evidence that the clinician has internalized a good workflow.

Do not claim causal learning when event timing cannot be reliably established.

---

# 25. Revised baseline methodology

The former methodology was:

```text
5 manual form pilot
→ freeze
→ 30 uncoached baseline
```

This is superseded because the product owner has already identified the manual duplicate-entry workflow as operationally unacceptable and because live visit guidance is a primary product function.

New sequence:

```text
C1 finalization integrity deployed/smoked
→ minimum dynamic Clinical Guidance
→ PR-1 transcript extraction
→ minimum PR-2 inline review/population
→ 5 real system-assisted pilot encounters
→ one deliberate refinement
→ freeze Guidance/Capture/KPI applicability contracts
→ 30 consecutive unique system-assisted scored encounters
→ baseline lock
→ clinician-facing Practice Review/intervention
→ re-measure
```

## 25.1 Five-case pilot objective

Test:

- dynamic relevance correctness;
- longitudinal data reuse;
- transcript extraction recall/precision at clinically meaningful level;
- ambiguous/conflict handling;
- manual correction burden;
- completion time;
- persistence/finalization integrity;
- cognitive friction;
- whether the Visit Plan matches the actual encounter.

Pilot cases are not included in the locked scored baseline because guidance/capture may change.

## 25.2 Thirty-case baseline policy

During the 30-case baseline:

- Clinical Guidance remains active and stable;
- audit calculations may run in background;
- routine KPI score feedback remains hidden;
- red/green performance coaching remains hidden;
- routine clinician-facing Practice Review remains hidden by default;
- safety-critical feedback remains allowed;
- guidance exposure is recorded where reliable;
- the cohort is explicitly labelled **system-assisted baseline**.

This baseline measures the stabilized clinical workflow and supplies denominators/patterns for later improvement. It is not an untouched pre-intervention measure of unaided clinician behavior.

---

# 26. Audit interpretation

Existing core interpretation remains:

```text
clinical process
!= formal documentation trace
!= transcript capture quality
```

Additional distinction:

```text
clinically correct process after system cue
!= proven spontaneous clinician behavior before cue
```

Both can be valuable, but they answer different questions.

One safety failure may generate a Signal even with a small denominator.

---

# 27. Evidence and guidance governance

Osteoporosis domains remain:

1. Diagnosis & case finding
2. DXA / VFA / imaging
3. Fracture-risk assessment
4. Secondary osteoporosis & laboratory evaluation
5. Pharmacologic treatment selection
6. Sequential therapy / treatment transitions
7. Monitoring / treatment response / adherence
8. Falls, frailty, exercise & nutrition
9. Communication / shared decision making / continuity

For every material Clinical Guidance rule or Practice Review claim, preserve where applicable:

```text
source framework / evidence / approved clinic policy
version/year
exact criterion or recommendation
applicability/trigger
strength/certainty
reviewed_on
freshness state
```

No silent hybridization of incompatible frameworks.

No exact treatment milestone/cadence rule should be invented without reviewed authority.

---

# 28. Gap classification and intervention

Accepted negative Signals use the existing four-class root-cause model.

## Knowledge gap

```text
targeted source
→ short test
→ spaced repetition
→ clinical case
→ re-measure
```

## Reasoning gap

```text
case review
→ challenge / red team
→ alternative reasoning
→ deliberate practice
→ re-measure
```

## Execution gap

```text
workflow analysis
→ reduce friction
→ Clinical Guidance/task redesign
→ re-measure execution
```

## Communication/system gap

```text
wording / teach-back / handoff / process redesign
→ re-measure
```

Positive observations can become sustained-strength Signals after repeated evidence.

---

# 29. Longitudinal Practice Review

Single-case critique is insufficient.

The system should aggregate reviewed observations and distinguish:

```text
isolated event
recurrent gap
improving trend
stable weakness
sustained strength
prompt-dependent correct behavior
increasingly spontaneous correct behavior
```

No stable pattern should be declared from inadequate denominators without a reliability warning.

---

# 30. Learning / mastery engine

Canonical learning progression remains:

```text
UNREAD
→ STUDIED
→ TESTED
→ MASTERED
→ RETENTION CHECK
```

Adaptive priority should use:

- accepted Practice Review Signals;
- audit gaps;
- high-confidence errors;
- active Improvement Projects;
- new evidence;
- weak domains;
- advanced challenge for sustained strengths.

Full learning-product breadth is not required to close Module 01; enough intervention machinery to demonstrate a real closed loop is required.

---

# 31. Patient Voice

Patient feedback remains an important later source for:

- understanding condition;
- understanding plan;
- understanding treatment rationale/duration/risks;
- whether questions/preferences were addressed;
- free-text confusion/concern/praise/suggestion.

It is not a default Module 01 closure blocker unless later evidence makes it necessary for a critical loop.

---

# 32. Safety and privacy

The public repository must never contain identifiable patient data or real raw transcripts.

Production transcript handling must ensure:

- raw transcript transient by default;
- logs do not emit transcript text/evidence snippets/candidate values;
- accepted structured data enters protected clinical storage only after clinician review;
- provider secrets/config stay out of source;
- provider privacy/data-control posture is reviewed before identifiable transcript use;
- access/audit trail/retention/GDPR concerns remain explicit.

Safety rules:

- safety/event triggers outrank routine visit convenience;
- delayed/missed time-critical therapy state must be based on exact treatment timeline when possible;
- system guidance and clinician decision/override remain separately traceable.

---

# 33. Calendar / CareTasks — independent deferred track

Calendar foundation exists, but live Digital Secretary/Setmore feed is paused.

Permanent architecture:

```text
Appointment = scheduled attendance
CareTask = clinical action that may exist without an appointment
```

Paused Calendar integration must not block Clinical Guidance, transcript capture, Practice Review, audit, standards or learning.

---

# 34. Clinical Excellence Home

The eventual Home should answer:

1. Where am I?
2. What improved?
3. Biggest current gap?
4. What needs action today?

Candidate sections:

```text
safety / overdue care
Clinical Guidance due/unresolved items
Practice Review signals
Audit / run charts
strengths and gaps
learning due
evidence freshness
active Improvement Projects
what the system learned this month
```

Do not build polished summary scores before the data contracts and baseline are sufficient.

---

# 35. Revised implementation stages

## Stage G-0 — Dynamic Guidance design / methodology rebase

Deliverables:

- product purpose clarification;
- revised system-assisted baseline methodology;
- `EncounterContextV1`;
- `VisitPlanV1`;
- `GuidanceRuleV1`;
- `GuidedCardStateV1`;
- `GuidanceExposureV1`;
- therapy milestone architecture;
- exact rule-priority semantics;
- preservation of current runtime/storage schema beneath the presentation layer.

## Stage G-1 — Dynamic Guidance runtime foundation

Deliverables:

- deterministic encounter-context resolver;
- visit-plan/rule evaluation;
- current coarse archetype compatibility;
- longitudinal due/event/unresolved inputs;
- `why now` reason rendering;
- no treatment-decision automation.

## Stage PR-1 — Transcript paste + candidate extraction

Deliverables remain based on corrected archived v3:

- protected paste/intake;
- ephemeral raw transcript;
- structured semantic candidates;
- strict validation;
- deterministic osteoporosis runtime mapping;
- no authoritative write.

## Stage PR-2 — Inline clinician review/population

Deliverables:

- provisional candidate overlay in destination cards;
- Accept/Reject/Edit;
- conflict handling;
- authoritative merge only after clinician review;
- provenance;
- compact extraction/applicable-gap summary.

## Stage G-2 — Guided clinical-card UX

Deliverables:

- clinically ordered visible flow;
- prior data reuse;
- due/event/milestone/unresolved emphasis;
- explicit Close;
- baseline/audit schema remains underneath.

## Stage PILOT — Five real system-assisted encounters

Deliverables:

- usability/capture/guidance evidence;
- one deliberate refinement;
- contract freeze.

## Stage PR-3 — Quick Practice Review shadow validation

Deliverables:

- review engine using structured encounter + transcript candidates/accepted data + audit + evidence;
- structured PracticeObservations;
- evidence/provenance;
- no routine performance feedback during scored baseline.

## Stage BASELINE — 30 scored system-assisted encounters

Deliverables:

- stable Guidance/Capture contract;
- denominator-aware baseline;
- guidance-exposure context where feasible;
- baseline lock.

## Stage PR-4/5/6 — Review → Signal → Intervention

Deliverables:

- clinician disposition;
- longitudinal pattern aggregation;
- gap classification;
- sustained-strength detection;
- Signal promotion;
- targeted intervention.

## Stage RE-MEASURE

Deliverables:

- later encounters assess whether targeted behavior improved/persisted;
- prompt-dependence trend assessed only when technically valid;
- at least one real improvement loop closed.

## Later stages

Deep Review/Red Team breadth, Patient Voice, full Home, benchmarking, Calendar/Secretary integration and Module 02 generalization remain later unless evidence elevates them.

---

# 36. Phase exit criteria / Module 01 closure evidence

This phase is sufficiently complete when:

- critical encounter finalization integrity is production-smoke verified;
- dynamic Clinical Guidance can construct a visit-relevant flow from structured longitudinal context;
- transcript extraction safely reduces duplicate manual entry;
- transcript candidates can be reviewed inline before authoritative write;
- five real system-assisted pilot encounters have informed one deliberate refinement;
- Guidance/Capture/KPI applicability contracts are frozen after pilot;
- 30-case scored system-assisted baseline is locked or an explicit methodology revision is approved;
- Quick Practice Review can produce structured evidence-traceable observations;
- clinician can accept/modify/dismiss important observations;
- repeated observations become denominator-aware Signals;
- Signals trigger root-cause-appropriate interventions;
- later encounters show whether at least one targeted improvement persisted;
- the visible osteoporosis workflow is informed by real use rather than a universal static checklist;
- a fresh conversation can reconstruct project truth entirely from the six canonicals without chat history.

Only then may `MODULE 01 CLOSED` be declared.
