# CLINICAL_EXCELLENCE_PLAN.md — Clinical Excellence architecture v2

> **STATUS:** ACTIVE detailed phase plan.
> **PHASE:** Baseline measurement foundation → Clinical Practice Review / learning-health loop.
> **CANONICAL HOME:** `athpapachr-cmd/osteoporosis`.
> **MODULE 01:** Osteoporosis.
> **UPDATED:** 2026-08-25 Asia/Nicosia.

This document owns the **detailed phase architecture and stage sequence**. Operational PR/branch/deploy state belongs in `CURRENT_OPERATIONAL.md`; exact current-slice design belongs in `SLICE_PLAN_CURRENT.md`.

---

# 1. Purpose

Build a reusable **Personal Clinical Excellence System** whose central outcome is not merely better documentation or higher audit scores, but measurable improvement in the clinician’s real practice:

- what is asked;
- what is noticed;
- how evidence is interpreted;
- how clinical decisions are made;
- how uncertainty is handled;
- what is explained to the patient;
- how preferences alter the plan;
- how follow-up is closed;
- what is learned from each encounter;
- whether the change persists in later practice.

Module 01 — Osteoporosis is the proving ground for a Core Engine that can later generalize to other clinical domains.

Canonical loop:

```text
STANDARD
→ LEARN
→ TEST / MASTER
→ APPLY
→ MEASURE
→ AUDIT
→ CLINICAL PRACTICE REVIEW
→ GAP OR STRENGTH
→ INTERVENE / REINFORCE
→ RE-MEASURE
→ SYSTEM LEARNS
```

The system should continuously answer:

1. Where is performance strong or weak?
2. Why?
3. What should change next?
4. Did the change actually improve future encounters?

---

# 2. Core architectural distinction — measurement, review and intervention

Three instruments must remain distinct.

## 2.1 Audit

Purpose:

> Determine whether an applicable defined process/standard was met.

Properties:

- deterministic where possible;
- explicit numerator/denominator/applicability;
- transparent missing/N/A handling;
- no black-box score;
- neutral during baseline collection;
- suitable for longitudinal measurement and re-audit.

## 2.2 Clinical Practice Review

Purpose:

> Critically review how the consultation was actually conducted and identify the highest-value improvement opportunities and strengths.

It can evaluate dimensions not captured well by a binary KPI, including reasoning sequence, communication accuracy, option framing, consultation flow, information overload, uncertainty and whether the final plan logically follows the available evidence.

## 2.3 Learning / Improvement

Purpose:

> Convert accepted review/audit Signals into the correct intervention and later test whether it worked.

The intervention depends on root cause rather than on the surface finding.

```text
knowledge gap → targeted learning/testing
reasoning gap → case challenge/red team/deliberate practice
execution gap → workflow/interface/task change
communication/system gap → communication/process redesign
strength → preserve/reinforce/advance
```

---

# 3. Reusable Core architecture

```text
PERSONAL CLINICAL EXCELLENCE SYSTEM
│
├── Standards / Competencies
├── Evidence / Guidelines
├── Learning / Mastery / Calibration
├── Clinical Practice / Encounters
├── Transcript-assisted Capture
├── Clinical Practice Review
├── Audit / Measurement
├── Signal Engine
├── Patient Voice
├── Safety
├── Benchmarking
├── Improvement Projects
├── CareTasks / Continuity
└── Clinical Excellence Home

MODULE 01 — OSTEOPOROSIS
└── domain standards, schemas, workflows, cases, metrics, evidence
```

Core owns reusable mechanics. Module 01 owns osteoporosis-specific content.

---

# 4. Core domain objects

Existing architectural objects remain valid and are extended rather than replaced.

## 4.1 `Signal`

```text
id
created_at
source_type
module
domain
direction: positive / negative / neutral / warning
importance: low / moderate / high / critical
confidence / reliability
summary
details
requires_action
linked_object_ids
status: new / reviewed / actioned / reassessed / closed
resolution
```

Signal sources include encounter, audit, Practice Review, patient feedback, learning/test, evidence, safety and benchmark.

## 4.2 `ClinicalStandard`

```text
standard_id
module
domain
statement
applicability
source_framework
source_version
strength_or_certainty
reviewed_on
status
```

## 4.3 `Competency`

```text
competency_id
module
domain
title
description
level: core / advanced / specialist
linked_standard_ids
linked_learning_ids
linked_assessment_ids
```

## 4.4 `EvidenceItem`

```text
evidence_id
type
title
source/authors/provider
year/version
reference
module/domain
priority
impact_classification
key_messages
practice_implications
reviewed_on
next_review_due
status
```

## 4.5 `LearningItem`

```text
learning_id
type
linked_evidence_id
assigned_reason
priority
status: unread / studied / tested / mastered / retention_due
mastery_score
next_review_due
linked_signal_or_project
```

## 4.6 `AssessmentAttempt`

```text
assessment_id
attempt_id
score/correctness
confidence_before_answer
response_time_optional
error_type
linked_competency
linked_signal
```

High-confidence errors are high-priority learning Signals.

## 4.7 `ClinicalEncounter`

Represents a real consultation and links structured clinical data, decisions, tasks, audit state, Practice Review, patient feedback and Signals.

## 4.8 `ClinicalDecision`

```text
decision_id
encounter_id
decision_question
options_considered
clinician_recommendation
final_decision
patient_acceptance
rationale
frameworks_considered
uncertainty/confidence
override_yes_no
override_reason
future_criterion_that_could_change_decision
```

## 4.9 `AuditMetric`

Formal KPI object preserving numerator, denominator, exclusions, target, baseline, current result, trend, sample size, reliability and completeness.

## 4.10 `ImprovementProject`

```text
project_id
problem
signal_sources
baseline
target
root_cause_class
interventions
started_at
re_audit_due
results
next_action
status
```

## 4.11 `ActionTask` / later `CareTask`

Generic due-date action object. Clinical tasks remain conceptually separate from scheduled appointments.

---

# 5. New Core object — `PracticeReview`

A Practice Review is an AI-assisted but clinician-governed critical review of one encounter.

Suggested v1 contract:

```text
review_id
encounter_id
module
review_mode: quick / deep / red_team / learning
source_set
created_at
review_status: generated / clinician_reviewed / finalized
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

`source_set` may include:

```text
structured encounter data
Heidi transcript (ephemeral source)
audit/KPI calculation
evidence/standards
patient feedback later
follow-up outcome later
```

Practice Review is not itself a KPI score.

---

# 6. New Core object — `PracticeObservation`

Each important review statement becomes an explicit observation rather than untraceable prose.

```text
observation_id
review_id
dimension
direction: strength / gap / safety / uncertainty
root_cause_class_optional
importance: low / moderate / high / critical
confidence
statement
why_it_matters
encounter_provenance
linked_standard_ids
linked_evidence_ids
suggested_change
suggested_intervention_type
clinician_disposition: pending / accepted / modified / dismissed
clinician_note_optional
linked_signal_id_optional
```

Important rule:

> A Practice Review observation should never present model opinion as established clinical truth merely because the model is confident.

For material claims, preserve encounter evidence/provenance and relevant external standard/evidence where available.

---

# 7. Practice Review dimensions

The first review engine should reason across at least nine dimensions.

## 7.1 Clinical completeness

Was clinically important history/examination/investigation/monitoring omitted or left unresolved for this archetype?

## 7.2 Clinical reasoning

Did the sequence of inference make sense? Were downstream calculations/decisions made before prerequisite facts were adequately characterized?

## 7.3 Decision quality

Were reasonable options considered, contraindications/safety reviewed, uncertainty acknowledged and the final choice supported by the available evidence?

## 7.4 Risk interpretation

Were DXA, fracture classification, FRAX/FRAXplus, VFA, treatment-response or other risk data interpreted correctly and reproducibly?

## 7.5 Safety

Did the consultation contain an unsafe omission, inaccurate high-consequence statement, medication timing risk or unresolved critical issue?

## 7.6 Communication accuracy

Were patient-facing statements scientifically accurate, proportionate and understandable? Detect overstatement, understatement and misleading certainty.

## 7.7 Shared decision making

Were patient priorities/preferences elicited? Did they materially influence the plan? Was “discussed option” kept distinct from final accepted decision?

## 7.8 Consultation flow / efficiency

Was the encounter coherent or repeatedly cycling between risk/results/treatment? Was information density excessive? Could the same clinical work be performed with less cognitive burden?

## 7.9 Follow-up execution

Did the consultation close with explicit prerequisites, actions, owner, timing, communication plan and unresolved items?

---

# 8. Review modes

## 8.1 Quick Post-Visit Review

Target reading time: approximately 2–3 minutes.

Default output:

```text
3 strengths
3 highest-value improvements
safety concern if present
1 reasoning issue
1 communication issue
1 concrete change for the next similar encounter
```

The goal is behavior change, not exhaustive commentary.

User actions:

```text
Accept Signal
Modify
Dismiss
Challenge me
Create Learning
```

## 8.2 Deep Review

Reconstruct the encounter in more detail, identify decision points and compare reasoning with standards/evidence.

## 8.3 RED TEAM

Assume the current clinical decision may be wrong and build the strongest evidence-based counter-case, then compare the counter-case with the actual decision.

## 8.4 LEARNING

Turn the encounter into a structured educational exercise: questions, uncertainty points, targeted reading, case challenge and retention plan.

---

# 9. Decision Reconstruction — anti-hindsight architecture

The review engine should reconstruct the actual decision path:

```text
DATA AVAILABLE AT TIME T
↓
INTERPRETATION
↓
OPTIONS CONSIDERED
↓
NEW INFORMATION
↓
OPTION ACCEPTED / REJECTED / DEFERRED
↓
PATIENT PREFERENCE
↓
FINAL DECISION
```

The critical question is:

> Given only the information available at that point, what decision was reasonable?

This reduces hindsight bias and prevents later results from making an earlier reasonable decision look retrospectively irrational.

---

# 10. Communication Review architecture

Communication review should assess:

```text
scientific accuracy
clarity
risk framing
overstatement / understatement
terminology
certainty vs uncertainty
information density
unnecessary repetition
patient questions
preference elicitation
teach-back
written information
```

The engine should be capable of distinguishing:

- “information was provided”;
- “information was accurate”;
- “information was understood”;
- “patient preference influenced the decision”.

These are not interchangeable.

---

# 11. Longitudinal Practice Review

Single-case criticism is insufficient. The system should aggregate reviewed observations across encounters.

Example output only after adequate denominators:

```text
fracture characterization incomplete: 5/15
VFA consideration missed when applicable: 3/8
explicit treatment rationale: 14/15
patient preference elicited: 13/15
explicit follow-up timing: 9/15
teach-back used: 1/15
```

The engine then asks whether the pattern is:

```text
isolated event
recurrent gap
improving trend
stable weakness
sustained strength
```

No stable pattern should be declared from inadequate sample size without a reliability warning.

---

# 12. Gap classification and intervention

Every important accepted negative observation should be mapped to the existing four-class root-cause model.

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
→ alternative framework/expert reasoning
→ deliberate practice
→ future blinded re-review when useful
```

## Execution gap

```text
workflow analysis
→ reduce friction
→ Cockpit/consultation-flow/task change
→ re-measure execution
```

## Communication/system gap

```text
wording / teach-back / handoff / documentation / process redesign
→ patient feedback or process audit
→ re-measure
```

Positive observations can become sustained-strength Signals after repeated evidence.

---

# 13. Transcript-assisted capture — purpose and boundary

Heidi transcript paste can reduce manual data entry and supply richer review evidence, but transcript text must not become unreviewed patient truth.

Canonical flow:

```text
PASTE HEIDI TRANSCRIPT
        ↓
EPHEMERAL PROCESSING
        ↓
STRUCTURED EXTRACTION CANDIDATES
        ↓
CLINICIAN REVIEW / EDIT / ACCEPT / REJECT
        ↓
ACCEPTED NORMALIZED ENCOUNTER DATA
        ↓
AUDIT + PRACTICE REVIEW
```

Raw transcript is not persisted by default.

---

# 14. Transcript extraction semantic categories

The extractor must preserve the distinction between:

```text
PATIENT / HISTORY FACT
OBJECTIVE RESULT
CLINICIAN INTERPRETATION
OPTION DISCUSSED
CLINICIAN RECOMMENDATION
PATIENT PREFERENCE
FINAL DECISION
PATIENT ACCEPTED
PATIENT DECLINED
PATIENT UNDECIDED
FOLLOW-UP TASK
UNCERTAIN / NEEDS REVIEW
```

This directly addresses a known failure mode of generic summaries: collapsing every discussed treatment into the final plan or converting a negative history statement into a negative laboratory result.

Extraction guardrails:

- preserve negation;
- preserve temporality;
- preserve speaker/source;
- preserve uncertainty;
- do not infer exact dates from vague timing;
- do not invent diagnosis/treatment/history/preferences;
- do not merge original FRAX with adjusted FRAXplus outputs;
- do not infer treatment acceptance from discussion alone.

---

# 15. Transcript review UX

Each candidate may show temporarily:

```text
field / object
candidate value
confidence
short source snippet
Accept
Reject
Edit
```

The evidence snippet is for the clinician review session and should not be stored by default after the raw transcript is discarded.

Accepted structured values retain provenance such as:

```text
source = heidi_transcript
clinician_reviewed = true
extraction_confidence = ...
```

An `Accept all high-confidence safe candidates` action is optional later and requires field-class safety thresholds; it is not the default first implementation.

---

# 16. Initial extraction scope

Start with more objective/high-yield information:

1. encounter reason/context;
2. anthropometrics;
3. fracture history/events;
4. falls/risk factors;
5. DXA/VFA/imaging;
6. laboratory values/dates/units where explicit;
7. treatment episodes/administrations;
8. treatment decision components;
9. follow-up tasks.

Only after reliable extraction is demonstrated add more interpretive domains such as patient understanding, communication quality and Practice Review inference.

---

# 17. Adaptive consultation-flow architecture

The existing Steps 1–6 are useful **storage/audit structure** but should not necessarily remain the visible clinical consultation sequence.

Permanent principle:

```text
CLINICAL WORKFLOW PRESENTATION
!=
STORAGE / AUDIT SCHEMA
```

The UI may present the same canonical data in the order that supports good clinical reasoning. Data should still be entered once and reused everywhere.

---

# 18. Candidate normalized osteoporosis consultation flow

For a broad assessment/review encounter:

```text
1. WHY TODAY / WHAT CHANGED?

2. FRACTURE + FALLS + FUNCTION
   new fractures?
   fracture mechanism/fragility classification
   falls / balance / frailty / functional change

3. DXA / VFA / IMAGING
   current DXA
   technical validity / longitudinal comparability
   vertebral-fracture/VFA consideration

4. SECONDARY CAUSES + LABS
   targeted history
   prior work-up adequacy
   current relevant labs
   unresolved causes/safety prerequisites

5. FRACTURE-RISK SYNTHESIS
   formal risk assessment
   framework/country model
   unadjusted vs adjusted outputs
   overall clinical risk/problem list

6. TREATMENT HISTORY / RESPONSE
   current/prior agent
   adherence/tolerance
   administration dates
   fracture on therapy / response

7. OPTIONS → RECOMMENDATION → PREFERENCE → DECISION
   options discussed
   safety/contraindications
   clinician recommendation
   patient preference
   final accepted/deferred decision

8. LIFESTYLE / COMMUNICATION
   exercise
   calcium/vitamin D/nutrition
   targeted additional counselling
   written information

9. CLOSE
   decision
   prerequisites
   patient tasks
   clinician tasks
   communication/results plan
   timing / next review
   unresolved critical item
   teach-back when appropriate
```

The purpose is not to force a rigid questionnaire; it is to reduce circular consultation flow and ensure reasoning prerequisites are resolved before downstream decisions.

---

# 19. Archetype-adaptive consultation flows

The visible flow should change according to the existing encounter archetype rather than forcing a new patient workflow onto stable follow-up.

## New / uncertain diagnosis

Emphasis:

```text
fracture characterization
DXA/VFA
aetiology/secondary causes
risk synthesis
treatment decision
education/close
```

## Known osteoporosis/osteopenia initial-to-service

Emphasis on verifying prior diagnosis/work-up, longitudinal DXA, treatment history and current management decision.

## Routine stable follow-up

Start with:

```text
interval fracture/falls/function
adherence/tolerance
administration/treatment status
due labs/DXA
response
next plan/close
```

Do not repeat full secondary-cause or condition education without a reason.

## Treatment start

Emphasize decision rationale, medication-specific safety/prerequisites, acceptance, administration instructions and monitoring.

## Treatment continuation/due monitoring

Emphasize adherence/tolerance, exact administration/timing, due monitoring and next due date.

## Treatment change/transition

Emphasize why change, prior last dose/end, sequencing safety, next agent/timing and tracking.

## Post-fragility fracture / fracture on treatment

Emphasize fracture verification/context, adherence/exposure, risk reassessment, secondary causes, treatment escalation/change and falls/function.

## Adverse effect/intolerance

Emphasize suspected relationship, severity, safety, alternative options and revised plan.

## Completion/consolidation

Emphasize response, course completion, consolidation/exit strategy and exact next timing.

---

# 20. Risk-synthesis gate

A major consultation-flow objective is to prevent downstream treatment reasoning from outrunning unresolved source facts.

Before final treatment selection, the interface should make the following synthesis visible:

```text
FRACTURE CHARACTERIZATION
+ DXA / VFA / IMAGING
+ SECONDARY CAUSES / LABS
+ FALLS / FUNCTION
+ FORMAL RISK ASSESSMENT
= CURRENT RISK / PROBLEM SYNTHESIS
```

Examples of issues the Practice Review engine should detect:

- fracture used as a risk modifier before fragility mechanism resolved;
- unclear femoral-neck vs total-hip BMD/T-score provenance;
- FRAX country/surrogate rationale unclear;
- adjusted FRAXplus output replacing rather than supplementing original FRAX;
- VFA indication not considered when relevant;
- medication decision made before a necessary safety prerequisite is resolved.

These observations may become learning or execution Signals; the workflow itself should not silently make the clinical decision.

---

# 21. Close card architecture

The encounter should end with a concise, explicit close rather than scattered plans across the transcript.

Suggested UI:

```text
Σήμερα αποφασίσαμε:
...

Πριν ξεκινήσει / prerequisites:
...

Εκκρεμεί:
...

Ο ασθενής πρέπει να κάνει:
...

Εμείς πρέπει να κάνουμε:
...

Αποτελέσματα / επικοινωνία:
...

Επόμενος επανέλεγχος:
...

Unresolved critical item: Yes / No
```

Optional final teach-back:

> “Πείτε μου με δικά σας λόγια τι θα κάνετε από εδώ και πέρα.”

This Close card should generate/normalize follow-up tasks rather than create duplicate free-text plans.

---

# 22. Baseline methodology and Practice Review exposure

Approved Baseline Audit sequence remains:

```text
5 usability/capture pilot encounters
→ one deliberate refinement
→ freeze form + KPI contract
→ 30 scored consecutive unique encounters
→ baseline lock
→ systematic intervention/re-audit
```

## 22.1 During 5-case pilot

Primary objective: capture usability, branching, persistence, timing and data interpretation.

Practice Review/transcript infrastructure may be tested for **engineering/design purposes**, but the pilot should record whether coaching was shown because that changes clinician exposure.

## 22.2 During 30-case scored baseline

Default policy:

- audit calculations may run in background;
- routine Practice Review coaching remains hidden;
- no red/green KPI prompts;
- safety-critical alert path may remain active;
- any unavoidable intervention exposure is recorded.

If the product owner intentionally activates systematic coaching before the 30-case baseline, the methodological baseline definition must be explicitly revised rather than pretending the cohort is an untouched pre-intervention baseline.

---

# 23. Existing Baseline Audit / KPI architecture remains authoritative

Steps 1–6 and the calculation contract remain the current measurement foundation.

Key rules retained:

- clinical process = Steps 1–5;
- formal documentation/capture trace = Step 6 separate axis;
- Heidi use itself is not a quality-success metric;
- missing formal documentation is not automatically a clinical omission;
- KPI 12/13 remain manual review where defined;
- Patient Voice remains the future source for KPIs 14/15;
- no live baseline score.

Reliability context remains denominator-aware; one safety failure may still generate a Signal even when sample size is small.

---

# 24. Existing post-pilot refinement backlog — preserved

The external review backlog remains candidate work **after pilot evidence**, not a mandate to expand the form before the five cases.

## 24.1 Encounter/adaptive architecture

- shared archetype registry;
- archetype-specific required-field gating where justified;
- background consistency flags;
- `other` archetype specifier;
- clarify new-patient vs sample semantics.

## 24.2 Fracture risk / FRAX

- completeness state;
- derive fracture number/recency from events;
- secondary-osteoporosis context;
- contextual adjustment reasons;
- optional TBS when actually used;
- distinguish framework output from management category;
- alcohol-unit definition;
- neutral framework/category coherence checks.

## 24.3 DXA / VFA / labs / function

- laboratory status/tri-state so blank is not ambiguous;
- units discipline including vitamin D and CTX;
- lowest T-score/diagnostic category;
- spine–hip discordance;
- derived VFA indication reasons;
- renal/mineral safety context;
- optional corrected calcium when justified;
- LSC-aware neutral trend descriptor;
- BTM timing/context;
- VFA-positive → fracture-event reconciliation;
- provenance hints;
- later sarcopenia depth only if useful.

## 24.4 Treatment / sequencing

- denosumab exit/delay safety derivation;
- next-due/overdue administration logic;
- renal/calcium/vitamin-D safety gates;
- post-anabolic consolidation;
- bisphosphonate duration/holiday review;
- treatment failure vs adherence-limited apparent failure;
- holiday restart trigger;
- anabolic duration limits;
- decision→episode/task linking;
- plan-complete vs unresolved-critical coherence;
- date reconciliation;
- decision confidence distinct from visit confidence.

## 24.5 Communication

- medication-specific applicability;
- completeness over applicable domains only;
- preserve teach-back as distinct evidence;
- preference-chain coherence;
- misunderstanding → communication Signal;
- information-type completeness;
- structured Step-5 Signal preparation;
- archetype-specific emphasis;
- free-text privacy reminders.

## 24.6 Documentation/provenance

- clinical-process-present read-only column;
- discrepancy hint when clinical process occurred but formal trace absent;
- missing-domain derivation;
- readiness/coherence gates;
- Heidi coherence;
- provenance auto-feed;
- objective completion-time fallback;
- taxonomy mapping;
- PII reminder consistency.

## 24.7 Cross-cutting

- central store helper when pilot evidence justifies refactor;
- shared registries;
- clear-on-collapse invariant;
- accessibility roles;
- unit metadata as first-class information.

Prioritization:

```text
safety/data loss
→ data interpretability
→ pilot usability
→ clinical safety derivations
→ provenance/reproducibility
→ structured Signals / Practice Review intelligence
→ polish
```

---

# 25. Evidence / standards / competency architecture

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

Each domain eventually links:

```text
Standards
↔ Evidence
↔ Competencies
↔ Learning
↔ Assessment
↔ Clinical encounters
↔ Audit
↔ Practice Review
↔ Patient Voice
↔ Signals
↔ Improvement
```

---

# 26. Evidence-to-practice lifecycle

```text
NEW EVIDENCE
→ relevance/authority/quality
→ impact classification
→ affected standard/rule?
→ clinician/system review
→ approved change
→ implementation in workflow/learning/communication/KPI
→ re-measure practice
```

Never silently hybridize incompatible guideline thresholds.

---

# 27. Learning / mastery engine

Canonical learning progression:

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

---

# 28. Patient Voice

Patient feedback should capture the patient’s own report of:

- understanding condition;
- understanding plan;
- understanding treatment rationale/duration/risks;
- whether questions/preferences were addressed;
- free-text confusion/concern/praise/suggestion.

Repeated themes can become Signals and Improvement Projects and should be re-measured after intervention.

---

# 29. Safety and privacy

The public repository must never contain identifiable patient data or raw real transcripts.

Production transcript handling must be designed so:

- raw transcript is transient by default;
- logs do not emit transcript text;
- accepted structured data enters protected clinical storage only after review;
- secret/API/provider configuration stays out of source;
- access, audit trail, retention and GDPR/privacy controls are treated explicitly.

Legacy public patient routes/CORS exposure remains a separate security-hardening concern; protection of `/clinical/*` alone is not whole-service compliance.

---

# 30. Clinical Calendar / CareTasks — independent deferred track

Calendar foundation exists, but live Digital Secretary/Setmore feed is paused.

Permanent architecture remains:

```text
Appointment = scheduled attendance
CareTask = clinical action that may exist without an appointment
```

Paused Calendar integration must not block transcript capture, Practice Review, audit, standards or learning work.

---

# 31. Clinical Excellence Home

The eventual Home should answer quickly:

1. Where am I?
2. What improved?
3. Biggest current gap?
4. What needs action today?

Candidate sections:

```text
Attention / safety / overdue care
Clinical Practice Review signals
Audit / run charts
Strengths and gaps
Learning due
Evidence freshness
Active Improvement Projects
Patient Voice themes
What the system learned this month
```

Do not build polished summary scores before the data contracts and baseline are sufficient.

---

# 32. Stage sequence for the new Practice Review program

## Stage PR-0 — Governance and data-contract design

Deliverables:

- six-canonical control-plane upgrade;
- `PracticeReview`, `PracticeObservation`, `DecisionReconstruction`, `CommunicationReview`, `WorkflowReview` contracts;
- transcript privacy/provenance contract;
- baseline exposure policy.

## Stage PR-1 — Transcript paste + structured candidate extraction

Deliverables:

- paste UI;
- ephemeral transcript endpoint;
- structured JSON extraction;
- no raw persistence/logging;
- objective-domain candidates first;
- validation against existing schemas.

## Stage PR-2 — Clinician review/acceptance workflow

Deliverables:

- candidate review screen;
- Accept/Reject/Edit;
- authoritative field merge;
- provenance and clinician-review state;
- conflict handling against existing values.

## Stage PR-3 — Quick Practice Review in shadow mode

Deliverables:

- review engine consuming structured encounter + transcript + audit + evidence;
- structured PracticeObservations;
- concise strengths/improvements/safety/reasoning/communication output;
- not shown routinely during scored baseline.

## Stage PR-4 — Deep Review / Red Team / Decision Reconstruction

Deliverables:

- chronological decision reconstruction;
- counter-case generation;
- evidence-linked critique;
- uncertainty/override handling.

## Stage PR-5 — Longitudinal pattern and Signal engine integration

Deliverables:

- aggregate reviewed observations;
- recurrence/reliability logic;
- gap-class assignment;
- sustained-strength detection;
- Signal promotion.

## Stage PR-6 — Intervention / Learning linkage

Deliverables:

- Signal → LearningItem / Challenge / workflow change / ImprovementProject;
- re-measurement due state;
- calibration where useful.

## Stage PR-7 — Adaptive consultation-flow presentation layer

Deliverables:

- archetype-specific visible workflow;
- same underlying canonical schema;
- risk-synthesis gate;
- explicit Close card;
- reduced duplicate entry.

Implement after pilot/review evidence identifies the highest-value workflow changes rather than redesigning the form from one transcript alone.

## Stage PR-8 — Patient Voice / outcome enrichment

Use patient feedback and follow-up outcome as additional Practice Review/Signal inputs without converting raw outcomes into simplistic competence penalties.

---

# 33. Phase exit criteria

This phase is sufficiently complete when:

- Baseline Form/KPI contract is frozen after pilot;
- 30-case scored baseline is locked or the methodology is explicitly revised;
- transcript extraction safely reduces duplicate entry;
- Practice Review can produce structured, evidence-traceable observations;
- clinician can accept/modify/dismiss them;
- repeated observations become denominator-aware Signals;
- Signals trigger root-cause-appropriate interventions;
- later encounters can show whether improvement persisted;
- adaptive osteoporosis workflow is informed by real evidence rather than one-off preference;
- a fresh conversation can reconstruct current project truth entirely from the six canonicals without chat history.
