# CLINICAL_EXCELLENCE_PLAN.md — Blueprint v1

> **STATUS:** ACTIVE detailed phase plan.
> **PHASE:** Clinical Excellence Core architecture + Module 01 Osteoporosis blueprint.
> **CANONICAL HOME:** `athpapachr-cmd/osteoporosis`.
> **UPDATED:** 2026-08-22 Asia/Nicosia.

---

# 1. Purpose

Build a reusable **Personal Clinical Excellence System** in which the existing Osteoporosis Cockpit becomes the point-of-care execution component of a larger continuous-improvement loop.

Module 01 is **Osteoporosis**. The reusable Core Engine must later support additional domains such as low-back pain, neck pain, knee pain, hip pain and shoulder pain without duplicating the architecture.

The system is designed to answer four questions continuously:

1. **Where is current performance strong or weak?**
2. **Why is it strong or weak?**
3. **What action should follow?**
4. **Did the action actually improve practice?**

Canonical loop:

```text
STANDARD
→ LEARN
→ TEST / MASTER
→ APPLY
→ MEASURE
→ AUDIT
→ GAP OR STRENGTH
→ INTERVENE / REINFORCE
→ RE-MEASURE
→ SYSTEM LEARNS
```

---

# 2. Current phase non-goals

This phase does **not** yet aim to:

- rewrite the existing Cockpit;
- finalize patient leaflets;
- build all future musculoskeletal modules;
- create a production patient-data platform;
- generate a fake overall excellence score before baseline data exists;
- automate clinical decisions without transparent rules and clinician review.

The output of this phase is an approved architecture and measurement foundation that can safely drive implementation.

---

# 3. System architecture

```text
PERSONAL CLINICAL EXCELLENCE SYSTEM
│
├── CORE ENGINE
│   ├── Standards
│   ├── Evidence / Guidelines
│   ├── Learning
│   ├── Testing / Mastery / Calibration
│   ├── Clinical Practice
│   ├── Patient Voice
│   ├── Audit / Measurement
│   ├── Safety
│   ├── Benchmarking
│   ├── Improvement
│   ├── Signal Engine
│   └── Personal Adaptation
│
├── MODULE 01 — OSTEOPOROSIS
│   └── existing Osteoporosis Cockpit + new domain content
│
├── MODULE 02 — future
├── MODULE 03 — future
└── ...
```

Core owns the reusable mechanics. Clinical modules own domain-specific standards, rules, evidence, competencies, cases, KPIs and workflows.

---

# 4. Core domain objects

The first implementation should be built around explicit objects rather than ad-hoc fields.

## 4.1 `Signal`

The central event object that connects the system.

Suggested fields:

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

Sources may include Clinical Encounter, Patient Feedback, Audit, Learning/Test, Evidence, Safety or Benchmark.

## 4.2 `ClinicalStandard`

Defines what good practice means.

```text
standard_id
module
domain
statement
applicability
source_framework
source_version
strength / certainty if available
effective_from
reviewed_on
status
```

## 4.3 `Competency`

Defines what the clinician should know or be able to do.

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

May represent a guideline, consensus, article, book chapter, lecture, course or authoritative resource.

```text
evidence_id
type
title
source/authors/provider
year/version
url/reference
module/domain
priority
status
impact_classification
key_messages
practice_implications
reviewed_on
next_review_due
```

## 4.5 `LearningItem`

Tracks the educational object and learning state.

```text
learning_id
type: paper / chapter / guideline / course / seminar / case / video / podcast
linked_evidence_id
priority
assigned_reason
status: unread / studied / tested / mastered
started_at
completed_at
mastery_score
next_review_due
linked_gap_or_project
```

## 4.6 `AssessmentItem` / `AssessmentAttempt`

Supports MCQs, clinical cases, open questions and image interpretation.

Attempt data should include:

```text
correctness / score
confidence_before_answer
response_time if useful
error_type
linked_competency
linked_case/signal
```

High-confidence errors become high-priority signals.

## 4.7 `ClinicalEncounter`

Represents a real clinical episode at a structured, privacy-safe application layer.

It should link to:

- applicable standards;
- completed/omitted/overridden items;
- clinical decisions;
- tasks/follow-up;
- patient feedback;
- generated signals.

## 4.8 `ClinicalDecision`

```text
decision_id
encounter_id
question / decision point
options considered
decision
rationale
framework(s) considered
recommendation_concordance
override: yes/no
override_reason
uncertainty/confidence
follow-up criterion that could change the decision
```

## 4.9 `PatientFeedback`

Core categories:

```text
understanding_condition
understanding_plan
understanding_rationale
questions/preferences_addressed
free_text
linked_encounter
```

Repeated patterns can generate system-level Signals.

## 4.10 `AuditMetric`

Every KPI must have a formal definition.

```text
metric_id
title
domain
definition
numerator
denominator
exclusions
target
source / standard
measurement_period
sample_size
current_value
baseline_value
trend
reliability
data_completeness
```

## 4.11 `Benchmark`

```text
benchmark_id
metric_id
source
country
population
setting
year
definition
value
comparability: high / moderate / low / context_only
notes
```

## 4.12 `SafetyEvent`

Types:

- error;
- near miss;
- unsafe omission;
- potential failure mode / FMEA item.

Should preserve root cause, action and reassessment.

## 4.13 `ImprovementProject`

```text
project_id
problem
signal_source
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

## 4.14 `ActionTask`

A generic actionable unit linking learning and clinical operations.

```text
task_id
type
source
priority
due_date
owner
status
completed_at
result
linked_signal/project/encounter
```

---

# 5. Signal Engine

The Signal Engine is the primary adaptive mechanism.

```text
EVENT
  ↓
SIGNAL CREATED
  ↓
CLASSIFY SOURCE / DOMAIN / DIRECTION
  ↓
ASSESS IMPORTANCE + RELIABILITY
  ↓
LINK TO STANDARD / COMPETENCY / KPI / PROJECT
  ↓
ACTION NEEDED?
  ├─ no → retain for trend/strength evidence
  └─ yes
       ↓
     ROOT-CAUSE CLASSIFICATION
       ↓
     INTERVENTION
       ↓
     REASSESSMENT / RE-AUDIT
       ↓
     CLOSE OR ITERATE
```

Important patterns to detect:

- repeated omissions;
- high-confidence learning errors;
- patient-feedback clusters;
- sustained strengths;
- declining performance after previous mastery;
- new evidence affecting active standards;
- overdue follow-up/safety tasks;
- conflict between clinical decision and framework;
- apparent score improvement caused by denominator/data-definition changes.

---

# 6. Gap classification algorithm

Every important negative signal should be classified into one or more root-cause families before recommending an intervention.

## 6.1 Knowledge gap

The clinician does not know or retain the required information.

Preferred response:

```text
short targeted reading/course
→ test
→ spaced repetition
→ clinical case
→ re-measure
```

## 6.2 Reasoning gap

Information is available, but interpretation/decision quality is weak.

Preferred response:

```text
case review
→ challenge / red-team analysis
→ compare frameworks / expert reasoning
→ deliberate practice
→ blinded re-review when useful
```

## 6.3 Execution gap

The clinician knows what should happen but does not execute it reliably.

Preferred response:

```text
workflow analysis
→ Cockpit/checklist/task change
→ reduce friction
→ measure compliance
```

## 6.4 Communication/system gap

Correct clinical thinking does not reliably reach the patient, record, follow-up chain or next professional.

Preferred response:

```text
teach-back / handoff redesign / documentation support
→ patient feedback
→ process audit
```

---

# 7. Strength loop

Positive performance is not simply marked green.

```text
POSITIVE SIGNAL
→ repeated?
→ stable over time?
→ adequate sample/data completeness?
→ audit confirmation?
→ external comparison if valid?
→ SUSTAINED STRENGTH
```

Actions for a sustained strength:

- preserve the successful process;
- decrease basic repetition;
- increase case difficulty;
- benchmark externally where meaningful;
- periodic surveillance;
- consider translating the pattern into a reusable best-practice protocol.

---

# 8. Module 01 — Osteoporosis competency map v0.1

Initial domains:

1. **Diagnosis & case finding**
2. **DXA / VFA / imaging**
3. **Fracture-risk assessment**
4. **Secondary osteoporosis & laboratory evaluation**
5. **Pharmacologic treatment selection**
6. **Sequential therapy / treatment transitions**
7. **Monitoring / treatment response / adherence**
8. **Falls, frailty, exercise & nutrition**
9. **Communication / shared decision making / continuity**

Each domain must eventually connect:

```text
Standards
↔ Evidence
↔ Learning
↔ Tests/Cases
↔ Clinical encounters
↔ Patient feedback
↔ KPIs/Audit
↔ Benchmarks
↔ Improvement projects
```

This taxonomy is provisional and should be revised when the first complete standards/competency inventory is built.

---

# 9. Existing Cockpit position inside Module 01

The current `index.html` + `main.py` Cockpit provides substantial point-of-care structure. It is not discarded.

It becomes the **Clinical Practice / Encounter Execution layer** of Module 01.

Planned architectural corrections before deeper integration include:

- separate guideline frameworks rather than a hybrid risk engine;
- remove unvalidated custom risk scores from treatment-decision authority;
- rebuild longitudinal DXA around BMD/LSC/scanner comparability rather than arbitrary T-score deltas;
- structure fractures as events;
- structure treatment episodes and administrations with exact dates;
- convert follow-up into due-date tasks;
- add visit audit/coverage logic;
- replace brittle evidence string matching with rule-level evidence metadata;
- add privacy/authentication controls before identifiable patient use.

These corrections belong to later implementation slices after the current blueprint/baseline design is approved.

---

# 10. Clinical Excellence Home — wireframe v0.1

The Home screen should answer within roughly 20–30 seconds:

- Where am I now?
- What improved?
- What is the biggest current gap?
- What needs action today?

Suggested top-level structure:

```text
CLINICAL EXCELLENCE

Overall state: Baseline / Improving / Stable / Needs attention
Overall score: unavailable until baseline is adequate
Improvement velocity
Current strongest domain
Current priority gap
Evidence freshness

WHAT NEEDS MY ATTENTION?        MY DEVELOPMENT
Clinical safety                 Knowledge Mastery
Overdue clinical tasks          Guideline Mastery
Practice gaps                   Clinical Execution
Learning due                    Decision Quality
Evidence updates                Continuity
Re-audits due                   Patient Communication
                                Safety
                                Evidence → Practice
                                Calibration

TODAY'S LEARNING                CLINICAL PRACTICE
Adaptive questions              Encounter performance
Case linked to recent gap       Key KPI gaps
Article/course linked to gap    Data completeness
Estimated time                  Current run-chart movement

ACTIVE IMPROVEMENT PROJECTS     AUDIT & SAFETY
Baseline → current → target     Last audit
Interventions                   Omissions / near misses
Next re-audit                   Safety actions

KNOWLEDGE & EVIDENCE            COURSES & EVENTS
Guidelines current/review due   Online + in-person
Articles pending appraisal      Relevance to active gaps
Knowledge debt                  Registration/deadline
```

---

# 11. Progress bars and measurement model

The dashboard should show multiple domain bars plus an eventual overall summary.

Candidate domains:

- Knowledge Mastery
- Guideline Mastery
- Clinical Practice / Execution
- Decision Quality
- Continuity of Care
- Audit & Safety
- Evidence → Practice Translation
- Patient Understanding / Communication
- Calibration

For each score preserve:

```text
current
baseline
change
trend
sample size / denominator
reliability
standard / target
benchmark + comparability when available
```

Progress bars show **current state**. Run charts show **trajectory**.

Potential later metric:

```text
Improvement velocity = longitudinal rate of improvement
```

Do not interpret plateau mechanically; distinguish a stable high-performing domain from an unresolved stagnant gap.

---

# 12. Baseline Osteoporosis Audit v1 — required next design artifact

Before claiming improvement, define and perform an initial audit over an appropriate sample of consecutive or clearly sampled cases.

The audit specification must include:

- sampling method;
- inclusion/exclusion criteria;
- exact KPI dictionary;
- numerator/denominator for each metric;
- data-completeness rules;
- treatment of `not applicable`;
- minimum sample-size / reliability display rules;
- baseline lock date;
- re-audit interval.

Initial candidate audit domains:

- fracture history/documentation;
- DXA interpretation completeness;
- vertebral-fracture/VFA consideration when indicated;
- fracture-risk assessment framework/documentation;
- secondary-cause evaluation;
- treatment-history completeness;
- treatment decision/rationale;
- adherence/adverse-effect assessment;
- falls/frailty assessment;
- follow-up plan and due tasks;
- patient understanding/communication where data are available.

The final Baseline Audit v1 specification is the next major design task after review of this blueprint.

---

# 13. Learning engine

Learning state is not a binary completed checkbox.

Canonical progression:

```text
UNREAD
→ STUDIED
→ TESTED
→ MASTERED
→ RETENTION CHECK
```

Adaptive learning should use:

- spaced repetition;
- recent clinical signals;
- high-confidence errors;
- active improvement projects;
- new evidence affecting practice;
- weak domains;
- advanced cases for sustained strengths.

Suggested cadence remains configurable, but the engine should support:

- brief weekly question sets;
- monthly case-based assessment;
- periodic larger milestone assessment;
- complex open clinical cases;
- case-triggered learning after real encounters.

---

# 14. Evidence-to-practice lifecycle

```text
NEW EVIDENCE
→ relevance
→ authority/quality
→ impact classification
→ does it change a standard/rule?
  ├─ no → supportive/archive/current evidence base
  └─ yes
      ↓
    identify affected objects
      ↓
    review/approve change
      ↓
    implement in Cockpit/curriculum/patient communication/KPI
      ↓
    re-measure practice
```

The system should track **Evidence Responsiveness** without rewarding reflexive adoption of every new publication.

---

# 15. Patient Voice loop

Patient feedback should be capable of producing system change.

Example:

```text
patient reports treatment rationale unclear
→ PatientFeedback
→ Signal
→ look for repeated pattern
→ communication gap confirmed
→ targeted evidence/learning + handoff/teach-back change
→ measure next cohort
→ retain or revise intervention
```

A single comment may remain an individual signal; repeated similar feedback may become a system-level Improvement Project.

---

# 16. Personal adaptation

The system should fit the working style of a clinician who values precision, challenge, explicit reasoning and continuous refinement.

Supported modes:

- **STANDARD** — routine support;
- **CHALLENGE** — identify omissions/alternatives;
- **RED TEAM** — strongest evidence-based counter-case;
- **LEARNING** — convert encounter into deliberate practice.

Design safeguards:

- high-confidence errors receive priority;
- reasoned overrides are recorded, not automatically punished;
- improvements are ranked by clinical importance;
- distinguish critical defect from optional refinement;
- stop expanding validation/scope once evidence is sufficient for the approved objective.

---

# 17. Repository structure direction

Current root files remain:

```text
index.html
main.py
osteoporosis-qa-handout.html
Dockerfile
requirements.txt
```

Canonical control files are now root-level by design.

As implementation expands, prefer organized directories rather than root-file sprawl. Candidate future structure:

```text
core/
modules/osteoporosis/
docs/evidence/
docs/audits/
docs/benchmarks/
docs/learning/
archive/
tests/
```

Do not move current runtime files merely for aesthetics during the Blueprint phase. Repository reorganization should be a deliberate implementation step with working deployment preserved.

---

# 18. Privacy/security boundary

The GitHub repository is public and therefore must contain no identifiable patient data.

Before any production persistence of identifiable clinical information, the implementation plan must explicitly cover:

- authentication;
- authorization / access control;
- encrypted transport/storage as applicable;
- audit logging;
- secrets management;
- data minimization;
- GDPR/privacy requirements;
- retention/deletion policy;
- separation of public code from private clinical data.

---

# 19. Current phase deliverables

This phase exits only when the following are reviewed/approved:

- [x] canonical five-file control plane created in the Osteoporosis repository;
- [x] Core Engine architecture documented;
- [x] Signal-first feedback model documented;
- [x] first Osteoporosis competency taxonomy documented;
- [x] Home Dashboard wireframe documented;
- [x] transparent score/run-chart principles documented;
- [ ] Core object schema v1 reviewed and frozen enough for implementation;
- [ ] Osteoporosis competency map expanded into standards/competencies;
- [ ] Baseline Osteoporosis Audit v1 specification completed;
- [ ] KPI dictionary v1 completed;
- [ ] first dashboard data contract defined;
- [ ] implementation sequence approved.

**Next major design action:** create **Baseline Osteoporosis Audit v1 + KPI Dictionary v1** while refining the Core object schema only where the audit requires it.
