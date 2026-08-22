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

# 12. Baseline Osteoporosis Audit v1 — active implementation state

The Baseline Audit design and KPI dictionary now exist and the Steps 1–6 prospective capture flow is implemented. The approved sequence is:

```text
pre-pilot hardening + smoke test
→ 5 consecutive pilot encounters
→ one deliberate usability/branching/calculation-contract refinement
→ freeze form + KPI applicability
→ 30 consecutive unique scored baseline cases
→ baseline lock
```

The audit specification includes sampling, inclusion/exclusion rules, KPI definitions, denominator/applicability rules, data completeness, reliability/sample-size conventions and separation of clinical process from formal documentation/capture quality.

During the 5-case usability pilot, the objective is not performance scoring. It is to verify capture completeness, branching, persistence, friction, timing and whether the field→KPI contract can classify cases without hidden clinician interpretation except where explicitly manual.

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
- [x] Baseline Osteoporosis Audit draft v1 + KPI Dictionary draft v1 defined;
- [x] field→KPI calculation contract defined for the pilot;
- [x] Steps 1–6 implemented;
- [x] pre-pilot hardening P1–P8 completed;
- [ ] final pre-pilot data-quality additions (`labs_date`, Step 6 conflict clear-on-collapse) deployed and smoke-tested;
- [ ] 5-case usability pilot completed;
- [ ] one deliberate post-pilot refinement completed;
- [ ] Baseline Form v1 + KPI applicability/calculation contract frozen;
- [ ] 30-case scored baseline completed and locked;
- [ ] Core object schema v1 reviewed and frozen enough for broader implementation;
- [ ] Osteoporosis competency map expanded into explicit standards/competencies;
- [ ] first dashboard data contract defined.

**Current next action:** complete `labs_date` + Step 6 conflict clear-on-collapse, deploy, run the explicit synthetic smoke test, then begin Pilot Case 1/5 on the next clinical workday if the smoke test passes.

---

# 20. Baseline Audit improvement backlog from external review

This section consolidates the complete external code/design review into one ordered backlog. It does **not** mean every item should be implemented before the 5-case pilot. The pilot remains the evidence gate for form burden and usability.

## 20.1 Pre-pilot data quality / integrity

Implement before the real pilot when small and low-risk:

- `labs_date` in Step 3 as a native calendar/date input so entered laboratory values can be distinguished as current vs historical;
- Step 6 source-conflict dependent fields (`conflict_resolution`, `conflict_note`) visible only when conflict is `yes`, with clear-on-collapse before persistence;
- preserve the completed P1–P8 regression protections: hidden-data hygiene, module ownership on save, whole-form progress, Step1→Step3 single source of truth, DXA machine normalization, inline Prior DXA/stable deletion, archetype applicability, BMI source behavior.

The smoke test is the stop/go gate. No additional feature should delay the pilot unless it reveals a safety, data-loss or material data-quality defect.

## 20.2 First deliberate post-pilot refinement — highest-priority candidates

Use the 5 pilot encounters to decide which of these are justified by observed friction or ambiguity:

### Encounter/adaptive architecture
- shared archetype registry: one canonical object for labels, context text, applicability, required fields and later consistency rules;
- archetype-specific required-field gating only where needed for capture completeness;
- background consistency flags for impossible/suspicious combinations;
- free-text specifier for `other` archetype;
- clarify `sample-first` versus `new patient` semantics.

### Step 2 — fracture risk / FRAX reproducibility
- FRAX input-completeness derived state;
- derive number/recency of prior fractures from structured events;
- derive secondary-osteoporosis context where appropriate, including later early-menopause detail;
- prefill contextual-adjustment reasons from existing structured/derived inputs while preserving clinician control;
- optional TBS field/context when actually used;
- explicitly distinguish framework output from overall management risk category;
- alcohol-unit definition/tooltip;
- neutral framework/category coherence checks only after denominator semantics are frozen.

### Step 3 — results/data quality
- tri-state/status model for laboratory panels so blank does not ambiguously mean not done/not available/not entered;
- units discipline, especially 25-OH vitamin D (`ng/mL` vs `nmol/L`) and CTX units;
- lowest T-score + diagnostic-category derived context;
- spine–hip discordance derived flag;
- VFA indication reasons derived/prefilled from existing height-loss, GC, T-score and vertebral-fracture context;
- renal/calcium/Vit-D safety background context;
- optional albumin/corrected-calcium support if justified;
- LSC-aware neutral trend descriptor;
- BTM timing/months-on-treatment context;
- VFA-positive result reconciliation with structured fracture events;
- provenance hints from Step 3 to Step 6;
- later EWGSOP2 staging only if the pilot shows sarcopenia depth is useful and muscle-mass data are available.

### Step 4 — treatment, sequencing and safety
- denosumab exit/delay safety derivation;
- administration next-due-date and overdue derivation;
- cross-step renal/calcium/Vit-D safety gates;
- post-anabolic consolidation signal;
- bisphosphonate duration/holiday review points;
- on-treatment fracture with adequate adherence → reassessment/failure candidate signal;
- distinguish adherence-limited apparent failure from true treatment failure;
- holiday review/restart trigger;
- anabolic duration limits;
- decision→episode linking and decision→task expectations where this reduces duplicate entry;
- reason↔decision coherence checks;
- `plan_complete` vs `unresolved_critical` contradiction flag;
- date reconciliation/validation across episodes, administrations and transitions;
- explicitly distinguish decision confidence from overall-visit confidence;
- keep ONJ/atypical-femur counseling in communication rather than duplicating it as Step-4 treatment state.

### Step 5 — communication / early Signal wiring
- medication/agent-specific applicability for medication plan, reason, alternatives, duration/timing, safety, missed dose and sequencing communication;
- communication completeness only over applicable items;
- preserve teach-back as a distinct evidence marker;
- preference-chain coherence and conditional `preferences_influenced_plan`;
- neutral teach-back/understanding coherence signal;
- unresolved misunderstanding → communication-signal candidate;
- information-given without information-type → capture-completeness flag;
- structured Step-5 Signal object compatible with the future Signal Engine;
- archetype-specific communication emphasis;
- privacy reminder consistency on free-text fields;
- explicit distinction between decision confidence and visit confidence.

### Step 6 — provenance/documentation axis
- read-only `clinical process present` column derived from Steps 1–5 beside manual GeSY/Heidi trace;
- candidate material-discrepancy hint when clinical process is present but formal trace is absent;
- derive/reconcile `missing_domains` from the trace matrix rather than duplicate manual state;
- coherence gates for reliability, major gaps, unresolved conflicts and `ready_for_audit`;
- Step1↔Step6 Heidi coherence checks;
- provenance auto-feed/hints from reviewed DXA/labs/imaging;
- objective completion-time fallback from timestamps, while retaining clinician-entered timing during pilot if useful;
- mapping between Step-5 signal taxonomy and Step-6 clinical-domain taxonomy;
- PII reminder consistency on conflict/formal-record notes.

## 20.3 Cross-cutting architecture after pilot evidence

- central store helper so module slices use one explicit merge/persistence contract rather than multiple ad-hoc localStorage writers;
- shared registries for archetypes, machines, risk options and clinical domains to prevent drift;
- clear-on-collapse as a general invariant for dependent fields;
- accessibility roles for segmented/radio-style choice lists;
- unit metadata as first-class data-quality information;
- preserve localStorage as prototype-only and move to authenticated/private production storage before identifiable use.

## 20.4 Implementation discipline

Prioritization order remains:

```text
safety / data loss
→ data interpretability
→ pilot usability and friction
→ clinical safety derivations
→ reproducibility / provenance automation
→ structured Signals / broader intelligence
→ polish
```

Do not implement the entire backlog before the pilot. The 5 real pilot encounters should determine which refinements materially improve capture and which are unnecessary complexity.
