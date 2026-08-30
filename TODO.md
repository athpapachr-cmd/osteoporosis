# TODO.md — Clinical Excellence long-range compass

> **ROLE:** permanent broad roadmap/checklist across phases.
> **CANONICAL HOME:** `athpapachr-cmd/osteoporosis`.
> **ACTIVE DETAILED PHASE:** `CLINICAL_EXCELLENCE_PLAN.md`.
> **ACTIVE SLICE DESIGN:** `SLICE_PLAN_CURRENT.md`.
> **OPERATIONAL NOW:** `CURRENT_OPERATIONAL.md`.
> **MODULE 01:** Osteoporosis.

This file answers **where the product is going and in what broad order**. It is not the operational lock and should not become a PR/deploy diary.

---

# 0. CLOSED FOUNDATIONS / CURRENT PRODUCTION BASE

- [x] Reframe project from isolated Osteoporosis Cockpit to reusable Personal Clinical Excellence System.
- [x] Establish Osteoporosis as Module 01.
- [x] Establish Signal-first feedback architecture.
- [x] Establish gap classes: knowledge / reasoning / execution / communication-system.
- [x] Establish sustained strengths as positive Signals.
- [x] Establish STANDARD / CHALLENGE / RED TEAM / LEARNING modes.
- [x] Define first Core objects and provisional osteoporosis competency domains.
- [x] Define transparent measurement principles and no-composite-score-before-baseline rule.
- [x] Define Baseline Osteoporosis Audit draft v1 and KPI Dictionary v1.
- [x] Implement Baseline Audit Steps 1–6.
- [x] Implement pre-pilot hardening P1–P8.
- [x] Add `labs_date` and Step-6 conflict clear-on-collapse.
- [x] Pass explicit 14-scenario synthetic form smoke.
- [x] Implement authenticated patient registry + PostgreSQL encounter/lab persistence.
- [x] Browser-smoke patient load/save/reload and longitudinal laboratory snapshots.
- [x] Remove duplicate top-level lab-history table and add `Νέες αναλύσεις` capture reset.
- [x] Add Clinical Calendar foundation/navigation/osteoporosis-only filtering.
- [x] Defer live Setmore/Secretary feed without blocking Clinical Excellence development.
- [x] Merge server-side encounter-finalization integrity rule: completed encounters cannot silently regress to draft; later material edits become `amended`.
- [x] Complete 3/3 live synthetic finalization smoke for server-side completed/amended semantics.
- [x] Upgrade documentation/control-plane architecture to six active canonicals.
- [x] Implement/test the bounded authoritative browser Finish correction on `fix/module01-c1-authoritative-finish-2026-08-30`.

The authoritative Finish correction remains **unmerged/undeployed** until separately authorized and production-smoke verified.

---

# 1. CURRENT PRIMARY PROGRAM — OSTEOPOROSIS AS A DYNAMIC CLINICAL EXCELLENCE CONSULTATION SYSTEM

The product owner has clarified the primary purpose:

```text
improve the current osteoporosis visit
+
reduce duplicate/manual capture
+
review what was actually said/reasoned/decided
+
improve the clinician over repeated real encounters
```

The Baseline Audit remains a measurement engine underneath the product. It is not the intended clinician-facing workflow.

The previous plan to run five real cases on the current largely manual Steps 1–6 workflow before transcript population/adaptive visit guidance is superseded.

## 1.1 Critical finalization integrity

- [x] C1 authoritative Finish fix implemented.
- [x] C1 focused browser/API regression passed.
- [ ] PR/review/merge C1 using exact tested ancestry after explicit merge authority.
- [ ] Allow normal Render auto-deploy from `main`.
- [ ] Production synthetic Finish → completed → reload smoke.

No real pilot against production until finalization integrity is deployed/smoke-verified.

## 1.2 Dynamic guided-visit foundation — BEFORE real pilot

- [ ] Freeze `EncounterContextV1` / `VisitPlanV1` / `GuidanceRuleV1` semantics.
- [ ] Replace coarse archetype-only `applicable/conditional/N/A` logic with layered relevance reasons:
  - safety/event override;
  - unresolved prior critical item;
  - treatment/agent-specific requirement;
  - evidence-defined milestone/due item;
  - archetype base flow;
  - contextual/optional item.
- [ ] Preserve the current encounter archetypes as coarse visit intent rather than multiplying one form per visit number.
- [ ] Add longitudinal treatment context:
  - active treatment episode/agent;
  - actual administration history;
  - administration count when reliable;
  - elapsed exposure;
  - next-due/delay state;
  - monitoring due state;
  - unresolved prior tasks/prerequisites;
  - new fracture/adverse-event/safety triggers.
- [ ] Define machine-readable therapy/milestone rule registry with evidence/approved-policy provenance.
- [ ] Do **not** hard-code clinical behavior for “4th/8th/10th Prolia” merely from ordinal number; use actual treatment timeline + reviewed milestone rules.
- [ ] Make every surfaced card able to answer `WHY NOW?`.
- [ ] Keep critical safety/event cards unhideable by a lower-priority archetype default.
- [ ] Preserve clinician override with explicit reason where appropriate.

## 1.3 Heidi-first capture + in-place provisional population — BEFORE real pilot

- [ ] Restart corrected PR-1 v3 transcript extraction design.
- [ ] Implement protected transcript paste/intake with raw transcript ephemeral by default.
- [ ] Implement reusable Core semantic candidates + deterministic osteoporosis target mapping.
- [ ] Preserve negation, temporality, speaker/source, uncertainty and discussion/recommendation/final-decision distinctions.
- [ ] Do not invent exact dates from vague timing.
- [ ] Do not force currently unmapped concepts into unrelated fields.
- [ ] Implement PR-2 minimum clinician review boundary.
- [ ] Render mapped candidates **inside the relevant clinical cards** as provisional values rather than requiring a second disconnected data-entry workflow.
- [ ] Support Accept / Reject / Edit and explicit conflicts with existing authoritative longitudinal data.
- [ ] Only accepted values become authoritative encounter data.
- [ ] Preserve provenance and clinician-review state.
- [ ] Add compact extraction coverage summary, e.g. captured / unresolved / needs review, without implying that unmentioned = negative.

## 1.4 Guided consultation UX — BEFORE real pilot

- [ ] Replace the visible Steps 1–6 mental model with an encounter-adaptive clinical sequence while keeping the storage/audit schema underneath.
- [ ] Surface only information that is:
  - needed today;
  - due today;
  - triggered by a new event;
  - unresolved from prior care;
  - uncertain/conflicting after transcript extraction;
  - required to close the current decision safely.
- [ ] Reuse prior authoritative data rather than repeatedly asking stable history.
- [ ] Support a concise explicit Close state: decision, prerequisites, unresolved items, patient tasks, clinician tasks, communication plan and timing.
- [ ] Ensure live Clinical Guidance structures the encounter but never silently makes the treatment decision.

## 1.5 Five-case real pilot — AFTER the real workflow exists

- [ ] Run 5 consecutive eligible real **system-assisted** osteoporosis encounters.
- [ ] Pilot the intended workflow: dynamic guidance + longitudinal reuse + transcript-assisted provisional population + clinician verification.
- [ ] For each case record:
  - time from transcript/visit close to authoritative completion;
  - manual fields/corrections required;
  - missed transcript facts;
  - false/incorrect extraction candidates;
  - ambiguous candidates;
  - duplicate questioning/data entry;
  - wrong/missing card relevance;
  - persistence/load/reload defects;
  - safety/data-integrity defects;
  - clinician friction/cognitive burden.
- [ ] Do not redesign after every individual case unless safety/data-loss/persistence requires it.
- [ ] After all five, make one deliberate refinement.
- [ ] Freeze `GuidanceRule` applicability semantics + capture/review contract + KPI denominator semantics for the scored baseline.

## 1.6 Thirty-case scored system-assisted baseline

- [ ] Run 30 consecutive unique eligible osteoporosis encounters under the frozen guided/capture contract.
- [ ] Keep the stabilized **Clinical Guidance** layer active.
- [ ] Keep routine KPI score feedback/red-green performance coaching hidden.
- [ ] Keep routine clinician-facing Practice Review hidden by default until the baseline policy permits it; safety-critical exceptions remain allowed.
- [ ] Record guidance exposure where technically feasible.
- [ ] Distinguish content already present before a system cue from content resolved after a cue where the event sequence can be established.
- [ ] Label this cohort accurately as a **system-assisted baseline**, not an untouched/unassisted clinician baseline.
- [ ] Lock denominator definitions and run-chart/reliability conventions.

## 1.7 Module 01 closure evidence

- [ ] Implement minimum viable Quick Practice Review with structured evidence-traceable observations.
- [ ] Implement clinician disposition on important review observations.
- [ ] Promote repeated observations into denominator-aware Signals.
- [ ] Classify negative Signals by root cause.
- [ ] Apply at least one root-cause-appropriate intervention.
- [ ] Re-measure in later encounters and record improved / unchanged / worsened / insufficient evidence.
- [ ] Demonstrate whether prompt dependence decreases or spontaneous correct clinical behavior increases where measurement is feasible.
- [ ] Perform final Module 01 closure review.
- [ ] Only then mark `MODULE 01 CLOSED` and generalize the proven Core.

---

# 2. CLINICAL GUIDANCE / DYNAMIC VISIT ENGINE

Purpose: help conduct the encounter, not merely document it afterward.

## 2.1 Core candidate objects

- [ ] `EncounterContextV1`.
- [ ] `VisitPlanV1`.
- [ ] `GuidanceRuleV1`.
- [ ] `GuidedCardStateV1`.
- [ ] `GuidanceExposureV1`.
- [ ] `TherapyMilestoneProfileV1`.

## 2.2 Visit-context inputs

At minimum:

- [ ] encounter archetype/visit intent;
- [ ] new vs established patient;
- [ ] active disease/risk state;
- [ ] active and prior therapies;
- [ ] actual administration dates;
- [ ] course/administration count when reliable;
- [ ] elapsed exposure;
- [ ] next due/overdue state;
- [ ] due DXA/lab/other monitoring state;
- [ ] new fracture/post-fracture state;
- [ ] fracture on treatment;
- [ ] adverse effect/intolerance;
- [ ] transition/exit/consolidation state;
- [ ] unresolved prior tasks/critical prerequisites;
- [ ] special populations/context modifiers.

## 2.3 Rule reasons / card states

- [ ] `critical_safety`.
- [ ] `event_triggered`.
- [ ] `unresolved_prior`.
- [ ] `agent_specific`.
- [ ] `milestone_due`.
- [ ] `archetype_core`.
- [ ] `contextual`.
- [ ] `not_applicable`.

Every surfaced item should retain the reason it is shown.

## 2.4 Archetype examples

- [ ] Initial new/uncertain diagnosis.
- [ ] Initial-to-service known osteoporosis/osteopenia.
- [ ] Results/work-up review with treatment decision.
- [ ] Routine stable follow-up.
- [ ] Treatment start.
- [ ] Treatment administration/continuation.
- [ ] Due-monitoring / milestone review.
- [ ] Treatment change/transition/exit.
- [ ] Post-fragility-fracture.
- [ ] Fracture-on-treatment.
- [ ] Adverse effect/intolerance.
- [ ] Completion/consolidation.

The examples may combine. A fracture event or safety trigger may override the normal routine-administration flow.

## 2.5 Denosumab/other repeated therapy modelling

- [ ] Build repeated-administration logic from actual treatment history rather than nominal appointment labels.
- [ ] Permit evidence/policy rules to trigger on:
  - every administration;
  - early-course administrations;
  - elapsed months/years;
  - administration count;
  - monitoring due state;
  - long-duration review point;
  - delayed/missed administration;
  - new fracture or treatment-response concern.
- [ ] If administration count and elapsed exposure diverge because of delays, preserve both and avoid pretending the ordinal count alone defines the clinical state.

---

# 3. HEIDI / TRANSCRIPT-ASSISTED CAPTURE

Purpose: remove duplicate manual entry while preserving clinical truth and clinician control.

## 3.1 Transcript intake/privacy

- [ ] Clear `Εισαγωγή από Heidi` workflow.
- [ ] Raw transcript ephemeral by default.
- [ ] No PostgreSQL/localStorage/sessionStorage/log persistence of raw transcript.
- [ ] Sanitized validation/error boundary.
- [ ] No identifiable real transcript in public repo/tests.
- [ ] Provider privacy/data-control gate before identifiable production transcript use.

## 3.2 Structured extraction

- [ ] Corrected PR-1 v3 composite candidate contract.
- [ ] Deterministic target mapping against actual persisted runtime registry.
- [ ] Provider-neutral Core extraction interface.
- [ ] Preserve semantic categories:
  - history fact;
  - objective result;
  - clinician interpretation;
  - option discussed;
  - recommendation;
  - patient preference;
  - final decision;
  - accepted/declined/undecided;
  - follow-up task;
  - uncertainty/needs review.

## 3.3 Clinician review / inline population

- [ ] Provisional values displayed in destination cards.
- [ ] Accept / Reject / Edit.
- [ ] Existing-value conflict handling.
- [ ] No silent overwrite.
- [ ] Provenance retained after acceptance.
- [ ] Optional safe bulk acceptance only after category-specific guardrails are proven.

---

# 4. CLINICAL PRACTICE REVIEW — PRIMARY LONGITUDINAL IMPROVEMENT ENGINE

Purpose: determine whether what was said, reasoned and decided was appropriate, not merely whether a checkbox was completed.

## 4.1 Foundation objects

- [ ] `PracticeReviewV1`.
- [ ] `PracticeObservationV1` with direction, gap class, importance, confidence, provenance, evidence linkage, suggested change and clinician disposition.
- [ ] `DecisionReconstructionV1`.
- [ ] `CommunicationReviewV1`.
- [ ] `WorkflowReviewV1`.
- [ ] `ReviewExposureV1`.

## 4.2 Review dimensions

- [ ] Clinical completeness.
- [ ] Clinical reasoning.
- [ ] Decision quality.
- [ ] Risk interpretation.
- [ ] Safety.
- [ ] Communication accuracy/clarity.
- [ ] Shared decision making.
- [ ] Consultation flow/efficiency.
- [ ] Follow-up execution.

## 4.3 Quick Review

- [ ] Concise post-visit review rather than exhaustive criticism.
- [ ] Strengths.
- [ ] Highest-value improvements.
- [ ] Safety concern when present.
- [ ] Reasoning issue.
- [ ] Communication issue.
- [ ] One concrete change for next similar encounter.
- [ ] Accept / Modify / Dismiss / Challenge me / Create Learning.

## 4.4 Deep Review / Red Team — not a Module 01 closure blocker by default

- [ ] Chronological decision reconstruction.
- [ ] Anti-hindsight evaluation.
- [ ] Evidence-based counter-case.
- [ ] Reasonable override vs genuine reasoning defect.

## 4.5 Longitudinal patterns

- [ ] Aggregate reviewed observations across encounters.
- [ ] Distinguish isolated event, recurrent gap, improving trend, stable weakness and sustained strength.
- [ ] Denominator/sample-size/reliability before declaring stable patterns.

---

# 5. AUDIT / QUALITY IMPROVEMENT

- [ ] Formal `AuditMetric` objects.
- [ ] Transparent numerator/denominator/applicability.
- [ ] System-assisted baseline → intervention → re-audit cycles.
- [ ] Run charts.
- [ ] `ImprovementProject` / PDSA-style iteration.
- [ ] Omissions vs reasoned overrides separate.
- [ ] Process audit + decision audit + later outcome review.
- [ ] Periodic random case review.
- [ ] Persistence of improvement after intervention.

---

# 6. OSTEOPOROSIS STANDARDS / EVIDENCE / COMPETENCY MAP

Domains:

- [ ] Diagnosis & case finding.
- [ ] DXA / VFA / imaging.
- [ ] Fracture-risk assessment.
- [ ] Secondary osteoporosis & laboratory evaluation.
- [ ] Pharmacologic treatment selection.
- [ ] Sequential therapy / treatment transitions.
- [ ] Monitoring / treatment response / adherence.
- [ ] Falls, frailty, exercise & nutrition.
- [ ] Communication / shared decision making / continuity.

For clinically active guidance rules:

- [ ] explicit evidence/policy source;
- [ ] version/year;
- [ ] applicability/trigger;
- [ ] recommendation/criterion;
- [ ] strength/certainty when available;
- [ ] reviewed date/freshness;
- [ ] no silent guideline hybridization.

Comprehensive completion of every future competency/learning resource is not a Module 01 closure blocker. Evidence required to support material Clinical Guidance and Practice Review claims is closure-critical.

---

# 7. LEARNING / TESTING / MASTERY ENGINE

- [ ] `unread → studied → tested → mastered → retention check`.
- [ ] MCQ/case/open response/image interpretation.
- [ ] Confidence-before-answer where useful.
- [ ] High-confidence errors prioritized.
- [ ] Spaced repetition.
- [ ] Case-triggered learning from accepted Practice Review Signals.
- [ ] Advanced cases for sustained strengths.

Only the intervention capability needed to close at least one real Module 01 improvement loop is closure-critical; the full future mastery product is not.

---

# 8. SAFETY

- [ ] Error / near-miss register.
- [ ] FMEA/potential failure-mode register for high-risk workflows.
- [ ] Safety Signals outrank educational convenience.
- [ ] Denosumab delay/exit safety logic with exact treatment timelines and explicit evidence provenance.
- [ ] Safety tasks/escalation lifecycle.
- [ ] Trace clinician override and AI/system recommendation separately.
- [ ] A critical safety/event trigger may override a routine visit plan.

---

# 9. PATIENT VOICE — POST-CLOSURE BY DEFAULT

- [ ] Compact patient-feedback instrument for understanding condition/plan/rationale and whether concerns/preferences were addressed.
- [ ] Free text where useful.
- [ ] Repeated theme detection.
- [ ] Theme → Signal/ImprovementProject.
- [ ] Re-measure after communication/process change.

Not a Module 01 closure blocker unless evidence later makes it necessary for a critical improvement loop.

---

# 10. BENCHMARKING — POST-CLOSURE BY DEFAULT

- [ ] Benchmark Registry with source/country/population/setting/year/definition/value.
- [ ] Comparability high/moderate/low/context-only.
- [ ] Avoid superiority/inferiority claims from non-comparable denominators.

---

# 11. CLINICAL EXCELLENCE HOME / ANALYTICS — LATER

- [ ] Attention-first panel: safety → overdue care → practice gaps → learning/evidence.
- [ ] Domain baseline/change/trend/reliability/sample size.
- [ ] Run charts.
- [ ] Current strongest domain / priority gap.
- [ ] Active Improvement Projects.
- [ ] Learning due.
- [ ] Evidence freshness.
- [ ] “What the system learned this month”.

Do not build polished summary scores before the relevant contracts/baseline are stable.

---

# 12. PRIVACY / PRODUCTION READINESS

- [x] PostgreSQL durable clinical storage implemented.
- [x] Browser-session authentication implemented for `/clinical/*`.
- [ ] Complete legacy-route/CORS exposure hardening before claiming whole-service protection for identifiable data.
- [ ] Add access/audit trail for sensitive actions/data access.
- [ ] Define retention/deletion/data-minimization approach.
- [ ] Review applicable GDPR/privacy requirements.
- [ ] Keep transcripts ephemeral by default.
- [ ] Never commit identifiable clinical datasets/transcripts.

---

# 13. CLINICAL CALENDAR / CARETASK / DIGITAL SECRETARY — DEFERRED, NOT ABANDONED

Already built:

- [x] Clinical Calendar storage/API/UI foundation.
- [x] Baseline sidebar navigation/root routing foundation.
- [x] Osteoporosis-only appointment filtering.

Deferred:

- [ ] structured live `visit_reason` feed;
- [ ] Setmore → Clinical Calendar live feed;
- [ ] CareTasks for labs/treatment/results/follow-up;
- [ ] reminders/notification workflow.

Permanent rule: **Appointment != CareTask**.

---

# 14. CLINIC UTILITIES / CLINICAL OPERATIONS — PARKED

## 14.1 Physiotherapy referral

Production CU-1 baseline is implemented/merged/deployed historically.

Later richer referral work remains preserved separately and must not be mutated/merged/deployed during Module 01 closure without explicit authorization.

## 14.2 Radiofrequency request/PDF workflow

Roadmap-preserved, not a Module 01 closure blocker.

---

# 15. PATIENT MATERIALS — LOWER PRIORITY

- [ ] Q&A refinements.
- [ ] Medication leaflets.
- [ ] Exercise materials.
- [ ] Other patient education assets.

---

# 16. GENERALIZE BEYOND OSTEOPOROSIS

Only after Module 01 proves the reusable engine in real use:

- [ ] freeze reusable Core APIs/data contracts;
- [ ] select Module 02 based on clinical priority/overlap;
- [ ] reuse Clinical Guidance / Signal / Learning / Audit / Practice Review / Patient Voice / Improvement machinery;
- [ ] distinguish domain-specific competence from global skills such as communication, calibration, safety and evidence responsiveness.

Clinic Utilities do not count as declaring a clinical Module 02.

---

# 17. BROAD IMPLEMENTATION ORDER — REVISED 2026-08-30

```text
1. authoritative Finish correction — IMPLEMENTED/TESTED, merge/deploy/smoke pending
2. dynamic guided-visit architecture + machine contract
3. PR-1 transcript extraction
4. PR-2 inline provisional population / clinician review
5. guided clinical-card runtime sufficient for real use
6. 5-case system-assisted usability/capture pilot
7. one deliberate refinement
8. freeze guidance + capture + KPI applicability contracts
9. minimum Quick Practice Review infrastructure/shadow validation
10. 30-case scored system-assisted baseline with KPI/performance coaching hidden
11. baseline lock
12. clinician-facing reviewed Signals/interventions
13. longitudinal pattern detection + one closed improvement loop
14. re-measurement / prompt-dependence trend where feasible
15. final Module 01 closure review
16. later Deep Review / Patient Voice / Home / benchmarking breadth
17. generalize Core to later clinical modules
```

If a safety/data-integrity defect appears, it outranks this sequence.
