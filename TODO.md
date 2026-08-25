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
- [x] Defer live Setmore/Secretary feed without blocking the rest of Clinical Excellence development.
- [x] Merge server-side encounter finalization integrity rule: completed encounters cannot silently regress to draft; later material edits become `amended`.
- [x] Upgrade documentation/control-plane architecture to six active canonicals with explicit slice/current operational separation.

---

# 1. CURRENT GATE — PRODUCTION INTEGRITY + BASELINE PILOT

## 1.1 Encounter-finalization validation

- [ ] Complete the small live synthetic smoke for the merged finalization guard:
  - completed + no-op Save → remains `completed`;
  - completed + content/date change → becomes `amended`;
  - reload/reopen → amendment persists and remains loadable.

## 1.2 Five-case usability/capture pilot

- [ ] Run 5 consecutive eligible real pilot encounters.
- [ ] Measure post-visit completion time, friction, ambiguous/missing fields and persistence behavior.
- [ ] Do not change the form after every case unless safety/data-loss/persistence requires it.
- [ ] After all 5, make one deliberate refinement.
- [ ] Freeze Baseline Form v1 + KPI applicability/calculation contract.

## 1.3 Thirty-case scored baseline

- [ ] Run 30 consecutive unique eligible osteoporosis encounters under the frozen contract.
- [ ] Keep routine KPI/practice coaching hidden during scored baseline; safety-critical exceptions only.
- [ ] Lock baseline and denominator definitions.
- [ ] Finalize run-chart conventions and reliability display.

---

# 2. CLINICAL PRACTICE REVIEW PROGRAM — PRIMARY IMPROVEMENT ENGINE

Purpose: move beyond “was the KPI met?” and create a structured system that improves **what the clinician asks, thinks, explains, decides, documents, follows up and learns** from real encounters.

## 2.1 Foundation objects / data contract

- [ ] Define `PracticeReview` v1 object.
- [ ] Define `PracticeObservation` v1 with:
  - dimension;
  - direction: strength / gap / safety / uncertainty;
  - gap class when negative;
  - importance;
  - confidence;
  - encounter provenance;
  - linked standard/evidence;
  - why it matters;
  - suggested change;
  - clinician accept/modify/dismiss state.
- [ ] Define `DecisionReconstruction` v1.
- [ ] Define `CommunicationReview` v1.
- [ ] Define `WorkflowReview` v1.
- [ ] Define `ReviewExposure`/baseline-contamination metadata.
- [ ] Link accepted review observations to reusable `Signal` objects rather than creating a parallel learning system.

## 2.2 Review dimensions

- [ ] Clinical completeness.
- [ ] Clinical reasoning quality.
- [ ] Decision quality / alternatives / rationale.
- [ ] Risk interpretation.
- [ ] Safety.
- [ ] Communication scientific accuracy and clarity.
- [ ] Shared decision making / preferences.
- [ ] Consultation flow / efficiency / cognitive burden.
- [ ] Follow-up execution / ownership / timing.

## 2.3 Quick Post-Visit Review

- [ ] Produce a concise 2–3 minute review rather than exhaustive criticism.
- [ ] Surface roughly:
  - 3 strengths;
  - 3 highest-value improvements;
  - safety concern if present;
  - one reasoning issue;
  - one communication issue;
  - one concrete behavior to change next similar visit.
- [ ] Support clinician `Accept Signal`, `Modify`, `Dismiss`, `Challenge me`, `Create Learning` actions.

## 2.4 Deep Review / Red Team

- [ ] Reconstruct encounter chronology and critical decision points.
- [ ] Evaluate what information was actually available at each decision point to reduce hindsight bias.
- [ ] Build strongest evidence-based counter-case to the clinician’s decision.
- [ ] Distinguish reasonable uncertainty/override from genuine reasoning defect.
- [ ] Link disagreement to explicit standards/evidence rather than model opinion alone.

## 2.5 Communication Review

- [ ] Review scientific accuracy of patient-facing statements.
- [ ] Detect overstatement / understatement / misleading certainty.
- [ ] Review risk framing and terminology.
- [ ] Detect excessive information density / unnecessary repetition.
- [ ] Review whether patient questions/preferences changed the plan.
- [ ] Review teach-back/understanding separately from merely giving information.

## 2.6 Longitudinal practice-pattern detection

- [ ] Aggregate repeated observations across encounters.
- [ ] Distinguish isolated event from recurring pattern.
- [ ] Convert recurring patterns into root-cause-classified Signals.
- [ ] Detect sustained strengths as well as repeated gaps.
- [ ] Display denominator/sample size and reliability before declaring a stable pattern.

## 2.7 Intervention and re-measurement

- [ ] Knowledge gap → targeted learning + test + retention check.
- [ ] Reasoning gap → case challenge/red team/deliberate practice.
- [ ] Execution gap → workflow/interface/task redesign rather than reading.
- [ ] Communication/system gap → wording/teach-back/handoff/process redesign.
- [ ] Re-measure each intervention in later encounters.
- [ ] Promote repeated confirmed improvement into sustained strength.

---

# 3. HEIDI / TRANSCRIPT-ASSISTED CAPTURE

Purpose: reduce duplicate manual entry while preserving clinical truth and clinician control.

## 3.1 Transcript intake

- [ ] Add a clear `Εισαγωγή από Heidi` / paste-transcript workflow.
- [ ] Treat raw transcript as ephemeral; do not persist it in PostgreSQL/localStorage/logs by default.
- [ ] Ensure request/application logs never print the transcript.
- [ ] Do not commit real transcripts to the public repo.

## 3.2 Structured extraction contract

- [ ] Extract candidates into existing normalized clinical schema rather than inventing a second patient model.
- [ ] Preserve categories:
  - patient/history fact;
  - objective result;
  - clinician interpretation;
  - option discussed;
  - clinician recommendation;
  - patient preference;
  - final decision;
  - accepted / declined / undecided;
  - follow-up task;
  - uncertainty / needs review.
- [ ] Preserve negation, timing, speaker/source and uncertainty.
- [ ] Never synthesize exact dates from vague time references.
- [ ] Never convert negative history into a negative investigation result.
- [ ] Never collapse “discussed option” into “final plan”.

## 3.3 Clinician review gate

- [ ] Show candidate value + confidence + short temporary evidence snippet where useful.
- [ ] Allow Accept / Reject / Edit per candidate.
- [ ] Optional `Accept all high-confidence safe candidates` only after category-specific guardrails exist.
- [ ] Only accepted values write to authoritative encounter data.
- [ ] Persist provenance such as `source=heidi_transcript` and `clinician_reviewed=true`.

## 3.4 Initial extraction priority

Start with relatively objective/high-value domains:

- [ ] fracture history/events;
- [ ] anthropometrics/risk factors;
- [ ] DXA/VFA/imaging;
- [ ] laboratory values with dates/units when explicit;
- [ ] treatment episodes/administrations;
- [ ] treatment decision and follow-up tasks.

Then add more interpretive domains:

- [ ] communication content;
- [ ] patient preferences;
- [ ] understanding/teach-back;
- [ ] Practice Review observations.

---

# 4. ADAPTIVE OSTEOPOROSIS CONSULTATION FLOW

Purpose: make the UI follow a clinically coherent osteoporosis consultation while keeping the existing audit/storage schema underneath.

Permanent principle:

```text
CLINICAL PRESENTATION FLOW != STORAGE/AUDIT SCHEMA
```

The same canonical datum is entered/accepted once and reused by audit, longitudinal review, treatment planning and learning.

## 4.1 General consultation sequence candidate

- [ ] Opening / what changed / reason for review.
- [ ] Interval fracture history + falls/function.
- [ ] DXA/VFA/imaging review.
- [ ] Secondary causes + relevant laboratory review.
- [ ] Formal fracture-risk synthesis.
- [ ] Treatment history/adherence/tolerance/response.
- [ ] Treatment options → clinician recommendation → patient preference → final decision.
- [ ] Exercise/nutrition/other communication relevant to the archetype.
- [ ] Explicit Close card: decision, prerequisites, patient tasks, clinician tasks, unresolved items, timing/next review.

## 4.2 Archetype-adaptive flows

- [ ] Initial new/uncertain diagnosis flow.
- [ ] Known osteoporosis/osteopenia initial-to-service flow.
- [ ] Routine stable follow-up flow.
- [ ] Treatment-start flow.
- [ ] Treatment-continuation/due-monitoring flow.
- [ ] Treatment change/transition flow.
- [ ] Post-fragility-fracture flow.
- [ ] Fracture-on-treatment flow.
- [ ] Adverse-effect/intolerance flow.
- [ ] Completion/consolidation flow.

## 4.3 Risk-synthesis gate

Before treatment selection, the system should make it easy to resolve:

```text
fracture characterization
+ DXA/VFA
+ secondary causes/labs
+ falls/function
+ formal risk assessment
= explicit clinical risk/problem synthesis
```

Do not use consultation-flow software to force a treatment recommendation; it structures evidence and reasoning while preserving clinician override.

## 4.4 Close card

- [ ] `Σήμερα αποφασίσαμε`.
- [ ] `Πριν ξεκινήσει / prerequisites`.
- [ ] `Εκκρεμεί`.
- [ ] `Ο ασθενής πρέπει να κάνει`.
- [ ] `Εμείς πρέπει να κάνουμε`.
- [ ] `Επικοινωνία / επανέλεγχος`.
- [ ] `Unresolved critical item`.
- [ ] Optional teach-back prompt when appropriate.

---

# 5. OSTEOPOROSIS STANDARDS / COMPETENCY MAP

Expand the current domains into explicit standards and measurable competencies:

- [ ] Diagnosis & case finding.
- [ ] DXA / VFA / imaging.
- [ ] Fracture-risk assessment.
- [ ] Secondary osteoporosis & laboratory evaluation.
- [ ] Pharmacologic treatment selection.
- [ ] Sequential therapy / treatment transitions.
- [ ] Monitoring / treatment response / adherence.
- [ ] Falls, frailty, exercise & nutrition.
- [ ] Communication / shared decision making / continuity.

For each:

- [ ] core / advanced / specialist competencies;
- [ ] linked guideline/framework/version;
- [ ] learning resources;
- [ ] assessment items/cases;
- [ ] clinical-practice KPIs;
- [ ] patient-feedback dimensions;
- [ ] safety/failure modes;
- [ ] external benchmarks where methodologically comparable.

---

# 6. EVIDENCE / GUIDELINE GOVERNANCE

- [ ] Structured Evidence Registry.
- [ ] Rule-level evidence metadata.
- [ ] Explicit framework separation; no silent hybridization.
- [ ] Reviewed date/version/freshness state.
- [ ] Evidence impact classification.
- [ ] Evidence → affected standard/rule → approved change → implementation → re-measurement lifecycle.
- [ ] Curated osteoporosis evidence backbone.

---

# 7. LEARNING / TESTING / MASTERY ENGINE

- [ ] `unread → studied → tested → mastered → retention check` states.
- [ ] MCQ, case-based, open-response, image interpretation.
- [ ] Confidence-before-answer where useful.
- [ ] High-confidence errors prioritized.
- [ ] Spaced repetition.
- [ ] Case-triggered learning from accepted Practice Review Signals.
- [ ] Advanced cases for sustained strengths.
- [ ] Evidence Responsiveness without rewarding reflexive adoption of weak evidence.

---

# 8. AUDIT / QUALITY IMPROVEMENT

- [ ] Formal `AuditMetric` objects.
- [ ] Baseline → intervention → re-audit cycles.
- [ ] Run charts.
- [ ] `ImprovementProject` / PDSA-style iteration.
- [ ] Omissions vs reasoned overrides kept separate.
- [ ] Process audit + decision audit + later outcome review.
- [ ] Periodic random case review.
- [ ] Persistence of improvement after initial intervention.

---

# 9. PATIENT VOICE

- [ ] Compact patient-feedback instrument for condition/plan/rationale understanding and whether concerns/preferences were addressed.
- [ ] Free text where appropriate.
- [ ] Repeated theme detection.
- [ ] Theme → Signal/ImprovementProject.
- [ ] Re-measure after communication/process change.

---

# 10. SAFETY

- [ ] Error / near-miss register.
- [ ] FMEA/potential failure-mode register for high-risk workflows.
- [ ] Safety Signals outrank educational convenience.
- [ ] Denosumab delay/exit safety logic with exact treatment timelines.
- [ ] Safety tasks/escalation lifecycle.
- [ ] Trace clinician override and AI recommendation separately.

---

# 11. BENCHMARKING

- [ ] Benchmark Registry with source/country/population/setting/year/definition/value.
- [ ] Comparability: high / moderate / low / context-only.
- [ ] Avoid superiority/inferiority claims from non-comparable denominators.
- [ ] Benchmark strengths as well as gaps.

---

# 12. CLINICAL EXCELLENCE HOME / ANALYTICS

Build after the relevant data contracts are sufficiently stable.

- [ ] Attention-first panel: safety → overdue care → practice gaps → learning/evidence.
- [ ] Domain state with baseline/change/trend/reliability/sample size.
- [ ] Run charts behind progress summaries.
- [ ] Current strongest domain / priority gap.
- [ ] Active Improvement Projects.
- [ ] Today’s Learning queue.
- [ ] Evidence freshness.
- [ ] Learning loop summary.
- [ ] “What the system learned this month”.
- [ ] Navigation to patient registry, encounters, Calendar/CareTasks when those feeds are ready.

---

# 13. PRIVACY / PRODUCTION READINESS

- [x] PostgreSQL durable clinical storage implemented.
- [x] Browser-session authentication implemented for `/clinical/*` layer.
- [ ] Complete legacy-route/CORS exposure hardening before treating the whole service as appropriately protected for identifiable production data.
- [ ] Add access/audit trail for sensitive clinical actions/data access.
- [ ] Define retention/deletion/data-minimization approach.
- [ ] Review applicable GDPR/privacy requirements.
- [ ] Keep transcripts ephemeral by default.
- [ ] Never commit identifiable clinical datasets or transcripts.

---

# 14. CLINICAL CALENDAR / CARETASK / DIGITAL SECRETARY — DEFERRED, NOT ABANDONED

Already built:

- [x] Clinical Calendar storage/API/UI foundation.
- [x] Baseline sidebar navigation and temporary root routing.
- [x] Osteoporosis-only appointment categories/filtering.

Deferred until Digital Secretary work is ready:

- [ ] structured `visit_reason` from Cal.com/telephone flow;
- [ ] transfer `visit_reason` into Setmore notes/comment;
- [ ] Setmore → Clinical Calendar live appointment feed;
- [ ] previous/current/next-week live smoke;
- [ ] CareTasks for labs/treatment/results/follow-up;
- [ ] Zadarma reminder/notification workflow.

Permanent rule: **Appointment != CareTask**.

---

# 15. PATIENT MATERIALS — LOWER PRIORITY CURRENTLY

- [ ] Patient Q&A refinements.
- [ ] Medication leaflets.
- [ ] Exercise posters/materials.
- [ ] Other patient education assets.

These remain useful but should not displace Core/Practice Review/measurement work unless priority changes explicitly.

---

# 16. GENERALIZE BEYOND OSTEOPOROSIS

Only after Module 01 proves the reusable engine in real use:

- [ ] freeze reusable Core APIs/data contracts;
- [ ] select Module 02 based on clinical priority/overlap;
- [ ] reuse Signal/Learning/Audit/Practice Review/Patient Voice/Benchmark/Improvement machinery;
- [ ] build cross-module Clinical Excellence Home;
- [ ] distinguish domain-specific competence from global skills such as communication, calibration, safety and evidence responsiveness.

---

# 17. BROAD IMPLEMENTATION ORDER

```text
1. close encounter-finalization smoke
2. 5-case usability/capture pilot
3. one post-pilot refinement
4. freeze Baseline Form + KPI contract
5. build transcript extraction / Practice Review infrastructure in shadow mode
6. 30-case scored baseline without routine coaching exposure
7. baseline lock
8. activate clinician-facing Quick Practice Review
9. Deep Review / Red Team / Decision Reconstruction
10. longitudinal Signals + targeted interventions
11. adaptive consultation-flow presentation layer informed by pilot/review evidence
12. Learning / Evidence / Patient Voice / Improvement loops
13. Clinical Excellence Home
14. resume Calendar/CareTask/Secretary integration when external dependency is ready
15. generalize Core to later modules
```

If a safety/data-integrity defect appears, it outranks this sequence.
