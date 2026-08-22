# TODO.md — Clinical Excellence long-range compass

> **ROLE:** permanent roadmap across phases.
> **CANONICAL HOME:** `athpapachr-cmd/osteoporosis`.
> **CURRENT ACTIVE PLAN:** `CLINICAL_EXCELLENCE_PLAN.md`.
> **MODULE 01:** Osteoporosis.

This file answers **where the project is going and in what order**. Detailed current architecture belongs in `CLINICAL_EXCELLENCE_PLAN.md`; exact current handoff belongs in `HANDOFF_CURRENT.md`; completed history belongs in `osteoporosis-change-log.md`.

---

# 0. CURRENT — BLUEPRINT → BASELINE PILOT IMPLEMENTATION

- [x] Establish canonical five-file control plane:
  - `AGENTS.md`
  - `TODO.md`
  - `CLINICAL_EXCELLENCE_PLAN.md`
  - `HANDOFF_CURRENT.md`
  - `osteoporosis-change-log.md`
- [x] Define the project as a reusable Personal Clinical Excellence System rather than an isolated Osteoporosis Cockpit.
- [x] Define Osteoporosis as Module 01.
- [x] Define Signal-first adaptive feedback architecture.
- [x] Define initial Core objects.
- [x] Define initial Osteoporosis competency taxonomy.
- [x] Define first Home Dashboard wireframe and progress-bar principles.
- [ ] Review/freeze Core object schema v1 sufficiently for broader Core implementation.
- [ ] Expand Osteoporosis competency map into explicit standards and measurable competencies.
- [x] Create Baseline Osteoporosis Audit **draft v1** specification.
- [x] Create KPI Dictionary **draft v1** with explicit numerators/denominators/exclusions/targets/sources.
- [x] Create neutral prospective `baseline_case_form_v1` schema.
- [x] Approve first narrow runtime slice: prospective baseline capture before full dashboard work.
- [x] Implement Baseline Audit pilot UI Step 1 in PR #1.
- [ ] Implement pilot UI Step 2 — fracture history + fracture-risk assessment.
- [ ] Implement pilot UI Step 3 — DXA/VFA + secondary causes + falls/frailty.
- [ ] Implement pilot UI Step 4 — treatment history/safety + decision + monitoring/follow-up.
- [ ] Implement pilot UI Step 5 — communication + immediate post-visit reflection.
- [ ] Implement pilot UI Step 6 — documentation trace + final Heidi/capture-source review.
- [ ] Define exact form-field → KPI status calculation contract.
- [ ] Run 5 pilot encounters and measure completion time/friction/missing fields.
- [ ] Revise the form once after pilot evidence.
- [ ] Freeze Baseline Form v1 + KPI applicability rules before scored baseline.
- [ ] Define first Dashboard Data Contract after baseline capture contract is frozen.

**CURRENT NEXT IMPLEMENTATION ACTION:** complete the remaining schema-driven pilot-form steps, then run the 5-case pilot before starting the 30-case scored baseline.

---

# 1. BASELINE / MEASUREMENT FOUNDATION

- [x] Replace unreliable retrospective-only baseline with prospective consecutive encounter capture.
- [x] Define 5-case usability pilot followed by 30 consecutive unique scored baseline cases.
- [x] Define minimum sample-size/reliability display rules.
- [x] Define data-completeness KPI as foundational metric.
- [x] Define treatment of `not applicable`, missing, undocumented and not-assessable values.
- [x] Separate clinical process from formal GeSY/documentation trace and capture quality.
- [x] Define Heidi as observational capture exposure during baseline; do not score or mandate its use.
- [ ] Establish baseline lock date after form/KPI freeze and 30 scored cases.
- [ ] Finalize run-chart conventions for longitudinal KPI display.
- [x] Define overall-score policy: no composite score before sufficient baseline data.
- [ ] Define Improvement Velocity cautiously and distinguish plateau-at-high-performance from unresolved stagnation.
- [ ] Define secure/private production data-store architecture before any identifiable patient-data persistence.

---

# 2. OSTEOPOROSIS STANDARDS / COMPETENCY MAP

Expand the provisional domains:

- [ ] Diagnosis & case finding.
- [ ] DXA / VFA / imaging.
- [ ] Fracture-risk assessment.
- [ ] Secondary osteoporosis & laboratory evaluation.
- [ ] Pharmacologic treatment selection.
- [ ] Sequential therapy / treatment transitions.
- [ ] Monitoring / treatment response / adherence.
- [ ] Falls, frailty, exercise & nutrition.
- [ ] Communication / shared decision making / continuity.

For each domain:

- [ ] define core/advanced/specialist competencies;
- [ ] link guideline/framework source and version;
- [ ] define learning resources;
- [ ] define assessment items/cases;
- [ ] define clinical-practice KPIs;
- [ ] define patient-feedback dimensions where applicable;
- [ ] identify available external benchmarks;
- [ ] define safety/failure modes where applicable.

---

# 3. EVIDENCE / GUIDELINE GOVERNANCE

- [ ] Create structured Evidence Registry.
- [ ] Separate guideline/framework outputs explicitly; do not silently hybridize thresholds.
- [ ] Add rule-level evidence metadata rather than string-matched evidence attachment.
- [ ] Track reviewed date, version and freshness status.
- [ ] Track new evidence impact as confirming / interesting-no-change / potentially practice-changing / practice-changing / conflicting-insufficient.
- [ ] Define change-control path: evidence → affected standard/rule → review → implementation → re-measurement.
- [ ] Curate initial osteoporosis sources and learning backbone from major professional organizations and authoritative literature.
- [ ] Keep online + in-person course/event tracking as an external educational feed, with relevance to active gaps.

---

# 4. LEARNING / TESTING / MASTERY ENGINE

- [ ] Implement learning states: unread → studied → tested → mastered → retention check.
- [ ] Support papers, book chapters, guidelines, online courses, in-person courses/seminars, congresses, cases, videos and podcasts.
- [ ] Support MCQ, case-based, open-response and image-interpretation assessments.
- [ ] Capture confidence before answer when useful.
- [ ] Prioritize high-confidence errors.
- [ ] Implement spaced repetition.
- [ ] Link real clinical encounters to targeted learning assignments.
- [ ] Reduce basic repetition for sustained strengths and progress to advanced cases.
- [ ] Track Evidence Responsiveness without rewarding reflexive adoption of weak new evidence.

---

# 5. EXISTING OSTEOPOROSIS COCKPIT — CORE CORRECTIONS BEFORE DEEPER INTEGRATION

- [ ] Separate guideline frameworks instead of hybrid risk classification.
- [ ] Remove unvalidated internal risk index from treatment-decision authority or relabel it clearly as non-validated support only.
- [ ] Rebuild DXA longitudinal model around BMD g/cm², percent change, scanner/cross-calibration and LSC.
- [ ] Add structured `FractureEvent` model.
- [ ] Add structured treatment episodes and exact administrations/due dates.
- [ ] Add care-task objects with due/overdue/completed lifecycle.
- [ ] Add structured visit audit/coverage objects.
- [ ] Replace evidence keyword/string matching with explicit rule metadata.
- [ ] Distinguish proposed / discussed / clinician-decided / patient-accepted decisions where relevant.
- [ ] Add unresolved critical-item close-visit check.
- [ ] Add pre-visit intelligence brief after data model supports it.

---

# 6. CLINICAL PRACTICE ↔ LEARNING FEEDBACK LOOP

- [ ] Convert meaningful encounter events into Signals.
- [ ] Classify negative signals as knowledge / reasoning / execution / communication-system gaps.
- [ ] Generate intervention type from root-cause class rather than defaulting to more education.
- [ ] Convert sustained good performance into strength signals.
- [ ] Preserve successful workflows and reduce unnecessary learning repetition.
- [ ] Add Challenge / Red Team / Learning modes for selected cases.
- [ ] Support blinded later re-review for decision consistency when useful.

---

# 7. PATIENT VOICE

- [ ] Define compact patient-feedback instrument for understanding condition, plan, rationale and whether concerns/preferences were addressed.
- [ ] Support free-text patient comments.
- [ ] Detect repeated communication themes.
- [ ] Turn repeated themes into Signals / Improvement Projects.
- [ ] Re-measure after communication/handoff changes.
- [ ] Keep patient feedback distinct from generic satisfaction scoring.

---

# 8. AUDIT / QUALITY IMPROVEMENT

- [ ] Implement formal AuditMetric objects.
- [ ] Implement baseline → intervention → re-audit cycles.
- [ ] Implement run charts.
- [ ] Implement ImprovementProject object and PDSA-style iteration.
- [ ] Track omissions, intentionally overridden items and reasons separately.
- [ ] Support process audit, clinical-decision audit and later outcome review.
- [ ] Add periodic random case review.
- [ ] Track which improvements actually persisted after initial intervention.

---

# 9. SAFETY

- [ ] Implement error / near-miss register.
- [ ] Implement potential failure-mode/FMEA register for high-risk workflows.
- [ ] Prioritize patient-safety signals above educational convenience.
- [ ] Add denosumab administration/delay/exit safety logic once exact treatment timelines are available.
- [ ] Add safety tasks and escalation lifecycle.
- [ ] Ensure clinician override and AI recommendations are traceable.

---

# 10. BENCHMARKING

- [ ] Create Benchmark Registry with metric/source/country/population/setting/year/definition/value.
- [ ] Add comparability rating: high / moderate / low / context only.
- [ ] Compare practice only against methodologically compatible denominators/settings.
- [ ] Surface external comparison for both gaps and sustained strengths.
- [ ] Avoid claims of superiority from inappropriate cross-setting comparisons.

---

# 11. HOME DASHBOARD / ANALYTICS

- [ ] Build first Clinical Excellence Home after data contract is frozen.
- [ ] Add attention-first panel: safety → overdue clinical care → practice gaps → learning/evidence updates.
- [ ] Add domain progress bars with current/baseline/change/trend/reliability/sample size.
- [ ] Add run charts behind bars.
- [ ] Add current strongest domain / priority gap.
- [ ] Add active Improvement Projects.
- [ ] Add Today's Learning adaptive queue.
- [ ] Add Evidence Freshness.
- [ ] Add Learning Loop summary: signals detected → actions → interventions → re-audits → confirmed improvements.
- [ ] Add “What the system learned this month”.

---

# 12. PRIVACY / PRODUCTION READINESS

The GitHub repository is public.

- [x] Keep current repository schemas/UI examples synthetic and prohibit identifiable inputs in the pilot UI.
- [x] Avoid public/server persistence in the first pilot slice; browser `localStorage` is prototype-only and explicitly unencrypted.
- [ ] Before identifiable production use: authentication and access control.
- [ ] Secure secrets and environment configuration.
- [ ] Add audit trail for sensitive clinical actions/data access.
- [ ] Define storage/encryption/data-minimization/retention approach.
- [ ] Review applicable GDPR/privacy requirements.
- [ ] Do not commit clinical datasets containing identifiable data.

---

# 13. PATIENT MATERIALS — DEFERRED CURRENTLY

- [ ] Patient Q&A handout refinements.
- [ ] Medication leaflets.
- [ ] Exercise posters/materials.
- [ ] Other patient education assets.

These remain valuable but are intentionally lower priority than Core Engine, measurement, audit and practice-integration architecture during the current phase.

---

# 14. GENERALIZE BEYOND OSTEOPOROSIS

Only after Module 01 proves the reusable engine in real use:

- [ ] Extract/freeze reusable Core APIs/data contracts.
- [ ] Define Module 02 based on priority and overlap analysis rather than arbitrary anatomy.
- [ ] Candidate domains: low-back pain, neck pain, knee pain, hip pain, shoulder pain.
- [ ] Reuse Signal/Learning/Audit/Patient Voice/Benchmark/Improvement machinery without duplication.
- [ ] Create overall clinical-practice dashboard across modules.
- [ ] Distinguish domain-specific competence from global clinical skills such as communication, calibration, safety and evidence responsiveness.
