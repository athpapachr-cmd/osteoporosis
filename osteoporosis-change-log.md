# osteoporosis-change-log.md — append-only project logbook

> **ROLE:** permanent chronological history of material project decisions and completed milestones.
> **CANONICAL HOME:** `athpapachr-cmd/osteoporosis`.
> **RULE:** append new entries; do not rewrite prior historical entries merely to match current architecture.

---

## 2026-08-22 — Project reframed from Cockpit to Personal Clinical Excellence System

The Osteoporosis Cockpit was explicitly reclassified as one component of a larger continuous clinical-improvement system rather than the entire project.

Approved conceptual direction:

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

The project is intended to become reusable beyond osteoporosis. Osteoporosis is Module 01 and will serve as the first proving ground for the reusable Core Engine before later modules such as low-back pain, neck pain, knee pain, hip pain and shoulder pain are considered.

The architecture was intentionally defined as dynamic: clinical practice, patient feedback, learning, audits, evidence, safety events and external benchmarks all produce signals that can change what the system teaches, measures or does next.

---

## 2026-08-22 — External educational / quality-improvement models reviewed conceptually

The design discussion incorporated useful patterns from major medical-education and bone-health ecosystems, including:

- longitudinal curriculum and mastery concepts;
- adaptive/spaced-repetition learning;
- case-triggered learning;
- article appraisal linked to practice change;
- formal osteoporosis and densitometry courses;
- standardized clinical quality indicators;
- benchmarking and re-audit;
- patient-experience input;
- PDSA/FMEA-style improvement and safety thinking.

The project decision was not to clone any one external platform. Instead, the Clinical Excellence System should integrate the strongest compatible concepts into one closed feedback loop connecting education to real clinical behavior and re-measurement.

---

## 2026-08-22 — Signal-first architecture approved

`Signal` was selected as the central adaptive object linking otherwise separate parts of the system.

Signal sources include:

- clinical encounters;
- patient feedback;
- audits;
- learning/tests;
- new evidence/guidelines;
- safety events/near misses;
- benchmarks;
- sustained good performance.

Negative signals must be classified before intervention as one or more of:

```text
KNOWLEDGE GAP
REASONING GAP
EXECUTION GAP
COMMUNICATION / SYSTEM GAP
```

This prevents the system from treating every poor result as an educational deficit.

Positive repeated signals can mature into `SUSTAINED STRENGTH` and should trigger reinforcement, advanced challenge and appropriate external comparison rather than endless basic repetition.

---

## 2026-08-22 — Patient Voice elevated to system-learning input

Patient feedback was defined as more than satisfaction measurement.

Initial feedback dimensions include:

- understanding the condition;
- understanding the plan;
- understanding treatment rationale/duration/risks;
- whether questions/preferences were addressed;
- free-text confusion, concern, praise or suggestion.

Repeated patient-feedback patterns may generate Signals, trigger improvement projects and require later re-measurement after communication/workflow changes.

---

## 2026-08-22 — Progress measurement principles approved

The dashboard should eventually show multiple progress bars but must also preserve longitudinal context.

For meaningful metrics, the design should carry:

```text
Current
Baseline
Change
Trend
Sample size / denominator
Reliability
Target / standard
External benchmark + comparability
Data completeness
```

Progress bars represent current state; run charts represent trajectory.

No composite Clinical Excellence score should be treated as real before an adequate baseline audit exists.

---

## 2026-08-22 — Personalized operating style approved

The system should adapt to the clinician rather than behaving as a generic CME tracker.

Approved interaction modes:

```text
STANDARD
CHALLENGE
RED TEAM
LEARNING
```

Approved personalized principles include:

- explain reasoning/source rather than giving only a verdict;
- allow clinician accept/modify/reject with rationale;
- prioritize high-confidence errors;
- measure calibration where useful;
- challenge confirmation bias;
- compare sustained strengths as well as gaps with external standards where methodologically valid;
- distinguish critical flaw from clinically meaningful improvement and cosmetic refinement;
- avoid endless refinement once the approved objective has sufficient evidence of completion.

---

## 2026-08-22 — Canonical GitHub control plane created

The public repository `athpapachr-cmd/osteoporosis` was designated as the canonical project home for Module 01 and the reusable Clinical Excellence architecture being developed through it.

A five-file active canonical set was created, following the same control-plane discipline used in the digital-secretary project:

```text
AGENTS.md
TODO.md
CLINICAL_EXCELLENCE_PLAN.md
HANDOFF_CURRENT.md
osteoporosis-change-log.md
```

`README.md` remains navigation only.

The permanent repository rule is one canonical project truth; chat history must not be the only place where durable decisions live.

No runtime code change was part of this control-plane bootstrap.

---

## 2026-08-22 — Next major design milestone selected

The next major design task is:

> **Baseline Osteoporosis Audit v1 + KPI Dictionary v1**

The baseline must be defined before any dashboard progress score is considered valid.

The audit design will specify sampling, inclusion/exclusion criteria, numerators, denominators, targets, data completeness, `not applicable` handling, reliability/sample-size rules, baseline lock criteria and re-audit timing.

---

## 2026-08-22 — Prospective baseline strategy adopted

A key operational constraint was identified: there is no reliable pre-existing osteoporosis patient registry or dedicated osteoporosis folder, and GeSY visit records may be incomplete relative to what actually occurred during a consultation.

The baseline strategy was therefore changed from a primarily retrospective chart audit to a **prospective post-visit encounter-capture baseline**.

Approved sequence:

```text
5 pilot cases
→ refine usability/branching
→ freeze form + KPI applicability rules
→ 30 consecutive unique scored baseline cases
→ lock baseline
→ interventions / re-audit
```

Heidi AI is currently recent and non-systematic. During baseline it is recorded only as an exposure/capture source; its use is not scored as good practice and is not forced before baseline lock.

The audit now explicitly separates:

```text
clinical process
formal GeSY/documentation trace
capture quality
```

---

## 2026-08-22 — Baseline audit / KPI / case-form schemas created

Machine-readable draft schemas were added:

```text
schemas/baseline_osteoporosis_audit_v1.yaml
schemas/kpi_dictionary_v1.yaml
schemas/baseline_case_form_v1.yaml
```

The baseline schema defines pilot/scored cohorts, reliability display, safety exceptions, baseline lock and re-audit rules.

The KPI dictionary defines the first 16 provisional osteoporosis KPIs, including data completeness, fracture history, risk assessment, DXA/VFA, secondary causes, falls/frailty, treatment history, decision documentation, continuity, denosumab timeliness, transition safety, fracture-on-treatment review and Patient Voice measures.

The case-form schema defines neutral prospective encounter capture with explicit separation between what occurred clinically and what is traceable in formal documentation.

---

## 2026-08-22 — Baseline Audit pilot UI Step 1 implemented in PR #1

The first code implementation slice was created on branch `feat/baseline-audit-pilot-v1` and opened as PR #1.

Added:

```text
static/baseline-audit/index.html
static/baseline-audit/styles.css
static/baseline-audit/app.js
```

The implemented first screen includes:

- pilot case identity/progress;
- encounter metadata;
- visit reason and osteoporosis status;
- adaptive sex/menopause context;
- structured Heidi exposure/output/review/correction capture;
- quick applicability signals;
- privacy warnings;
- local draft save/resume and case list using browser `localStorage` only;
- no live KPI coaching or red/green baseline performance feedback.

Steps 2–6 are intentionally placeholders. No server-side patient-data API, `main.py` rewrite or new production clinical-data storage was introduced.

The existing FastAPI static mount can serve the page at `/static/baseline-audit/` after merge/deploy.

---

## 2026-08-22 — Step 1 refined into adaptive encounter context

After clinical review, Step 1 was reworked to separate patient relationship (`new_to_service` vs `established_patient`) from encounter archetype. It added measured/source height, weight/BMI, derived height loss, fracture recency, glucocorticoid dose/duration, falls count, structured secondary conditions, frailty/immobility and conditional sarcopenia case-finding.

Heidi capture was simplified: no raw/corrected transcript and no manual diff; only exposure/review/material-correction metadata with optional one-click correction categories.

A formal encounter-archetype schema was added so the audit no longer applies one checklist to every visit.

---

## 2026-08-22 — Baseline Audit Step 2 implemented

Step 2 added structured fracture events and formal fracture-risk capture. It records review scope, interval fractures, event site/date/fragility/on-treatment status, FRAX/FRAXplus use, country/surrogate model, MOF/hip probability, FN BMD use, explicit risk framework, resulting category and contextual adjustment/override.

The system explicitly avoids internal FRAX-like surrogate scoring and silent hybridization of guideline thresholds.

---

## 2026-08-22 — Selective migration principle adopted for legacy Cockpit data

The new Clinical Excellence dashboard will not copy the legacy Cockpit field-for-field. Existing data are classified as useful, duplicate, outdated or context-specific. Useful fields are preserved and normalized with provenance/timing/applicability before being linked to KPI, audit, learning and improvement loops.

Examples: T-scores are retained but longitudinal DXA gains BMD/LSC/machine comparability; falls/frailty fields are retained but outpatient function and 12-month fall counts are prioritized; numeric labs remain optional while the audit measures whether relevant evaluation occurred; hospital-specific Morse fields are not automatically promoted into the outpatient osteoporosis baseline.

---

## 2026-08-22 — Baseline Audit Step 3 implemented

Step 3 added:

- DXA current-use context, BMD g/cm², T-scores, ROI/artifact review, Z-score relevance;
- longitudinal DXA comparability, machine/cross-calibration, facility LSC and BMD/LSC interpretation status;
- VFA/vertebral-imaging indication separated from action/result;
- secondary-cause process capture separate from optional numeric lab entry;
- legacy mineral/renal labs and bone-turnover markers as optional values;
- conditional secondary-cause lab fields/status;
- outpatient falls/frailty/function assessment with CFS, cognition, immobility, aid, gait/balance and optional TUG;
- conditional sarcopenia case-finding with SARC-F, chair stand, grip strength, gait speed, SPPB and TUG.

Step 3 seeds relevant values from Step 1 to reduce duplicate entry and preserves the baseline rule of no live KPI coaching.

---

## 2026-08-22 — Baseline Audit Step 4 implemented

Step 4 upgraded the legacy treatment/plan concepts into date-aware, auditable objects. It added repeated treatment episodes with exact start/end dates when known, adherence/tolerance/response context, repeated administration events with scheduled/actual/next-due dates, and a structured current clinical decision with rationale, patient preference and optional confidence.

It also added explicit transition/sequencing capture for denosumab exit, post-teriparatide, post-romosozumab and bisphosphonate holiday/restart scenarios, plus repeated follow-up tasks that are intended to become reusable `CareTask` objects later. Encounter close now records whether the plan was complete and whether an unresolved critical item remained.

The Step 4 baseline UI deliberately does not generate treatment recommendations or live guideline-concordance verdicts. Reasoned clinician overrides remain distinguishable from errors, and exact dates are preferred over approximate durations without forcing invented historical dates.
