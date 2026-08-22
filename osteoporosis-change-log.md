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

The implemented first screen includes pilot case identity/progress, encounter metadata, visit reason/status, adaptive context, structured Heidi metadata, quick applicability signals, privacy warnings, and local draft save/resume. No server-side patient-data API or production clinical storage was introduced.

---

## 2026-08-22 — Step 1 refined into adaptive encounter context

Step 1 was reworked to separate patient relationship from encounter archetype and added anthropometrics/BMI/height loss, fracture recency, glucocorticoid dose/duration, falls count, structured secondary conditions, frailty/immobility and conditional sarcopenia case-finding. Heidi capture was simplified to exposure/review/material-correction metadata without transcript or manual diff.

---

## 2026-08-22 — Baseline Audit Step 2 implemented

Step 2 added structured fracture events and formal fracture-risk capture: FRAX/FRAXplus, country/surrogate model, MOF/hip probabilities, FN-BMD use, explicit framework, resulting risk category and contextual adjustment/override. Internal FRAX-like surrogate scoring and silent guideline hybridization are avoided.

---

## 2026-08-22 — Selective migration principle adopted for legacy Cockpit data

The new Clinical Excellence dashboard does not copy the legacy Cockpit field-for-field. Useful data are preserved and normalized with provenance/timing/applicability before being linked to KPI, audit, learning and improvement loops.

---

## 2026-08-22 — Baseline Audit Step 3 implemented

Step 3 added DXA BMD/T-scores, ROI/artifact review, longitudinal comparability/LSC, VFA/vertebral-imaging indication/action/result, secondary-cause process, optional labs/BTMs, falls/frailty/function assessment and conditional sarcopenia testing.

---

## 2026-08-22 — Baseline Audit Step 4 implemented

Step 4 upgraded treatment/plan concepts into date-aware treatment episodes and administration events, with adherence/tolerance/response context, current clinical decision/rationale, patient preference, transition/sequencing capture, follow-up tasks and unresolved critical-item close.

---

## 2026-08-22 — Baseline Audit Step 5 implemented

Step 5 added encounter-specific communication capture and compact immediate post-visit reflection. It records condition/risk explanation, results/status discussion, exercise, nutrition, calcium/vitamin D/other supplements, medication/no-drug plan, rationale, alternatives/trade-offs, timing/review point, safety and sequencing communication, questions and patient preferences.

Clinician impression of patient understanding is recorded separately from the later Patient Voice instrument. Post-visit reflection remains low burden and can flag potential case-review, learning, communication/system or safety Signals without displaying a live baseline score.

---

## 2026-08-22 — Baseline Audit Step 6 implemented

Step 6 completed the prospective baseline capture flow by adding documentation provenance and capture-quality review.

It records capture sources, a domain-level matrix for formal GeSY trace versus Heidi trace, material discrepancies, formal-record completeness and missing-content domains, and a final clinician-reviewed Heidi summary seeded from Step 1 without requiring raw/corrected transcripts or manual diffs.

Capture reliability, remaining major information gaps, reasons for limited capture, optional completion time and readiness for later audit calculation are also recorded.

The key interpretation rule was frozen: clinical process is represented by Steps 1–5; formal documentation is a separate evidence axis; Heidi is a supplementary clinician-reviewed capture source and its use is not a quality-success metric. Missing formal documentation must not be silently converted into a clinical omission.

PR #8 was merged into `main` as commit `a14be3b9bfd393ccc245665c79bf700cf5eaff55`.
