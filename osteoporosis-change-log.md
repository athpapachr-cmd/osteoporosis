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

- understanding of the condition;
- understanding of the plan;
- understanding of treatment rationale/duration/risks;
- whether questions and preferences were addressed;
- free-text confusion, concern, praise or suggestion.

Repeated feedback patterns can create Signals and Improvement Projects.

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

A five-file active canonical set was created:

```text
AGENTS.md
TODO.md
CLINICAL_EXCELLENCE_PLAN.md
HANDOFF_CURRENT.md
osteoporosis-change-log.md
```

`README.md` remains navigation only and is not a sixth source of architecture truth.

---

## 2026-08-22 — Next major design milestone selected

The next major design task was selected as Baseline Osteoporosis Audit v1 + KPI Dictionary v1, because the baseline must be defined before any dashboard progress score is considered valid.

---

## 2026-08-22 — Prospective baseline strategy adopted

The baseline strategy was changed from a primarily retrospective chart audit to a prospective post-visit encounter-capture baseline.

Approved sequence:

```text
5 pilot cases
→ refine usability/branching
→ freeze form + KPI applicability rules
→ 30 consecutive unique scored baseline cases
→ lock baseline
→ interventions / re-audit
```

Heidi AI is recorded only as an exposure/capture source during baseline; its use is not scored as good practice and is not forced before baseline lock.

The audit explicitly separates clinical process, formal GeSY/documentation trace, and capture quality.

---

## 2026-08-22 — Baseline audit / KPI / case-form schemas created

Machine-readable draft schemas were added for the baseline audit, KPI dictionary, and neutral prospective case form.

---

## 2026-08-22 — Baseline Audit pilot UI Step 1 implemented in PR #1

The first code implementation slice added the baseline-audit pilot UI, including encounter metadata, adaptive context, structured Heidi metadata, quick applicability signals, privacy warnings, and local draft save/resume.

---

## 2026-08-22 — Step 1 refined into adaptive encounter context

Step 1 was reworked to separate patient relationship from encounter archetype and added anthropometrics/BMI/height loss, fracture recency, glucocorticoid dose/duration, falls count, structured secondary conditions, frailty/immobility, and conditional sarcopenia case-finding.

---

## 2026-08-22 — Baseline Audit Step 2 implemented

Step 2 added structured fracture events and formal fracture-risk capture, including FRAX/FRAXplus, country/surrogate model, MOF/hip probabilities, FN-BMD use, explicit framework, resulting risk category, and contextual adjustment/override.

---

## 2026-08-22 — Selective migration principle adopted for legacy Cockpit data

Useful legacy data are preserved and normalized with provenance/timing/applicability before being linked to KPI, audit, learning, and improvement loops; the old Cockpit is not copied field-for-field.

---

## 2026-08-22 — Baseline Audit Step 3 implemented

Step 3 added DXA BMD/T-scores, ROI/artifact review, longitudinal comparability/LSC, VFA/vertebral-imaging indication/action/result, secondary-cause process, optional labs/BTMs, falls/frailty/function assessment, and conditional sarcopenia testing.

---

## 2026-08-22 — Baseline Audit Step 4 implemented

Step 4 added date-aware treatment episodes and administration events, adherence/tolerance/response context, current clinical decision/rationale, patient preference, transition/sequencing capture, follow-up tasks, and unresolved critical-item close.

---

## 2026-08-22 — Baseline Audit Step 5 implemented

Step 5 added encounter-specific communication capture and compact immediate post-visit reflection while keeping clinician-estimated understanding separate from future Patient Voice.

---

## 2026-08-22 — Baseline Audit Step 6 implemented

Step 6 completed prospective capture with documentation provenance, formal GeSY versus Heidi trace, material discrepancies, final clinician-reviewed Heidi summary, capture reliability, remaining information gaps, optional completion time, and readiness for later audit calculation.

PR #8 was merged as `a14be3b9bfd393ccc245665c79bf700cf5eaff55`.

---

## 2026-08-22 — Pre-pilot Patch 2 fixes core save data integrity

The legacy Steps 1–2 save path was changed from full-object replacement to merge behavior, and the temporary pilot-completion snapshot/restore workaround was removed.

---

## 2026-08-22 — Pre-pilot Patch 1 prevents hidden stale dependent data

A central `data-hygiene.js` guard was added to clear hidden dependent values before persistence, including DXA detail/longitudinal fields, Step 4 transition fields, and Step 5 information/misunderstanding state.

---

## 2026-08-22 — Pre-pilot Patch 4 establishes one source of truth for shared Step 1/3 risk fields

Step 1 `risk_context` became canonical for shared falls/frailty/sarcopenia screening fields; Step 3 became a read-only projection plus Step-3-specific detailed assessment.

---

## 2026-08-22 — Pre-pilot Patch 5 normalizes and persists DXA machine identity

`dxa-machine-select.js` normalized current DXA machine identity, preserved optional `machine_label`, and retained unrecognized legacy free text under `other_unknown`.

---

## 2026-08-22 — Second review hardens P1/P2 and verifies P4/P5

A second review identified a residual module-ownership bug in core saves. The core save payload was changed to exclude module-owned slices, and the current-DXA trend path gained an independent `DXA used == yes` guard.

---

## 2026-08-22 — Final applicability ownership fix before smoke test

`applicability_review` was added to the app-core module exclusion list and the adaptive module's post-save repair shim was removed. Applicability state is now preserved through correct ownership rather than asynchronous repair.

---

## 2026-08-22 — External review backlog integrated and final pre-pilot data-quality additions prepared

The complete latest external review was consolidated into `CLINICAL_EXCELLENCE_PLAN.md §20` as a prioritized post-pilot backlog rather than being implemented wholesale before real usability evidence exists.

Two low-risk data-quality additions were approved before the smoke test:

- Step 3 now captures an optional `labs_date` through a native calendar/date input and persists it as `step3.labs.labs_date`, allowing entered laboratory snapshots to be distinguished as current versus historical.
- Step 6 source-conflict details are now conditional on `source_conflict_present=yes`; `conflict_resolution` and `conflict_note` collapse and are cleared before persistence when conflict changes away from `yes`.

The Step 3 and Step 6 schemas were updated to make these rules explicit. The next gate is deployment followed by the expanded 14-scenario synthetic smoke test. If it passes, the 5-case real pilot is scheduled to begin on Monday 2026-08-24.