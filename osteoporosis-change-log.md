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

`README.md` remains navigation only and is not a sixth source of architecture truth.

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

---

## 2026-08-22 — Pre-pilot Patch 2 fixes core save data integrity

External review identified that the legacy Steps 1–2 `saveDraft()` replaced the whole stored case object. Because Steps 3–6 and longitudinal review persist as additional top-level slices, a later core save could therefore remove those slices.

The root save behavior was changed from full replacement to merge-by-`internal_uuid`:

```text
stored case + current core state → merged stored case
```

The temporary `pilot-completion.js` snapshot/restore workaround was removed, including its asynchronous `setTimeout(0)` restoration path. Pilot completion remains intact but no longer carries responsibility for preserving module slices during ordinary saves.

This change is a pre-pilot data-integrity fix and should be retained before any real pilot case is collected.

---

## 2026-08-22 — Pre-pilot Patch 1 prevents hidden stale dependent data

External review identified that child fields hidden by parent toggles could retain and re-persist stale values. This was particularly dangerous for DXA because prior BMD/T-score values could remain available to longitudinal trends after `DXA used` was changed to `No`.

A central pilot data-hygiene guard was added at:

```text
static/baseline-audit/data-hygiene.js
```

It clears dependent DOM values before the step modules collect/persist state and sanitizes legacy stale values already present in the active localStorage case. Covered dependencies include DXA detail/longitudinal fields, Step 4 transition fields, Step 5 information-type selections and misunderstanding-correction state.

The guard is loaded by the baseline bootstrap before Step 6/pilot completion. This is a pre-pilot data-integrity measure so hidden/non-applicable child values cannot silently contaminate longitudinal review or later audit calculation.

---

## 2026-08-22 — Pre-pilot Patch 4 establishes one source of truth for shared Step 1/3 risk fields

External review identified that falls count, CFS, cognitive impairment, immobility and basic sarcopenia screening were persisted independently in Step 1 and Step 3, allowing the two values to diverge.

The canonical rule is now:

```text
Step 1 risk_context = source of truth
Step 3 = read-only projection + additional detailed functional assessment
```

A runtime module `static/baseline-audit/shared-risk-source.js` synchronizes the Step 3 projection from Step 1, disables editing of the shared controls in Step 3, and removes duplicate copies of those shared fields from persisted `step3` state. Detailed Step 3-only fields such as falls/function review, fall injury, ambulatory aid, gait/balance concern, TUG, chair stand, grip strength, gait speed, SPPB and actions remain independently editable.

This removes ambiguity for later derived signals/KPI logic and prevents one encounter from carrying two competing versions of the same clinical fact.

---

## 2026-08-22 — Pre-pilot Patch 5 normalizes and persists DXA machine identity

The DXA machine field was normalized before the longitudinal layer initializes. `static/baseline-audit/dxa-machine-select.js` converts the Step 3 control to a fixed machine select, preserves optional local machine identity in `machine_label`, maps recognized legacy labels to normalized keys and preserves unrecognized legacy free text under `other_unknown` rather than dropping it.

This removed the practical text→select race and made machine identity persistent across save/reload while keeping the current pilot data model backward-compatible.

---

## 2026-08-22 — Second review hardens P1/P2 and verifies P4/P5

A second independent code review correctly identified a residual P2 ownership bug: `currentCase` could contain stale module slices from load time, and a core merge could therefore overwrite fresher `longitudinal_review` state.

The root rule was tightened so `app-core` now excludes module-owned slices from its save payload:

```text
step3 / step4 / step5 / step6 / longitudinal_review / pilot_completion / audit_evaluation_v1
```

The P1 path was also hardened: `data-hygiene.js` now loads before `longitudinal.js`, and `currentDxaPoint()` independently refuses to expose a current DXA point unless `DXA used == yes`. Thus chart correctness no longer depends only on sanitizer timing.

The same review re-verified that P4 and P5 were already active through their dedicated runtime modules. They were not rewritten again during this hardening pass because there was no remaining pilot data-integrity defect demonstrated in those paths; unnecessary ownership refactoring was deferred until after smoke testing or pilot evidence if still warranted.

---

## 2026-08-22 — Final applicability ownership fix before smoke test

A later independent review found that Patch 7 introduced a new top-level `applicability_review` slice but the core save exclusion list had not been extended to include it. The adaptive module therefore relied on a capture-phase Save/Finish snapshot plus `setTimeout(persistReview)` repair shim to restore fresh applicability state after a core save.

The ownership rule was corrected at the source:

```text
applicability_review = module-owned
app-core must not write it
```

`applicability_review` was added to the `app-core` module exclusion list and the post-save repair shim was removed from `adaptive-applicability.js`. The smoke test now explicitly requires an applicability override to survive a Step 1/2 Save and reload without any repair listener.

---

## 2026-08-22 — External review backlog integrated and final pre-pilot data-quality additions prepared

The complete latest external review was consolidated into `CLINICAL_EXCELLENCE_PLAN.md §20` as a prioritized post-pilot backlog rather than being implemented wholesale before real usability evidence exists.

Two low-risk data-quality additions were approved before the smoke test:

- Step 3 now captures an optional `labs_date` through a native calendar/date input and persists it as `step3.labs.labs_date`, allowing entered laboratory snapshots to be distinguished as current versus historical.
- Step 6 source-conflict details are now conditional on `source_conflict_present=yes`; `conflict_resolution` and `conflict_note` collapse and are cleared before persistence when conflict changes away from `yes`.

The Step 3 and Step 6 schemas were updated to make these rules explicit. The next gate is deployment followed by the expanded 14-scenario synthetic smoke test. If it passes, the 5-case real pilot is scheduled to begin on Monday 2026-08-24.

---

## 2026-08-23 — Synthetic pre-pilot form smoke passed

The expanded 14-scenario synthetic smoke passed. It covered FRAX/FRAXplus persistence, DXA stale-data hygiene, shared Step1→Step3 risk projection, DXA machine persistence, archetype applicability, applicability override persistence, completion denominator behavior, prior DXA stable IDs, BMI source behavior, `labs_date`, Step-6 conflict clear-on-collapse and full Save/Finish/reload persistence.

This closed the planned pre-pilot form-integrity gate.

---

## 2026-08-23 — Patient Registry + durable encounter/lab persistence implemented

PR #21 introduced the patient-centric clinical persistence layer while preserving legacy runtime composition.

Added PostgreSQL-backed objects/endpoints for:

```text
clinical_patients
clinical_encounters
clinical_lab_snapshots
```

The browser workspace gained patient search/create/open, encounter create/load/update, server Save/Finish synchronization and longitudinal lab snapshots keyed by actual `labs_date`.

`localStorage` became working cache rather than durable clinical source of truth.

---

## 2026-08-23 — Session authentication and online PostgreSQL storage verified

PRs #22/#23 added browser-session clinical authentication and a non-sensitive storage-status path/startup log.

Live startup evidence verified:

```text
dialect=postgresql
database_url_configured=True
storage_mode=online_database
clinical_key_configured=True
```

This proved that the production clinical layer was using an online PostgreSQL database rather than local SQLite fallback.

The verification did not constitute a claim that every legacy route or the whole service was privacy/GDPR-compliant.

---

## 2026-08-23 — Patient persistence and longitudinal laboratory browser smoke passed

Using a synthetic test patient, login, patient reload, retained encounter values and new laboratory persistence were verified. Newly entered laboratory values appeared in the comparative longitudinal table.

This closed the main online persistence proof needed before real pilot use.

---

## 2026-08-23 — Longitudinal laboratory UI simplified

PR #24 removed the duplicate laboratory-history table from the Patient Registry area and retained the comparative history only in Step 3.

Added `Νέες αναλύσεις`, which clears the current laboratory-entry form without deleting historical server snapshots. Both UI smokes passed.

---

## 2026-08-23 — Clinical Calendar foundation and workspace navigation added

PR #25 added the first `clinical_appointments` store/API and weekly Clinical Calendar foundation.

PR #26 added Calendar access from the Baseline workspace and changed the service root to the current Baseline/Clinical Excellence workspace while keeping the legacy Cockpit separately accessible.

The architecture froze:

```text
Appointment != CareTask
```

---

## 2026-08-23 — Clinical Calendar restricted to osteoporosis-related appointments

PR #27 restricted the Module 01 Calendar to:

```text
osteoporosis_first
osteoporosis_review
osteoporosis_unspecified
prolia
aclasta
```

Unrelated appointments are excluded. Explicit semantics outrank duration, and duration alone cannot classify a generic appointment as osteoporosis.

The live Setmore/Digital Secretary feed remained unimplemented and was later intentionally paused.

---

## 2026-08-23 — Encounter finalization integrity hardened

PR #29 was squash-merged as:

```text
0a2147b8ae5fb8316bde16c8fbb4c0d96aba2194
```

A completed encounter can no longer silently revert to `draft` on a later ordinary Save. Material edits after completion become `amended`, and amended encounters remain amended on later saves.

Focused deterministic transition tests were added. A final live synthetic browser smoke remains an explicit operational verification gate.

---

## 2026-08-25 — Clinical Practice Review established as a first-class product program

A real Heidi transcript was used as a design case to distinguish traditional audit from deeper practice improvement.

The product objective was expanded from “did the KPI occur?” to structured review of:

- clinical completeness;
- reasoning sequence;
- decision quality;
- risk interpretation;
- safety;
- scientific accuracy of patient communication;
- shared decision making;
- consultation flow/efficiency;
- follow-up execution.

The new design introduces `PracticeReview` and `PracticeObservation`, Quick Review, Deep Review/RED TEAM, Decision Reconstruction, Communication Review, longitudinal pattern detection and explicit conversion of accepted observations into existing Signal/root-cause/intervention/re-measurement machinery.

Transcript-assisted capture is designed as:

```text
raw Heidi transcript
→ ephemeral structured extraction
→ candidate values
→ clinician review/accept/reject/edit
→ authoritative structured encounter data
```

The raw transcript is not persisted by default. Discussion, recommendation, preference and final decision must remain distinct.

---

## 2026-08-25 — Adaptive clinical consultation flow direction approved

The visible order of the consultation was explicitly separated from the underlying Steps 1–6 storage/audit schema.

Canonical principle:

```text
CLINICAL WORKFLOW PRESENTATION != STORAGE / AUDIT SCHEMA
```

The candidate osteoporosis flow becomes archetype-adaptive and generally moves through:

```text
why today / interval change
→ fracture + falls/function
→ DXA/VFA/imaging
→ secondary causes/labs
→ risk synthesis
→ treatment history/response
→ options/recommendation/preference/final decision
→ targeted lifestyle/communication
→ explicit close/tasks/timing
```

The design includes a risk-synthesis gate before final treatment choice and a concise Close card for prerequisites, patient tasks, clinician tasks, communication, timing and unresolved critical items.

The actual post-pilot workflow will be informed by multiple encounters/audit/Practice Review evidence rather than treating one transcript as the universal template.

---

## 2026-08-25 — Canonical control plane upgraded for cross-conversation continuity

The Clinical Excellence documentation architecture was aligned with the stronger operational pattern already used in the Digital Secretary control plane.

The previous five-file set was superseded by six active canonical authorities:

```text
AGENTS.md
TODO.md
CLINICAL_EXCELLENCE_PLAN.md
SLICE_PLAN_CURRENT.md
CURRENT_OPERATIONAL.md
osteoporosis-change-log.md
```

`HANDOFF_CURRENT.md` became a compatibility redirect only.

`CURRENT_OPERATIONAL.md` is now the sole operational NOW/writer lock. `SLICE_PLAN_CURRENT.md` owns the exact approved design of one active implementation slice.

A fresh session must verify remote `main`, bootstrap from the active canonicals and reconstruct the current writer/status/next action rather than relying on chat history.

The first active Practice Review implementation slice was frozen as **PR-1 Transcript Intake + Candidate Extraction v1**, deliberately stopping before any extracted value can write into authoritative patient data.
