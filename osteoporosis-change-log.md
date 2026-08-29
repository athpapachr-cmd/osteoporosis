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

- Step 3 now captures an optional `labs_date` through a native calendar/date input and persists it as `step3.labs.labs_date`, allowing entered laboratory snapshots to be distinguished as current versus historical results.
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

---

## 2026-08-26 — Encounter finalization live verification closed

The product owner completed the agreed three synthetic browser checks for PR #29:

```text
completed + no-op Save → completed
material change + Save → amended
reload/reopen → amended and loadable
```

All three passed. The encounter-finalization integrity gate is therefore closed and no longer blocks PR-1 design work.

---

## 2026-08-26 — PR-1 pre-code review triggered in-slice REPLAN

A fresh, read-only design review bootstrapped from the canonical `main` and inspected actual runtime/schema seams before implementation. It identified three material design defects:

1. the YAML/documentation target vocabulary does not always match the actual browser payload persisted in `clinical_encounters.payload_json`;
2. a singular candidate `value + target_mapping` cannot safely represent composite clinical assertions such as fracture events, treatment episodes and final decisions;
3. transcript PHI requires a dedicated sanitized request-validation/error boundary rather than relying on ordinary validation-error representation.

The product decision was **REPLAN within PR-1**, not a new roadmap phase and not permission to start runtime code.

The corrected v3 design freezes:

- composite `components[]` candidates;
- deterministic module-owned `target_mappings[]`;
- `osteoporosis_runtime_targets_v1` based on actual persisted runtime paths;
- explicit mapped/ambiguous/unmapped behavior;
- provider-neutral extraction with no provider-authored application paths;
- PHI-safe validation/logging;
- no implicit provider SDK retries;
- ephemeral browser state including BFCache cleanup;
- deterministic tests plus synthetic provider eval gates;
- explicit product-owner `IMPLEMENT` approval before any runtime writer/branch.

The canonical correction is carried by docs-only PR #32. No PR-1 runtime code is part of that correction.

---

## 2026-08-26 — Near-term Clinic Utilities detour registered

The product owner requested a bounded near-future detour to bring two existing standalone clinic tools into the Personal Clinical Excellence workspace.

The first tool is a **physiotherapy referral text generator**. The intended direction is to inspect the current source first, preserve/refine its useful generation behavior, integrate it into the Clinical Excellence navigation/workspace and align its presentation with the shared visual system.

The second is a **radiofrequency treatment request/PDF workflow**. After source inspection and visual integration, the intended workflow includes a protected durable request registry with minimum lifecycle states:

```text
pending
approved_awaiting_application
completed
```

The future UI should make pending requests, approved requests awaiting application and historical completed procedures easy to review. A repeat request must be created by cloning reusable data from a prior request into a **new request identity**, with reconfirmation/editing before submission; the historical request must remain unchanged.

This work is classified as cross-module **Clinic Utilities / Clinical Operations**, not as Osteoporosis Module 01 clinical logic and not as a new clinical Module 02. It is roadmap-approved but not the active runtime slice. Before activation, the source websites must be located/provided and inspected read-only, `CURRENT_OPERATIONAL.md` must explicitly switch the active slice, and a small dedicated design slice must be frozen before implementation.

---

## 2026-08-27 — CU-1 Physiotherapy Referral v2 pre-code design reached DESIGN-COMPLETE

The bounded Clinic Utilities detour completed the full CU-1 physiotherapy referral pre-code design.

All planned regional/shared v1.1 clinical profiles were frozen, followed by machine-contract hardening of typed state, canonical route/gateway identities, route ownership/precedence, ID/enumeration normalization, formatter behavior and safety-result semantics.

Repeat completeness review v2 identified two remaining declarative gaps:

```text
R1 — safety/consistency trigger conditions
R2 — route-specific required/conditional validation
```

PR #52 added the declarative rule catalog, route-requirements catalog, canonical context value sets, validation-error policy, typed safety/neurological-screen semantics and focused synthetic fixtures. A post-merge review then found two exact shared-muscle transcription defects; PR #53 corrected them without reopening clinical content:

```text
MRI/ultrasound confirmation remains optional context
major-avulsion/rupture concern remains the canonical safety-input flag
```

`clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW_V3.md` classifies the resulting pre-code design as:

```text
DESIGN-COMPLETE
```

The single normative machine entrypoint remains:

```text
clinic_utilities/contracts/cu1_contract_manifest_v1.yaml
```

The frozen first implementation boundary is:

```text
ephemeral ReferralDraftV1
→ deterministic validation/rule evaluation
→ ShortReferralFormatter / DetailedReferralFormatter
→ generated text
→ copy / print
```

This milestone does **not** authorize runtime implementation. No CU-1 runtime writer exists, no production CU-1 code has been written, and persistence remains outside the first implementation scope. A future implementation requires a fresh six-canonical bootstrap, explicit product-owner authorization and a new implementation slice/branch.

---

## 2026-08-28 — CU-1 Physiotherapy Referral v2 runtime implemented and deployed

The bounded CU-1 runtime slice was completed against the frozen `cu1_contract_manifest_v1.yaml` contract without reopening the clinical taxonomy or adding referral persistence.

PR #56 was independently reviewed at exact head `e04004add617afa7222c51d0d669c2134dd8f575`, passed the final focused GitHub Actions suite at **29/29 tests**, and was squash-merged as `c1da07f581cf8ccf1159d18bb63c23b674cbe9bd`.

Delivered runtime includes the protected Clinical Excellence physiotherapy-referral page/API, deterministic normalization/route validation, declarative safety/consistency evaluation, short/detailed formatting, copy/print workflow, exact frozen shared-gateway enforcement, and fail-closed validation of forged gateway/safety acknowledgement/disposition state.

The first implementation remains deliberately ephemeral: referral drafts and generated text are not persisted to PostgreSQL, localStorage or sessionStorage.

Render auto-deploy `dep-da8afeuk1f9s73f5sr6g` completed successfully and reported `live` at the exact merge commit. External route-level HTTP smoke from the assistant execution sandbox could not be executed because DNS resolution failed before reaching the Render host; this is recorded as not proven rather than as an application failure.

CU-1 runtime v1 is closed after canonical closeout/writer-lock release. CU-2 and PR-1 are not authorized by this completion.

---

## 2026-08-29 — CU-1 history/evidence/rehabilitation design hardening advanced through cervical C4

PR #63 on `design/cu1-history-evidence-timeline-2026-08-28` established the v1.1 structured-history, evidence-authority and rehabilitation-sequence semantics as a design-only extension of the deployed CU-1 referral utility. Runtime evidence-aware recommendation generation remains explicitly unauthorized.

The evidence corpus was normalized into a reviewed sharded registry. Tranche 2 and tranche 3 both passed their promotion gates, the shard-integration gate passed, and route-specific native coverage was then advanced with matching history prompts, evidence profiles/sequences or explicit evidence-gap behavior, formal route reviews and regression fixtures.

Reviewed native coverage now includes:

```text
calcific rotator-cuff tendinopathy
glenohumeral instability/dislocation context split
glenohumeral osteoarthritis management-context split
degenerative meniscal lesion — conservative rehabilitation
patellar tendinopathy
thumb CMC-1 osteoarthritis
C1 nonspecific neck pain
C2 neck pain with radiating upper-limb symptoms
C3 headache with cervical musculoskeletal features / formal-CGH split
C4 cervical-dizziness presentation / clinician-established split
```

Material evidence-governance corrections preserved during the work include:

- posterior shoulder-instability Part-II postoperative rehabilitation/RTS evidence is not nonoperative authority;
- multidirectional-instability absence of eligible comparative RCTs is not relabelled as a very-low treatment-effect estimate;
- GHOA best-practice opinion is not represented as comparative efficacy, and the 2026 review confirming no nonsurgical physiotherapy RCTs is preserved as an evidence gap rather than a negative effect estimate;
- degenerative meniscal MRI findings do not automatically establish symptom causation or surgical indication, and true locking remains distinct from clicking/catching;
- contemporary patellar-tendinopathy evidence does not justify freezing eccentric, HSR, isometric or PTLE loading as a universal physician protocol or inventing numeric RTS thresholds;
- thumb-CMC assessment measures are not converted into rehabilitation progression thresholds and orthosis type/wear schedule remain execution detail;
- C1 generic nonspecific-neck authority cannot absorb radiating, headache, dizziness or post-traumatic cervical routes;
- C2 preserves subjective radiating symptoms, objective neurological findings and formal radiculopathy as distinct concepts, and cervical-radiculopathy-specific synthesis does not automatically apply to symptom-only cases;
- C3 cervical headache features do not automatically establish formal cervicogenic headache, ICHD-3 remains diagnostic-boundary authority rather than an autonomous diagnostic engine, and low-certainty network rankings/manual-therapy signals are not turned into a mandatory protocol;
- C4 dizziness plus neck symptoms, cervical/sensorimotor testing or treatment response does not establish cervical causation. The Bárány Society no-routine-criteria position is preserved, presentation-only C4 has an explicit blocked evidence gap, and only a clinician-entered cervical-dizziness context receives a cautious one-phase evidence-bounded sequence. Alternative vestibular/migraine/neurological/vascular/cardiovascular/otological causes are not silently declared excluded, vestibular rehabilitation is not an automatic default, and outcome-specific low/very-low GRADE evidence is not flattened into one synthetic certainty.

The global CU-1 gate remains:

```text
SHARD INTEGRATION             PASS
REVIEWED ROUTE COVERAGE       PASS for completed routes/contexts
GLOBAL ROUTE COVERAGE         FAIL / IN PROGRESS
ROUTE-HISTORY PROMPTS         FAIL globally until all routes curated
ROUTE FIXTURE CORPUS          FAIL globally until all routes curated
DESIGN-COMPLETE               NO
RUNTIME AUTHORIZED            NO
```

The next authorized route after this reconciliation is `post_traumatic_neck_pain` (C5), followed by the remaining wrist/hand and elbow routes and then the remaining routine registry. PR #63 remains draft and must not be merged merely because individual route gates pass.

---

## 2026-08-29 — C5 post-traumatic / whiplash-associated cervical route closed as a context split

The C5 route `post_traumatic_neck_pain` passed exact evidence/applicability review without being collapsed into either a generic nonspecific-neck pathway or a single undifferentiated whiplash protocol.

The normative C5 design now separates:

```text
recent explicit uncomplicated WAD
→ rep_c5_recent_whiplash_wad_v1
→ seq_c5_recent_whiplash_wad_v1
→ sequence_complete

persistent explicit WAD
→ rep_c5_persistent_whiplash_wad_v1
→ seq_c5_persistent_whiplash_wad_v1
→ sequence_complete

explicit WAD with unclear/not-stated temporal phase
→ rep_c5_whiplash_phase_unresolved_v1
→ blocked_evidence_gap

other post-traumatic cervical pain without explicit matching WAD context
→ rep_c5_other_posttraumatic_neck_pain_v1
→ blocked_evidence_gap

unresolved structural / neurological safety context
→ rep_c5_unresolved_posttraumatic_safety_v1
→ routine sequence blocked
```

The current SIRA third-edition 2014 acute-WAD guideline remains the active recent/acute WAD authority at this review date. The proposed fourth edition remains draft/non-approved and is not used as normative authority.

Evidence-strength boundaries were preserved rather than hybridized:

- SIRA stay-active advice and neck exercise remain separate Level-B recommendations for recent/acute uncomplicated WAD;
- manual therapy remains a selected limited-evidence Level-C adjunct rather than automatic core care;
- SIRA's advice against prolonged reduction of usual activities remains a consensus clinical-practice point, while the recommendation against routine immobilisation collars is Level A;
- the 2024 guided neck-specific exercise meta-analysis does not convert observed study frequency/duration patterns into a universal referral schedule;
- the 2025 education-plus-exercise GRADE synthesis remains very-low certainty and does not establish a mandatory superior combined bundle;
- persistent objective neurological signs with ongoing disability use the OPTIMa/Côté 2016 medical-review context after a pre-PASS source-attribution correction.

C5 also freezes the following safety/applicability boundaries:

```text
post-traumatic neck pain != automatic WAD
recent WAD != persistent WAD
vague duration != inferred WAD temporal phase
C5 selection != fracture/dislocation/instability excluded
WAD/QTF grade != CU-1 inferred classification
other cervical trauma != WAD authority != generic C1 fallback
post-traumatic headache/dizziness/arm symptoms != automatic C3/C4/C2 disease-specific authority
patient-specific structural/healing restriction > conflicting uncomplicated-WAD default
```

Both nonblocked WAD sequences remain single-phase and deliberately contain no universal numeric progression threshold, fixed visit frequency, total PT course duration or elapsed-time-only progression rule.

Focused GitHub Actions passed at the exact C5 review head before manifest activation. The reviewed shard and matching fixtures were then activated in the normative manifest and coverage matrix.

The overall CU-1 gate remains:

```text
SHARD INTEGRATION             PASS
C1-C5 CERVICAL COVERAGE       PASS for reviewed contexts
GLOBAL ROUTE COVERAGE         FAIL / IN PROGRESS
ROUTE-HISTORY PROMPTS         FAIL globally until all routes curated
ROUTE FIXTURE CORPUS          FAIL globally until all routes curated
DESIGN-COMPLETE               NO
RUNTIME AUTHORIZED            NO
```

PR #63 remains draft/unmerged. No runtime recommendation logic, persistence change, CU-2 work or PR-1 restart was authorized.

The exact next route is now `lateral_elbow_tendinopathy`, followed by the remaining wrist/hand and elbow routes and then the remaining routine registry.

---

## 2026-08-29 — Lateral elbow tendinopathy route closed as single-phase evidence-bounded authority

The route `lateral_elbow_tendinopathy` passed exact evidence/applicability review and was activated without converting the existing incomplete two-phase seed into a fabricated staged protocol.

The existing core profile identity was retained:

```text
rep_lateral_elbow_tendinopathy_v1
```

but its normative sequence now resolves, through the reviewed route activation amendment, to:

```text
seq_lateral_elbow_evidence_bounded_v1
→ sequence_complete
→ single required phase
→ progression_criteria: []
```

The 2022 APTA/JOSPT lateral-elbow CPG remains the route-specific clinical-practice authority. The reviewed route preserves:

- Grade-B resisted wrist-extensor exercise for subacute/chronic LET, with isometric, concentric and/or eccentric loading but no universal dose;
- Grade-F high-demand reintroduction as a conditional work/sport/hobby direction rather than a second phase requiring an invented transition threshold;
- Grade-C proximal shoulder/scapular training only when an actual proximal impairment is present;
- selected Grade-B local manual therapy, dry needling and rigid taping recommendations without confusing evidence direction with automatic adjunct selection;
- Grade-F counterforce/wrist-support orthosis authority only for selected aggravating-activity/immediate contexts;
- education, activity/load modification and ergonomic/workstation context without a fixed conservative-care package.

Two 2024 syntheses were added as current effect-context authority. The Cochrane manual-therapy/exercise review found low-certainty, generally small and non-durable effects, while the Campos non-invasive-therapy review found mostly small-to-no effects with evidence commonly low or very low certainty. These sources narrow claims about magnitude, durability and universal superiority; they do not transform individualized exercise into a `do_not_offer` recommendation.

A pre-PASS audit corrected a synthetic outcome-measure strength. The final logical registry keeps:

```text
PRTEE / DASH / PSFS or high-demand activity-specific function measures
→ CPG Grade A

ROM / pressure-pain threshold / pain-free grip / maximum grip
→ CPG Grade B
```

without flattening the two families into one hybrid strength. These measures remain tracking/assessment tools and are not automatic loading-progression, discharge, return-to-work or return-to-sport thresholds.

Hard LET boundaries now include:

```text
lateral elbow pain or local tenderness != automatic LET diagnosis
Cozen/Mill/Maudsley-type finding != automatic diagnosis
imaging common-extensor abnormality != automatically symptomatic diagnosis
acute/highly irritable LET != automatic subacute/chronic Grade-B loading authority
objective PIN/radial motor deficit != routine LET
cervical/radicular pattern != routine LET
substantial mechanical block or material trauma/instability != routine LET
CPG recommendation direction != automatic treatment selection
newer low-certainty synthesis != proven no-effect claim
```

Matching route-history prompts and `cu1_lateral_elbow_fixtures_v1.yaml` were added. The route fixture corpus explicitly tests acute-vs-subacute/chronic applicability, high-demand Grade-F behavior, diagnosis-vs-finding separation, PIN/cervical owner boundaries, adjunct selection, CPG-grade preservation, outcome-measure grading, 2024 low-certainty interpretation, ESWT non-auto-authority and missing-history semantics.

The exact route review passed focused GitHub Actions before activation. The route shard, activation amendment, manifest and coverage matrix were then reconciled. Runtime evidence-aware generation remains unauthorized.

The global CU-1 gate remains:

```text
SHARD INTEGRATION             PASS
LATERAL ELBOW ROUTE           PASS
GLOBAL ROUTE COVERAGE         FAIL / IN PROGRESS
ROUTE-HISTORY PROMPTS         FAIL globally until all routes curated
ROUTE FIXTURE CORPUS          FAIL globally until all routes curated
DESIGN-COMPLETE               NO
RUNTIME AUTHORIZED            NO
```

PR #63 remains draft/unmerged. No runtime recommendation logic, persistence change, CU-2 work or PR-1 restart was authorized.

The exact next route is now `medial_elbow_tendinopathy`, followed by the remaining wrist/hand and elbow routes and then the remaining routine registry.

---

## 2026-08-29 — Medial elbow tendinopathy route closed as low-certainty evidence-bounded authority

The route `medial_elbow_tendinopathy` passed exact evidence/applicability review and was activated as a distinct medial-elbow authority rather than as a mirrored copy of the lateral-elbow pathway.

The normative objects are:

```text
rep_medial_elbow_tendinopathy_v1
→ seq_medial_elbow_evidence_bounded_v1
→ sequence_complete
→ single required phase
→ progression_criteria: []
```

No current medial-specific rehabilitation clinical-practice guideline with graded recommendations equivalent to the 2022 APTA/JOSPT lateral-elbow CPG was identified. The primary treatment-effect authority is the 2026 See/Loo/Jaafar systematic review of eccentric exercise for medial epicondylitis. It included five small clinical studies totaling 143 patients, used heterogeneous protocols, could not support meta-analysis, and was judged overall low certainty.

The normative interpretation is deliberately narrow:

```text
eccentric flexor-pronator loading
→ may be considered
→ low certainty
→ not mandatory
→ not universally superior
→ no universal dose/frequency/duration

lateral-elbow CPG grades
→ not medial-elbow authority by analogy
```

The 2023 clinical overview and 2024 medial-elbow differential review are retained as history, management-context and differential authority rather than comparative treatment-effect evidence. Narrative staged management descriptions are not converted into validated rehabilitation phases or elapsed-time progression rules.

Hard medial-elbow boundaries now include:

```text
medial elbow pain/tenderness != automatic tendinopathy diagnosis
pain with resisted wrist flexion/pronation != automatic diagnosis
imaging common-flexor abnormality != automatically symptomatic diagnosis
subjective ring/small-finger paresthesia != objective ulnar deficit != formal ulnar neuropathy
material valgus/UCL instability != routine tendon-only pathway
progressive objective ulnar motor deficit != routine tendon-only pathway
major trauma or substantial mechanical block != routine medial tendinopathy
```

Manual therapy, dry needling, taping, orthosis and ESWT are not automatically labelled medial-route evidence merely because analogous interventions appear in lateral-elbow literature or the frozen UI. A separate reviewed evidence claim or clinician-instruction pathway would be required.

Matching route-history prompts and `cu1_medial_elbow_fixtures_v1.yaml` were added. The fixtures test medial-specific profile resolution, diagnosis-vs-finding separation, low-certainty eccentric evidence, no lateral-CGP grade borrowing, subjective-vs-objective ulnar-neural semantics, valgus/UCL owner boundaries, no fixed rest protocol, no narrative-phase promotion, adjunct non-auto-authority and missing-history behavior.

Focused GitHub Actions run #203 passed at exact review head `531e94ce2a92857247962f1d68be20b01e6a05c3` before manifest activation. The medial shard, fixtures, manifest and coverage matrix were then reconciled. Runtime evidence-aware generation remains unauthorized.

The global CU-1 gate remains:

```text
SHARD INTEGRATION             PASS
LATERAL ELBOW ROUTE           PASS
MEDIAL ELBOW ROUTE            PASS
GLOBAL ROUTE COVERAGE         FAIL / IN PROGRESS
ROUTE-HISTORY PROMPTS         FAIL globally until all routes curated
ROUTE FIXTURE CORPUS          FAIL globally until all routes curated
DESIGN-COMPLETE               NO
RUNTIME AUTHORIZED            NO
```

PR #63 remains draft/unmerged. No runtime recommendation logic, persistence change, CU-2 work or PR-1 restart was authorized.

The exact next route is now `ulnar_neuropathy_at_elbow`, followed by the remaining wrist/hand and elbow routes and then the remaining routine registry.

---

## 2026-08-29 — Ulnar neuropathy at the elbow route closed as mild-conservative + nonmild/safety context split

The route `ulnar_neuropathy_at_elbow` passed exact evidence/applicability review without being converted into a generic cubital-tunnel physiotherapy protocol.

The normative context split is:

```text
explicit mild clinical context
+ objective ulnar motor status actually assessed without material deficit
+ no intrinsic atrophy/clawing
+ no unresolved structural/alternative localization owner
→ rep_une_mild_sensory_predominant_v1
→ seq_une_mild_conservative_v1
→ sequence_complete

nonmild / severity unresolved / motor status not sufficiently assessed
→ rep_une_nonmild_or_severity_unresolved_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap

progressive objective motor weakness / intrinsic atrophy / clawing / material neurological worsening
or material trauma / structural compression / nerve instability / discordant localization
→ rep_une_progressive_motor_or_structural_safety_v1
→ rehabilitation_sequence_id: null
→ routine sequence blocked + reassessment/correct owner
```

The 2025 Cochrane review supplies only a narrow conservative signal in mild UNE: information about movements or positions to avoid may reduce subjective discomfort. This is represented as cautious education and individualized provoking-position/movement modification; it does not establish a universal splint, exercise, nerve-gliding, treatment-frequency or course-duration protocol.

The 2025 night-splint systematic review remains very-low certainty and insufficient to determine benefit over advice. The route therefore does not auto-render night splinting or generate a device type, elbow angle, nightly duration or total course. The 2022 physiotherapy systematic review remains heterogeneous and cannot establish nerve gliding, manual therapy, electrical modalities or another method as best or mandatory physiotherapy.

The AANEM neuromuscular-ultrasound guideline was source-identity corrected during pre-PASS review to its 2022 publication year. Its Level-B recommendation remains diagnostic-framework authority to help confirm/localize UNE; it is not treatment-effect certainty, does not replace clinical/electrodiagnostic evaluation and does not create an autonomous diagnosis. The 2025 diagnostic Delphi remains expert-consensus candidate criteria requiring further validation/weighting rather than a CU-1 diagnostic scale.

Hard route boundaries include:

```text
subjective ulnar paresthesia != objective sensory deficit != objective motor deficit != formal UNE diagnosis
positive Tinel or elbow-flexion provocation != definitive diagnosis
formal UNE diagnosis != mild severity
not_assessed motor status != normal != mild conservative eligibility
mild-case conservative evidence != nonmild or unknown-severity complete sequence
progressive motor weakness/atrophy/clawing != routine mild sequence
cervical/plexus/wrist-level ulnar/trauma/structural localization concern != cubital-tunnel fallback
```

The mild sequence contains `progression_criteria: []`; no numeric progression/discharge threshold, fixed visit frequency or total physiotherapy course is manufactured. CU-1 also does not generate an autonomous surgical threshold or procedure choice.

Matching route-history prompts and `cu1_ulnar_elbow_fixtures_v1.yaml` were added. The fixtures explicitly test `not_assessed != normal`, mild-vs-nonmild applicability, symptom-vs-diagnosis separation, Delphi nonvalidation, AANEM ultrasound scope, night-splint certainty, no best physiotherapy method, structural/localization fail-closed behavior and missing-history semantics.

Focused GitHub Actions run #214 passed at exact review head `fc25283bbe8cb5dde0efebb670650990ac4db782` before activation. The UNE shard, fixture extension, manifest and coverage matrix were then reconciled. Runtime evidence-aware generation remains unauthorized.

The global CU-1 gate remains:

```text
SHARD INTEGRATION             PASS
LATERAL ELBOW ROUTE           PASS
MEDIAL ELBOW ROUTE            PASS
ULNAR NEUROPATHY ELBOW ROUTE  PASS as context split
GLOBAL ROUTE COVERAGE         FAIL / IN PROGRESS
ROUTE-HISTORY PROMPTS         FAIL globally until all routes curated
ROUTE FIXTURE CORPUS          FAIL globally until all routes curated
DESIGN-COMPLETE               NO
RUNTIME AUTHORIZED            NO
```

PR #63 remains draft/unmerged. No runtime recommendation logic, persistence change, CU-2 work or PR-1 restart was authorized.

The exact next route is now `posterior_interosseous_nerve_supinator_syndrome`, followed by the remaining wrist/hand and elbow routes and then the remaining routine registry.

---

## 2026-08-29 — Posterior interosseous nerve / supinator route closed as explicit evidence-gap + safety/context split

The route `posterior_interosseous_nerve_supinator_syndrome` passed exact evidence/applicability review without manufacturing a route-specific physiotherapy sequence.

The reviewed model has four machine-distinct contexts:

```text
pain-predominant radial-tunnel / lateral-forearm presentation without material PIN motor deficit
→ rep_pin_pain_only_or_radial_tunnel_mismatch_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap / route mismatch

PIN-pattern motor presentation with diagnosis/etiology/localization unresolved
→ rep_pin_motor_presentation_unresolved_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap

explicit clinician-established spontaneous non-traumatic/non-compressive PIN palsy
+ no space-occupying lesion
+ no progressive motor safety trigger
→ rep_pin_spontaneous_noncompressive_established_v1
→ rehabilitation_sequence_id: null
→ conservative-management context supported, specific PT sequence not established

demonstrable compression/entrapment or mass
or trauma / iatrogenic / inflammatory structural cause
or progressive/materially worsening motor deficit
→ rep_pin_compressive_structural_or_progressive_v1
→ rehabilitation_sequence_id: null
→ specialist/correct-owner/safety behavior
```

The 2026 prospective Japanese multicenter cohort of spontaneous PIN palsy supports initial conservative management in its narrow spontaneous, non-traumatic, non-compressive, no-space-occupying-lesion population. It does not define a nerve-gliding, splinting, manual-therapy, electrical-stimulation, strengthening, visit-frequency or PT-duration protocol. The route-specific 2006 systematic review found no randomized or controlled clinical trials, and the current review did not identify a later high-quality controlled rehabilitation evidence base sufficient to create a criteria-based `RehabilitationSequenceV1`.

Hard PIN boundaries preserved by the review include:

```text
lateral forearm pain / radial-tunnel provocation != motor PIN diagnosis
motor pattern != formal PIN diagnosis != etiology != supinator/Frohse compression site
formal diagnosis != spontaneous/noncompressive etiology
not_assessed motor status != normal or pain-only classification
route selection != cervical / plexus / proximal radial / multifocal motor neuropathy exclusion
investigation finding != rehabilitation protocol
spontaneous conservative-management signal != specific physiotherapy effectiveness evidence
```

The 2026 cohort's six-month motor-recovery signal remains source/population-specific management-course context. CU-1 does not convert it into a universal PT course duration, elapsed-time progression rule, automatic surgical threshold or procedure choice. New/progressive objective motor weakness, demonstrable compression/mass, trauma/iatrogenic/structural cause or other discordant localization requires reassessment or the correct owner.

A pre-PASS source audit corrected the 2006 systematic-review citation to `J Peripher Nerv Syst. 2006;11(2):101-110`, DOI `10.1111/j.1085-9489.2006.00074.x`, corrected the electrodiagnostic review issue year/citation to `HSS J. 2012;8(2):184-189`, DOI `10.1007/s11420-011-9238-8`, and removed a motor-deficit-only safety claim from the pain-only mismatch profile. These were source-identity/applicability corrections only and did not broaden clinical authority.

Matching route-history prompts and `cu1_pin_supinator_fixtures_v1.yaml` were added. The fixture oracle verifies pain-only-vs-motor separation, motor-pattern-vs-diagnosis separation, `not_assessed != normal`, formal-diagnosis-vs-etiology separation, non-universalization of the 2026 six-month signal, fail-closed structural/traumatic/progressive-motor contexts, alternative localization/MMN boundaries, diagnostic-test-vs-treatment separation, non-auto-rendering of nerve gliding/splint/manual/electrical modalities, no generic LET/elbow/peripheral-nerve fallback, and missing-history semantics.

Focused GitHub Actions run #236 passed at exact review head `ce56ea9bcd1774547627e9a46d8470ab2a4053c0` before activation. The reviewed PIN shard and mandatory narrow amendment were activated in manifest `cu1_evidence_manifest_v1_19`; the coverage matrix was reconciled to `cu1_evidence_coverage_matrix_v1_19` and correctly records PIN as `profile_curated / blocked_evidence_gap`, not `sequence_complete`.

The global CU-1 gate remains:

```text
SHARD INTEGRATION             PASS
LATERAL ELBOW ROUTE           PASS
MEDIAL ELBOW ROUTE            PASS
ULNAR NEUROPATHY ELBOW ROUTE  PASS as context split
PIN / SUPINATOR ROUTE         PASS as explicit evidence-gap + safety/context split
GLOBAL ROUTE COVERAGE         FAIL / IN PROGRESS
ROUTE-HISTORY PROMPTS         FAIL globally until all routes curated
ROUTE FIXTURE CORPUS          FAIL globally until all routes curated
DESIGN-COMPLETE               NO
RUNTIME AUTHORIZED            NO
```

PR #63 remains draft/unmerged. No runtime recommendation logic, persistence change, CU-2 work or PR-1 restart was authorized.

The exact next route is now `distal_biceps_tendon_disorder_nonoperative`, followed by the remaining wrist/hand and elbow routes and then the remaining routine registry.