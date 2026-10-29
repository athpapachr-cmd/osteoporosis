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

Positive repeated signals can mature into `SUSTAINED_STRENGTH` and should trigger reinforcement, advanced challenge and appropriate external comparison rather than endless basic repetition.

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

## 2026-08-30 — Module 01 C1 + G-1 progressive guidance released

The product owner authorized release of the accepted Module-01 ancestry containing the C1 authoritative Finish correction and G-1 progressive-guidance foundation after the R1/R2 state-integrity blockers were corrected and the complete G-1+C1 regression suite passed.

Release PR #64 was reviewed at exact head:

```text
e7f400b3a99810a5667cf89899f7db91424ea253
```

At that exact head both relevant GitHub Actions checks passed:

```text
g1-guidance            SUCCESS — run 33331923695
baseline-finalization   SUCCESS — run 33331923682
```

PR #64 was squash-merged to `main` as:

```text
a6ba9ef1719a18a48a1756bf08bbd157d448a63e
```

The released runtime includes:

- the single authoritative C1 Finish path with protected completed/amended confirmation semantics;
- the G-1 read-only longitudinal guidance projection;
- coarse clinician-declared visit intent plus quick-notes context;
- deterministic Visit Plan / `WHY NOW` guidance;
- G1-R1 explicit history availability states so unavailable history is never presented as zero history;
- G1-R2 live-DOM-over-persisted semantics so explicit live clearing cannot resurrect stale cached context.

Render auto-deploy was triggered by the merge commit without manual duplication. Deploy:

```text
dep-daa8iv0ae00c73b7eudg
```

reached `live` at the exact runtime release SHA `a6ba9ef1719a18a48a1756bf08bbd157d448a63e`.

Production browser smoke is **not yet proven**. The assistant execution sandbox could not resolve the Render production hostname, so no route-level or authenticated browser request reached the service. This is recorded as a tooling/network limitation, not as evidence of application failure.

Accordingly:

```text
MERGED = YES
DEPLOYED = YES
PRODUCTION-SMOKE-VERIFIED = NO
```

An authenticated synthetic production smoke of C1 Finish/reload and G-1 interactive guidance/history-state behavior remains required before the production-readiness gate is considered fully closed. PR-1/PR-2, real pilot collection, new medication-specific milestone rules and parked physiotherapy/RF work remain outside this release.

---

## 2026-08-31 — G-1 production smoke verified after WHY-NOW correction

Authenticated production smoke by the product owner first confirmed that the released C1/G-1 workflow was working but exposed one bounded presentation defect: the deterministic G-1 WHY-NOW explanation existed but was not sufficiently discoverable in the top `Σημερινή ροή` summary.

PR #66 added the explicit summary prefix:

```text
Γιατί τώρα: <existing deterministic item.why_now>
```

without changing guidance rules, priorities, treatment logic, taxonomy, persistence, schema or KPI semantics.

The correction passed exact-head G-1 workflow runs:

```text
33333512964  SUCCESS
33333526378  SUCCESS
```

and was squash-merged as:

```text
d9423f4dcf6bebd056e83407132c6ce3e25d2280
```

Render auto-deploy `dep-daa93ljncjis739ssef0` reached `LIVE` at that exact runtime correction SHA.

The product owner then directly re-smoked the corrected production path and confirmed:

```text
`Τύπος σημερινής επίσκεψης`
→ visible literal `Γιατί τώρα:` in `Σημερινή ροή`
→ guidance changes dynamically with current visit context
→ surfaced information is experienced as informative / guiding
```

Therefore the C1/G-1 production-readiness smoke gate is closed:

```text
IMPLEMENTED = YES
TESTED = YES
MERGED = YES
DEPLOYED = YES
PRODUCTION-SMOKE-VERIFIED = YES
PILOT-VALIDATED = NO
```

The positive usefulness observation is recorded as product-owner production feedback only. It is not a substitute for the planned real system-assisted pilot or evidence-from-use refinement. No PR-1/PR-2 runtime, new medication-specific milestone content, physiotherapy/RF mutation or real pilot collection was authorized by this smoke closeout.

---

## 2026-08-31 — G-2 evidence-backed osteoporosis guidance design completed

The first evidence-backed clinical-content layer for the G-1 dynamic visit engine reached **DESIGN-COMPLETE** on branch `design/module01-g2-evidence-backed-guidance-2026-08-31`.

The frozen design adds versioned machine contracts for:

```text
evidence registry
→ guidance rules
→ visit profiles
→ therapy milestones
→ normative activation manifest
```

with NOGG 2024 as the primary clinical framework, current EMA/EU regulatory safety sources where applicable, and explicitly separate supporting/context evidence including Endocrine Society, ECTS, ASBMR/BHOF and recent denosumab-discontinuation studies.

Final clinical review corrected two important applicability details before runtime work:

- the NOGG-specific `>7 months` denosumab rebound-risk escalation now requires at least **two reliable actual denosumab doses**, rather than firing after a first dose;
- the NOGG FRAX evidence rule now requires explicit formal-risk indication and NOGG scope, rather than treating every initial visit as evidence that FRAX is mandatory.

The design also freezes explicit non-rules:

```text
no automatic CTX 280/300 ng/L → second zoledronate command
no generic Prolia 4th/8th/10th dose milestone
no automatic treatment-failure label after fracture on treatment
no automatic cardiology/vascular referral for romosozumab without approved clinic policy
```

Medication safety content is classified as checklist guidance when the existing data model cannot prove full/current safety clearance. Two denosumab-exit rules remain evidence-valid but blocked from first runtime activation: post-exit CTX follow-up until reliable denosumab-exit→zoledronate linkage exists, and the no-CTX fallback until CTX-monitoring availability is explicitly represented.

The G-2 machine contract passed GitHub Actions at:

```text
run   33358433732
head  6a40a4a87882a4531c69ce9dff5e0ecd46011d84
result SUCCESS
```

This milestone means the evidence/content contract is reviewed and frozen for bounded implementation. It does **not** mean G-2 is implemented, runtime-tested, merged, deployed, production-smoke-verified or pilot-validated.

---

## 2026-08-31 — G-2 evidence-backed guidance runtime implemented and tested

The bounded G-2 runtime slice was completed on branch:

```text
feat/module01-g2-evidence-guidance-runtime-2026-08-31
```

against the design-complete ancestry:

```text
0395e52ed75f835d49713504df3df4ce51183edf
```

The exact runtime head tested by the complete regression gate was:

```text
e0657ba5924db87b38a0e05514613fbadf45bcd9
```

The implementation adds a pure deterministic osteoporosis evidence-guidance layer over the existing G-1 mechanics rather than replacing G-1 with a monolithic treatment engine.

Runtime flow is:

```text
G-1 protected longitudinal projection
+ live current encounter snapshot
→ G-2 evidence context
→ deterministic evidence contributions
→ merge with existing G-1 Visit Plan
→ existing Σημερινή ροή / Γιατί τώρα renderer
```

The implementation also extends the live snapshot so current Step 1–4 controls and repeated treatment/administration rows outrank stale browser cache, including explicit blank/empty state. Evidence-backed cards display concise provenance, while medication-specific safety rules classified as `checklist_only` are visibly labelled as requiring clinical confirmation rather than automatic clearance.

Key tested clinical/runtime safeguards include:

- NOGG-specific framework/scope guards;
- explicit formal-risk indication before the NOGG FRAX evidence rule;
- exact actual dates for denosumab timing and no counting of scheduled-only doses;
- six-month denosumab evidence due as an ephemeral derived value;
- specific >7-month NOGG rebound escalation only after at least two reliable actual doses;
- fail-closed suppression of exact denosumab milestones when history conflicts;
- no automatic treatment-failure/switch label after fracture on treatment;
- no automatic selected-agent write during denosumab exit guidance;
- exact-start requirements for oral-bisphosphonate 12–16-week / ≥5-year and zoledronate ≥3-year milestones;
- R15/R16 remaining inactive;
- no CTX 280/300 automatic retreatment command;
- no generic Prolia 4th/8th/10th milestone.

GitHub Actions evidence:

```text
workflow: G2 evidence guidance runtime
run:      33403182604
head:     e0657ba5924db87b38a0e05514613fbadf45bcd9
result:   SUCCESS
```

The single workflow job passed the frozen G-2 contract validator, focused G-2 core/live-state/wiring regressions, all inherited G-1 core/wiring/UI-state/WHY-NOW regressions, the authoritative Finish browser regression and the server finalization lifecycle regression.

Exact-head comparison found the branch ahead of `main` with merge base exactly `5182d250e244b2ed9e086138cb3b2edcdb967e25`, behind by 0, and no PR-1/PR-2 or parked physiotherapy/RF leakage.

This milestone closes only the bounded implementation/test gate:

```text
IMPLEMENTED = YES
TESTED = YES
PRODUCT-OWNER RELEASE REVIEW = NO
PR = NONE
MERGED = NO
DEPLOYED = NO
PRODUCTION-SMOKE-VERIFIED = NO
PILOT-VALIDATED = NO
```

The runtime/canonical writer lock was released after closeout. A separate fresh release-readiness review and explicit product-owner release authority are required before any PR/merge/deploy action.

---

## 2026-09-02 — G-3 released, production visibility hotfix deployed, re-smoke verified

G-3 guidance salience and deterministic longitudinal patient summary were released through PR #70 after exact-head regression review. PR #70 was squash-merged to `main` as:

```text
ef17367c7b8959f51e05b80909226804951d1bc7
```

Render auto-deploy `dep-dabj38s9v7es73fmq800` reached `live` at that exact commit. Initial product-owner production smoke then failed both new presentation acceptance criteria: explicit `Νέο` salience and `Σύνοψη ασθενούς` were not visible.

Investigation identified three bounded presentation/delivery defects rather than a clinical-rule defect:

1. static baseline assets lacked explicit no-store release-coherency headers;
2. the summary root was hidden before an active protected patient existed;
3. `Νέο` presentation compared only card IDs, so a new material evidence trigger on an already-visible card was not treated as newly surfaced.

PR #71 corrected those defects without changing G-2 evidence rules, treatment thresholds, patient persistence, DB schema or C1 Finish semantics. It was squash-merged as:

```text
ab94c6286bdc49cb8304b072e557c5eb0a96b0c6
```

Render auto-deploy:

```text
dep-dabolap5efls739s9am0
```

reached `live` at the exact hotfix merge SHA. The product owner then re-smoked production and reported that all intended G-3 elements were visible and working well, specifically including `Νέο` and `Σύνοψη ασθενούς`.

Therefore:

```text
G-3 IMPLEMENTED = YES
G-3 TESTED = YES
G-3 MERGED = YES
G-3 DEPLOYED = YES
G-3 PRODUCTION-SMOKE-VERIFIED = YES
G-3 PILOT-VALIDATED = NO
```

The successful re-smoke is production validation of the released UI behavior only; it is not the planned real-clinic system-assisted pilot.

---

## 2026-09-02 — G-4 workspace ergonomics and RF utility navigation implemented and tested

Product-owner evidence from use after successful G-3 re-smoke identified three bounded workspace needs: collapse/expand for the longitudinal patient summary and current visit flow, sticky access to the patient summary while scrolling, and Cockpit access to the existing radiofrequency-treatment PDF utility.

G-4 was implemented on branch:

```text
feat/module01-g4-collapsible-sticky-summary-rf-utility-2026-09-02
```

The runtime adds only presentation/navigation behavior:

- native accessible `Σύμπτυξη / Ανάπτυξη` controls for `Σύνοψη ασθενούς` and `Σημερινή ροή`;
- sticky positioning for the existing single patient-summary surface;
- per-browser `sessionStorage` UI preference only, with no clinical-data write;
- a Cockpit `Clinic Utilities` group retaining the existing physiotherapy referral route and adding `Ραδιοκύματα — PDF` navigation to the existing protected RF utility in a new tab.

The RF PDF engine, templates, request state and authentication remain owned by the existing clinic reception service; G-4 does not duplicate them into osteoporosis encounter state.

Exact tested runtime head:

```text
942d4e06944ebd6de97891cb8e2739c88ba85a38
```

GitHub Actions:

```text
workflow: G3 guidance salience longitudinal summary
run:      33599860151
result:   SUCCESS
```

The gate passed the focused G-4 ergonomics/RF-navigation regression plus inherited G-3, G-2, G-1 and C1 finalization regressions.

Exact-head review against production `ab94c628...` found the tested runtime branch ahead with merge base exactly production `main`, behind by 0 and no C2, PR-1/PR-2, evidence-rule, treatment-threshold or DB-schema leakage. Subsequent branch commits before release review are canonical documentation only.

This milestone closes only the implementation/test gate:

```text
G-4 DESIGN = COMPLETE
G-4 IMPLEMENTED = YES
G-4 TESTED = YES
G-4 EXACT-HEAD REVIEW = PASS
G-4 MERGED = NO
G-4 DEPLOYED = NO
G-4 PRODUCTION-SMOKE-VERIFIED = NO
```

A final canonical reconciliation and exact-head release-readiness check are required before a release PR may be opened. Merge/deploy still require separate explicit product-owner authority.

---

## 2026-09-02 — G-4 released/deployed; RF auth gap isolated; bounded gateway hotfix implemented/tested

The original G-4 release completed after bounded PR #72 exact-head checks passed. PR #72 was squash-merged to `main` as:

```text
338830340f6fed2ae1a3f08f6fdb0b8059932a66
```

Render auto-deploy:

```text
dep-dac27kojo6nc739biu80
```

was reported `live` by the product owner at that exact merge source. Production smoke verified the G-4 workspace behavior: independent collapse/expand for `Σύνοψη ασθενούς` and `Σημερινή ροή`, sticky patient summary and physiotherapy utility navigation all passed. RF navigation reached the intended existing service but the protected RF page correctly rejected the unauthenticated cross-origin browser with `401` / `Απαιτείται εξουσιοδοτημένη πρόσβαση.`

The failure was classified as:

```text
CROSS-SERVICE AUTHENTICATION INTEGRATION GAP
```

not as RF PDF/template/business-rule failure.

A fresh control-plane inspection showed that `ortho-reception-backend-v2` was independently frozen by its own governed active slice, so the hotfix did not mutate that runtime, its authorization code or its secrets/configuration. The selected correction remained entirely in `athpapachr-cmd/osteoporosis`:

```text
authenticated clinical_session browser
→ same-origin /clinical/clinic-utilities/rf gateway
→ server-only RF_GATEWAY_ACCESS_KEY
→ X-RF-Key to fixed existing RF service
```

The existing RF service remains source of truth for form HTML, history/persistence, validation, PDF templates/coordinates and generated PDFs. No RF payload enters osteoporosis encounter state and no RF credential is exposed to JavaScript or browser URLs.

Independent review of the first gateway candidate identified one material privacy problem before release: keeping RF history lookup as a local GET would create an additional Osteoporosis URL/access-log surface containing identity/GeSY identifiers. The final runtime changed the local browser contract to bounded `POST application/x-www-form-urlencoded`; only `identity_number`, `gesy_number` and `application_location` are accepted and translated server-side to the existing upstream GET. Local GET `/history` is not implemented. The gateway also fails closed if the upstream form's expected create/history transport seams change.

Final exact tested runtime head:

```text
29140a6cd4c9f57b454daa6e4a2883ec0345b53f
```

Exact GitHub Actions evidence:

```text
workflow: G3 guidance salience longitudinal summary
run:      33640110048
result:   SUCCESS
```

That run passed RF gateway Python syntax, the focused authenticated-gateway and POST-only history privacy regressions, the G-4 workspace regression and all inherited G-3/G-2/G-1/C1 gates. Final source/security/scope review found no remaining release-blocking defect in the bounded hotfix.

Closeout state:

```text
HOTFIX DESIGN                    COMPLETE
HOTFIX IMPLEMENTED               YES
HOTFIX TESTED                    YES
HOTFIX EXACT-HEAD REVIEW         PASS
HOTFIX PR                        NONE
HOTFIX MERGED                    NO
RF_GATEWAY_ACCESS_KEY CONFIGURED NO / NOT VERIFIED
HOTFIX DEPLOYED                  NO
HOTFIX PRODUCTION-SMOKE-VERIFIED NO
G-4 PILOT-VALIDATED              NO
```

The implementation/runtime writer lock was released during canonical closeout. Product-owner authority in this step covered canonical closeout and final docs-only drift verification only. A later release requires separate authorization for PR, merge and production configuration/deployment. No manual Render deploy, reception-backend mutation, production secret mutation or production RF smoke occurred in this closeout.


---
## 2026-09-05 — Native RF v2 release candidate completed; PR path authorized

The authoritative RF eligibility form changed materially before the old G-4 gateway create/PDF path was fully production-smoked. The obsolete external RF form contract was therefore not validated further. Product-owner REPLAN approval moved RF UI/business/PDF/history ownership into native Clinical Excellence Clinic Utilities for this clinician's Category-A workflow: A.1 new treatment and A.2 continuation.

Release-candidate runtime evidence:

```text
branch: feat/clinic-utilities-rf-v2-native-2026-09-05
exact tested runtime head: aa2f92cce5d4cd2cfd02cafc59413be7bdc0d5fb
workflow: RF v2 native clinic utility
run: 33988642002
result: SUCCESS
```

The exact supplied 12-page official PDF is packaged at `clinic_utilities/rf/templates/rf_official_form_v2.pdf` and guarded by byte identity: 310238 bytes, SHA-256 `998e99e6b0a51d4a19431dd2e31e595282d7adf17eb29f5f91eeab94e3647252`, Git blob `c6c234e99095be38c47a1c6f078dacdd47f4199f`. The final gate generated real A.1/A.2 packages against that binary and preserved inherited CU-1/G4/G3/G2/G1/C1 regressions.

Exact-head review corrected three release-relevant issues before the final tested runtime: A.1 now fails closed unless exactly 3 NSAID trials + 3 other analgesic trials are resolved; raw medication paste, raw physiotherapy-date paste and medication `source_text` are not persisted; and manual actual-procedure history uses `clinician_manual` provenance rather than `legacy_manual`.

The persistence boundary remains explicit:

```text
RF APPLICATION REQUEST
!=
ACTUAL RF PROCEDURE
```

Generating an approval/request PDF does not infer that treatment occurred.

Stale docs-only PR #74 was closed unmerged as superseded by the authoritative-form/native-ownership replan. Product-owner authority was then granted to proceed through opening one bounded native RF v2 release PR. Merge, production configuration, deploy and production smoke remain separate decisions.

---

## 2026-09-07 — RF v2 production smoke closed; Clinical Learning Hub L-0 activated

Native RF v2 reached production through PR #75, correction PR #76 and imaging semantic-guard PR #77. Current production identity after #77 is `1d26195c77e186cff98086283252af2eb499dd17`, Render deploy `dep-daeliaks728c7384f3fg` LIVE.

Product-owner authenticated smoke subsequently confirmed the intended A.1/A.2 flow, unilateral target with system-derived location, 0..3 medication-table semantics and corrected parser behavior, laboratory-PDF rejection, real/poorly extractable imaging fallback with explicit clinician confirmation, and final PDF generation. RF v2 is therefore `PRODUCTION-SMOKE-VERIFIED`, not `PILOT-VALIDATED`, and is closed for now unless new material evidence appears.

The official PDF still contains a clinician-signature area. Automated signature-image storage in the public repository was explicitly rejected. Manual/external signing is accepted for now. Any future online signing requires a separate protected security/e-signature design; a PNG/SVG signature must not become a public version-controlled asset.

The product owner then approved moving the primary roadmap back to the Clinical Excellence learning/capture program. `CLINICAL_LEARNING_HUB_DESIGN_V1.md` was promoted into canonical design direction and slice `CORE-LEARNING-HUB-L0-2026-09-07` activated for contract/design freeze only. L-0 covers Challenge, Fact Ledger, Daily Case Review, Foundation Map, due-state, PHI/provenance, Signal and baseline-intervention contracts. No learning runtime implementation is authorized by this design activation.

---

## 2026-09-07 — Clinical Learning Hub design merged to main

PR #78 (`Clinical Learning Hub: activate L-0 canonical design`) was squash-merged to `main` as `6fa2099647513e4a6b0e71ec30eb5275164626ed`. Render auto-deploy `dep-dafdb695efls73anlt6g` reached `LIVE`; the PR was documentation/design-only and did not change runtime behavior.

The repository now carries `CLINICAL_LEARNING_HUB_DESIGN_V1.md` as product-owner-approved canonical design direction and activates slice `CORE-LEARNING-HUB-L0-2026-09-07` for contract/design freeze. L-1 runtime implementation remains unauthorized until L-0 field-level contracts, provenance/revision semantics, PHI boundaries, due-state behavior and owner seams pass exact design review.

---

## 2026-09-07 — Clinical Learning Hub L-0 independent review reached CLOSURE PASS

An independent exact-head L-0 review did not accept the prior active-writer PASS as authority. On the original candidate it found material but bounded contract defects involving immutable reference-verification ownership, Challenge-delete/due referential integrity, PHI scan coverage and sanitized rejection, internal cross-object references, Daily Case revision semantics, Signal authority, Foundation source-deletion semantics and deferred due-state reactivation.

Those findings were corrected within the granted L-0 design authority. The corrected substantive contract head is:

```text
afaf9d5d0c7df9d53f8ec714f74d7b44e3a69f96
```

The full `Clinical Learning L0 contract gate` then passed at run `34147429373`. `CLINICAL_LEARNING_L0_INDEPENDENT_REVIEW_V1.md` records the independent disposition:

```text
CLOSURE PASS
MATERIAL OPEN FINDING NONE
L-0 CONTRACT ELIGIBLE TO FREEZE / COMPLETE
L-1 IMPLEMENTATION NOT AUTHORIZED
```

Canonical closeout and a final exact-head gate remain required before L-0 is declared complete and placed in design/release HOLD. No runtime/database/API/UI implementation, patient-data mutation, raw-transcript persistence, production configuration or RF mutation occurred in this review.

---

## 2026-09-07 — Clinical Learning Hub L-0 contract frozen; PR #80 opened in HOLD

After the independent L-0 CLOSURE PASS, canonical closeout was completed and all temporary closeout helpers were removed. The clean freeze head was:

```text
f947b12ce77db2ad1e5ff9117d7a4f794e224b60
```

The full `Clinical Learning L0 contract gate` passed on that clean head at run `34148333413`. The L-0 field-level learning contracts are therefore **FROZEN / COMPLETE** as a design milestone.

Bounded design PR #80 (`Clinical Learning Hub: freeze L-0 contracts and L-1 boundaries`) was then opened against `main`. It remains unmerged and is in **RELEASE / DESIGN HOLD**. Opening the PR does not authorize merge and does not authorize L-1 runtime implementation.

Lifecycle at this milestone:

```text
L-0 CONTRACT FROZEN / COMPLETE   YES
INDEPENDENT REVIEW               CLOSURE PASS
CLEAN FREEZE GATE                PASS
PR #80                           OPEN
MERGED                           NO
L-1 IMPLEMENTATION               NOT AUTHORIZED
```

---

## 2026-09-07 — Clinical Learning Hub L-0 contracts merged to main

After explicit product-owner merge authority, PR #80 (`Clinical Learning Hub: freeze L-0 contracts and L-1 boundaries`) was squash-merged to `main` as:

```text
43ed3090c5d4fe849e5f500edf22182fade36cd8
```

The merge contains the independently reviewed/frozen L-0 learning contracts, machine-validation fixtures/tests, Foundation Map v1, L-1 ownership boundary and canonical closeout. It contains no Learning Hub runtime/database/API/UI implementation and no patient/transcript write path.

Render auto-deploy triggered from the `main` commit and reached `LIVE` as:

```text
dep-daffttrbc2fs73d7v45g
```

This deploy is lifecycle evidence for the merged repository state; it does not convert L-0 design contracts into an implemented Learning Hub runtime.

Current lifecycle:

```text
L-0 CONTRACT FROZEN / COMPLETE   YES
INDEPENDENT REVIEW               CLOSURE PASS
PR #80                           MERGED
MAIN                              43ed3090c5d4fe849e5f500edf22182fade36cd8
RENDER                            LIVE
L-1 IMPLEMENTATION               NOT AUTHORIZED
```

The next gate is a separate product-owner decision to authorize the bounded L-1 Challenge Import + History + Foundation Map MVP implementation slice.

---

## 2026-09-07 — Clinical Learning L-1 implementation exact-head regression checkpoint

On branch `feat/clinical-learning-l1-challenge-foundation-mvp-2026-09-07`, the bounded L-1 Clinical Learning runtime was implemented for protected Challenge import/history/revisions/delete, reference-verification overlay, the 14-node Osteoporosis Foundation Map, explicit Foundation assessments/state, reviewed due-state scheduling and the first protected clinician-facing Learning Hub UI.

Hardening corrected timezone-aware UTC normalization and prevented an older Foundation assessment from overwriting a newer materialized state. Clinician-facing UX was then improved for Challenge revision flow and structured Foundation assessment.

Exact runtime/UX head:

```text
b53a39f12363e08a7be952e1d7a25fbb5511bca8
```

Full `Clinical Learning L1 regression gate`:

```text
run 34156659908
SUCCESS
```

The gate covered Python/JavaScript syntax, L-1 runtime/hardening tests, inherited L-0 contract regressions, adjacent-owner isolation and diff hygiene. Physiotherapy/CU-1 and RF owners were not mutated.

Lifecycle at this checkpoint:

```text
L-1 CORE IMPLEMENTED              YES
L-1 CLINICIAN-FACING UX           IMPLEMENTED
L-1 EXACT-HEAD REGRESSION GATE    PASS
FINAL EXACT-HEAD REVIEW           PENDING
PR                                NONE
MERGED                            NO
DEPLOYED                          NO
```

The product owner paused the separate physiotherapy workstream until this L-1 checkpoint is completed. The next legitimate action is final exact-head review, then canonical closeout and a bounded PR/release HOLD if clean.

---

## 2026-09-07 — Clinical Learning L-1 final exact-head review passed; release-candidate closeout

A fresh final review bootstrapped from remote `main` and the six canonicals, verified the exact active branch rather than choosing by recency/name, claimed the bounded L-1 writer lock, and reviewed the **full branch vs current main** across architecture/runtime, privacy/safety, concurrency/integrity and clinician-facing product workflows.

The review found no frozen-contract REPLAN requirement, but did find bounded material defects that were corrected before release closeout.

Clinically meaningful UX findings corrected:

- Challenge History exposed a `Νέα revision` control without a functioning clinician revision workflow;
- Foundation assessment HTML and JavaScript were mismatched, including a nonexistent `assessmentJson` dependency;
- Foundation evidence controls contained values outside the frozen assessment-method enum and lacked explicit per-evidence demonstrated result/note capture;
- prior immutable Challenge revisions were not meaningfully inspectable in History;
- Fact Ledger progressive-disclosure provenance was insufficiently visible.

The clinician-facing correction now preserves immutable revision semantics end-to-end: History can inspect a selected revision; `Νέα revision` creates a candidate from the latest accepted revision, resets imported/server authority, requires fresh server preview + observation disposition + explicit confirmation, then appends through server-owned `PUT`. Foundation assessment is structured around only frozen methods/states/results, explicit proposed/final state, clinician-reviewed evidence, notes and optional next-review scheduling. UI-contract regression coverage now detects missing DOM bindings, invalid Foundation methods and missing provenance seams.

Data-integrity / concurrency findings corrected:

- Challenge mutation/delete/reference operations now serialize on authoritative revision rows on the production Postgres path; delete locks revisions before purging overlays/due/content and creating the tombstone;
- residual database races fail closed as sanitized HTTP `409 learning_write_conflict_retry` rather than leaking driver/constraint errors;
- Foundation state materialization row-locks an existing node before replacement so a concurrently older assessment cannot overwrite newer state;
- a distinct equal-timestamp Foundation attempt fails closed because no deterministic ordering exists, while an exact same `attempt_id` remains idempotent.

No patient/transcript/Signal/DailyCase/PracticeReview ownership was introduced, no production config/secrets changed, no frozen L-0 schema was modified, and the separate physiotherapy/CU-1/RF owners remained untouched.

Reviewed substantive head:

```text
ada16afb573609cd555b99c1cc62a4a160d4215f
```

Complete `Clinical Learning L1 regression gate`:

```text
run 34158892560
SUCCESS
```

That exact run passed Python syntax, browser JavaScript syntax, L-1 runtime/hardening/clinician-UI contract tests, inherited L-0 contract regression, scope/adjacent-owner guard and diff hygiene.

Fresh comparison at substantive review close:

```text
main / merge-base  5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6
ahead / behind     35 / 0
physio/CU1/RF leak NONE
frozen-schema diff NONE
REPLAN              NO
```

Canonical closeout after that reviewed substantive head is documentation-only. The final PR head must still pass the complete L-1 gate exactly at that final docs head before a bounded PR may be opened.

Lifecycle at this closeout entry:

```text
L-1 IMPLEMENTED                 YES
L-1 TESTED                      YES
L-1 FINAL EXACT-HEAD REVIEW     PASS
L-1 PHYSIO/RF ISOLATION         PASS
L-1 PR                          NONE — NEXT AFTER FINAL DOCS-HEAD PASS
L-1 MERGED                      NO
L-1 DEPLOYED                    NO
L-1 PRODUCTION-SMOKE            NO
```

The next legitimate boundary is final docs-head verification → bounded PR to `main` → **RELEASE HOLD** for separate explicit product-owner squash-merge/release authority. PR-1/PR-2, Daily Case Review, Signal work and physiotherapy remain out of scope.

---

## 2026-09-12 — Knee-OA v4 post-smoke refinement released

The released Knee-OA physiotherapy referral received a bounded post-release v4 correction after Product Owner use exposed weak visual linkage between the four clinical-picture parents and their dependent qualifiers.

The v4 release uses a four-column desktop / 2×2 mobile clinical-picture surface with focused contextual sheets, preserves dependent-state hygiene, corrects deterministic Greek referral prose, and retains the existing CU-1/evidence/safety authority. During final PR gating, inherited test paths were migrated and two real accessibility defects were caught before merge: 42 px mobile touch targets and horizontal overflow under large text enlargement. Both were corrected without weakening acceptance criteria.

Exact tested PR head `602740d12da0de8ad9a6134f8adf48f4b6f7c691` passed the v4, inherited prototype, Cockpit integration, CU-1 and evidence gates. PR `#89` was squash-merged as `bf527e3836a18491b2758fd293b42f82e0924382`. Render auto-deploy `dep-daiivp0jo6nc73bl6pug` became live at that exact runtime SHA.

Temporary no-secret external smoke run `34689920602` passed: current/v4 assets returned `200`, the live CSS contained the ≥44 px + large-text wrapping correction, production transport remained non-synthetic, and the protected physiotherapy route/bootstrap returned `401` without authentication. No credential, session or patient data was supplied. Full authenticated live E2E, receiver validation, clinical pilot, paid validation, Cyprus/GeSY item-level activation and second-diagnosis expansion remain unproven/deferred.

---

## 2026-09-15 — Physio Knee-OA V5.1/post-use release chain reconciled

The Physio Referral product track was reconciled to current production truth without changing the parallel root Medical Report operational lifecycle.

Verified sequence:

- PR #106 merged V5.1 at `8064999ea70e0a90f6073fc6d66a0c8caaba1538`;
- PR #107 merged the Cyprus/clinical-review post-use refinement at `a19f4b9d52c3076f715fd864a5f7664d2b140c81`; authenticated live smoke `34957778592` succeeded;
- PR #109 merged receiver prose compression and optional symptom chronicity at `b962485c741f558121e8daabfcf1d20c84f31f63`; authenticated live smoke `35019200920` succeeded.

Current live smoke preserved the protected `clinical_excellence_cockpit` context, explicit `CY_GESY`, context-only chronicity, compressed functional-retraining wording, no structured previous-physiotherapy/response control and no browser draft persistence.

Root `CURRENT_OPERATIONAL.md` / `SLICE_PLAN_CURRENT.md` were intentionally not rewritten because their active lifecycle remains the parallel Medical Report release/smoke track.

---

## 2026-09-15 — Atomic canonical checkpoint governance adopted

Repository continuity rules were hardened after a real long-conversation failure mode showed that correct runtime work could outpace product-local canonical closeout.

The permanent operating model now treats each material lifecycle transition as an atomic checkpoint barrier: perform one transition, collect its exact evidence, persist the resulting state in the correct canonical/workstream `CURRENT`, verify that durable checkpoint, and only then continue to the next deliberate material action.

Parallel workstreams keep `CURRENT_OPERATIONAL.md` as the single repo-wide writer lock while maintaining their own product-local `CURRENT.md` where one exists. A release-affecting parallel PR must update its applicable workstream current-state file even when root NOW correctly belongs to another lifecycle.

GitHub governance now includes a pull-request Canonical Impact Declaration plus a `Canonical impact guard` Action. The guard validates declared canonical effects against the actual PR diff and requires a workstream current-state update for release-affecting PRs. Because `main` is not currently protected by a required-check rule, a red guard is defined as a merge prohibition by repository policy; the same check can later be made server-enforced through branch protection.

The durable objective is explicit:

```text
CHAT END / CONTEXT LOSS MUST NEVER BE REQUIRED FOR PROJECT STATE RECOVERY
```

---

## 2026-09-16 — Medical Report V1.1 production lifecycle closed

Clinical Documents Medical Report V1.1 completed its release lifecycle after the previously frozen implementation candidate was released through PR #104.

Release identity:

```text
implementation exact tested head: 4c30867e8aea00842772b6df7d624a79bbf3f19e
implementation gate:              34740684612 — SUCCESS
release PR:                       #104
release commit:                   2b6ccf56f77c646fc475318e4d76396dd95a2bee
Render deploy:                    dep-daj3l00ae00c73drs850 — LIVE
```

The first authenticated synthetic-only production smoke (`35052415555`) failed inside one combined verification step. Because that disposable harness lacked stage markers, the result was checkpointed as failed/non-diagnostic rather than being rationalized into a PASS.

A stage-labelled diagnostic re-smoke (`35052740386`) then verified the protected V1.1 contract/auth and no-persistence boundary, deployed source/refinement assets, live GPT-5.6 structured draft, `REFERRAL != COMPLETED RESULT` behavior, session-only refinement with immutable original evidence/source summaries, and valid preview/final PDF generation. The run also exposed two malformed disposable `jq` usage assertions; its own emitted metadata showed positive GPT-5.6 token use, so those were harness defects rather than runtime defects.

The same diagnostic run returned a non-empty research synthesis for a referral-only analysis but no web-search queries/citations. That ambiguity was not silently accepted. A targeted generalized synthetic diagnosis/prognosis smoke (`35053140054`) was therefore run against the live research endpoint and passed with:

```text
HTTP:             200
research text:    14,487 characters
URL citations:   13
captured queries: 8
web-search calls: 10
model:            GPT-5.6
usage:            80,451 total tokens
```

This established that the web-backed literature-search path is functioning when a material research question exists. No Medical Report runtime correction was required from the verification sequence.

The released V1.1 state therefore reached:

```text
IMPLEMENTED                    yes
EXACT-HEAD TESTED              yes
MERGED                         yes
DEPLOYED                       yes
AUTHENTICATED PRODUCTION SMOKE yes
PILOT-VALIDATED                no / not claimed
ACTIVE WRITER                  none
```

Production verification used only synthetic/non-identifiable data. No real patient file or secret value was committed or printed. The no-persistent-case boundary remains unchanged.

Medical Report V1.1 is closed. Reopen only for a material production defect, authoritative form/requirement change, safety/data-integrity issue or explicit new workflow requirement.


## 2026-09-26 — Sick Leave selectable templates and color themes released

- Released PR #117 (`ad68935750e7ab3101b253fad4d3b94725204461`) for the Clinical Documents Sick Leave generator.
- Added five presentation-only layouts: Classic Medical, Modern Clinic, Minimal, Compact and Formal.
- Added six controlled professional color themes: Navy, Teal, Graphite, Burgundy, Forest and Monochrome.
- Preserved the existing SickLeaveDraftV1 clinical/document model, PDF metadata schema, previous-document reuse, signature semantics and no-patient-browser-storage boundary.
- Exact implementation evidence: Clinical Documents P1 `36267760233` SUCCESS with 32 sick-leave tests; Clinical Documents P2 `36267760213` SUCCESS.
- PR #117 checks passed Canonical impact, P1, P2 and CU-1 before squash merge.
- Render auto-deploy `dep-das28crtqb8s739ddpkg` deployed the exact merge commit and reached `live`; no manual redeploy was triggered.
- Authenticated/browser production user-flow smoke remains to be observed during normal clinical use.


## 2026-09-27 — Global Cockpit Home v1 released

- Released PR #118 as squash merge `b7b8779d943eb1d8db1f8966a81796bc69647c3b`.
- The service root now enters the reusable Clinical Excellence Cockpit Home instead of routing directly into Osteoporosis Module 01.
- Cockpit Home exposes Today/Clinical Calendar, Clinical Modules, Learning & Improvement, one global Clinic Utilities surface, and a link to the separately-owned Reception/call dashboard.
- Osteoporosis navigation is now module-specific: the general Clinic Utilities group and top-level Heidi AI navigation item were removed; Heidi remains an in-encounter capture/exposure concept inside Module 01.
- The physiotherapy referral has one global Cockpit entry rather than duplicate Module-01/global entries.
- Home calendar presentation is privacy-minimized to aggregate category counts and does not render patient identity fields.
- Release hardening also corrected stale CI ownership guards/tests that incorrectly treated global `main.py`/navigation changes as Physio- or Learning-only work; inherited product regressions continue to run while strict product-only scope assertions apply only when the corresponding product owner is actually mutated.
- Final PR head `ba2de644f4fd16e70d12f04831b28f5fd42730ec` passed all 16 checks.
- Render auto-deploy `dep-das41frncjis73e6q47g` reached `live` for the exact merge commit.
- Production smoke verified `GET /` -> 307 and `GET /static/cockpit/` -> 200.
- Cal.com visit-reason ingestion into Clinical Calendar remains a separate next integration slice.


---

## 2026-10-01 — Cockpit Today Context Strip V1 released

- Released PR #130 as squash merge `db42903e08b11dfe52d427a410f8b15547fb5507`.
- Cockpit Home now shows protected read-only **Previous / Current / Next** appointment context from the existing Clinical Calendar while preserving **«Άνοιγμα εβδομαδιαίου ημερολογίου»**.
- Clinician attention follows appointment `start_at` order; Aclasta is the bounded lawful overlap exception, while other simultaneous active overlaps fail closed visibly; Next remains the earliest future `start_at`.
- D1 does not use phone or protected patient ID for identity/linkage and adds no booking, Calendar write, Secretary, Setmore, Zadarma or clinical-state writer.
- Final independent closure accepted the runtime/product behavior; the only residual BLOCK was two stale canonical statements, corrected as an R0 docs-only delta with no runtime/test semantic change. The review chain was then stopped under `PROCEDURES.md`.
- Render deploy `dep-davbs48473hc73eust00` reached **live** from source `794d96dc4e0676691a4d6333b8f47bdd335cdb6c`. No additional production smoke is claimed in this checkpoint.
- The next Cockpit step is the D2 Relevant Communication Context Product Owner checkpoint; D2 implementation is not yet authorized.

---

## 2026-10-03 — Bounded review execution and D1.1 pre-code BLOCK reconciliation

- Product Owner reported an independent D1.1 review continued for over seven hours before interruption; no execution trace was supplied, so the specific long-running loop is unproven.
- Exact original prompt/request were inspected. `PROCEDURES.md` P5.1 now defines when evidence is sufficient: map declared questions to affected sources, expand only for a concrete dependency or contradiction, treat COMPLETE and NO ADDITIONAL MATERIAL FINDING as bounded claims, and stop once the decision is supported. No time limit was adopted as the solution. The Work-mode chat exposes no intermediate execution trace; a separate Codex run of the same prompt completed in about four minutes, so the specific seven-hour cause remains unproven.
- Supplied complete-scope BLOCK (0:2:0) was consumed. D1.1-PRE-01 requires snapshot-only `other` admission with preserved legacy discard/removal; PRE-02 requires phone/link minimization on every effective-`other` mutation, including clearing manual classification.
- Corrected design blob `261ee0e7f2fc8b9ebc59310bb1c5e0517439fe33` and one delta + affected cumulative closure request are on `design/cockpit-d1-1-global-context-2026-10-02`. Original request is archived unchanged; source/finding provenance and limitations are retained.
- Governance consistency audit completed; **independent design closure remains PENDING**. No D1.1 runtime/test implementation, release, deployment or Visit Brief implementation occurred. PR-1 root slice/phase/roadmap and other workstreams remain preserved.


## 2026-10-03 — D1.1 pre-code closure consumed; bounded implementation candidate tested

Product Owner supplied independent closure PASS / COMPLETE_FOR_DECLARED_SCOPE / PRE-01 CLOSED / PRE-02 CLOSED / 0:0:0 against design blob `261ee0e7f2fc8b9ebc59310bb1c5e0517439fe33` and explicitly authorized implementation. The pre-code chain stopped. Branch `feat/cockpit-d1-1-global-context-2026-10-03` starts from fresh main `d5da6271567a4141b2708d9fa12e673dfce37131`.

The implementation reuses the complete snapshot producer, default-off legacy import helper, one normalized appointment store and protected auth. Effective other rows retain minimal global context with empty phone/null clinical link; manual-clear minimization, weekly filtering and complete same-source/window reconciliation are preserved. A minimal server global projection supplies Home Previous/Current/Next, cross-day Next and the accepted Aclasta exception; other overlaps fail closed. Global Today osteoporosis counters are removed; weekly Calendar link remains.

Focused author evidence: 30 Calendar/snapshot/Home/shared Surgery UI tests PASS, syntax/navigation/diff hygiene PASS. One independent exact-implementation-head post-code review is being prepared; not launched, no verdict yet. Merge/deploy/production smoke and Visit Brief remain unperformed and unauthorized pending their separate gates. Implementation writer is released into review HOLD. Operational identity belongs in the two CURRENT files, rather than being backfilled into this historical entry.

## 2026-10-06 — D1.1 freshness paired release deployed; Product Owner smoke pending

- Independent exact-head R2 fidelity handback: PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0; technical review chain CLOSED. Historical daily-source HOLD request remains historical without a verdict.
- Reception PR #157 merged as `8d2827514457a1da54f33c6109b22e6106532301`; reviewed tree preserved; Render deploy `dep-db25omh7lnhs73dn82o0` LIVE.
- Cockpit PR #136 merged as `12800be3a51f716b2ccd72715c6e0e637f2e43ab`; all six reviewed runtime/test blobs preserved; Render deploy `dep-db25uae7bikc73dkrku0` LIVE.
- Cockpit `RECEPTION_SCHEDULE_CONTEXT_URL` installed for the protected Reception route; existing matching server-to-server ingest-key pair retained without secret rotation. Configuration deploy `dep-db25qh7lot8c73duu8vg` LIVE before Cockpit code merge.
- Production smoke is PENDING PRODUCT OWNER. No live Cal/patient-data smoke or booking mutation was performed by this release task. Visit Brief implementation remains deferred.


---

## 2026-10-07 — Calendar Unification smoke closed; two-entry Cockpit design replan

The Product Owner confirmed Calendar Unification SMOKE VERIFIED and explicitly continued to design-only Visit Brief / Clinical Inbox work. The prior paired release identities remain Reception `e6babb5ab758d282166767c36dd7311024406afb` / `dep-db2m42mi0phs7393jms0` and Cockpit PR #137 merge `3f4d82280e3a58f25383ccd896ebcf806ec755c2`, final main `045798dfa28612b268f16a99262c6ecc9ca4829d` / `dep-db2m7sbl550s73binhig`. Calendar's independent R2 PASS / 0:0:0 remains CLOSED. This is Product Owner smoke evidence, not a new agent-run production test or pilot validation.

The product correction establishes two first-class entrances over one clinical-attention lifecycle: appointment-driven floating Visit Brief and independent Clinical Inbox for daily Gmail clinical inputs. Labs, MRI, X-ray/imaging and GESY notifications retain distinct semantics. Same-day clinician review/communication does not depend on an appointment. Always-on intake/durable delivery replaces on-open Gmail discovery as the correctness path; free Cockpit remains a clinical consumer.

The revised design preserves verified/clinician-confirmed identity, candidate-vs-authoritative results, existing clinical owners/G3 reuse, Reception communication orchestration/Zadarma transport, clinician approval, truthful failure/idempotency and configured non-booked callback windows. Original Current/design/request bytes were archived unchanged from the verified base before replacement. Root PR-1 release HOLD and slice design remain untouched.

The new design candidate is `cockpit/VISIT_BRIEF_CLINICAL_INBOX_DESIGN_2026-10-07.md`. One bounded independent R2 pre-code request is prepared; no review was dispatched or verdict claimed. No runtime/schema/test/config/provider changes, live mailbox access, patient document processing, message send, merge or deploy occurred in this design replan.


---

## 2026-10-09 — Visit Capture synthetic implementation independently closed (not released)

The Product Owner approved the bounded first-code Visit Capture workflow after the independent Dia/Heidi/GESY → Cockpit delta pre-code R2 PASS. The implemented candidate extends the existing protected `clinical_encounters` owner and uses an explicit clinician Save, patient-bound capture context, structured three-level projections (Snapshot / Visit Brief / Encounter Detail), minimum first-class pending persistence, and server-side no-silent-edit after signoff. No live Dia, Heidi, GESY, Gmail, Zadarma or real-patient input was accessed during the code/evidence work.

The original exact-head post-code R2 returned BLOCK / P0:P1:P2 = 0:0:1 solely for P2-F2-01: an unbounded nested `external_dependency`. Product Owner authorized a single bounded correction. `clinical_data.py` now accepts only `actor` (1–80 chars) and `condition` (1–240 chars) with `extra=forbid`; `test_visit_capture.py` proves valid persistence and rejects unknown/oversized nested input before encounter/pending write.

Corrected exact runtime/test head: `07f68a2adc4664a4ab050b3fa10a663de591b662`. Focused CI run `37963002895` SUCCESS (20 focused/UI/Home tests, Python and JS syntax, diff hygiene). Product Owner supplied independent focused C1–C3 closure PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0, **original P2-F2-01 CLOSED**. The older F1/F3/F4/F5/F6 PASS findings were inherited, and the review chain STOPPED. Durable result: `cockpit/reviews/VISIT_CAPTURE_P2_F2_01_FOCUSED_CLOSURE_PASS_RECEIPT_2026-10-09.md`.

No implementation/design review is pending for the declared scope. This is **implemented, focused tested, independently review-closed, not merged, not deployed and not live-patient qualified**. Release sequencing, design PR #138 ancestry, any production feature/data-access fence, and the Product Owner's explicit merge/deploy authorization remain separate gates. Root PR-1 release HOLD and the independent Clinical Inbox/provider activation gates remain unchanged.


---

## 2026-10-09 — Visit Capture controlled release merges (deployment pending)

Product Owner elected to finish release into the existing Cockpit rather than create a second Render preview or introduce a new feature flag. Identifiable Dia/Heidi/GESY use is NOT thereby qualified or authorized. Parent reviewed documentation/design PR #138 squash-merged as `9ec24a9a5f756b31ee8e6031b5d587c90ce91ac6`; Render auto-deploy `dep-db4jtlrncjis73dn58u0` reached LIVE. The independently reviewed Visit Capture implementation PR #139 was reconciled to `main` without file/tree changes and squash-merged as `dd14cd0df1ca84d051ba7bf784b5b33226ed20b0`. Git merge tree `75b82b79b1af1ee37bed628e44c45cf4f6496723` exactly matches implementation candidate tree; reviewed runtime/test blobs unchanged from corrected `07f68a2adc4664a4ab050b3fa10a663de591b662`; original review closure PASS / 0:0:0 remains final. Cockpit Render automatic deploy `dep-db4k1l942hec73dll8j0` started from implementation merge; live status and product smoke were not yet verified when recorded. Once deployed, Visit Capture is reachable for holders of existing clinical authentication (no OFF-by-default feature flag). Neither real patient data nor live provider systems were touched during release operations. Product Owner real-use/privacy qualification remains a separate open gate.


---

## 2026-10-09 — Visit Capture controlled production auto-deploy LIVE (product smoke pending)

Design PR #138 merged `9ec24a9a5f756b31ee8e6031b5d587c90ce91ac6`; implementation PR #139 merged `dd14cd0df1ca84d051ba7bf784b5b33226ed20b0`, merge tree equal to original implementation candidate tree, no new runtime/test bytes. Checkpoint source `8dbc9a78d032d710be4530875d4fa02a15628268` changed only root/Cockpit CURRENT, TODO and changelog. Render auto-deploy `dep-db4k2r3ncjis73dnc6p0` reached LIVE from that exact source; prior code-merge deploy `dep-db4k1l942hec73dll8j0` was LIVE before being replaced. Corrected runtime/test bytes remained independent R2 PASS/0:0:0 with original P2-F2-01 closed. No clinical patient data/provider activity, authenticated browser smoke or actual GESY use was performed; product-owner visual smoke remains pending. Existing clinical authentication makes Visit Capture Save accessible on the deployed system; separate identifiable Dia/Heidi/GESY processing and retention/privacy qualification remains OPEN. Release writer released.

## 2026-10-10 — Hybrid Cockpit PR #142 merged and initially deployed LIVE

- Product Owner explicitly approved the existing-Cockpit Stage A controlled release, not live provider processing.
- PR #142 squash merged into main at `56bb7fe5d1566fbe3f213cbdf5b6eed2272a502e` after exact-source reconciliation with released #140 and #143.
- Reviewed Stage A Hybrid UI and two protected 3+3 worklist read projections reused independent R1/R2 PASS; 13 applicable tests successful; inherited L0 design-only diff-scope check unrelated and red.
- Render auto-deploy `dep-db56637avr4c73eprc5g` for source `87bd157a203a668b605e319b3e79ebf06953bcfa` achieved LIVE at 2026-10-10T16:10:17Z. Final docs-only closeout creates a successor automatic deployment to verify, without manual deploy.
- No real patient/provider access or clinical write during release. Anonymous asset/browser smoke failed on DNS (not validated). Identifiable Dia/Heidi/ΓεΣΥ qualification, live Clinical Inbox/Gmail/Zadarma, derived ToDos and signed plaintext Save remain separate.

## 2026-10-11 — Cockpit clinician-first Home V4 PR #145 released

- Product Owner expressly approved production release after the independent affected R1 and sole A2/P1 correction-closure review returned **PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0**; review chain stopped.
- PR [#145](https://github.com/athpapachr-cmd/osteoporosis/pull/145) squash merged as `70d7ae4ea19c48bbd7cdd4374f189dcf747859c3`, retaining all ten frozen code/test/workflow blobs; corrected Home accurately labels the on-demand clinical day popover as partial and links to the full Reception schedule.
- Render service `srv-d5qfk31r0fns73di596g` automatically deployed code-identical main checkpoint `262c90090287ee420c5445455c62ba97ff9c713e` as `dep-db5gp60ae00c739mimtg`, terminal **LIVE** at `2026-10-11T04:13:50.389459Z`. Initial auto-deploys superseded by newer main documentation checkpoints canceled normally. The present final documentation-only release record triggers another native auto-deploy; observe terminal status separately, do not assert it LIVE in advance.
- Synthetic Chromium desktop and 680px half-width CI, Visit Capture, Surgery and canonical-impact gates succeeded on reviewed code; inherited nonapplicable Learning L0 design-only check remained red. External no-PHI HTTP smoke was blocked by executor DNS, so no end-user browser validation is claimed.
- No actual identifiable patient data, Gmail/Zadarma processing, new protected writes, calendar authority changes or typed clinical Save. Release writer released. Future full Reception day/Clinical Inbox/signed clinical excerpt and real-data qualification are separate scoped decisions.

## 2026-10-11 — Native RF v2 managed physicians / Horos JPEG #146 production release

- Product Owner explicitly approved implementation completion, controlled squash merge and deployment after independent delta + affected-cumulative R2 **PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0** closed the sole doctor issuer-switching P1-01. Previous JPEG integrity closure remains PASS.
- PR [#146](https://github.com/athpapachr-cmd/osteoporosis/pull/146) squash merged to `main` as `c2f9fafd3fc0873acfb8877dd9761942e63bb75a`. Reviewed doctor/UI, doctor protected API/catalog/persistence, JPEG PDF and JS regression Git blobs preserved on merged main. RF CI, synthetic browser handler, official A1/A2/PDF, JPEG and Canonical Impact gate SUCCESS; unrelated Learning-only diff scope checks excluded from RF validity.
- Render existing `osteoporosis` service (main auto-deploy) reached LIVE from exact commits `4d66b80fcaf64198227662a0364a79304dac6d8d` (`dep-db5ks93rjlhs73as4ha0`, 08:53:30Z) and `f5120776ddb97583bc9419247d04fd2db3b9a416` (`dep-db5ku6k9v7es7396qfg0`, 08:57:50Z). Superseded deployments canceled normally after newer documentation checkpoints; no manual deploy or config changes.
- External non-identifiable GET smoke of `/`, `/clinical/clinic-utilities/rf` and `/clinical/clinic-utilities/rf/api/contract` failed locally **at DNS resolution**, before an HTTP response; web-based fetch was also inaccessible. Therefore live HTTP/authenticated clinical browser smoke is **NOT VERIFIED** (not an observed server failure). No real patient or provider data accessed and no protected writes performed.
- Release writer is released. This final documentation-only closeout triggers a successor automatic Render deploy whose exact SHA must be verified before claiming latest production LIVE. Real clinical-use qualification and Product Owner authenticated synthetic/browser verification remain separate.

## 2026-10-11 — RF Horos JPEG selection presentation fix PR #147

- Product Owner reported that accepted JPEG previews misleadingly coexisted with the native `No file chosen` filename and explicitly requested a quick presentation-only fix with no additional independent review.
- PR [#147](https://github.com/athpapachr-cmd/osteoporosis/pull/147) changed only the existing RF HTML/CSS upload label, grammatical retained-image status in `app.js`, and focused `test_rf_v2_ui.js`, together with root/RF CURRENT operational checkpoints. The actual native multiple-file input remains keyboard-focusable; JPEG file acceptance, 0–10 limits, previews, removal, official PDF assembly, backend and physician data remain unchanged.
- Exact tested code/test commit `5950b27a2db210dd7133e34115efe6f6b9f17e7f`; RF focused regression [38129264416](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38129264416) SUCCESS; mandatory canonical-impact [38129380222](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38129380222) SUCCESS. PR squash merged into main at `5e0515ad9adbfda8cc11777cd4da88f5b42dcc35`, preserving exact affected source/test Git blobs.
- Initial automatic Render deploy of squash commit returned `update_failed` after a successful build; next normal main auto-deploy [dep-db5llfn40ujc73cj8no0](https://dashboard.render.com/web/srv-d5qfk31r0fns73di596g), source `f75cb28c97b08bf7aebca3e27d940b677f3c8af5`, became **LIVE** at 2026-10-11T09:47:27Z. Uvicorn startup was observed. The final documentation-only closeout commit will trigger another automatic deploy, whose exact identity remains to verify independently; do not manually trigger duplicate deployment.
- No authenticated hands-on browser smoke or real patient/provider data access; direct clinician visual confirmation remains separate. A2 continuation historical-PDF import/autofill is not part of this fix. Scoped writer released and no additional independent review requested.

## 2026-10-11 — Native RF A2 source-first, prior RF reuse and structured drafts #149 released

- Product Owner separately authorized controlled release after independent pre-code R2 PASS and independent corrected post-code affected R2 **PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0**; findings P1-01 product provenance, P1-02 patient name/identity reconciliation and P2-01 A2 source-first UI all CLOSED.
- PR [#149](https://github.com/athpapachr-cmd/osteoporosis/pull/149) squash **MERGED** into main as `d69c49b7ad65fba4965049a242e19eb0d1c1d12c`. All eight RF reviewed runtime/test blobs matched original R2 frozen commit `390296d421f3d1f0c47e6475ec327c91f02b8367` after merge. Focused RF CI and canonical impact PASS. Design-only #148 remains separate.
- Existing Render service `srv-d5qfk31r0fns73di596g` auto-deployed code-identical main checkpoint `ba7f2f305cacdebf3dec75ee8574d09729e4e703` via `dep-db5vd4f40ujc73ctj1f0`, **LIVE** 2026-10-11T20:52:06Z. Previous initial squash deploy canceled when superseded by automatic documentation checkpoint; no manual deploy. Application startup complete recorded in Render app logs.
- A2 now supports old official RF PDF/source-first one-field search and explicit source/person selection; partial A1/A2 drafts persist *structured fields only*, never uploaded PDFs/JPEGs. Clinical actual prior procedure and issuing doctor remain distinct. No identifiable patient data or provider writes performed during release. Anonymous external browser GET check was inaccessible from executor, so authenticated end-user smoke remains **NOT VERIFIED**; clinician product visual smoke is next optional action.
- This append-only release record itself advances main and triggers an additional normal auto-deploy, whose terminal exact-main LIVE state must be independently verified. No new review or application code changes.

## 2026-10-13 — Native RF v2 manual outcomes, rejected corrections and explicit personal follow-up PR #150 released

- Product Owner separately authorized production release following one independent corrected post-code affected R2 **IMPLEMENTATION-FIDELITY PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0**, original P1-01 manual chronology/decision correction, P1-02 stale selected RF, P2-01 manual decision provenance all CLOSED. Approved pre-code R2 reused, no new review chain.
- [PR #150](https://github.com/athpapachr-cmd/osteoporosis/pull/150) was transitioned READY then squash **MERGED** as `c84b5d66654a7b51faec9595dd33f93261abbf38`. Exact READY head `0a46943adb4b2accf77daaeeec57a15ad5fd76cb`; frozen corrected code/test `d72290b0c83be76c9f59448f05bc20b7ed0e5d87`, tree `ff9051d556163f846c6fc607bc798aad4021870f`; all eight checked blob identities preserved through merged main. [RF CI 38463372271](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38463372271), [Canonical Impact 38463381870](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38463381870), [CU1 38463381833](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38463381833) SUCCESS, including 45 RF native tests/browser actual A→B regression and source/diff hygiene. Clinical Learning L0/L1 failures were strict non-RF diff-scope guards rather than RF runtime failures.
- Production enables protected manual RF submitted/approved/rejected/scheduled/performed recording, optimistic audited corrections and real-event chronology, linked rejection correction, server-attested verified own submission tracking versus voluntary opt-in for colleagues/unknown, and visible audit-source/event-time/entry-time. Issued official RF/PDF/JPEG, doctor snapshots, A1/A2, saved structured drafts and existing auth unchanged. No Home/Calendar/Reception/external GeSY writes or new provider integration.
- Existing `main` Render service `srv-d5qfk31r0fns73di596g` automatically deployed source `11ae46249a98282451f7c5b3ff4cd87d9dc10bd4` via `dep-db79aplbedkc738b1460`, **LIVE 2026-10-13T20:34:27.648292Z**. The initial squash source deployment was normally canceled when a new code-identical docs checkpoint superseded it; no manual second deploy. Two unauthenticated executor HTTP probes failed at DNS before service contact. Consequently LIVE Render source/status proven, independent HTTP and authenticated user-facing/browser smoke **NOT VERIFIED**, no patient or protected write, no assertion of actual clinician validation.
- Release writer released. This final documentation-only checkpoint itself advances main and causes one additional ordinary automatic deployment; verify that successor read-only before claiming its exact source LIVE. No new independent R2 or design round required.

## 2026-10-14 — Cockpit shared Home/Shortcuts/Clinical Key and Reception local-calendar date PR #151 released

- Product Owner requested controlled production release after the fresh independent final affected R2 **PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:0:0**, with original Visit Capture unsaved Dia P1-01 **CLOSED** and shared-navigation real Chromium evidence gap **CLOSED**. Reviewed release governance source `cb948e04965c59f03b92629c8d34a60dea2a83d9` and frozen code/test `b905c5fc213f112e408381555f50ca3734671b4d`; [independent receipt](cockpit/reviews/COCKPIT_GLOBAL_NAV_KEY_CALENDAR_FINAL_INDEPENDENT_R2_PASS_RECEIPT_2026-10-14.md). Eight source/test file blobs matched review after merge.
- [PR #151](https://github.com/athpapachr-cmd/osteoporosis/pull/151) exact ready head `036813e444aa69424b7f6ebaf850bb7389081fec` was squash **MERGED** as `56031bf8c9b84b2adbce8643173be7167e6b851a`. Five applicable Home/Chromium, Visit Capture, Clinical Calendar, Surgery and Canonical Impact CI workflows SUCCESS at both tested and governance-release heads. Other-workstream Learning/Physio strict diff-scope and Clinical Documents old inline HTML-link expectations remain separately red and are not falsely reported as full-repository CI green.
- Runtime change includes consistent top-right Home/Shortcuts/Clinical Key on nine core Cockpit/clinic entrypoints, existing cookie authentication and protected on-page refresh without discarding unsaved Dia, and precise Cyprus-local `target_date` for Reception-owned Cockpit calendar source with fail-closed coverage/freshness and source-vs-auth distinction; no changed patient Save authority, new credentials backend, external provider write or real patient data migration.
- Production `osteoporosis` Render service `srv-d5qfk31r0fns73di596g` automatically deployed code-identical post-merge checkpoint `1e23ac22269bae4b087cd3a0c53e31c80405883d` through [dep-db7frnlbedkc738ie71g](https://dashboard.render.com/web/srv-d5qfk31r0fns73di596g/deploys/dep-db7frnlbedkc738ie71g), **LIVE at 2026-10-14T03:59:59.449177Z**. The initial exact-squash deployment was normally canceled by the newer governance-only checkpoint. No manual duplicate deployment.
- One anonymous GET from executor failed at DNS before HTTP; public endpoint smoke and authenticated clinician hands-on smoke remain **UNVERIFIED**, NOT a negative health finding. Real identifiable patient/Reception/provider testing is not authorized by this release. Release writer released and independent review stopped. This final docs-only source update auto-triggers one further normal deployment; verify successor exact-source LIVE separately without a manual duplicate.

## 2026-10-15 — Native RF Tracking #157 issued-copy, exact-ID deletion and UX production release

- Product Owner instructed **“Passed. Release”** after final fresh independent affected POST-CODE R2 **IMPLEMENTATION-FIDELITY PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:0:0**, closing P1-01. Reviewed implementation+test HEAD `3281466017b6c1677624bb180c7db1b2f7240ac6`; prior unaffected PASS reused, no further review loop.
- [PR #157](https://github.com/athpapachr-cmd/osteoporosis/pull/157) squash **MERGED** as `5d256961ca4f7b9830f561d974fac3e96df873a1`. Reviewed source/test blobs unchanged by final RF CURRENT-only checkpoint commits. Scope: Tracking and RF creator view exclusivity, initial PDF browser download without popup, protected selected immutable RF-field restore for fresh PDF with new issuer/imaging (no recovered historical binaries), exact-ID optimistic-revision deletion with confirmation and deletion receipt, A2 atomic real prior-procedure ID and clinically compatible historical deduplication.
- Existing Render `osteoporosis` service `srv-d5qfk31r0fns73di596g` auto-deployed code-identical post-merge source `823fc4a94717a4cd3b2f20b7d99ddd118afe1154` through [deploy dep-db8g8tp42hec73ecctng](https://dashboard.render.com/web/srv-d5qfk31r0fns73di596g/deploys/dep-db8g8tp42hec73ecctng), **LIVE at 2026-10-15T16:52:49.397717Z**. Initial merge-SHA deploys were canceled by newer standard docs checkpoint, no duplicate manual deploy.
- Production HTTP no-PHI smoke from this executor **UNVERIFIED** due to DNS resolution failure before an HTTP response, not a confirmed application failure; authenticated browser/Clinical Key and real-patient workflow smoke **NOT CLAIMED VERIFIED**. No real RF deleted, no clinical patient records accessed, no external GeSY/provider action or manual database mutation by the release operator. Release writer released. This final docs-only record advances main and triggers another ordinary Render auto-deploy whose exact status requires read-only confirmation.



## 2026-10-17 — #161 persistent synthetic browser access release

PR #168 squash-merged as `21011b666d554ccc13aa1011078e6d65c928d2f7`. Existing Cockpit shell and Clinical Key now expose a fixed synthetic-only path through the same booking-link/Brief transaction and existing persisted clinical owners; complete stored chart and idempotent scenario preparation added. Seven new backend/browser tests and affected inherited CI passed. Production staff gates, M8 and CP1 unchanged. Merge is proven; Render availability and Product Owner acceptance are separate, still pending at this checkpoint.


## 2026-10-17 — #161 live persistent synthetic browser proof / PO acceptance pending

Bounded chart-selection correction #169 merged `7b10f063551710ed98b2a9ed8373515eb7167f82` after #168 `21011b6…`. Existing Render final code LIVE as `dep-db9tprm7bikc73e46png` / source `af509c2…`. Live one-off Actions run [39068175789](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39068175789) SUCCESS: two independent real Chromium sessions against the existing PostgreSQL-backed service, repeatable seed, persistent booking confirmation, previous plan/Pending, full two-visit chart, refresh and patient-switch isolation. Existing secret reused without disclosure; no new configuration, credentials, services, migrations, provider bookings, M8 or CP1 work. URL `/clinical/synthetic-161/`; fixed synthetic subset excluded from ordinary Home search/recent encounters and rejected by live booking confirmation. IMPLEMENTED / TEST DEPLOYED / READY FOR PRODUCT OWNER TEST; ACCEPTED remains pending. Writer released and CP1 explicitly stopped until PO acceptance.


## 2026-10-18 — #161 PO rejection and bounded usability correction

PO rejected duplicate identity checks and an inert Pending status despite technical PASS. Fixed synthetic appointment identity now opens the existing Brief projection directly; registry selection opens stored chart. Added inline received/undo on the existing synthetic Pending row, timestamp conflict protection and synthetic audit. Shared real-patient authentication/verification unchanged. 35 affected Python/Chromium tests and four Node suites PASS. Not yet deployed or accepted; CP1 remains stopped.

#161 UX correction #170 released as `3684ee5` on existing Render at source `9466375` (LIVE `dep-db9uapnlk1mc73cbij7g`). Live run `39070316620` SUCCESS proves direct opening and received/undo persistence across two independent browsers and refresh. PO rejection remains authoritative until explicit acceptance. CP1 stopped; bounded writer released.


## 2026-10-18 — #161 obligation provenance and usable patient chart

PO rejected unstructured chart and hidden obligation origin despite direct-opening correction. Bounded synthetic correction reuses structured signed encounter, existing Pending and audit records for clinical overview, source-linked obligation creation and dated lifecycle. Existing Home Clinical Updates/obligations panels read these same records; no Gmail/unread claims, new source or storage. 36 affected Python/Chromium and four Node suites PASS, nine focused synthetic tests. Release and PO acceptance pending; CP1 stopped.

#171 released as `b93d47c` on existing Render `dep-db9ujlbm8hqs73atet0g`, source `cbad563` LIVE. Live run `39071664714` SUCCESS verifies source encounter, lifecycle, structured chart and Clinical Updates navigation in two independent browser sessions. Prior received status restored. PO acceptance remains pending; CP1 stopped, writer released.


## 2026-10-18 — CP1 normal Cockpit natural-summary capture deployed and live verified

PO accepted the #161 experience and authorized CP1 in the normal application. #172 (`e5910a4`) reused VisitCapture context/preview/atomic Save, signed Encounter and ClinicalPending for natural Dia extraction through the existing configured provider, editable approval and source-linked chart/Clinical Updates. Raw source remains ephemeral; accepted bounded excerpts map to Pending IDs inside the signed encounter. A separately marked test patient uses the same normal application and storage, excluded from real registry/recent views. No new provider, credentials, schema or CP2 lifecycle was introduced.

Initial live extraction failed because the existing provider secret returned credit_balance_exhausted. #173 (`b046185`) added sanitized, actionable quota messaging; PO restored the balance. Live run `39074882064` SUCCESS then proved actual provider conversion, clinician editing/rejection, single atomic persisted visit/accepted obligation, duplicate refusal and persistence/source navigation across independent desktop/mobile sessions. #174 (`f84e501`) compacted review fields and wrapped update rows. Final Render `dep-db9vdfe1egvs73d7dn7g` LIVE, final read-only run `39075421312` SUCCESS. Synthetic screenshots inspected. 62 affected local Python/browser tests, four Node suites and affected CI passed; bounded independent review chain closed. No real patient data used in tests.

CP1 technical delivery is ready for personal PO testing, not PO-accepted. Writer released; exact next action is CP1 feedback/acceptance, not CP2–CP5 activation. Separate PR1 privacy/release HOLD unchanged.


## 2026-10-18 — Normal Cockpit patient follow-through installed (#175)

- Product Owner requested restoration/reuse of the existing Pending checkpoint implementation and a dot timeline that opens a popup, then the source visit via «Άνοιγμα».
- Reused #166 event-owner semantics over existing ClinicalPending and signed Encounter owners: explicit plan adoption/observation, independent receipt and image review, timestamped not-yet, exact-event undo, revision uniqueness/idempotent replay. Real writes retain existing clinician authority; shared-key synthetic actions require the fixed patient ID and marker.
- Agent installed the reviewed append-only PostgreSQL table through the existing Aiven administrator using verify-full TLS. Verified 13 columns, 5 indexes and SELECT/INSERT-only application grants. No network/IP filter, application environment, credential, plan or cost change; no patient rows read or changed by migration. Earlier PG Studio command-count and browser-IP obstacles were resolved through direct administrative execution.
- PR #175 squash: `ed979bf60ca876f044f917d997728e4354bcd2fc`; Render `dep-dba64jlg1s2s73b10nbg` LIVE at that source, 06:09:48 UTC.
- Independent bounded event-owner and popup/draft-protection review closure PASS. Eight CP1/CP2 API/browser tests and 22 inherited capture tests PASS; affected Capture/Home/Brief and canonical CI PASS. Unrelated existing Learning L0 design-only allowlist failure unchanged.
- Live proof [39103120886](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39103120886) SUCCESS: actual PostgreSQL receipt/not-yet/review/completion/undo and idempotent replay, unchanged signed visit, popup/source navigation, matching chart/Clinical Updates after refresh in independent 1280/390 sessions. No provider request or new encounter. Desktop/mobile screenshots inspected.
- Persistent fixture «Δοκιμαστικός ασθενής · Σύνοψη Dia» remains available in normal Cockpit, CD receipt confirmed and image review pending. Installed and synthetic-live-verified; personal Product Owner acceptance remains pending. CP3–CP5 are not activated by this release.


## 2026-10-18 — Visit-first trajectory and finite follow-through installed (#177)

Product Owner liked the installed experience and clarified its visual hierarchy: visits are the main trajectory, while receipt/review are subordinate follow-through of the originating visit. Implemented visits-only continuing main rail and source_encounter_id-linked finite obligation branches, reusing existing checkpoint controls and server state. Completion and exact undo update the branch endpoint; the top obligations card navigates to the branch. Existing visit popup → Άνοιγμα remains.

Runtime d83830f4eac362d6c8ba08b54d5c8535a26e9542 deployed LIVE through native Render dep-dba6sibm8hqs73b3hs10. Eight local CP1/CP2 tests PASS, including two visits/exact source association, independent stages, completion/undo, draft safeguards, refresh and independent1280/390 browsers. Independent bounded R1 PASS on371ea37; affected Capture39105910461, Home39105910432, Brief39105910418 and canonical39105910466 PASS. Existing unrelated L0 design-only allowlist rejection verified separately.

Live synthetic39106123898 PASS: primary visit hierarchy, finite branch, receipt/notyet/review/completion/undo, idempotent replay, unchanged signed visit, popup/open source and same persisted chart/Clinical Updates across desktop/mobile. Screenshots inspected under cockpit-v4-browser-evidence/visit-followthrough-live/. Persistent test fixture left receipt-confirmed/image-review-pending. No new encounter/provider request, backend/schema/auth/environment/network change or user technical step. Correction is implemented, merged, deployed and live synthetic verified; personal PO review pending. CP3–CP5 and separate PR1 HOLD unchanged.


## 2026-10-18 — One connected visit trajectory matching PO sketch (#179)

The PO clarified with a drawing that #177’s separate branch cards did not meet the intended geometry. The installed correction draws one continuous primary visit line and a finite subordinate line descending from each actual source visit dot. Branches share their source visit’s coordinate system, with real CD/review state and the same existing confirmations/undo. Unknown preadoption obligations also connect; cancellation remains distinct from completion. Popup → Άνοιγμα retains exact source visit navigation, and mobile uses internal horizontal scrolling with a hint.

Reviewed runtime c388eb2; independent R1 cancellation-label finding corrected and closed (ALL MATERIAL FINDINGS CLOSED / COMPLETE_FOR_DECLARED_SCOPE). Real Chromium two-visit proof measures actual source/start/end coordinates and verifies main alignment, collapse/reopen, mobile navigation, independent stages/completion/undo, draft guard, refresh/second session and no document overflow. Capture39107312287, Home39107312259, Brief39107312243, Demo39107312239 and canonical39107312227 PASS. Existing unrelated L0 design-only allowlist failure separately verified.

Squash a8c629673a7f6a2ff468f7ac033b32564d5c06f3 LIVE through Render dep-dba77davcj2c739u9f60. Live normal-app synthetic39107552980 PASS: measured source-dot fork/finite end, receipt/notyet/review/completion/undo, replay, immutable signed source, popup/open and persisted chart/Clinical Updates across independent1280/390 sessions. Final screenshots connected-journey-1280.png and390.png inspected under cockpit-v4-browser-evidence/connected-journey-live/ (early extra pre-popup desktop capture blank; final valid readback retained). Fixture remains receipt-confirmed/images-pending. No provider/new encounter or backend/schema/auth/env/network change. Installed and live synthetic verified; personal PO visual acceptance pending. CP3–CP5 and separate PR1 HOLD unchanged.


## 2026-10-18 — Shared real patient creation/opening installed (#181)

PO screenshots established the real-use entry gap: Home had no create, real appointment Brief could not open an independent manual chart path, and the capture URL still exposed the obsolete JSON page. One shared native-dialog selector now serves Home/New patient/no-result create, booking preview, dated-schedule public appointment caller and Capture. Name search shows stored records for explicit selection; new profiles require name with optional DOB/phone. Appointment names are editable lookup drafts, never automatic identity/link authority. Existing protected registry, chart and natural CP1 composer/context/Save owners reused; no extra identity checkbox introduced for manual chart opening.

A strict create-only adapter beside existing upsert uses a derived UUID ID, privately bounded validation and normalized immutable replay. Reuse conflicts cannot overwrite a profile; primary-key race rollback/readback produces one record. No visit/Pending/provider call on profile creation. Legacy capture redirects before its old UI loads. Draft guards cover navigation before reset; close/auth/context invalidation prevents late create/search from reopening or exposing hidden data. Optional staff Brief401/403 probes independent registry status before revoking a valid registry session; protected Brief/link authority remains gated. Separate linking action is visually optional.

Independent R2 predesign PASS on8fb60f9; fidelity finding corrected/closed on8a653ee; missed dated-schedule consumer bound and reviewed PASS on6361a69. ALL MATERIAL FINDINGS CLOSED / COMPLETE_FOR_DECLARED_SCOPE. Three new API/browser tests PASS (concurrent replay, conflicting input, roles, no clinical side effects, namesakes, failed Brief401/403, actual public dated caller, all entries, refresh, draft and late-result privacy1280/390). Existing41affected Python and3Node suites retained; final Capture39109581262, Home39109581326, Brief39109581397, Demo39109581350, Baseline39109581456 and canonical39109580892 PASS. Existing unrelated L0 design-only allowlist rejection separately verified.

Squash51532021fc7c809f0d6325b34083d0c17c10add8 LIVE via Renderdep-dba7ol142hec73dav160. Live39109917285 PASS at normal /static/cockpit/ in independent1280/390 sessions: new form, private invalid/no-write422 validation, actual displayed appointment→manual name prefill in both sessions, legacy capture existing marked chart→naturalcomposer and new selector/no JSON. No real appointment names/results logged/captured; safe screenshots inspected under cockpit-v4-browser-evidence/patient-entry-live/. No valid production creation, encounter, Pending, link or provider write was performed; real creation/race/persistence proven locally and in CI, not labelled live production-write proof. No schema/env/IP/account change. Installed and live navigation verified; personal PO use pending. CP3–CP5 and separate PR1 HOLD unchanged.


## 2026-10-18 — Dia instruction restored beside the actual natural summary (#183)

PO requested visible instruction text/copy after the prompt control had remained in a hidden prototype surface. The actual normal composer now always shows “Αντιγραφή οδηγίας για το Dia”, with the identical static instruction in expandable/selectable text. Existing clipboard helper extended/exported, preserving old callers. Clipboard denial reveals manual text without losing the active summary draft. The instruction asks faithful natural Greek source summarization, explicit uncertainty, planned/performed/results and receipt/image-review distinctions; no identifiers, invented clinical facts or new advice. No mandatory JSON/special headings or new provider/model/write authority.

Independent bounded R1 PASS on a0cf7c6. Actual Chromium1280/390 proves clipboardwrite/read exact displayed prompt, no patient name/DOB, denial/manual fallback, draft preservation and nooverflow; the existing CP1 edit/reject/Save/refresh/second-session flow remains PASS. Affected Capture39111019531, Home39111019480, Brief39111019518, Demo39111019549 and canonical39111019508 PASS. Existing unrelated L0 design-only allowlist failure verified separately.

Squash7a7f445a05575b70d4d48a4039cdad41eead59cb LIVE via Renderdep-dba835e7bikc73ec74o0. Live39111291323 PASS against the normal capture route/existing marked chart: actual clipboard matches visible static instruction, no identifiers, draft preserved, readable expandable text/nooverflow independent1280/390. Safe screenshots inspected under cockpit-v4-browser-evidence/dia-prompt-live/. No patient/clinical/provider writes, schema/auth/env/network change. Installed/live copy verified; personal PO use pending. Later milestones unchanged.


## 2026-10-18 — Real obligation actions, popup and reversible deletion installed (#185)

PO reported real chart obligations were informational and a completed second visit offered no way to close the prior follow-up. Source investigation proved an authentication mismatch: normal VisitCaptureSave accepted protected Clinical Key while Pending allowed staff clinician or an exceptional marked synthetic fixture. This meant prior synthetic success did not cover real-key capability. Pending now reuses normal protected clinical authority with truthful shared_clinical_key attribution; any staff-cookie presence, including empty/invalid/rotated/secretary, still requires valid staff clinician without fallback. Booking/staff-labelled Visit Brief access remains unchanged. Removed synthetic actor exception so the fixed fixture tests exactly the same authority as ordinary charts.

Overview obligation, source-branch title/dot and Clinical Updates open the same native popup and existing checkpoint actions. Follow-up completion is explicit, never inferred from another visit. CD receipt and image review remain independent. Delete removes an irrelevant open obligation from active overview/branches/Updates and exposes restore under collapsed deleted history. Cancellation/restoration append to the existing event table, enforce actor/target/replay/revision and exact active cancellation, preserve prior state and support later plan adoption with monotonic revision. Row locks serialize writes; signed visit content remains unchanged. No schema/environment/network/provider change or hard deletion.

R2 design3167c9c and exact-runtime fidelitya4c0ac9 PASS COMPLETE_FOR_DECLARED_SCOPE.6CP2API/browser tests PASS including unmarked normal-key chart, second-visit prior follow-up closure, notyet/undo/delete/keep/restore/refresh/independent1280/390, strict staff-cookie exclusion, revision/replay/source immutability.56 existing affected Python and3Node suites PASS. Final Capture39113385516, Home39113385463, Brief39113385434, Demo39113385450 and canonical39113385452 SUCCESS; known unrelated L0 design-only scope rejection separately verified.

Squash7e0e42a0a5af4268f7ae7c2e775feb099b5449d0 LIVE via Renderdep-dba8kdjm8hqs73b4sc8g. Live39114026743 PASS in independent1280/390: same normal auth, popup overview/branch/actualInbox Updates, notyet/completion/undo, deletion/keep/persistedrestore, nooverflow and immutable signed source. Initial markedfixture was actuallyresolved with both receipt+review confirmed; proof restored that observed completed state at the end of each session. Earlier39113802021/39113912647 stopped before mutation because remembered baseline assumptions were stale, not application failure. Safe popup screenshots inspected undercockpit-v4-browser-evidence/obligation-actions-live/. No provider/newvisit/real-patient writes. Installed and same-authority live synthetic verified; personal PO use pending; later milestones unchanged.


## 2026-10-18 — One-login appointment-to-chart continuity installed (#187)

PO reported that appointment preview demanded another connection and remained disconnected from the existing patient chart/latest visit. First divergences: legacy VisitBrief imposed separately configured staff credentials and every-opening independent confirmation; manual entry dropped booking context instead of retaining its chart selection. Normal physician chart-* adapter now reuses existing protected Clinical Key/valid staff-clinician authority with strict staff-cookie precedence and honest shared actor, while preserving old staff/secretary route contracts. Same BookingPatientLinkORM/link+access audits reused, no new store/schema/environment/credential. First explicit chart selection persists for one canonical booking UID and observed attendee label; namesakes/new UIDs never auto-select. Read/write source qualification, observation, blank/name drift, revision/replay/CAS/revoke checks preserve clinical access boundaries and source records.

Latest signed stored visit has its own labelled projection, preserving old appointment-asof Brief semantics. Exact visit/chart and current Pending popup actions are available in the connected appointment. Home slots, dated schedule and weekly-calendar URL use the same controller/session; Cyprus date is carried across UTC midnight. Shared selector callback preserves a created profile after link failure, retries linking that same ID, and ignores late closed/auth/context results. Signed encounters and provider bookings remain unchanged.

Independent R2 design5caddbd PASS. Fidelity47cbdf4 found blank source attendee accepted as linkage evidence; corrected before release, with initial/saved-link blank read/select rejection and preserved association. Cumulative closurea3134b0 PASS ALL MATERIAL FINDINGS CLOSED / COMPLETE_FOR_DECLARED_SCOPE. Final bounded local-day caller correction4201294 independently PASS with actual midnight click regression4108102. Three new API/actual-browser tests at1280/390 PASS: explicit choice, canonical UID aliases, namesakes, persistence/reopen/independent session, latest exact visit vs asof, source drift/blank, auth/replay/CAS/revoke, create-link failure/retry without duplicates, late close and calendar URL. Existing CP1/CP2/entry/VisitBrief/Home/Calendar/Node/Chromium gates retained. Final Capture39115590050, Home39115590010, Brief39115590018, Calendar39115590045, Demo39115590040 and canonical39115590029 SUCCESS; known unrelated L0 design-only allowlist failure separately verified.

Squash28e7038ec2f7295445c2c747e0c4f1841286400a LIVE through Renderdep-dba943bm8hqs73b58sp0. Live39115856958 PASS using actual displayed appointments and one normal login per independent1280/390: chart-read200/no-store, shared selector, close/reopen and weekly-link entry reuse session. Actual names/history were not captured/logged; emptied form screenshots inspected undercockpit-v4-browser-evidence/appointment-chart-live/. No real-person association/profile/encounter/provider/model write performed by proof. True persistent mapping and latest-visit behavior are verified through local/CI writes, not mislabelled production real-person write proof. Installed/live navigation verified; personalPOselection/use pending. Later milestones unchanged.


## 2026-10-18 — Simple obligation controls and completed-task removal installed (#189)

PO rejected universal CD/report tracking choices on unrelated reminders and could not delete an already-completed BP task. Unplanned ordinary tasks now show only “Ολοκληρώθηκε η ενέργεια; — Ναι / Όχι ακόμη”, adopting the existing completion plan on the first explicit observation. Existing approved CD/image plans and independent confirmations remain intact; no silent plan migration or description-based classification.

The existing append-only cancellation owner now accepts resolved obligations, records previous status/resolution pointer and clears the active pointer atomically. Restore requires the exact current cancellation and revision, restores that exact status/pointer and preserves confirmation events and signed source. Delete/keep/restore is available inline and in the popup, including completed tasks, under unchanged auth/draft/busy/late-response guards. No automatic deletion of the actual BP task was performed. Extraction and copyable Dia instructions explicitly distinguish missing/blank fields from agreed future actions; this is source fidelity, not a specialty-wide BP policy.

R2 predesignb9d022b and implementation1071b19 PASS COMPLETE_FOR_DECLARED_SCOPE.8CP2 tests verify unmarked1280/390 simple controls, resolved inline/popup remove/keep/restore, legacy no-plan resolution pointer, replay/stale/source immutability and existing independent CD stages;80 affectedPython+4Node PASS. FinalCI exposed a preexisting bookingURL deferred-script startup race:0.3s delayed patient-entry download reproduced the exact failure, and initial activation now waitsDOMContentLoaded. Corrected3appointmenttests/BriefNodePASS, independentdelta88640e4 PASS with prior obligation approval retained. Final published30b88c6 Capture39125843005/Home39125843045/Brief39125842976/Demo39125842984/canonical39125842975 SUCCESS. Known L0design-onlyallowlist rejection separately verified as unrelated.

Squash6d46974e26dde13fe81d93bb5d7e532748730a74 LIVE via Renderdep-dbabcne1egvs73dgfoeg at12:08:41UTC. Live39126229636 PASS on fixed marked synthetic chart: completed CD delete/keep/persistedrestore with approved receipt+review retained in independent1280/390 and unchanged signedvisits. Two real-provider synthetic drafts PASS: blankBP yields noPending; explicitly agreed BP/pulse measurement yields sourcedtask, no encounterSave. Firstlive39126097681 stopped before mutation because the verification script read template.textContent instead of template.content.textContent; verification-only581e4e2 corrected it. Safe screenshots inspected undercockpit-v4-browser-evidence/simple-obligations-live/. No realpatient/profile/visit writes or schema/auth/env/network changes. Installed and live synthetic verified; personalPOuse pending, later milestones unchanged.


## 2026-10-18 — Unique recent patients and canonical Inbox/task distinction restored (#191)

PO reported the same patient repeatedly occupying the recent rail and Clinical Updates and asked to check canonicals. Investigation found two distinct causes: recent-encounters limited three visits rather than three unique patient IDs; doctor-shell routed both the mail Inbox and task entry to the same ClinicalPending list. Phaseplan§34, Product Constitution and the independentInboxdesign reserve ClinicalInbox for incoming results/documents/notifications and review/action independent of appointments. The binding CP3plan calls for selective3–4relevantHomeitems, not a repeat of all visits or tasks.

Recent projection now ranks latest completed/amended encounter per exactpatientID before limit3 and the rail is labelledΠρόσφατοιασθενείς; identical names remain separate identities. Existing patient tasks have their own Home entry and task view, grouped by exactpatientID with each PendingID/nativeaction retained. Optionalactive_onlyreadfilter runs before100limit so resolved/cancelled rows do not displace open tasks; default APIhistory remains compatible. MailInbox truthfully shows its unconnected source status and makes no Pendingread; refresh replaces rather than duplicates its placeholder. No Gmailintake or fullCP3ranking/DailyBrief was activated.

51affectedPython and3Node suites PASS, including actual1280/390 browser grouping, namesakes, taskpopup routing, repeatedrecentvisits, minimal protected projection, active/defaulthistory and InboxnoPendingread. IndependentR1 exact73db71d PASS COMPLETE_FOR_DECLARED_SCOPE; one bounded idempotentrefresh delta2e9cd8c PASS. Final published df16e3f Capture39127376276, Home39127376263, Brief39127376300, Demo39127376295, baseline39127376243 andcanonical39127376235 SUCCESS. Known unrelatedL0design-onlyallowlistfailure separately verified.

Squashcd632b8f879d4c2978df08859ce096ae1ddbe8dd LIVE via Renderdep-dbabnd61egvs73dgpj50 at12:31:29UTC. Actual-source readonlylive39127659489 PASS1280/390: protectedrecentIDsunique/railcount, incomingInboxwithoutPendingread, actualactivePendingIDs exactly preserved in patientgroups and nooverflow. Only safeemptyInbox screenshots captured/inspected undercockpit-v4-browser-evidence/attention-separation-live/. No actual names/history logged/captured, no patient/task/providerwrites or schema/auth/env/networkchanges. Installed and live-read verified; POusepending. FullInbox andCP3remain planned; no new milestone or laterphaseactivation.


## 2026-10-18 — Compact obligation branches and permanent removal from trash (#193/#194)

PO requested less visual weight for follow-through, deletion beside undo, removal of repeated source text/origin buttons and permanent deletion beyond reversible trash. Inline branches now omit duplicated source details, retain those details in the native popup, use smaller gaps and share a compact action row. Source-linked visit geometry and independent CD confirmations remain unchanged.

Existing lifecycle now supports cancelled → terminal deleted, using the exact active cancellation/revision and append-only permanently_deleted event. Exact replay is safe; restoration, plan/step writes and legacy updates cannot resurrect it. Current task reads exclude terminal rows, including trash, before their limits. Signed visits and private audit/tombstone remain: this is irreversible operational task removal, not physical clinical-record erasure. The UI offers it only from trash and explains that it cannot be restored and the stored visit remains. No actual task was automatically deleted.

Pre-code R2 found the legacy synthetic writer could revive a terminal row using its current timestamp. Design c27a858 closed this with an atomic status exclusion and explicit regression; implementation d3c616c independently PASS. 55 affected Python and3 Node suites passed, with final13CP2 tests covering planned/unplanned/resolved-then-cancelled deletion, replay/stale/identity/source/other-task preservation, legacy rejection/seed preservation and actual1280/390 compact controls and permanent-confirmation persistence. #193 squash eea7e83d9ee4dbf4f3b9475aa21752a6cbb6ca63 deployed through dep-dbac1udg1s2s73b5r89g. Final affected Capture39128945167/Home39128945258/Brief39128945169/Demo39128945183/baseline39128945288/canonical39128945322 passed; known unrelated L0 allowlist failure separately verified.

Read-only live probes39129251840 and39129465429 exposed actual CD action wrapping. #194 reduces only scoped rail button font/padding/gap and adds a multistage alignment regression;13CP2PASS and independent R1 65baa49 PASS. Final Capture39129754911/Home39129754921/Brief39129754931/Demo39129754897/canonical39129754953 passed. Squash4797b60f002b90e903ff7b89b375870387304325 LIVE via dep-dbac79gae00c739jivog at13:05:10UTC. Live39130033288 PASS1280/390: same-row undo/delete, no inline duplicate source, popup source retained, no overflow and unchanged fixed marked chart. Safe screenshots inspected undercockpit-v4-browser-evidence/compact-obligations-live/. Permanent mutations were verified locally/CI, not performed on production fixtures or real tasks. No schema/grant/auth/env/network/provider changes. Installed/live layout verified; PO use pending, later milestones unchanged.


## 2026-10-18 — Focused stored visit reader (#196)

PO reported that opening a past visit left the latest clinical summary at the top and accumulated all expanded visits below it. Timeline Open, obligation-source and appointment-latest now converge on one exact-encounter selector. A separate focused reader shows only the chosen stored visit/date/reason; chart overview, trajectory and other bodies are hidden. Back restores the existing chart, scroll and source-dot focus. Existing signed content, protected loading and patient/draft/busy/generation/auth safeguards remain unchanged. No backend, schema, clinical state or provider delta.

Independent R1 e8ecef3 PASS / COMPLETE_FOR_DECLARED_SCOPE, no material findings. 34 affected CP1/CP2/patient-entry/appointment/Home Python tests and three Node suites passed. Actual Chromium1280/390 proves two-date switching without accumulation, obligation-source closure, appointment latest, Back and active-draft preservation; screenshots inspected. Final affected Capture39131271753/Home39131271765/Brief39131271684/Demo39131271630/canonical39131271604 SUCCESS; existing unrelated L0 design-only allowlist39131271769 failure inspected.

Squash58281ea331ae13e6c4a79c06cd875666e6163c75 LIVE via Renderdep-dbach6qvcj2c73a2s910 at13:26:26UTC. Read-only fixed marked live39131610970 on verification-onlyb921f5d PASS1280/390: one exact selected visit, overview hidden, timeline/source/shared public reader, Back, nooverflow, full chart unchanged. Safe screenshots inspected atcockpit-v4-browser-evidence/focused-visit-live/. Live fixture has one visit; two-date fidelity and actual booking caller are local/CI evidence, not represented as real booking writes or two-date live proof. No production patient/task/provider mutation. Installed/live read verified; personal PO use pending, CP3/Inbox/PR-1 remain unchanged.


## 2026-10-18 — Explicit audited correction of stored visits (#198)

PO showed an old knee follow-up with an unsupported BP task derived from a missing checklist field and requested editing visits from their beginning, including captures made before the improved Dia instruction. Existing extraction already distinguishes missing fields from explicitly agreed future tasks. No real clinical content or task was changed automatically.

Focused signed structured visits now offer Επεξεργασία επίσκεψης with the stored date/full candidate prefilled. Clinician can correct/add/remove fields, optionally paste and re-extract natural text, inspect kept/new/replaced/removed task effects, and explicitly save corrections on the same encounter. Direct overwrite remains blocked; explicit amendment appends previous complete signed snapshots and honest actor/time/request identity in flat Encounter JSON history, updates the effective revision/status/date and exposes read-only history. No new table/grant/auth/provider settings.

Unchanged bound task ids, confirmations and status remain. Removed/replaced tasks use the shared existing lifecycle cancellation helper atomically; replacements begin unfulfilled, cancelled/deleted rows are never implicitly restored. Historical source excerpts resolve exact task ids across prior snapshots. Full encounter+task token, strict actor/cookie precedence, existing context, row locks/CAS/context consumption, transactional rollback and actor-bound exact replay protect writes. Current latest chart uses effective revision; appointment-as-of selects the effective historical signed revision before its clinical-date cutoff.

R2design300eb8a PASS. Initiald579051 fidelity BLOCK found superseded as-of fallback, old Save continuation crossing a reopened draft, and422 rejection locking a correctable draft. Boundedf94d67f fixes and direct temporal/delayed-preview/rejected-task-status browser regressions independently PASS ALL MATERIAL FINDINGS CLOSED / COMPLETE_FOR_DECLARED_SCOPE. Earlier79affectedPython/3Node and final24amendment/Brief tests passed; API covers same-id history, unchanged completion, replace/remove/new task semantics, permanent-state preservation, roles/private errors, stale/replay/context and rollback; actualChromium1280/390 covers edit/cancel/add/remove/re-extract/preview/Save/refresh/independent read. Final Capture39133420891/Home39133420842/Brief39133420846/Demo39133421024/canonical39133420896 SUCCESS; known unrelated L0design-onlyallowlist39133420799 failure inspected.

Squashc2ed41b9a3da197b5394c6925b8e35506dc16bd3 LIVE through Renderdep-dbacv161egvs73di0fvg at13:55:51UTC. Verification-onlyc5b8950 run39133756681 PASS1280/390: fixed marked edit prefill/date, unsaved change cancel/reopen, nooverflow/pageerrors, full chart identical and zero amendment/extraction/save requests. Safe screenshots inspected undercockpit-v4-browser-evidence/visit-amendments-live/. Actual write behavior verified locally/CI, never performed on real records or the established production fixture. Installed/live UI verified; personal PO use pending, CP3/Inbox/PR-1 unchanged.


## 2026-10-18 — Next-day preparation and full day agenda (#200/#201)

PO requested an arrow in the top-right date popover to prepare tomorrow's visits. #200 adds previous/next day navigation, full Cyprus date with Today/Tomorrow label, Today return and reset to today on reopening. Same appointment caller carries the booking's own Cyprus date and never infers patient identity. Each load invalidates older results; navigation remains available during empty/error/loading and keyboard focus survives repaint. R1review0395665 PASS,14Home/appointmentPython/3Node and actualChromium1280/680/390 PASS, including DST23/25hours/year rollover, stale-day rejection, failure recovery and nooverflow. Home39152573177/Brief39152573358/Capture39152573467/Demo39152573297/canonical39152573414 passed. Squashf2bdba77bbbe38d12fdea4867b8b7bf2c6fee970 deployed throughdep-dbagm3psrm7s73aemitg at18:09:54UTC.

Read-only39153152179 proved arrows but returned filtered0tomorrow. Inspection showed the inherited /appointments intentionally excludes non-Osteoporosis categories; this was not accepted as complete day-preparation evidence. Bounded R2designdc1bfc0 andimplementation6c4bed6 independently PASS. #201 extracts the same validated dated source reader; default weekly /appointments still filters, while new protected /agenda includes all actual categories and returns only seven CockpitAppointment fields, UTC-aware and no-store. No phone/contact/patientlink/rawhistory fields or new source/store/write/auth/schema/env authority. Range/freshness/full coverage/invalid interval/unavailable checks remain shared. Only day popover rebinds.37calendar/Home/appointmentPython/3Node/actualbrowserPASS; mixedcategories, unchangedweeklyfilter, minimalkeys/noDBwrites and qualifiedempty/sourcefailure covered. Final Calendar39153829143/Home39153829180/Brief39153829205/Capture39153829246/Demo39153829211/canonical39153829230 SUCCESS; known unrelatedL0design-onlyallowlist39153829098 rejection inspected.

Finalsquasha8b0a59eabf611ca18ebfa1f9bb4a6877bcd4317 LIVE through Renderdep-dbagu50ae00c739oua3g at18:27:14UTC. Correctedactualreadonly39154287877 on verification-only42fb63e PASS1280/390:8tomorrowappointments, matchingCyprusday/count, same-dayHome-next inclusion, Todayreturn and nooverflow/pageerrors/clinical-linkwrites. Safe date/arrows-only screenshots inspected undercockpit-v4-browser-evidence/full-agenda-live/. Actual names/reasons never logged/captured. Finalfullagenda proof supersedes filteredzero for all-category preparation; bookingcaller/failure/stale/DST evidence local/CI retained. Installed and actual-source live verified; personal POusepending, CP3/Inbox/PR-1 unchanged.


---

## 2026-10-22 — Dedicated patient obligations screen (#205), source-preserving UX

The clinician asked to move all clinical patient obligations out of the small Tasks icon popover into a full first-class Cockpit workspace with grouping by exact patient identity, documented category and Cyprus review date, while maintaining an independent Clinical Inbox in which email arrives for review **before** any clinician-approved linkage/action. No new task registry, email ingest/auto-close, role/API/DB or patient write.

PR #205 implementation frozen source+test `f63eb7d84f60a104dcd8761d2a1b0d4cf0fe11b9`. An affected R1 found and corrected three P1 UI/privacy/data-authority issues (surgical badge mislabeled as obligations, hidden typed search value, unsaved draft hiding); focused delta+affected closure PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:0:0. Home+actual Chromium 1280/680/390 run 39715477276, Visit Capture+CP2 39715477212, Visit Brief/#161 39715477224, Surgery 39715477270, canonical 39715477155 all SUCCESS. Full-clinic 100+ pagination remains separate; the first screen clearly exposes current-100 limitation.

The PO authorized installation. Squash merge `56d37eeef5d331b53bb13afbd014160cc6855c9f` to `main`, native Render `dep-dbco16ou01pc73bhu1fg` LIVE same commit at 2026-10-22T03:20:54.956Z, no app runtime errors in the observed startup log window. Production use/individual PO experience not yet assessed. No Aiven/data rows, PHI, secrets, Gmail or other provider operations performed. Release writer released; next clinician UI acceptance, then scoped Clinical Inbox source intake and server pagination.

## 2026-10-23 — Cockpit PR #207 obligation category/due-date and autonomous in-list actions released

- The clinician explicitly approved **production release** after final independent **R2 PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0**. [Final review/approval receipt](cockpit/reviews/COCKPIT_INLIST_OBLIGATIONS_R2_FINAL_PASS_RELEASE_GATE_2026-10-23.md).
- Guarded squash merged [PR #207](https://github.com/athpapachr-cmd/osteoporosis/pull/207) from exact HEAD `a1334e013ccc62b64d6040606cac4059b578e677` into prior main `8708a37c0d4083c3f811ced6b186e9ef7e817355`, creating `c4a6df54d9f4cf9e2005fdf491881a672d3ce6ac`. All 15 related GitHub Actions passed; last code/test `13bd91426d1f7f09ce5c0a9d17fc153df6eefce4`.
- Native Render `osteoporosis` main auto-deploy `dep-dbdf1j95efls73apdta0` LIVE exact merge SHA. A docs-only merge checkpoint automatically deployed at exact `86b2de271bb8ef5f3d4de71990414eacf492d48c` as `dep-dbdf2oe1egvs738jnkf0`, LIVE 2026-10-23T05:34:31.368317Z, with unchanged runtime code. No manual duplicate deploy.
- No-PHI startup smoke at latest LIVE: 29 app log lines inspected, three startup confirmations, zero startup/runtime error signatures. No clinic patient record read/write, provider or data-authority change. Personal clinician workflow acceptance remains pending.
- Released clinician-confirmed category/date editor and within-category due/undated task groups, same-screen explicit metadata Save, reversible cancel/restore and optional original chart route. The original signed encounter text, task identity and checkpoint meaning remain owned by existing protected clinical Pending source and audit.
- Out of scope: full-clinic pagination beyond current first-100-active-row projection, Gmail/Clinical Inbox provider intake, automatic task completion, database schema, secrets, permissions, real-patient test writes. Detailed deployment evidence: [release receipt](cockpit/releases/COCKPIT_OBLIGATIONS_PR207_PRODUCTION_RELEASE_2026-10-23.md).

## 2026-10-24 — Clinical Inbox Phase A synthetic-only #161 preview — independent R2 closed, bounded production release

- Product Owner supplied independent focused **POST-CODE R2 PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:0:0**, P1-01 (patient re-link old review/disposition) and P1-02 (eligible-patient current/stale HTTP 401) both CLOSED at code+test `36387ec4696de07b847f52fae891a79d3db279a6`. [Receipt](cockpit/reviews/CLINICAL_INBOX_PHASE_A_SYNTHETIC_POSTCODE_R2_FINAL_CLOSURE_PASS_2026-10-24.md). Existing PRE-CODE Q1–Q6/P1/P2 PASS reused.
- After Product Owner's **“Προχωράμε!!!! Engage”**, release was limited to the existing protected **#161 synthetic test preview**; no approval for real patient data or mailbox integrations. Exact PR HEAD `2c8a565ae0112cbbe354b848540b9a24aa6fb198` had **15/15 PASS** GitHub Actions including core Chromium/Visit Brief and canonical.
- [PR #208](https://github.com/athpapachr-cmd/osteoporosis/pull/208) was guarded squash-merged from pre-release main `32a87ec8757b21b64dd8837e362810b5a2b9b510` as `a35f10e099e41f603ee7c9dfc8a947d9407051ba`. Native autoDeploy of merge SHA canceled when the mandatory docs-only merged checkpoint followed; latest reviewed runtime (unchanged) was **LIVE** under docs-only `b6698b306cfad7a86ebb130f88c2e0a86722da32` / Render `dep-dbe38m4s728c738b7ntg`, finished 2026-10-24T04:32:34.375812Z.
- No-PHI app log smoke on LIVE: **41 app entries, six standard startup signatures, zero error/Traceback/Exception signatures**. No real patient record, email/provider/source document read or write. No authentic clinician user-experience acceptance asserted.
- Implemented capabilities: synthetic-only separate Inbox via top ✉ inside protected #161 test route, account-shaped Gmail/iCloud/Outlook fixtures, explicit identity verification+review, CAS/idempotent revisions, mixed-subject Inbox-only, shared confirmed synthetic Visit Brief read. The **normal** Cockpit continues to show an unconnected Inbox without fake badge; ephemeral in-memory attention state disappears on process restart.
- **Open future gates**: protected persistent ClinicalAttention schema/role/audit/retention/grants; non-synthetic clinician manual/source workflow; independently qualified Gmail/Outlook/iCloud provider connectors; always-on Reception ingestion; medical document processing; clinical record acceptance; outgoing SMS. These are separate phase decisions. [Detailed release receipt](cockpit/releases/COCKPIT_CLINICAL_INBOX_PHASE_A_SYNTHETIC_PR208_PRODUCTION_RELEASE_2026-10-24.md).
---

## 2026-10-25 — Cockpit Priority 2 Notion task code deployed, feature OFF

[PR #219](https://github.com/athpapachr-cmd/osteoporosis/pull/219) delivered a separate topbar peek of the existing **Οι εργασίες μου** Notion data source, direct page links, editable manual confirmation and optional short voice proposal. The existing ClinicalPending owner remains separate. Confirmed source notes/status update precedes any follow-up creation; a durable page/revision attempt guard prevents duplicate confirmation. Independent corrected pre-code and post-code R2 PASS / P0:P1:P2=0:0:0. Exact code CI 17/17 and final PR head CI 18/18 SUCCESS; synthetic local 29 Python, Node and actual Chromium 1280/680/390 PASS.

Merged code `4b61af01e4b65c2020cc8df8ed9bb231fe29a027` is LIVE at Render deploy `dep-dbf6mum7bikc73btbid0`. No-PHI smoke: Cockpit assets 200, deployed JS source hash matches, storage PostgreSQL online, unauthenticated task routes 401, zero error-level app logs. The Notion integration token is absent and task voice provider approval is not configured, so state is **DEPLOYED_FEATURE_OFF**. No real Notion task or provider call occurred. Activation needs a dedicated internal connection shared only with the existing database and task-specific voice approval; first actual task action belongs to the clinician after one compact review. Priority 3 remains on hold.

---

## 2026-10-26 — Cockpit Priority 2 source and voice activated for clinician smoke

The Product Owner installed the dedicated Notion token in Render and approved OpenAI processing of short task audio/transcripts. The bounded task and voice flags were enabled. An authenticated production peek first exposed a `42501` guard-table privilege blocker, so the application failed closed before any Notion call. [PR #220](https://github.com/athpapachr-cmd/osteoporosis/pull/220) added safe SQLSTATE-only diagnostics; [PR #221](https://github.com/athpapachr-cmd/osteoporosis/pull/221) supplied an independently reviewed least-privilege Aiven migration, exact-head CI 2/2 PASS. The existing administrator installed the empty unique guard table through one verify-full TLS `psql` transaction; metadata and app privileges verified 8/8, no patient rows accessed or network rule changed.

Live authenticated peek then returned four active Notion rows with direct links/revisions and `voice_available=true`. The Cockpit dropdown displayed all four with manual **Ενημέρωση** and **Υπαγόρευση** actions. State is **READY_FOR_PRODUCT_OWNER_SMOKE**. The agent did not send patient audio/content to OpenAI or confirm any Notion write; those first real outcomes await a clinician-selected task and explicit review. Priority 3 remains on hold.

---

## 2026-10-26 — Cockpit Priority 2 new-task and voice-prefill correction released

The clinician's first selected-task dictation exposed an empty editable proposal for future-tense speech and prompted a general new-task create/dictate action. [PR #222](https://github.com/athpapachr-cmd/osteoporosis/pull/222) adds both actions above the Notion preview, auto-fills editable new-task title/notes/due/category/priority from short dictation, and fills an existing task note for planned work without marking Done. All writes still require one clinician confirmation; new task creation reuses the durable one-shot guard and freezes unknown outcomes. The independent post-code R2 review found and closed a microphone cleanup issue; final verdict PASS / P0:P1:P2=0:0:0. Synthetic 18 Python, Doctor Shell Node and Chromium 1280/390 PASS; final PR head CI 11/11 PASS.

Squash merge `5663b9863439bdc79b31291d9c14d9668b652fa8` is LIVE on Render deploy `dep-dbfciamq1p3s73f35vt0`. Bounded no-write smoke matched deployed JS/CSS hashes to reviewed files, rejected anonymous task reads/proposals/invalid create with 401 and found zero error-level app logs. No real task was created by the agent; the clinician's first confirmed creation and its Notion result remain to be verified. Priority 3 remains on hold.


---

## 2026-10-26 — Cockpit Priority 3 Office Cases released for clinician acceptance

The Product Owner accepted Priority 2, requested the next roadmap priority and specified separate entries for each accident payment. [PR #223](https://github.com/athpapachr-cmd/osteoporosis/pull/223) adds one Office Cases screen with surgery room-entry time, accident register and individual audited payments, and exact eligible RF-to-surgery transfer. Independent post-code R2 passed 0:0:0 and final PR CI 19/19. The reviewed Aiven additive schema was installed and verified before code activation; temporary allowlist access was removed.

Squash merge `bd6133fef96a4e0f4a88e96907ec03ef5a16a263` is LIVE on native Render `dep-dbfecim7bikc73c4dbj0`. No-write smoke matched deployed Cockpit HTML/JS/CSS bytes, found 401 on protected anonymous read/write probes and zero error-level app logs. No real clinical or payment row was used by the agent. State is **READY_FOR_PRODUCT_OWNER_SMOKE**; first real clinician use and acceptance remain pending. Priority 4 is on hold. [Release receipt](cockpit/releases/COCKPIT_OFFICE_CASES_PRIORITY3_PRODUCTION_RELEASE_2026-10-26.md).

---

## 2026-10-26 — Priority 3 RF surgery prefill correction released

The clinician clarified that the requested date is the patient's date of birth and chose to enter DOB and phone in future RF Tracking records. [PR #225](https://github.com/athpapachr-cmd/osteoporosis/pull/225) adds optional protected RF fields and exact eligible per-RF surgery prefill, with manual completion for old RF. Independent exact-code R2 passed 0:0:0; RF, Office Cases, surgery and shared Cockpit checks passed. Three unrelated Clinical Learning jobs stopped in their pretest scope guard on the authorized RF file changes and were qualified for this release.

Squash merge `043bb8ecd0d0916488cd5681d5b1278c8ba15558` is LIVE on Render `dep-dbff88i9lscs739u3l60`. The merge message retained `[skip render]` strings, so one manual deploy was needed after no automatic deploy appeared. No-write smoke matched five served static files to source, received 401 on protected anonymous routes and found no error-level app logs in the release window. No real clinical row was read or written by the agent. Clinician visual/first-use acceptance is pending; Priority 4 remains on hold. [Release receipt](cockpit/releases/COCKPIT_RF_PREFILL_CORRECTION_PRODUCTION_RELEASE_2026-10-26.md).

---

## 2026-10-26 — Priority 4 local-only visit audio released and installed

The Product Owner chose a fully local Mac workflow after Apple Dictation performed poorly on mixed Greek/English clinical speech. [PR #227](https://github.com/athpapachr-cmd/osteoporosis/pull/227) adds informed-patient Start/Stop in protected Visit Capture, loopback-only audio handling, pinned local Whisper transcription, identifier-only Krikri classification, deterministic exact-source review, explicit risk/unclassified accounting, atomic patient-context renewal, copy-only GeSY text and existing preview/explicit Save as the sole clinical write. Independent corrected pre-code and post-code R2 passed 0:0:0; exact-head CI passed 15/15.

Initial stable installation found two fail-closed target-Mac portability issues before patient use. [PR #228](https://github.com/athpapachr-cmd/osteoporosis/pull/228) normalizes only exact Ollama SHA-256 digest wire forms and skips only macOS's exposed-but-unimplemented `RLIMIT_AS`, retaining decoder CPU/file bounds and every identity/cleanup guard. Correction CI passed 11/11; 55 affected Python, three Node and 37 shared Chromium regressions passed. Feature merge `1c46da558e88f14f8974de862ff5eebf6a5766f3` and final correction `a467ae510ef0fc792845b65ddcf895b30d4684ce` are LIVE on Render `dep-dbfsm3e0tbcc73du91qg`.

Final production no-write smoke matched three served assets to merged bytes, returned anonymous 401 on context/context-renew/preview/save and found zero error or HTTP 5xx logs. FileVault and exact pinned FFmpeg/Whisper/Krikri/Ollama assets are installed locally. A final merged-tree real loopback HTTP run on a 287.76-second synthetic mixed Greek/English recording returned 200 with strict schema, 55 exact-source units and 11 explicit unclassified items, then cleaned temporary content. No real patient audio/data, cloud AI request or clinical write occurred. State is **READY_FOR_PRODUCT_OWNER_SMOKE**; first informed-patient use and clinician accuracy/timing acceptance remain. [Release receipt](cockpit/releases/COCKPIT_LOCAL_VISIT_AUDIO_PRIORITY4_PRODUCTION_RELEASE_2026-10-26.md).

---

## 2026-10-27 — Priority 4 first-use local-audio entry corrected

The clinician's first launcher attempt reached the ordinary Dia-summary composer instead of local Start/Stop. Product Owner screenshots and live route inspection isolated the cause: the dedicated Visit Capture entry script discarded the exact nonce-bearing local-audio fragment by redirecting unconditionally to the integrated shell.

[PR #230](https://github.com/athpapachr-cmd/osteoporosis/pull/230) retains the dedicated page only for the exact 43-character base64url local nonce, while ordinary patient entry and malformed/manual fragments keep the previous safe redirect. Independent affected R1 passed 0:0:0; final exact-head CI passed 14/14. Squash merge `20b6e2a0120a3e18ecb89b33e29d65a433d3180c` became LIVE through native Render auto-deploy `dep-dbg1oqnf3r2c73b4d6ug` at 2026-10-27T03:39:08.372Z.

Production `entry.js` bytes exactly matched the merged source. Hidden real-browser no-write smoke passed exact local entry, ordinary synthetic-patient redirect and malformed-fragment rejection; release-window error and HTTP 500/502/503 logs were empty. No manual deploy, real patient audio/data or clinical write occurred. The entry blocker is closed; the first informed-patient recording, compact clinical review, identity confirmation, preview and explicit Save remain clinician-owned acceptance.

---

## 2026-10-27 — Guarded web Visit Capture audio code released

The Product Owner superseded the local-only processing direction with an integrated web path and required verified Zero Data Retention on the exact OpenAI project before real patient audio or transcript transmission. [PR #232](https://github.com/athpapachr-cmd/osteoporosis/pull/232) adds informed-patient chart recording, protected temporary audio upload, OpenAI `gpt-transcribe` and `gpt-6-luna` source-unit classification, exact-source editable draft, focused review, explicit recorded-patient confirmation and the existing one-shot Visit Capture Save. Independent pre-code and post-code R2 passed 0:0:0; final exact-head CI passed 17/17.

Squash merge `667d76f3d1c3b636d65a869df85d02dff10fe800` reached LIVE through one native Render deploy `dep-dbg4m68473hc73fcj4k0`. No-write production smoke matched five served assets, received 401 on anonymous protected audio/context/preview/save routes, and found no error-level app or HTTP 5xx logs in the release window. The Render environment lacks all three new purpose/ZDR gate variables, so real web audio remains disabled pending project proof and clinician first-use acceptance. No real patient audio, provider request or clinical write was made by the agent. [Release receipt](cockpit/releases/COCKPIT_WEB_VISIT_AUDIO_PRODUCTION_RELEASE_2026-10-27.md).

---

## 2026-10-27 — Web Visit Capture configured for standard OpenAI retention

The Product Owner replaced the earlier exact-project ZDR condition with standard OpenAI API retention. [PR #235](https://github.com/athpapachr-cmd/osteoporosis/pull/235) adds truthful clinician/patient notice, explicit per-request acknowledgement, a dedicated purpose/retention/project gate and `OpenAI-Project` headers on both provider requests. Responses retains `store:false`; standard abuse-monitoring retention is not described as deletion. Corrected independent pre/post-code R2 passed 0:0:0 and exact-head CI passed 14/14.

Squash merge `bb33258ea31a961560c3c34c2d93d7d50fbbcbe3` reached LIVE on one manual Render code deploy `dep-dbgelj0n74is739bjjog` after `[skip render]` in the squash body caused the automatic commit to be ignored. The three approved purpose/standard/project settings were then merged without changing the existing masked API key, and same-code config deploy `dep-dbgen2gn74is739bon1g` reached LIVE. After each transition, two static assets matched source, five anonymous protected routes returned 401 and release-window app error/HTTP 5xx logs were empty. The feature is configured for authenticated clinician use; the masked key/project association, authenticated capability, provider routing and first clinical result remain unverified. No agent patient audio, provider request or clinical write occurred. [Release receipt](cockpit/releases/COCKPIT_WEB_VISIT_AUDIO_STANDARD_RETENTION_RELEASE_2026-10-27.md).

[Docs receipt PR #236](https://github.com/athpapachr-cmd/osteoporosis/pull/236) passed 9/9 effective exact-head checks and merged as `5a83c40bc9df843a41429d78686980ff0ec7dfcc`. Render event `evt-dbgfj18473hc739j346g` ignored this `[skip render]` documentation commit, leaving the configuration deploy LIVE without duplication. Technical writer released; clinician first-use acceptance remains.

---

## 2026-10-28 — Web visit audio failure diagnostics released

The clinician's synthetic-chart recording reached protected context renewal (200) but the audio draft returned 502. The original request had no failure-stage marker and its temporary audio was removed, so its cause remains unknown. [PR #238](https://github.com/athpapachr-cmd/osteoporosis/pull/238) adds only private fixed stage/kind/provider-HTTP-status logging for future failures. It does not log clinical content or change the generic public error, fail-closed processing, cleanup or clinician Save. Eight focused local synthetic tests, independent Security Review, Product Owner-selected Bugbot and final 10/10 PR checks passed. One inherited appointment browser timeout on an intermediate docs head passed on a single rerun; the final head passed directly.

Squash merge `803f82ebb1a1395a34e19764dc688199c455b482` is LIVE via native Render deploy `dep-dbgqpcp5efls73au76s0`. No-PHI smoke matched two static assets, returned 401 on five anonymous protected routes and found no app ERROR/5xx in the release window. No agent patient or synthetic audio/provider call or clinical write followed deployment. Next clinician-owned step is one short fictional recording in the synthetic chart, without Save, to identify any new failure stage. [Release receipt](cockpit/releases/COCKPIT_WEB_VISIT_AUDIO_FAILURE_DIAGNOSTICS_RELEASE_2026-10-28.md).

---

## 2026-10-28 — Web visit audio compact review released

The Product Owner's successful fictional recording exposed an unusable review: the entire internal schema appeared and provisional source units felt like nine separate confirmations. [PR #240](https://github.com/athpapachr-cmd/osteoporosis/pull/240) replaces only the web-audio presentation with populated grouped fields, a closed deduplicated attention disclosure and one final patient/content confirmation after the fields. Provisional statements remain explicit uncertainties; any edit invalidates confirmation/Preview. Manual Dia/amendment editing, providers, retention, selected-patient context, backend schema and explicit Save remain unchanged.

Independent pre/post-code R2 passed 0:0:0 and final exact-head CI passed 14/14, including 46 Visit Capture tests in Chromium. Squash merge `73f091059f482f726abc3fc77b089422189bfb01` is LIVE through native Render deploy `dep-dbgrcc142hec73988oqg`. Exact JS/CSS hashes matched, five anonymous protected routes returned 401 and release-window app ERROR/HTTP 5xx logs were empty. No config, recording, provider request or clinical write occurred after deployment. Next is one Product Owner fictional recording without Save to accept the corrected UX. [Release receipt](cockpit/releases/COCKPIT_WEB_VISIT_AUDIO_COMPACT_REVIEW_RELEASE_2026-10-28.md).

---

## 2026-10-28 — Protected native/local audio patient confirmation released

The Product Owner chose local native processing on the Mac. A prerequisite gap existed in protected Visit Capture: recorded-patient confirmation was required for web audio but not for the already allowed local-audio source role. [PR #242](https://github.com/athpapachr-cmd/osteoporosis/pull/242) applies the same Preview/Save requirement to both audio roles and binds the existing local page's identity checkbox to the protected requests. Withdrawal invalidates Save readiness, including delayed Preview/context-renewal races. Non-audio Visit Capture remains on its established path.

Independent pre-code and post-code R2 closed at P0/P1/P2 = 0/0/0 after one bounded Save-readiness correction. Final exact-head CI passed 14/14. Squash merge `c090846a8258f2b4beb5a77e5c10d97b9ae7bfd3` reached LIVE on native Render deploy `dep-dbgsaaq9lscs73922j00`. Three served assets matched source; five anonymous protected routes returned 401; the release window contained no app ERROR or HTTP 5xx. No agent patient audio, provider request or clinical write was made. The installed native Mac app remains synthetic-only while natural-speech ASR, the missing observed macOS mic indicator and local model quality are investigated. [Release receipt](cockpit/releases/COCKPIT_NATIVE_AUDIO_CONFIRMATION_GUARD_RELEASE_2026-10-28.md).

Docs receipt [PR #243](https://github.com/athpapachr-cmd/osteoporosis/pull/243) passed 9/9 exact-head checks and merged as `a8c9945d6c4d2afcebd67866b3c1a5d62b955046`. Render event `evt-dbgsf7p42hec7399gurg` ignored this `[skip render]` commit, leaving the code deploy LIVE without duplication. The guard release writer was released; the next bounded work is native recording/ASR and model-quality qualification.

---

## 2026-10-28 — Web visit audio semantic completeness released

The Product Owner's fictional Dia comparison showed that the compact web-audio draft lost clinically material clauses from compound speech and offered no source-informed next-visit checks. [PR #245](https://github.com/athpapachr-cmd/osteoporosis/pull/245) adds validated non-overlapping exact-source multi-fact extraction, deterministic mapping for symptoms, therapy, decisions, explicit obligations and follow-up, and a separate bounded next-visit suggestion block. Suggestions remain outside Preview/Save unless the clinician uses one bulk add-to-plan action; one final patient/content confirmation remains.

Corrected independent pre/post-code R2 passed 0:0:0 and final exact-head CI passed 16/16, including both full Visit Capture Chromium jobs. Squash merge `d4e66a0147771834e27b1f6d1639ecaae1909f8e` reached LIVE on native Render deploy `dep-dbgt23ijnfac739m9ipg`. Three served assets matched merged bytes, five anonymous protected routes returned 401 and release-window app ERROR/HTTP 5xx logs were empty. No duplicate deploy, config change, recording, provider request or clinical write occurred. Next is one Product Owner fictional recording without Save to assess clinical completeness and suggestion usefulness. [Release receipt](cockpit/releases/COCKPIT_WEB_VISIT_AUDIO_SEMANTIC_COMPLETENESS_RELEASE_2026-10-28.md).

---

## 2026-10-28 — Web visit audio no-loss recovery and contextual review released

The Product Owner's fictional use showed that a successful transcript could be discarded when structuring failed, both transcript and processed text occupied the full page, and the draft needed bounded clinical considerations based on combined context. [PR #247](https://github.com/athpapachr-cmd/osteoporosis/pull/247) adds protected transcript-only recovery/reprocessing, browser-memory audio rescue, default-collapsed transcript/processed views and display-only contextual considerations. Transcript and inferred prose remain outside Preview/Save; one final patient/content confirmation and explicit Save remain.

The first native deploy failed safely before cutover because runtime DDL met the app role's correct lack of schema CREATE; prior production stayed LIVE. [PR #248](https://github.com/athpapachr-cmd/osteoporosis/pull/248) moved DDL to an independently reviewed least-privilege administrator migration. The exact empty opaque table was installed over verified TLS without reading patient rows or widening schema rights. PR #248 passed 10/10 final checks and merged as `3500cb49f27a7c5a89641a6bb9ce4bb9488bcae8`.

Native Render deploy `dep-dbh0muivcj2c73e00gj0` reached LIVE on the exact correction. Three served assets matched merged hashes, anonymous capability/draft/reprocess returned 401, and release-window app ERROR/HTTP 5xx logs were empty. No agent recording, provider request or clinical write occurred. Technical release is closed; one Product Owner fictional recording without Save is next. [Release receipt](cockpit/releases/COCKPIT_WEB_VISIT_AUDIO_RECOVERY_CONTEXTUAL_REVIEW_RELEASE_2026-10-28.md).

## 2026-10-28 — Direct audio summary and no-loss paste correction released

PR #250 removes the audio identity checkbox, displays the validated summary automatically after Stop, and uses one explicit Save with fresh current-candidate preview. Paste unlocks after processing; failed manual extraction preserves transcript recovery and any prior completed summary. Generic age/sex descriptions are explicitly distinguished from identifying contradictions in the model prompt. Independent R2 closed both fallback findings, 47 local affected tests and all 15 PR checks passed. Merge `796b11cd` is LIVE on `dep-dbh29tfb9u1s73e71900`; two assets matched and anonymous auth/error-log smoke passed. No provider or clinical write by the agent. [Receipt](cockpit/releases/COCKPIT_WEB_AUDIO_DIRECT_SUMMARY_RELEASE_2026-10-28.md).

## 2026-10-28 — Complete stored clinical facts restored to the final chart

PR #252 corrects source-proven UI omissions: history, findings, recommendations, follow-up and existing coding now render alongside the reason/assessment/plan in latest card, visit reader and prepared draft cards. Real provider and protected saved-synthetic-chart reads proved the four clinician-reported facts were present; production Chromium run40743751686 proves they are visible after the correction at1280/390 without changing the stored visit. Independent R1 PASS0:0:0, local62 plus focused4/4 and all23 PR checks passed. Merge77b1656c is LIVE on Render dep-dbh3lpss728c73bvk4d0; asset/auth/log smoke passed. No clinical Save or patient-data rewrite; clinician only refreshes the page. [Receipt](cockpit/releases/COCKPIT_COMPLETE_CLINICAL_SUMMARY_RELEASE_2026-10-28.md).


## 2026-10-28 — Complete prepared-note import and retained AI capture released

PR #254 adds default direct prepared-note import without a second AI call, arbitrary full headings, verbatim content/copy/export, automatic chart draft and one final Save. Recording/text AI remains the explicit secondary path, retaining recovery. The complete ChatGPT prompt is copyable in app. Only reserved explicit unchecked obligations create tasks; indented AI headings are safely excluded after independent correction closure.

Independent pre-code and final exact-code R2 PASS0:0:0; local79 affected, focused7 correction and one CP2 visibility check PASS. Final17 PR checks PASS. Merge a0c9ad8f024009df7dfca9a53979b9ecf10be52a is LIVE on Render dep-dbh5adcs728c739smeog at20:06:10Z. Three asset hashes, six anonymous401 routes, fresh production browser and empty ERROR/5xx logs passed. No provider/production clinical write. [Receipt](cockpit/releases/COCKPIT_DIRECT_PREPARED_NOTE_RELEASE_2026-10-28.md).


## 2026-10-29 — Purpose-focused clinical capture released after actual synthetic proof

PRs #256/#257/#258 merged compact/default clinical overview with optional complete source, completed-task history, labelled source/purpose/orthopaedic-role planning, neutral reasonless-audio recovery and isolated incompatible optional hints. Revised ChatGPT/provider prompts preserve completed communication versus real new unfinished obligations and request concise complete questions. One final Save and both deterministic no-AI prepared-note import/secondary AI recording remain. No model/config/schema change.

Final2f874f694d2d2c5372d8069d5035021c20648f20 LIVE on Render dep-dbhqpf29lscs73a6l4a0 at20:31:54Z. All final10 PR checks/141 capture tests and independent closures0:0:0 PASS. Same invented TTS proof40949970481 PASS14/14/94.52s after human output inspection; source facts and conditional therapy retained, DEXA/endocrine follow-up preserved, no disc/renal priority or new task for completed advice/referral. Three assets exact, six anonymous routes401, desktop/mobile browser clean and ERROR/5xx/audio failure logs empty. No real patient replay or production clinical Save. Full receipt: `cockpit/releases/COCKPIT_PURPOSE_FOCUSED_CAPTURE_RELEASE_2026-10-29.md`. Technical release complete; clinician acceptance/real pilot remains separate.
