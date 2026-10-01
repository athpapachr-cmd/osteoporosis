# R3 — Point-of-Care Clinical Interaction Review

> **Workstream:** OST-UI · **Mode:** fresh separate, documentation-only review · **Date:** 2026-10-01 Asia/Nicosia  
> **Decision question:** When a clinician must decide now, what interaction exposes decision-changing longitudinal information while separating fact, derivation, guidance, uncertainty, decision and future obligation?  
> **Disposition:** **R3 COMPLETE / R4 ELIGIBLE FOR COORDINATOR RECONCILIATION**

## A. Exact source identity

### Canonical Bootstrap Manifest

| Required item | Fresh R3 identity and boundary |
|---|---|
| Repository / live `main` | `athpapachr-cmd/osteoporosis`; remote `refs/heads/main` freshly verified as `63e903e05c1bfe22ca925374b8994355f6c92baf` on 2026-10-01. The contextual SHA in the governing prompt was verified, not assumed. |
| Six active root canonicals | Read fully, in `AGENTS.md` order: `AGENTS.md` → `TODO.md` → `CLINICAL_EXCELLENCE_PLAN.md` → `SLICE_PLAN_CURRENT.md` → `CURRENT_OPERATIONAL.md` → `osteoporosis-change-log.md`. They do not differ between the reviewed local R2 ancestry and its reconciled head. |
| Major phase / active root slice | Module-01 closure: dynamic guided consultation and transcript-assisted capture, then Practice Review and measurement/improvement. `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16` is the active, design-verified, implementation-authorized bounded root slice. |
| Writer / permitted mutation | Root `CURRENT_OPERATIONAL.md` retains the PR-1 writer lock. OST-UI is a parallel review sidecar. R3 changes only this review and `programme/OST-UI/CURRENT.md` on its separate R3 branch; no overlapping runtime or root-canonical mutation. |
| R2 coordinator input | Remote `refs/heads/docs/ost-ui-r2-longitudinal-trajectory-review-2026-10-01` freshly verified at `c706976820fd0a4eaa45210ca76eae95f661295c`. The R3 branch was created directly from this exact reconciled R2 head. |
| Relevant release state | Current-main S1 fracture/fragility semantics are merged. R1 and R2 are coordinator accepted. OST-UI remains review-only; no R3 deploy/smoke applies. R4 and synthesis are not authorized by R3 completion. |
| Permanent invariants | Protected patient history; completed/amended history distinct from current draft; scheduled/planned administration distinct from actual; unknown distinct from no; generic fracture distinct from confirmed fragility; current guidance distinct from final clinician decision; public repository contains no identifiable patient data. |
| Exact action / stop | Review eight point-of-care journeys, write this artifact and local OST-UI checkpoint, return exact identity and disposition to the coordinator, then stop. No UI, schema, clinical-rule, PR #121, PR-1/H-12, R4, synthesis, merge or deploy action. |

**Governing inputs consumed fully at `c706976…`:** `programme/OST-UI/PROJECT-INDEX.md`, `CURRENT.md`, `R1-CURRENT-PRODUCT-CONSTITUTION-AUDIT.md`, and `R2-LONGITUDINAL-CLINICAL-TRAJECTORY-REVIEW.md`. R1 establishes the hybrid current product; R2 establishes the reliable, partial, ambiguous and missing longitudinal meanings. Neither review is repeated here.

**Current interaction evidence:** source at live `main` `63e903e…`, especially `static/baseline-audit/index.html`, `progressive-guidance-core.js`, `progressive-guidance-ui.js`, `osteoporosis-evidence-guidance-core.js`, `osteoporosis-longitudinal-summary-core.js`, `step3.js`, `step4.js`, `patient-registry.js`, `g4-workspace-ergonomics.js`, and `longitudinal.js`. The R3 branch is based on R2 review ancestry and does not itself update runtime; where its four S1-related files differ from live `main`, this review used `main` for executable interaction/fragility evidence. Physio evidence was limited to `clinic_utilities/physio_referral_product/UX_CONTRACT_CURRENT.md` and current `static/clinic-utilities/physio-referral/{product-app,product-clinical-sheet-v4}.js` as interaction patterns. No patient data or live clinical encounter was inspected.

**Evidence labels:** **OBSERVED** = directly supported by current source or accepted R1/R2 source findings; **INFERRED** = point-of-care consequence of those observations; **UNVERIFIED** = requires use testing, clinical-owner decision or missing persisted semantics. Interaction recommendations below are requirements for evaluation, not an implementation design or a new clinical rule.

## B. Current point-of-care mental model

**OBSERVED:** the clinician opens Module 01's Baseline Audit / pilot-framed six-step encounter shell, chooses or opens a protected patient, reads the read-only `Σύνοψη ασθενούς`, then `Σημερινή ροή` with `Γιατί τώρα`, and edits facts in Steps 1–4 before decision/communication/documentation in later steps. G1/G2 can reorder and highlight cards and update guidance from live inputs; G3 summarizes selected completed/amended history; G4 makes the two summary surfaces collapsible/sticky. The top flow lists domain names and explanations, while the editable controls remain in step panels (`progressive-guidance-ui.js:393-444, 621-689`; `index.html:85-91`).

**INFERRED:** the clinician has to translate a highlighted *domain* into the correct Step, then mentally join the read-only patient snapshot, current draft, historical encounter rows, Step-3 trends and Step-4 decisions/tasks. The system already knows some of today's context, but does not yet offer a single, direct path from a decision-changing state to its source, correction and resulting decision. This is the current cognitive bottleneck. R1's encounter/audit-first diagnosis and R2's semantic limits explain why a full automatic life-course story cannot simply be drawn on top.

The justified interaction model is a **patient-and-problem entry point with a small decision-changing current-state readout**, followed by context-specific questions, editable source facts, a visibly recalculated derived state, linked guidance/evidence, explicit clinician decision and bounded future actions. It is a projection over present reliable truth, with visible limits. It does not assert an unrecorded epoch, recommendation, preference, goal or task disposition.

### Visible truth hierarchy

| Kind of statement | Point-of-care treatment | Current reliability |
|---|---|---|
| Patient data / fact | Show value, date and source; current entry is labelled current/provisional until authoritative completion. | **RELIABLE** when explicitly captured and sourced; otherwise **PARTIAL**. |
| Historical event | Identify actual event and source encounter; never turn a planned event into an actual event. | **RELIABLE** for captured exact actual administrations and consistent fracture facts; completeness **PARTIAL**. |
| Current derived state | Label as calculated from named inputs and current rules, with freshness and invalidating conflicts. | **RELIABLE** only within satisfied input/rule guards; otherwise **AMBIGUOUS**. |
| Missing / unknown information | Say “unknown / not recorded / history unavailable” and ask for a fact when decision-relevant. | **NOT CURRENTLY REPRESENTED** for some negative-vs-unasked distinctions. |
| Conflict / uncertainty | Present the competing facts, source and effect on interpretation; do not choose silently. | **RELIABLE** for G1/S1 detected conflicts; **PARTIAL** for cross-source DXA/lab/task conflicts. |
| Guideline position | Name framework, version and applicability separately from patient facts. | **RELIABLE** for reviewed active G2 scope. |
| System-derived guidance | Say what to review/verify and why now; do not visually pose it as a decision already made. | **RELIABLE** only under G2 activation guards; checklist-only safety remains a checklist. |
| Clinician recommendation | Separate from the system's guidance and from the final choice. | **NOT CURRENTLY REPRESENTED** as a distinct durable semantic fact. |
| Alternative option | Show only if actually documented/discussed; absence of content is not “no alternative.” | **NOT CURRENTLY REPRESENTED** losslessly; Step-5 process flag is **PARTIAL**. |
| Patient preference / constraint | Show actual content and whether it affected choice only if recorded; process flag is insufficient. | **PARTIAL** flags, content **NOT CURRENTLY REPRESENTED**. |
| Final decision | Label as clinician's recorded decision, with encounter/date and optional rationale. | **RELIABLE** when explicitly entered; decision-to-action link **PARTIAL**. |
| Contraindication / warning | Distinguish a known patient-specific contraindication from a general safety checklist or unresolved prerequisite. | **PARTIAL**; current G2 medication safety often `checklist_only`. |
| Hard stop | Reserve for an existing authoritative safety/validation rule; no new stop is inferred from missing data here. | **BLOCKED ON CLINICAL OWNER** for any new stop criterion. |
| Future obligation | State whether it is a recorded open prior task, a current derived due state, or an ambiguous possible duplicate; do not imply stable cross-visit identity. | **PARTIAL / AMBIGUOUS** across edits and rescheduling. |
| Evidence / provenance | Attach source, framework/version and rule identity to the specific guidance statement; expand for detail. | **RELIABLE** for present G2 rule provenance; historical as-at-decision evidence **NOT CURRENTLY REPRESENTED**. |

## C. Eight journey interaction reviews

The first-20-second tiers in §D are journey-specific. Here each review states the current path, cognitive burden, safe behavior and information class. “Next step” means the next *interaction or clinician review*, not a prescribed treatment.

### A. Patient currently receiving denosumab

**First question:** Is active denosumab asserted, what actual administration is reliably recorded most recently, and is any monitoring or prior obligation relevant to today's decision? **OBSERVED:** G3 can show active treatment, last actual administration and count; G1 carries exact actual events, separate next-due and unresolved tasks; G2 can derive an evidence due state when its prerequisites are satisfied. A scheduled row does not prove a dose. Current Step-4 episodes/administrations and Step-3 labs are still separate editing contexts.

**Cognitive path/burden:** summary → current flow → Step 4 for exact exposure and planned rows → Step 3 for monitoring → Step 4 for tasks/decision. The clinician holds treatment assertion, captured actual exposure, derived timing and safety verification in memory; summary and Step 4 repeat some data, while exact source and action path are not adjacent. Full six-step form completion is not inherently relevant to a routine administration. **Safe interaction:** show one compact treatment state with last *actual* date and source, separately labelled derived due status, only currently relevant monitoring and unresolved recorded tasks, and an explicit “actual date unknown/history incomplete” state. Expose episode/admin detail on request. **Class:** captured actual **RELIABLE**; treatment course completeness and tasks **PARTIAL**; inferred complete course **NOT CURRENTLY REPRESENTED**.

### B. Delayed/missed denosumab dose

**First question:** What is the last actual dose, what dose was planned, what was recorded missed/delayed, and what timing state is derived now? **OBSERVED:** Step-4 rows contain scheduled/actual/status/next due; G1 actual-event projection excludes scheduled-only missed rows; G2 timing depends on reliable actual history. The G3 summary favors last actual and the current-flow item can signal timing, but the four concepts are not displayed as a single causal view. R2's synthetic probe showed a missed scheduled row did not increase actual count.

**Cognitive path/burden:** the clinician moves from G3 last actual → G1/G2 why-now text → old Step-4 rows/encounter history to locate the missed plan → current Step 4 to act. The missed plan and later actual action are not linked; a chronology assembled visually would be **AMBIGUOUS**, even if dates are present. **Safe interaction:** align four clearly titled facts/states in view: last recorded actual dose, recorded planned date, recorded missed/delayed status, and today's derived timing (with rule/input provenance). If the actual date is absent, do not show a confident timing state. Ask the clinician to verify source records; do not mark a missed plan resolved because a later actual row exists. **Class:** exact captured dates **RELIABLE**, plan-resolution linkage **NOT CURRENTLY REPRESENTED**, timing **RELIABLE only when guarded**.

### C. Denosumab discontinuation planning

**First question:** Is cessation a current intent or a past final decision, what exposure is actually documented, what prompted the change, and what facts/prerequisites remain unknown? **OBSERVED:** Step-4 decision/transition fields and episodes can record stop/plan and future agent/date; G2 has reviewed exit guidance without automatically selecting treatment; Step-3 investigations and tasks are elsewhere. R2 found no durable chain across reason → options → recommendation → preference → final decision → actual successor dose → monitoring/outcome; R15/R16 remain inactive for missing linkage/availability semantics.

**Cognitive path/burden:** the clinician reads summary last decision and treatment, then opens Step 4 transition/administrations, Step 3 labs/BTMs, perhaps prior encounters, and returns to Step 4 close. The distinction between intended next agent/date and actual action is easy to lose. **Safe interaction:** keep current/previous asserted treatment and captured last actual exposure visible; show current discontinuation intent, reason if recorded, relevant investigations with dates, unresolved safety verification, recorded transition plan and its *planned* status, and current G2 source-backed guidance. Ask for missing information. Do not infer a successor strategy, execution, or monitoring closure. **Class:** explicit current fields **RELIABLE** as assertions; full transition continuity **PARTIAL/AMBIGUOUS**; recommendation/preference chain **NOT CURRENTLY REPRESENTED**.

### D. New fragility fracture

**First question:** What fracture event is reported, when/site, what is known about trauma mechanism, and is fragility confirmed under current S1 semantics? **OBSERVED:** current `main` S1 requires a structured current event with explicit `low_trauma=yes` for confirmed current fragility in G2; G3 distinguishes generic events, confirmed fragility and legacy unconfirmed claims, and fails closed on conflicting explicit mechanisms. The coarse visit archetype or legacy Step-1 checkbox does not by itself confirm mechanism.

**Cognitive path/burden:** the clinician may see a “fracture” or post-fracture visit intent in the top flow but must inspect Step-2 event details for mechanism, Step-4 exposure/decision and Step-3 results. A generic label risks anchoring the clinician to fragility before mechanism is established. **Safe interaction:** show event fact and source first; place mechanism and confirmed/unknown/conflicting interpretation immediately beside it; show treatment context separately and only decision-changing investigations/risks. Unknown remains unknown. Editing mechanism must update the derived fragility interpretation and guidance; the fact that a fracture happened remains visible. **Class:** structured event **RELIABLE** if captured; mechanism **RELIABLE** only if explicit/consistent, otherwise **AMBIGUOUS**; no new fragility rule is defined.

### E. Fracture while treated

**First question:** What event occurred, was actual exposure at its time established, and what interpretation is still open? **OBSERVED:** event `occurred_on_treatment` and Step-4 treatment/administration history can be recorded, while G2 event guidance activates; R2 found no stable event-to-epoch linkage. Step-3 DXA/labs, Step-4 adherence/tolerance/response and the event may live in different panels.

**Cognitive path/burden:** history and current event, exact actual exposure, adherence, DXA/labs and current decision are at least four conceptual contexts. Repeated summary and card text do not establish whether the treatment was taken at the fracture date. **Safe interaction:** present event + trauma mechanism, captured actual dates, asserted active episode, adherence/tolerance/response *as recorded*, relevant comparable investigations and explicit uncertainty before any assessment. “Treatment failure” can appear only as a clinician-recorded interpretation or a question for reassessment, never an automatic label. **Class:** event and captured administration **RELIABLE**, exposure-at-event **PARTIAL**, failure interpretation **BLOCKED ON CLINICAL OWNER** for any new automatic rule.

### F. Monitoring / possible treatment failure

**First question:** Which *comparable* change or new event is prompting reassessment, and what exposure/measurement context qualifies it? **OBSERVED:** Step-3 DXA comparison/LSC, manual DXA history, lab/BTM context and charts exist; G3 latest DXA and lab snapshot are concise but can omit unique manual DXA or reflect unreconciled lab rows; Step-4 captures exposure, response/adherence and decision. R2 found DXA/FRAX overlap and lab dual representation.

**Cognitive path/burden:** summary → Step-3 trends and detailed quality/LSC → lab table → Step-4 exposure/adherence/last decision → fracture history. The clinician must remember comparability and source while interpreting a simple-looking trend. **Safe interaction:** foreground only the change that triggered review, comparability/LSC and source/date, new fracture status, captured actual exposure, adherence/tolerance and prior final decision. Put the full chart and measurements behind expansion. A descriptive numerical change is not labelled clinically significant when comparability/LSC is unknown. Conflicting scan/lab provenance must qualify or suppress the apparent trend. **Class:** captured values **RELIABLE**, trend interpretation **PARTIAL/AMBIGUOUS** pending consolidation; automatic treatment-failure judgment **BLOCKED ON CLINICAL OWNER**.

### G. CKD constraint

**First question:** What current renal fact or dated trend is available, and which safety verification is relevant to today's contemplated action? **OBSERVED:** Step-1 secondary condition can mark CKD; Step-3 contains dated creatinine/eGFR and protected lab snapshots; G2 can show medication-specific safety content labelled checklist-only. R2 found no complete constraint/interpretation trajectory and possible lab lifecycle/provenance duplication.

**Cognitive path/burden:** a condition checkbox, labs, treatment choice and safety checklist are dispersed across Steps 1, 3 and 4. The clinician must join date/source to the action and remember whether data are current enough; the UI does not prove a current safety clearance. **Safe interaction:** surface the dated renal facts and known CKD assertion beside the currently relevant safety *checklist*, with missing/stale/unavailable and provenance states explicit. Do not derive a renal threshold or contraindication from this review. Detail and longer trend open on demand. **Class:** dated sourced lab **RELIABLE if reconciled**; condition trajectory and safety clearance **PARTIAL/NOT CURRENTLY REPRESENTED**; any new threshold/stop **BLOCKED ON CLINICAL OWNER**.

### H. Glucocorticoid-associated osteoporosis

**First question:** Is current systemic exposure explicitly recorded, with what dose/duration assertion, and how does it alter today's existing guidance? **OBSERVED:** Step-1 stores current GC checkbox, dose and duration; G2 can use these current values; historical encounters provide snapshots. R2 found no precise start/stop/dose-change interval and checkbox defaults can blur unasked versus negative.

**Cognitive path/burden:** the clinician navigates Step 1 exposure, Step 2 fracture/risk, Step 3 results and Step 4 decision; multiple historical snapshots must be compared manually. Showing a precise cumulative exposure timeline would falsely reduce uncertainty. **Safe interaction:** expose current explicitly entered dose/duration and date of source, note that older snapshots do not establish intervening intervals, and show only presently applicable reviewed guidance. Ask for current exposure when unknown. Historical details expand when useful. **Class:** current explicit assertion **RELIABLE as recorded**; longitudinal exposure course **PARTIAL/AMBIGUOUS**; exact intervals **NOT CURRENTLY REPRESENTED**.

Across all eight, the repeated burden is **navigation between summary, flow, historical encounters and fixed Steps**, not lack of any guidance engine. Form fields unrelated to the current decision should not dominate the initial view; safety/event and unresolved decision-changing information must remain visible. These are qualitative source-based findings, not measured usability scores or pilot claims.

## D. First-20-seconds information model

The first 10–20 seconds should establish the patient's present problem, reliable current state, one or two decision-changing facts, uncertainty and the next *review/action*. The table classifies **candidate content by relevance**, not by whether the current UI can already render it. “Hide unless asked” never applies to a material active warning.

| Journey | MUST SEE IMMEDIATELY | SHOW WHEN RELEVANT | EXPAND ON DEMAND | HIDE UNLESS ASKED |
|---|---|---|---|---|
| A. Receiving denosumab | Active assertion; last captured actual date; guarded due state or unknown; blocking history conflict | Relevant dated monitoring; recorded open prior task | Full administrations, episode reasons, lab history, rule rationale | Unrelated risk/form domains and old routine notes |
| B. Delayed/missed dose | Last actual, planned date, missed status, current derived timing, missing actual/conflict | Relevant recorded follow-up and monitoring | Source rows and date/rule trace | Full treatment inventory and unrelated audit fields |
| C. Discontinuation | Intent versus recorded final stop; actual last exposure; unresolved prerequisite/plan status | Dated investigations, recorded reason, current reviewed guidance | Full prior episodes, alternative discussion if actually recorded, evidence | General routine follow-up cards |
| D. New fracture | Event site/time; mechanism known/unknown/conflicting; confirmed fragility status; treatment context | Decision-changing imaging/risk and new-event guidance | Event source, older fractures, evidence | Routine stable-follow-up content |
| E. Fracture on treatment | Event and mechanism; asserted treatment; captured actual exposure; interpretation unresolved | Adherence/tolerance and relevant DXA/labs | Full administrations, outcome chronology and source rows | Automatic failure/switch label; unrelated work-up |
| F. Monitoring/failure concern | Triggering result/event; comparability/LSC qualification; actual exposure completeness | Dated labs/BTMs, adherence, previous final decision | Full trends, source reports, rule rationale | Noncomparable trend as a confident headline; routine form completion |
| G. CKD | Dated renal fact/absence; known CKD assertion; current safety checklist status | Trend or medication-specific context when a treatment action is contemplated | Full lab history/provenance, evidence | Unrelated measurements and generic safety lists |
| H. Glucocorticoids | Explicit current exposure and dose/duration or unknown; current decision-relevant context | Relevant fracture/risk and reviewed guidance | Prior encounter snapshots and evidence | Invented continuous exposure interval and unrelated cards |

**Availability rule:** an item enters “must see” only if it changes today's interpretation/action or its absence could mislead that decision. A long timeline is not a default requirement. A conflict, unavailable history or missing actual dose can raise a normally collapsed item into immediate view.

## E. “What Matters Today” assessment

**What works — OBSERVED:** `Σημερινή ροή` is already dynamic by visit intent, prior tasks, treatment/due and events; G2 adds rule-backed reasons, G3 marks newly surfaced guidance, and `Γιατί τώρα` is visible both above and inside surfaced cards. The history-unavailable message explicitly warns against treating a failed load as no history. This is a strong existing relevance and trust mechanism (`progressive-guidance-ui.js:271-298, 322-429, 621-689`).

**What is missing — OBSERVED/INFERRED:** the top list names up to 12 domains and explains salience, but it is rendered as noninteractive boxes; it does not itself show an issue's fact/derived-state/uncertainty/next-action chain or take the clinician to the source field. The clinician still has to remember which Step holds the issue. G3's “latest” summary and G1/G2 why-now operate beside, not as a unified point-of-care decision path. Some top text can look conclusive when the underlying provenance is partial (R2 §§J–L).

**R3 judgment:** make the *patient/problem-specific current-state and action path* the **primary entry point for clinical work**, using today's flow as its relevance engine and explicit rationale. Keep the existing step shell accessible for detailed entry and audit/documentation. This is an interaction requirement, not a proposed screen or implementation. It should be tested in real use; R3 does not claim the new entry point has already reduced time or error.

## F. Longitudinal orientation requirements

Minimum orientation is a **selective present-state strip**, not an always-open full timeline: current asserted treatment, last relevant actual action, the event/result that changed today's problem, last recorded final decision, unresolved recorded obligations, and a visible “history available / incomplete / conflicting” marker. Each item needs date/source and a route to detail. Major transitions appear when they explain today's state; otherwise they stay collapsed. “Changed since last completed encounter” is useful only where source comparison is reliable; it is not proof of a continuous epoch or that an unrecorded event did not occur.

The current G3 summary already contains many of these parts, but its latest-only treatment/DXA/lab/decision views can hide intermediate uncertainty and unique manual history (R2 §§B, K). A patient-level orientation may therefore describe a *bounded recorded history*, never a certified complete 10–15-year story. Historical evidence and outcome associations remain explicitly unavailable where not captured.

## G. Progressive disclosure findings

**Supported hypothesis:** a restrained default surface is justified by the eight journeys: most need a handful of facts and one current problem, while full Step-3/4 detail varies widely. G4 proves summaries can collapse, and Step-3/4 already reveal dependent details. Physio shows a current working pattern of compact defaults, contextual qualifiers and evidence sheets. This supports progressive disclosure as an **interaction principle**, subject to clinical safety priority.

Keep collapsed by default: full historical encounter list, complete treatment rows, whole DXA/FRAX/lab charts, general secondary-cause inventory, audit/capture metrics, and detailed evidence text. Open them when the clinician asks, a current issue makes them relevant, a conflict/warning requires source inspection, or rationale is requested. Keep visible: the patient/problem identity, decision-changing fact and date, current derived-state qualification, missing input, unresolved active warning and next review/action. Collapse preference must never hide a newly material safety/event issue without an evident cue.

## H. Dynamic interaction / editability findings

**Current capability — OBSERVED:** G1/G2 read live Step-1–4 controls and schedule re-evaluation on input/change; changing encounter intent, fracture fields, episode/admin date, treatment decision or renal/GC context can alter card surfacing. Step-3/4 fields are editable in their panels. S1 corrects fracture mechanism semantics on current `main`; the re-evaluation must preserve unknown/conflict rather than let a visit label prove fragility. G4 collapse is presentation-only.

**Current friction — INFERRED:** the clinician can edit a source field, but must navigate from the summary/why-now to its Step. A treatment state, last actual date, fracture mechanism, lab/DXA datum or task may require separate panels/history. The current top flow is not an edit/navigation control. A changed task date can create a second projected obligation rather than a safe update. An unreconciled lab/DXA edit can leave another historical representation apparently current. These last two are semantic/provenance dependencies, not mere UI latency.

**Required behavior:** an issue should lead directly to its authoritative editable input or a clearly labelled source record; upstream correction should immediately withdraw stale downstream *derived* guidance and show recalculation/pending/uncertainty, without erasing the underlying recorded fact. If the input source is historical or dual-owned, correction must be explicit and provenance-aware rather than silently overwriting. For treatment state, fracture, renal context, actual date and discontinuation intent, dynamically reveal only relevant questions and current reviewed guidance; never create a new clinical branch or rule in presentation code. For tasks, show ambiguity until obligation continuity is semantically resolved.

## I. Missing, uncertain and conflicting information interaction

| Situation | Safe visible interaction | Priority / current-state limit |
|---|---|---|
| Protected history unavailable | Prominent “history unavailable; previous encounters/obligations unknown”, retry/source-status affordance; suppress confident history-dependent conclusions. | **Interrupt reliance on longitudinal guidance**, not all data entry. **RELIABLE** existing fail-closed behavior. |
| Low-trauma mechanism unknown | Fracture event stays visible; fragility interpretation says unconfirmed/unknown; ask for mechanism if decision-changing. | Warning/query at fracture decision. **RELIABLE** S1 semantics. |
| Actual administration date missing | Keep scheduled/missed record separate; timing shown as indeterminate when a rule requires actual date; request verification. | Interrupt any action that would rely on a confident timing state. **PARTIAL** capture. |
| DXA provenance/comparability ambiguous | Display source/date and disagreement or unknown LSC; withhold “significant change” label; open competing scans. | Warning by monitoring decision, expandable source detail. **AMBIGUOUS** dual representation. |
| Renal data missing/stale | Show absent/date-unknown result and checklist-only safety status; ask to verify before relying on safety interpretation. | Contextual warning; exact hard stop needs clinical owner. **PARTIAL**. |
| Recommendation content never recorded | Say “recommendation not recorded”; do not reconstruct it from final decision or Step-5 discussion flag. | Expandable decision-history limitation unless current choice depends on it. **NOT CURRENTLY REPRESENTED**. |
| Task continuity ambiguous | Show recorded open items and “possible duplicate/rescheduled; continuity unverified”; require review before claiming completion. | Warning at obligation/close action, not a false duplicate suppression. **AMBIGUOUS**. |

**Conflict triage:** a conflict that invalidates a computed due state, treatment state or confirmed fragility interpretation must interrupt *reliance on that conclusion* and leave the recorded facts inspectable. A stale lab or duplicate DXA should warn beside the affected decision/result; source detail can expand on demand. An ambiguous old obligation should warn in the obligations/close context, not globally block the whole encounter. A new hard-stop severity rule is outside R3 and requires the clinical owner. Existing G1 treatment/administration conflict records and S1 fracture-mechanism conflict behavior are useful; cross-owner DXA/lab and obligation conflicts are not comprehensively detected today.

In every case: **UNKNOWN ≠ NO** and **NOT RECORDED ≠ DID NOT HAPPEN**. Blank checkboxes for GC/CKD and historical absence are not reliable negative evidence where R2 found that distinction unrepresented.

## J. Guidance, evidence and decision separation

The safe visible sequence is **patient fact → derived current context → applicable guideline position → system guidance/verification → clinician recommendation → patient preference/constraint → final decision**. The first four can be supported today within G1/G2 guards; the last three cannot be retrospectively filled from a final decision. Alternative options remain distinct. A contraindication is a patient-specific clinical assertion, a warning may flag uncertainty, and a hard stop requires an existing authoritative rule; none is equivalent to a generic checklist.

**OBSERVED:** G2 attaches `rule_id`, rule class, source refs, strength and activation mode to current contributions; the UI shows compact source labels and explicitly says `Safety checklist — απαιτεί κλινική επιβεβαίωση, όχι automatic clearance` (`progressive-guidance-ui.js:322-352`). **Cognitive risk:** evidence labels and “why now” are adjacent to editable decision cards, so the clinician could read a rule-backed prompt as the system's treatment choice or treat checklist display as safety clearance. A derived due date could be mistaken for an administered or booked action. The distinction must be in wording and placement, not color alone.

Evidence should have a compact source/framework/version cue linked to the specific guidance statement, with rule trigger/limitations and fuller rationale available on request. If frameworks differ, show the distinct source positions rather than a blended “consensus.” Current G2 provenance is suitable for *today's* guidance; it does not establish what evidence was shown at a historical decision. R3 does not change evidence governance or guideline content.

## K. Future-obligation interaction

Today the UI can safely show four different statuses, each explicitly labelled:

1. **Recorded open task from a prior completed/amended encounter:** source date, task type, due date/timeframe, and its stored status; it is not proof that the action did not occur elsewhere.
2. **Derived treatment due state:** rule/input-based current projection, distinct from an appointment, planned dose and actual administration.
3. **Ambiguous / possibly duplicated task:** display source rows and uncertainty, particularly after a due-date/timeframe edit. R2's executable probe produced two planned tasks from one reused UUID when the semantic tuple changed.
4. **Recorded completed or not applicable:** show only when the exact persisted disposition supports it; do not infer from a later blank close or nearby action.

The present `type|due_date|timeframe_text` cross-encounter key and narrow `planned/already_done/not_applicable` statuses do not support a stable “rescheduled/deferred/declined/superseded/resolved by alternative” story. A longitudinal task tracker that asserts such a story is **blocked on missing semantics**. The point-of-care surface should still expose current recorded open items and derived due states with their limits.

## L. Decision / preference capture requirements

At a decision visit, the clinician needs distinct places in the interaction to establish **options considered**, **clinician recommendation**, **patient preference or constraint**, and **final decision**, plus which actual action later occurred. The current Step-4 final decision/selected agent/rationale is useful, and Step-5 can record that alternatives or preferences were discussed, but R2 showed the option/recommendation/preference *content* is not separately durable. Showing a process checkmark as the content would be misleading. A proposed PR-1 transcript candidate is non-authoritative and does not repair history.

R3 therefore identifies separate capture as an **interaction requirement blocked on persisted semantic ownership**, not a schema prescription. A clinician should be able to say “unknown/not recorded” without inventing an option or preference. The same applies to an explicit **“What are we trying to achieve?”** goal/target-state concept: it would help interpret future monitoring and decisions, but there is no general durable owner and no target definition approved here. This remains an interaction requirement pending clinical semantic owner; no threshold or target is set.

## M. Six-step-shell disposition

**SECONDARY EDITING STRUCTURE.** This is the R3 interaction conclusion across all eight journeys. The Steps organize detailed current data capture, documentation and existing audit fields, and remain useful for editing and close. They should not be the primary starting sequence when the clinician's immediate problem is delayed treatment, fracture, transition, monitoring concern, CKD or GC exposure. The patient/problem and decision-changing state should orient work first; the relevant Step can then serve as the detail editor. This does not decide whether the shell is retained, moved or retired in implementation, which belongs downstream.

Problem-first is supported **within the bounded reliable substrate** because these journeys start from different triggers and share no single sensible Step 1→6 order. It is not yet user-tested against real clinicians, and it does not license a false complete trajectory. Routine stable follow-up may need only a minimal state and close; a new fracture may need multiple expanded domains. The appropriate depth is journey-dependent even though the primary entry principle is common.

## N. Physio pattern-transfer assessment

Current Physio evidence is a Knee-OA referral contract and a current synthetic-only prototype runtime, not the Osteoporosis product or a medical-rule authority. It demonstrates interaction mechanics: a compact starting state, contextual clinical sheets, immediate server-owned projection, stale-output gating, and evidence sheets (`UX_CONTRACT_CURRENT.md` §§2, 5–9; `product-app.js:67-87, 182-200, 205-228`; `product-clinical-sheet-v4.js`). Its document explicitly says it does not implement the contract by itself; runtime evidence is limited to the inspected prototype.

| Pattern | R3 transfer judgment | Osteoporosis adaptation / boundary |
|---|---|---|
| Progressive disclosure | **TRANSFERABLE WITH ADAPTATION** | Keep a small current-state default, but safety/event/unknown information must override collapse; no single-diagnosis default plan. |
| Contextual dependent options | **TRANSFERABLE WITH ADAPTATION** | Upstream patient facts should reveal relevant questions and existing reviewed G2 guidance, while missing semantics stay missing and clinical branches remain owned by their rules. |
| Live output | **TRANSFERABLE WITH ADAPTATION** | Recalculate *derived state and guidance* on edits, visibly invalidate stale output; do not auto-generate a final treatment decision/referral analogue. |
| Direct manipulation | **TRANSFERABLE WITH ADAPTATION** | Source facts need direct correction routes and provenance/authority guardrails for historical or dual-owned records. A click must not silently rewrite patient history. |
| Restrained default surface | **TRANSFERABLE WITH ADAPTATION** | Eight journeys support selective facts; conflicting, safety-critical or unavailable state remains prominent. |
| Evidence on demand | **TRANSFERABLE** | Keep concise provenance next to each applicable guidance statement and expand rationale/source detail; retain G2 framework/activation distinctions. |

Physio's preselected rehabilitation plan, editable generated referral and single-diagnosis task model are **NOT APPROPRIATE FOR OSTEOPOROSIS** as a clinical product model. R3 imports no Physio treatment content or semantics.

## O. Evidence-supported interaction principles

1. **Patient and problem first:** orient to the active patient's present issue before asking for form completion (R1 shell diagnosis; eight distinct journeys).
2. **Decision-changing state first:** show only facts/derivations that can alter today's interpretation or next review; expand a full timeline only when it matters (journeys A–H; G3 asset).
3. **Truth labels and provenance:** distinguish fact, event, current derivation, guidance, clinician decision and future obligation; include dates and source where available (R2 truth map; G2/G3).
4. **Visible uncertainty:** history unavailable, unknown mechanism/date, conflict and ambiguous task continuity stay explicit, and invalidate conclusions they cannot support (G1/S1; R2 gaps).
5. **Contextual progressive disclosure:** reveal details/questions according to current issue, but let safety/event signals and source conflicts override a collapsed preference (G1/G4; Physio pattern evidence).
6. **Editable upstream facts with live downstream consistency:** connect a surfaced issue to its source and retract stale derived guidance after correction (current live G1/G2; Physio stale-output pattern).
7. **Evidence on demand, linked to guidance:** concise current source cue, detailed rationale on request, and checklist-only safety never mistaken for clearance (G2 UI; Physio pattern evidence).
8. **Future obligations with honest continuity:** expose recorded open tasks and derived due separately; label duplicate/changed-date uncertainty (G1; R2 task probe).
9. **Clinician owns recommendation and decision:** the system surfaces considerations, while options, recommendation, preference and final choice remain distinct; missing content remains absent (R2 §F; AGENTS.md §§6, 12).

## P. Current-state versus semantic-dependency matrix

The reliability column qualifies the *information behind* the proposed interaction. A current-state interaction may still require UI work; this matrix says whether today's data can honestly support its meaning. “After consolidating” means existing facts need provenance/reconciliation, not that R3 chooses the owner or mechanism.

| Desired interaction | Underlying information class | Dependency classification | Limit that must remain visible |
|---|---|---|---|
| Patient header with protected history availability and current draft distinction | **RELIABLE** | **CAN DO WITH CURRENT RELIABLE STATE** | Loading/unavailable is not no history. |
| Current asserted active treatment + captured last actual administration | **RELIABLE** captured facts; course **PARTIAL** | **CAN DO WITH CURRENT RELIABLE STATE** | “Recorded” does not prove complete exposure. |
| Current rule-derived due state next to actual/planned dates | **RELIABLE** only when guarded | **CAN DO WITH CURRENT RELIABLE STATE** | Due is derived, not a booked or administered event. |
| Recorded missed plan beside last actual dose | Plan/event **RELIABLE**, link **AMBIGUOUS** | **CAN DO WITH CURRENT RELIABLE STATE** for separate rows | Do not say a later dose resolved that plan. |
| Resolved missed-plan → later-action chain | **NOT CURRENTLY REPRESENTED** | **BLOCKED ON MISSING SEMANTICS** | Temporal proximity is not linkage. |
| Fracture fact, known/unknown mechanism and confirmed S1 fragility state | **RELIABLE** if explicit/consistent | **CAN DO WITH CURRENT RELIABLE STATE** | Archetype/legacy claim alone does not confirm fragility. |
| Fracture-at-treatment exposure/epoch interpretation | **PARTIAL / AMBIGUOUS** | **BLOCKED ON MISSING SEMANTICS** for confident link | `occurred_on_treatment` is an assertion, not full exposure proof. |
| Automatic treatment-failure or switch decision | **NOT CURRENTLY REPRESENTED** as an approved rule | **BLOCKED ON CLINICAL OWNER** | Current G2 intentionally avoids automatic failure/switch. |
| Latest dated dedicated lab facts and checklist-only safety | **RELIABLE** if source lifecycle sound; safety clearance **PARTIAL** | **CAN DO WITH CURRENT RELIABLE STATE** for labelled result/checklist | No automated clearance. |
| Unified lab snapshot after date edit/clear or source disagreement | **AMBIGUOUS** | **CAN DO AFTER CONSOLIDATING EXISTING STATE** | Show stale/duplicate possibility until reconciled. |
| DXA/FRAX trend with manual/encounter source reconciliation | **PARTIAL / AMBIGUOUS** | **CAN DO AFTER CONSOLIDATING EXISTING STATE** | Preserve unique manual facts; no significance claim without comparability/LSC. |
| CKD-specific threshold, contraindication or new hard stop | **NOT CURRENTLY REPRESENTED** as R3 authority | **BLOCKED ON CLINICAL OWNER** | Present facts/checklist, not an invented cutoff. |
| GC current assertion and prior snapshots | **RELIABLE** current entry; historical course **PARTIAL** | **CAN DO WITH CURRENT RELIABLE STATE** | Snapshots do not establish intervals. |
| Precise GC exposure interval/cumulative course | **NOT CURRENTLY REPRESENTED** | **BLOCKED ON MISSING SEMANTICS** | Do not draw a continuous course from visit dates. |
| Current G2 why-now with linked evidence and checklist label | **RELIABLE** within activation scope | **CAN DO WITH CURRENT RELIABLE STATE** | Guidance is not a clinician recommendation or final choice. |
| Historical evidence/guidance as seen at the old decision | **NOT CURRENTLY REPRESENTED** | **BLOCKED ON MISSING SEMANTICS** | Today's rule rerun is not the evidence clock “then.” |
| Last explicit final decision with encounter source | **RELIABLE** if entered | **CAN DO WITH CURRENT RELIABLE STATE** | Later actual action remains separately proved. |
| Options, recommendation and preference content as separate durable facts | **NOT CURRENTLY REPRESENTED** losslessly | **BLOCKED ON MISSING SEMANTICS** | Step-5 checkmarks and optional prose cannot certify content. |
| Explicit clinical goal/target state | **NOT CURRENTLY REPRESENTED** | **BLOCKED ON MISSING SEMANTICS** and **BLOCKED ON CLINICAL OWNER** for target meaning | No target threshold defined here. |
| Recorded prior open task and explicit derived treatment due | **PARTIAL** continuity, reliable individual rows | **CAN DO WITH CURRENT RELIABLE STATE** | Label record-derived versus rule-derived; no stable identity claim. |
| Rescheduled/deferred/superseded/declined obligation history | **NOT CURRENTLY REPRESENTED** | **BLOCKED ON MISSING SEMANTICS** | UUID and semantic tuple do not prove continuity. |
| Selective patient/problem entry and direct path to existing edit fields | **RELIABLE** for sourceable current facts | **CAN DO WITH CURRENT RELIABLE STATE** | Interaction improvement only; historical edits still need source authority. |

## Q. Direct R3 conclusion

**Justified now:** a patient-and-current-problem point-of-care interaction that starts with a restrained, source-labelled orientation; separates recorded facts from current derived state; shows only decision-changing history, missing information and obligations; uses G1/G2 `Γιατί τώρα` and current evidence provenance; lets the clinician reach and correct relevant source facts; recalculates or withdraws derived guidance live; and keeps the clinician's decision distinct from the system's prompts. The existing protected history, actual administration projection, S1 fragility semantics, G2 rules, G3 summary and G4 workspace mechanics support this bounded model. The six-step shell is a secondary detail editor.

**Still blocked:** a certified complete treatment timeline, missed-plan resolution, stable task lifecycle, option/recommendation/preference content, explicit durable goal, historical evidence-as-seen, precise GC exposure intervals, confident cross-source DXA/lab trend and automatic failure/safety conclusions. Those require consolidation, missing persisted meaning or clinical-owner decisions as specified in §P. R3 does not conclude a rebuild, implementation architecture or final product scope.

## R. Disposition

**R3 COMPLETE / R4 ELIGIBLE FOR COORDINATOR RECONCILIATION**

The R3 branch contains this review and the local `programme/OST-UI/CURRENT.md` checkpoint only. Programme coordinator verification/reconciliation of the exact branch/head/artifact is next. R4, synthesis, UI implementation, clinical-rule change, schema/database mutation, root-canonical mutation, PR #121 mutation, merge and deploy were not performed.
