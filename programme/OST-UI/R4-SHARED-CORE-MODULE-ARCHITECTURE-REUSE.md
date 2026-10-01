# R4 — Shared Core / Module Architecture / Reuse Review

> **Workstream:** OST-UI · **Date:** 2026-10-01 Asia/Nicosia · **Scope:** architecture disposition only
> **Disposition:** **R4 COMPLETE / SYNTHESIS READY**

## A. Exact source identity

| Source / control | Freshly verified identity and use |
|---|---|
| Repository and remote `main` | `athpapachr-cmd/osteoporosis`, `63e903e05c1bfe22ca925374b8994355f6c92baf` (`git ls-remote`, 2026-10-01). |
| Six root canonicals | `AGENTS.md` → `TODO.md` → `CLINICAL_EXCELLENCE_PLAN.md` → `SLICE_PLAN_CURRENT.md` → `CURRENT_OPERATIONAL.md` → `osteoporosis-change-log.md`, fully read at verified `main`. |
| Root phase / writer | Module-01 closure; PR-1 transcript candidate extraction is the active, design-verified implementation slice and sole root operational writer. OST-UI has documentation-only scope; no root lock or runtime authority. |
| OST-UI base | Remote R3 coordinator-reconciled head `5a2d99bf9180470fb9b7fd1f907d26f2066b3dbb`, verified exactly; R4 branch `docs/ost-ui-r4-architecture-reuse-review-2026-10-01` was created from it. `PROJECT-INDEX.md`, `CURRENT.md`, R1, R2 and R3 were read fully at this base. R1–R3 are accepted evidence. |
| Programme | Remote programme registry branch `docs/ost-programme-lifecourse-bootstrap-2026-09-27` verified at `9f505ffc0d5b1460cc2a54dd49bc07d28b138808`. Reviewed P1 current-ownership target `b4f917b161f33b0bfb3507328df07cec2d7bd2b6` is in that ancestry; P1 is ownership evidence, not P2 design. Q1–Q12 remain parked. |
| Executable evidence | Read-only owner checks against verified `main`: `clinical_data.py`, `clinical_data_ext.py`, `patient-registry.js`, G1/G2/G3 modules, `longitudinal.js`, Cockpit and module navigation, and bounded Physio interaction source/contract. R3 ancestry is not treated as newer runtime than `main`. |
| PR-1 / release boundary | The programme registry records H13 semantic promotion independently passed; PR-1 remains a separate lifecycle. No PR-1 release engineering, clinical-rule, schema, database, runtime, root-canonical or PR #121 mutation here. No R4 deploy/smoke applies. |

**Canonical Bootstrap Manifest:** the root phase, slice, writer and allowed scope are above. Safety/privacy invariants are protected patient facts; completed/amended history distinct from current draft; actual distinct from scheduled/planned and derived due; unknown distinct from negative; current evidence output distinct from historical evidence and clinician decision; no identifiable data in the public repository. The exact authorized action is this R4 artifact plus the OST-UI `CURRENT.md` checkpoint and remote branch publication. Synthesis, implementation, prototypes, merge and deploy are deferred.

**Evidence notation:** **OBSERVED** denotes current source or accepted R1–R3 findings; **DECISION** denotes this R4 architectural disposition; **OWNER DEPENDENCY** denotes clinical or Product Owner meaning still to be supplied. No recommendation below claims that target semantics are implemented.

## B. Architecture principles derived from R1–R3

1. Keep one protected patient and encounter truth; clinical domains interpret sourced facts, while UI, AI and projections have no independent patient-fact authority.
2. Extend an existing factual owner before adding a durable owner. Preserve unique manually entered history until it is reconciled, even when it overlaps another representation.
3. Keep patient fact, derived context, reviewed guidance output, clinician recommendation/decision and actual action distinguishable by identity, source and time.
4. Reuse G1/G2/G3/G4 and the current evidence registry. Consolidate factual normalization where independent scanners can disagree; retain distinct products of that normalization.
5. Make a patient/current-problem interaction projection primary. Use the six steps to edit detailed source facts and documentation, with no clinical truth in the UI projection.

## C. Current architecture map

**OBSERVED:** `clinical_patients` → protected `clinical_encounters` with full dated payload and completed/amended status, plus dated `clinical_lab_snapshots`. A browser working case/cache syncs through `patient-registry.js`; it is not an additional durable clinical owner. Module 01 stores fracture/risk, Step-3 investigation and Step-4 treatment/decision/task assertions in encounter payloads. G1 derives reliable actual administration, latest asserted treatment, unresolved tuple-keyed tasks and conflicts from completed/amended encounters. `EncounterContext` combines historical and live visit state; G2 evaluates reviewed osteoporosis registries/rules; Visit Plan and `Γιατί τώρα` surface transient guidance. G3 independently scans some historical facts for its read-only latest summary. Step-3 longitudinal DXA/FRAX histories and dedicated lab rows supply additional factual history. Global Cockpit owns global entry/navigation; Module 01 still exposes the six-step shell and an injected Physio utility link. PR-1 candidates are proposed, transient and non-authoritative.

## D. Mechanism disposition matrix

Each row has exactly one **primary** R4 disposition. A future Core placement does not itself authorize a migration.

| Mechanism | Current owner / role | R4 disposition | Target responsibility and reason |
|---|---|---|---|
| Patient identity | `clinical_data.py` protected patient row; module browser selects active ID | **KEEP** | One authoritative patient ID; other modules reference it. |
| Active patient/context binding | `patient-registry.js` module-injected session context | **MOVE TO SHARED CORE** | One reusable protected patient-selection/context contract across modules; osteoporosis encounter interpretation stays local; removes parallel active-patient session models. |
| Encounter identity/context | `clinical_data.py` encounter ID/status/date; module visit intent | **KEEP + ADAPT** | Keep global encounter identity/finalization; expose a reusable context envelope, while osteoporosis archetype/problem meaning stays local. |
| Protected patient persistence | `clinical_patients` and protected API | **KEEP** | Existing factual owner and protection boundary. |
| Completed/amended encounter persistence | `clinical_encounters.payload_json` and status resolution | **KEEP + ADAPT** | Preserve owner and finalization; add bounded semantic identity/linkage within its governed payload/write path when implemented. No replacement store. |
| Generic lab persistence | `clinical_lab_snapshots`, `clinical_data_ext.py` | **KEEP + ADAPT** | Patient-level dated result owner; reconcile source encounter/date edit/clear lifecycle and provenance. Osteoporosis BTM meaning stays local. |
| Transcript/candidate capture boundary | PR-1 Core design and H13 promoted candidate semantics | **KEEP** | Transient proposal → deterministic mapping → clinician review → existing authoritative owner. No capture-owned truth. |
| Document/file mechanisms | `clinic_utilities/clinical_documents` and other separately owned utilities | **KEEP** | Retain their own protected document/source handling; link by provenance if clinically relevant, never copy a file into an osteoporosis fact by implication. |
| Navigation/Cockpit shell | `static/cockpit`, service root | **KEEP + ADAPT** | Keep global Home; offer shared patient/context entry and module route without making Cockpit a clinical-semantic owner. |
| Evidence/provenance presentation primitives | G2 UI and Physio evidence sheets | **MOVE TO SHARED CORE** | Generic source/status/time/uncertainty display and evidence-on-demand contract is cross-module; guideline meaning, applicability and source registry remain domain-owned. Avoid separate provenance visual languages that blur fact versus guidance. |
| Fracture event storage | `fracture_history.events[]` | **KEEP MODULE-LOCAL** | Structured fracture facts stay with current encounter owner; stable event ID/source enables reuse. No new fracture store justified. |
| Fragility mechanism representation | event `low_trauma`; S1 G2/G3 interpretation | **KEEP MODULE-LOCAL** | Keep reviewed S1 fail-closed meaning; legacy summary claim is not an independent mechanism fact. |
| Treatment episodes | `step4.treatment_episodes[]` snapshots | **KEEP + ADAPT** | Add stable epoch continuity and explicit start/stop/revision links within existing Step-4 owner; do not infer missing exposure. |
| Administration events/plans | `step4.administrations[]`; G1 actual projection | **KEEP + ADAPT** | Preserve separate actual/scheduled/planned rows; add explicit plan→actual/missed-resolution link only when established. Keep dose/agent meaning local. |
| Transition representation | `step4.transition` and decision/episode rows | **KEEP + ADAPT** | Link intent, decision, successor plan, actual action and monitoring by explicit provenance; planned successor is never actual. |
| Future tasks/obligations | `step4.tasks[]`; G1 tuple-keyed unresolved projection | **KEEP + ADAPT** | Extend task owner with stable obligation identity and explicit disposition history; replace tuple *matching* only after legacy reconciliation. |
| Cross-encounter task tuple matcher | G1 `type|due_date|timeframe_text` key | **REPLACE** | Stable obligation identity plus explicit disposition must govern continuity after legacy reconciliation; changing a date currently creates a second apparent obligation. Keep all source rows. |
| Generic obligation identity/provenance contract | Not currently a durable shared contract; Step-4 task row UUID only | **MOVE TO SHARED CORE** | A cross-module identity, source, due/timeframe and disposition envelope is reusable for clinical actions; osteoporosis owns triggers, priority and closure meaning. Prevents each module inventing a separate task continuity model. This is a proposed bounded seam, not an existing store or new rules engine. |
| Goals/target state | No general durable owner; plan/rationale may imply one | **KEEP MODULE-LOCAL** | Add explicit osteoporosis goal assertions/revisions linked to decision and review, only after bounded clinical target definitions. A generic goal engine is not evidenced. |
| DXA longitudinal representation | `step3.dxa` plus `longitudinal_review.dxa_history[]` | **KEEP + ADAPT** | Reconcile scans by source/date/machine/measurements; keep unique manual rows factual; derive trend and latest view from reconciled facts. |
| FRAX longitudinal representation | `risk_assessment` plus `longitudinal_review.frax_history[]` | **KEEP + ADAPT** | Reconcile formal assessments, preserve original versus contextual/adjusted values and framework/version; derive trends without duplicate authority. |
| Laboratory/BTM history | Step-3 labs plus dedicated dated lab rows | **KEEP + ADAPT** | Make Step-3 capture a sourced view/write through the dated lab owner; preserve BTM context and unique historical values. |
| CKD/GC constraints | `risk_context` snapshots, dated renal labs; G2 current adapter | **KEEP MODULE-LOCAL** | Explicitly distinguish asserted/unknown exposure and dated renal facts; no inferred continuous GC interval or new threshold. |
| Clinical decisions | `step4.decision`, Step-5 process flags | **KEEP + ADAPT** | Extend Step-4 durable decision owner with distinct option, recommendation, final choice and links to action/evidence context. |
| Patient preference/constraints | Step-4 flags/rationale, Step-5 process | **KEEP + ADAPT** | Record actual content, source/time and effect on decision through the existing decision owner; flags alone are not content. |
| Outcomes | fracture/results/response assertions across encounters | **KEEP + ADAPT** | Link an explicitly reviewed outcome/response interpretation to source events, epoch and decision where known; chronology alone is not causation. |
| `LongitudinalGuidanceProjectionV1` | G1 read-only completed/amended history projection | **KEEP + ADAPT** | Retain filtering, actual-event dedupe and fail-closed conflicts; consume reconciled factual identities, extend continuity coverage without fact writes. |
| `EncounterContext` | G1 current visit builder; G0 broader contract | **KEEP + ADAPT** | One derived current-context contract over normalized history + live facts; G2 context is an adapter, not a rival patient model. |
| Visit Plan | G1/G2 ordered card guidance | **KEEP + ADAPT** | Remain ephemeral current interaction plan, with source/uncertainty and problem-specific routing; never persist as a treatment decision. |
| `Γιατί τώρα` | G1/G2 reason rendering | **KEEP + ADAPT** | Keep rule/input explanation; bind it to the relevant current-state item and source, and withdraw stale derived output on edit. |
| G2 evidence/guidance evaluation | Reviewed registry/manifest → deterministic JS executor | **KEEP + ADAPT** | Retain reviewed clinical authority and deterministic ephemeral guidance; factor duplicate factual adapters/predicates only where they risk divergent meanings. No second evidence engine. |
| G3 longitudinal summary | Separate read-only scan of encounters/labs plus G1 projection | **KEEP + ADAPT** | Preserve useful summary; consume reconciled factual projection for shared facts, retaining G3 selection/presentation logic and incomplete-history labels. |
| G4 workspace behavior | Collapse/sticky/salience UI state | **KEEP + ADAPT** | Reuse density controls and live salience for primary workspace; no patient truth in session preference. |
| Six-step shell | `static/baseline-audit` Steps 1–6 | **KEEP + ADAPT** | Secondary structured editor/documentation and audit capture behind patient/problem entry; no forced primary sequence. |
| Patient history surfaces | protected registry encounter list and G3 summary | **KEEP + ADAPT** | Link selective trajectory item to the sourced encounter/fact; never claim complete history from missing records. |
| Trend surfaces | Step-3 DXA/FRAX/lab tables/charts | **KEEP + ADAPT** | Keep descriptive charts; feed reconciled sourced facts and preserve comparability/LSC qualifiers. |
| Sidebar/module navigation | module sidebar plus `calendar-link.js` injection | **KEEP + ADAPT** | Keep module routes; remove duplicate global Physio entry and stale audit/pilot affordances from primary path. |
| Module-injected global Physio link | `calendar-link.js` duplicate of Cockpit utility entry | **RETIRE** | The existing global Cockpit entry already owns this route; a second Module-01 link duplicates navigation without osteoporosis meaning. |
| Current step-specific editors | Steps 1–6, especially Step-3/4 | **KEEP + ADAPT** | Retain detailed entry and finalization owner; provide direct route from current problem to correct editor. |
| Global Cockpit shell | `static/cockpit` Home and global utilities | **KEEP** | Already appropriate global entry; no osteoporosis clinical logic moves into Home. |
| Physio interaction primitives | Physio product app, contextual sheets, evidence sheets | **KEEP + ADAPT** | Reuse behavior patterns only: progressive disclosure, dependent options, live updates, direct editing and evidence on demand. |
| Shared patient/context primitives | protected API plus module-specific browser binding | **MOVE TO SHARED CORE** | Shared identity/status/source envelope for all modules; module clinical interpretations remain separate; removes duplicate context plumbing. |
| Future Live Copilot interface | No Live implementation | **KEEP MODULE-LOCAL** | Future capture adapter targets the same module-owned semantic mapping/review/write seams; no Live-owned fact/rule model. Architecture constraint, not implementation authority. |

**REPLACE / RETIRE audit:** no working protected store, guidance engine, editor or trend mechanism meets the reuse-before-replace burden. The **tuple `type|due_date|timeframe_text` as authoritative cross-encounter obligation matching** is **REPLACE** once stable identity and legacy reconciliation exist: keeping or adapting that key cannot tell a reschedule from a new obligation, and merely consolidating old tuples cannot recover the clinician's intent; extending the task owner with stable identity therefore necessarily replaces the matcher. R2's executable probe demonstrated the duplicate. Individual task rows remain. The **duplicate global Physio sidebar link** is **RETIRE** from Module 01 because the current Cockpit already owns its global utility entry; retaining it serves no module-specific clinical function. Disabled KPI/Library and stale pilot/privacy copy are presentation cleanup in the retained shell, not removal of their underlying audit or privacy capabilities.

## E. Shared Core vs Osteoporosis-local map

**Core target:** protected patient and encounter identity/status; reusable active-patient context; dated generic result identity/source; generic clinical-event reference/provenance envelope; generic obligation identity/disposition envelope; transcript proposal/review transport; reusable evidence/provenance presentation; global navigation. A `MOVE TO SHARED CORE` above removes duplicate *identity or presentation contracts*, not an osteoporosis fact. Existing global patient/encounter/lab persistence is already a shared substrate and stays with its current owner.

**Osteoporosis target:** fracture/fragility interpretation, treatment epochs and agent-specific administration meaning, transitions, osteoporosis obligation triggers, goals, DXA/FRAX/BTM interpretation, CKD/GC relevance, evidence registry and G2 rule meaning, decision/response associations and disease-specific interaction content. Similar screens in Physio do not make these clinical semantics generic. A shared event reference may identify/source a fracture; it does not decide whether that fracture is fragility or treatment failure.

## F. Longitudinal truth architecture

**Treatment epoch choice — Option B.** Option A, reconstructing solely from unlinked encounter snapshots, loses stable course identity, reasons and interruption relationships. Option C, a separate durable treatment store, is not supported while Step-4 already persists episode rows and protected encounters. **DECISION:** extend the existing Step-4 treatment representation with stable longitudinal epoch identity and explicit continuity/transition links; G1 derives state from those reviewed facts. Historical snapshots with no provable link remain separate/uncertain, not silently stitched into an epoch. A later dedicated store would require a demonstrated owner limit, not just schema neatness.

**Administrations:** Step-4 remains factual owner for recorded actual events and scheduled/planned/missed assertions. Exact `actual_date` establishes captured action; scheduled, planned and missed are not actual. Link a planned/missed row to a later action only by explicit reviewed reference. G2's evidence-derived due date consumes reliable actual chronology and applicable reviewed rules; it is ephemeral and must never write an administration or booking. Generic event identity/source may be shared; agent, timing and due semantics remain osteoporosis-local.

**Obligations:** keep Step-4 tasks and add stable obligation identity across encounters, explicit status transitions (including rescheduled/deferred/superseded/declined or alternative disposition only when recorded), source, actor and due history. The existing UUID identifies a row, not currently a cross-encounter obligation. A generic Core envelope can serve other modules, while Module 01 owns clinical trigger and resolution meaning. The G1 tuple key must be retired as authoritative continuity **after** old rows are reconciled; until then label possible duplicates and preserve all original rows. `close.unresolved_critical` must not be erased by a later blank/no without an explicit resolution link.

**Decisions, goals, outcomes:** extend Step-4 decision authority to distinguish option considered, clinician recommendation, patient preference/constraint, final decision and later actual action, each with source/time and explicit relationship. Process checkboxes and prose remain supporting evidence, not replacements for missing content. An osteoporosis-local goal/target assertion may be linked to a decision and revised/reviewed over time; the clinical owner must define valid targets without this review inventing thresholds. Outcomes remain sourced events/results plus optional clinician-recorded interpretation/association; date order alone never proves response or causation.

## G. Investigation/provenance consolidation

| Domain | Factual owners and overlap | Architectural direction / preservation |
|---|---|
| DXA | `step3.dxa` is an encounter-scoped scan snapshot; `longitudinal_review.dxa_history[]` contains manual historical scan facts, sometimes unique and sometimes duplicate. | Retain both captured sources while matching only with adequate scan date, machine, measurements, source encounter/report and explicit conflict state. Extend existing source/provenance handling; project one reconciled trend and G3 latest view. Never delete or demote a unique manual scan; unknown comparability/LSC withholds significance language. |
| FRAX | `risk_assessment` owns encounter formal result/framework; manual `frax_history[]` may repeat it and can carry adjusted/contextual values. | Preserve original formal values separately from contextual adjustment, model/framework/version/date and source. Reconcile matching assessment identities before one projection feeds trends/summary; do not overwrite a formal result with an adjusted one. |
| Labs / BTMs | `step3.labs` is encounter capture; `clinical_lab_snapshots` is dated patient-level result owner. Current browser upsert matches source encounter **and** date; date edit/clear can strand a row. | Extend the existing dedicated row lifecycle and source link; Step-3 edits should update/reconcile that owner rather than create a second independent result. Retain unique dated values and BTM context, show conflicts rather than picking a latest row silently. No new laboratory store. |

Across all three, a source link and explicit conflict/unknown state precede any derived trend. A file/report attachment is evidence provenance, not a second result fact until reviewed and entered by the appropriate owner.

## H. Derived-state architecture

```text
protected patient + completed/amended sourced facts + dated lab rows
→ one reconciled factual normalization / conflict view (read-only)
→ G1 longitudinal projection
→ EncounterContext (history + explicit live current visit)
→ G2 reviewed-rule evaluation + G1 Visit Plan / Γιατί τώρα
→ G3 summary and current problem interaction projections
```

G1, G2 and G3 are legitimate **different projections**: G1 continuity/current-context extraction, G2 clinical rule evaluation, G3 concise historical presentation. They must share factual normalization for treatment/admin identities, fracture mechanism, DXA/FRAX/lab provenance and obligations where duplication currently risks divergent answers. This is a read-only seam over the existing owners, not a new patient database. `osteoporosis_evidence_context_v1` remains a G2 rule adapter. The reviewed evidence/registry → deterministic executor → ephemeral guidance architecture is **KEEP + ADAPT**; duplicated JS predicates may be consolidated under one reviewed rule meaning, but current rule output is never patient fact or historical evidence at a past decision.

For future decisions, record which reviewed rule/source version and relevant input/provenance context was actually available or shown **at decision time**, linked to the clinician's decision. Do not reconstruct that clock by rerunning today's registry on old encounters. Existing amended rows do not preserve every prior version; unsupported historical certainty remains explicitly unavailable.

## I. Point-of-care interaction support architecture

The primary patient/current-problem projection consumes the existing layers in this order: **durable sourced facts → reconciled longitudinal state → live current visit facts → reviewed current guidance → selective interaction state**. G1 supplies present continuity and fail-closed history; G2 supplies source-backed current guidance; G3 supplies selective summary; G4 supplies disclosure/salience. The new projection needs source-to-editor routing, visible unknown/conflict states and live invalidation after upstream edits. It must not become a fact writer, evidence author or clinical decision engine. A currently unavailable historical input suppresses only conclusions dependent on it; it does not turn history into zero.

## J. PR-1 / manual UI / future Live convergence

```text
manual UI | PR-1 transcript candidates | future Live Copilot candidates
→ same deterministic module mapping and clinician review/acceptance boundary
→ same protected patient/encounter/lab semantic owners
→ same normalized history, G1 context and reviewed G2 evidence engine
```

PR-1's proposed semantic candidate is not an authoritative fact. Any future Live stream uses the same proposal, conflict, review and owner write path; it has neither a separate database nor a parallel clinical brain. This is interface compatibility only. No Live implementation or PR-1 redesign is authorized.

## K. Six-step-shell disposition

**KEEP + ADAPT as secondary editor/documentation layer.** Steps 1–6 keep their structured capture, finalization and audit utility. Patient/problem-first state is the primary entry and navigation; a relevant state item opens its authoritative Step field or source encounter. The clinician need not traverse all six steps to inspect a delayed dose or fracture. Baseline/pilot copy and dead module links can be cleaned during a later UI slice, with no new storage owner.

## L. Physio reuse disposition

Physio supplies **interaction primitives**, not an osteoporosis domain model: progressive disclosure, contextual dependent controls, direct upstream editing, live downstream refresh with stale-output suppression, and evidence-on-demand. Adapt these to sourced longitudinal facts, safety/event priority, uncertainty and clinician decision ownership. Physio's Knee-OA clinical vocabulary, preselected rehabilitation plan, referral output and single-visit model stay with Physio. Shared UI code is optional; sharing semantic contracts is justified only where identity/provenance truly crosses modules.

## M. Anti-patterns / rejected alternatives

Reject a second patient or AI-owned truth store, UI-authored clinical facts, a Live-only database/rule engine, duplicated evidence interpretation, a confident timeline made from sparse encounter dates, automatic action from a plan or current guidance, a full rewrite for code cleanliness, literal Physio workflow transfer, and the six-step form as the sole product model. Reject event sourcing or a new treatment table as a prerequisite without evidence that the existing Step-4 owner cannot support stable identity. Reject any consolidation that discards a unique historical DXA/FRAX/lab fact. The two actual `REPLACE`/`RETIRE` targets in §D are narrower than these rejected wholesale alternatives.

## N. Minimal architectural delta

| Timing | Smallest justified change |
|---|---|
| **MUST CHANGE BEFORE PROTOTYPE** | Freeze a one-page owner/interface contract for patient/encounter IDs, factual source/provenance, actual vs planned vs due, stable epoch and obligation identity semantics, distinct decision roles, history-unavailable/conflict behavior and reviewed-rule authority. Define how both prototypes label unsupported historical links as unknown. Decide which existing owner receives any explicitly entered continuity/decision facts; no new store is presumed. Clinical owner supplies only the bounded meaning needed for the two scenarios. |
| **CAN ADAPT DURING PROTOTYPE** | Read-only normalization shared by G1/G3/G2 adapters; source-to-editor routing; patient/problem interaction projection; provenance-aware DXA/FRAX/lab display; direct-edit/live recalculation; secondary Step access; shared patient context and evidence UI primitives. Prototype may reveal an implementation detail, not open a new broad review. |
| **CAN WAIT** | Full legacy-history reconciliation/migration; exhaustive 15-year trajectory visualization; complete GC exposure intervals; generalized cross-module goal engine; broad document linkage; every outcome association; comprehensive historical guidance-exposure archive; full Core extraction across modules. Preserve existing facts and label limits meanwhile. |
| **DO NOT CHANGE** | Protected patient/encounter/lab substrate and completed/amended behavior; reviewed G2 medical rules/registry; S1 fragility semantics; PR-1/H13 lifecycle; actual-versus-scheduled guard; fail-closed history/conflict behavior; Physio domain model; root writer lock. |

## O. Two-prototype readiness

| Candidate slice | Reuse now | Minimum semantic prerequisite / hypothesis to test |
|---|---|---|
| **1. Denosumab longitudinal management** | Protected patient/encounter facts, Step-4 episode/admin/transition/task editors, G1 actual chronology/conflicts, G2 reviewed timing/exit guidance, G3 summary, G4 disclosure. | Define stable epoch and obligation links and clearly mark unlinked historical missed plans/actuals as unknown; retain actual/planned/derived separation. Test whether one sourced state/problem projection and secondary Step editor allow safe review of a due/delayed/transition situation without a parallel truth path. No new cadence or rescue rule. |
| **2. New fragility fracture / treatment reassessment** | S1 event/`low_trauma` semantics, Step-2 event and Step-4 treatment/decision facts, G1/G2 event priority, G3 fracture summary, Step-3 sourced investigations. | Define only the explicit event→epoch/decision reference when known, separate option/recommendation/preference/final choice, and retain unknown exposure or mechanism. Test whether the projection makes mechanism, treatment exposure, comparability, unresolved information and current reviewed guidance distinct without an automatic failure/switch claim. |

**Readiness:** architecture is sufficiently specific to proceed to the single programme synthesis and then these two bounded prototypes, subject to the pre-prototype owner/interface contract above. `UNKNOWN UNTIL PROTOTYPE` is limited to interaction effectiveness and the exact presentation/component factoring, not the patient truth or clinical-rule owner.

## P. Product Owner decisions required

1. **PRODUCT OWNER DECISION:** approve the two-slice pre-prototype boundary and whether the primary patient/problem entry should replace the six-step start on day one of the prototype or coexist as an explicit alternate entry during evaluation. The architecture keeps Steps as the editor either way.
2. **PRODUCT OWNER DECISION:** choose the minimum clinician-confirmed historical reconciliation workflow for ambiguous old epoch/task/DXA/FRAX/lab links before claiming a complete longitudinal story; default is to show uncertainty and preserve all source rows.

**BOUNDED CLINICAL OWNER DECISIONS:** define valid osteoporosis goal/target meanings and any new treatment/obligation closure meaning needed by the two slices. Existing reviewed G2/S1 rules remain unchanged. **PROTOTYPE HYPOTHESES:** whether the selective state projection reduces navigation and makes source/uncertainty/action clearer. **IMPLEMENTATION DETAILS:** field names, migration mechanics and component packaging after synthesis. None warrants another review lane.

## Q. Synthesis-ready architecture conclusion

The target is **one protected patient/encounter/lab substrate with module-owned osteoporosis facts and a small shared identity/provenance/obligation envelope**, extended within existing Step-4 owners for stable treatment, task and decision continuity. A read-only reconciled factual seam feeds G1 longitudinal state, current EncounterContext, reviewed G2 guidance, G3 summary and one patient/problem-first interaction projection. The current six steps become source editors and documentation. Manual entry, PR-1 candidates and future Live candidates converge through the same clinician-reviewed semantic owners. The Global Cockpit and Physio interaction mechanics are reused without importing Physio clinical meaning. This reaches the accepted R1–R3 product model with bounded extensions and consolidation, while preserving existing protected persistence, finalization and clinical guidance.

## R. Disposition

**R4 COMPLETE / SYNTHESIS READY**

Return this exact branch/head and the local `CURRENT.md` checkpoint to the programme coordinator. R4 stops after remote publication. No synthesis, prototype, implementation, clinical-rule change, schema/database change, PR #121 mutation, merge or deploy is part of this lane.
