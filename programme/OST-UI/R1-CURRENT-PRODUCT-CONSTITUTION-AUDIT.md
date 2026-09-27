# R1 — Current Product vs Product Constitution Audit

> **Workstream:** OST-UI
> **Review:** R1 — Current Product vs Product Constitution
> **Mode:** fresh separate READ-ONLY current-state audit
> **Runtime mutation:** none
> **Clinical-semantic mutation:** none
> **Root canonical mutation:** none
> **Disposition:** R1 COMPLETE / R2 ELIGIBLE FOR COORDINATOR RECONCILIATION

---

## A. Exact source identity

### Canonical Bootstrap Manifest

**Fresh current main**

- Repository: athpapachr-cmd/osteoporosis
- Fresh main SHA: **2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b**

The six active root canonicals were consumed in the order required by AGENTS.md:

1. AGENTS.md
2. TODO.md
3. CLINICAL_EXCELLENCE_PLAN.md
4. SLICE_PLAN_CURRENT.md
5. CURRENT_OPERATIONAL.md
6. osteoporosis-change-log.md

**Root operational writer / lock**

- Root NOW remains the bounded PR-1 Heidi-first transcript intake / candidate-extraction lifecycle.
- OST-UI did not take, alter or overlap that writer lock.
- No runtime, schema, database, root-canonical or clinical-rule mutation was performed.

**Governing OST-UI R1 contract**

- Draft PR: **#122**
- Branch: **docs/ost-ui-product-reconstruction-bootstrap-2026-09-27**
- Exact verified contract head: **9363ad5dbb5f7c059f351421599bd6ede4db6438**
- Consumed fully:
  - programme/OST-UI/PROJECT-INDEX.md
  - programme/OST-UI/CURRENT.md
- PR #122 remains open, draft and unmerged.

**Supporting Product Constitution**

- Draft PR: **#121**
- Branch: **docs/ost-programme-lifecourse-bootstrap-2026-09-27**
- Exact verified supporting head: **80851762f39105628ab9009406cc7828b3e9be19**
- Consumed: programme/PRODUCT-CONSTITUTION-V0.2.md
- This was used only as supporting product intent, not as current-main runtime authority.

### Major current runtime owners inspected

- main.py
- clinical_data.py
- clinical_data_ext.py
- static/cockpit/index.html
- static/cockpit/app.js
- static/baseline-audit/index.html
- static/baseline-audit/app.js
- static/baseline-audit/patient-registry.js
- static/baseline-audit/progressive-guidance-core.js
- static/baseline-audit/progressive-guidance-ui.js
- static/baseline-audit/osteoporosis-evidence-guidance-core.js
- static/baseline-audit/osteoporosis-longitudinal-summary-core.js
- static/baseline-audit/g4-workspace-ergonomics.js
- static/baseline-audit/calendar-link.js
- static/baseline-audit/longitudinal.js
- static/baseline-audit/prior-dxa-inline.js
- static/baseline-audit/lab-history-ui.js
- static/baseline-audit/step4.js

Relevant current tests inspected include:

- test_cockpit_home.py
- test_progressive_guidance_node.js
- test_g3_guidance_summary_node.js
- test_g4_workspace_ergonomics.js
- test_physio_knee_oa_cockpit_browser.py

Relevant Physio interaction contract inspected only as reuse evidence:

- clinic_utilities/physio_referral_product/UX_CONTRACT_CURRENT.md

### Evidence discipline used in this audit

- **OBSERVED** = directly established by current source/runtime contract or current regression test.
- **INFERRED** = reviewer interpretation of combined observed behavior.
- **UNVERIFIED** = not provable within R1's current-state/product-surface boundary.

This audit keeps separate:

CURRENT RUNTIME FACT
!=
CANONICAL DESIGN INTENT
!=
DRAFT FUTURE CONCEPT
!=
REVIEWER INTERPRETATION

---

## B. Current product map

### B1. What the clinician actually encounters today

**OBSERVED — global entry is now a real Cockpit Home.**

The service root redirects to /static/cockpit/. The Home presents:

- today's privacy-minimized Clinical Calendar summary;
- Clinical Modules;
- Learning & Improvement;
- global Clinic Utilities;
- a separately owned Reception dashboard.

Osteoporosis is presented as **Module 01**, not as the whole Clinical Excellence product.

**OBSERVED — opening Module 01 still enters the historical Baseline Audit surface.**

The Osteoporosis route is /static/baseline-audit/. Its visible shell still identifies itself as:

- “Osteoporosis Cockpit”;
- “Baseline Audit v1 · prospective encounter capture”;
- “PILOT CASE 1/5”;
- a six-step encounter flow.

Its persistent visible navigation begins with:

- Αρχική;
- Νέο Case;
- Cases;
- disabled KPI Overview;
- disabled Βιβλιοθήκη;
- Privacy.

**OBSERVED — protected patient context is injected into that encounter shell.**

patient-registry.js adds a protected Patient Registry that can:

- authenticate to the clinical-data boundary;
- search/create a patient by patient ID;
- load the active protected patient;
- show encounter and laboratory counts;
- list prior encounters by date/status;
- load a prior encounter;
- start a new visit for the active patient.

The active patient identifier itself is held in sessionStorage as browser session context, while the authoritative patient/encounter/laboratory records live server-side.

**OBSERVED — above the six steps, the current runtime now renders two state-aware surfaces.**

1. **Σύνοψη ασθενούς** — read-only longitudinal summary.
2. **Σημερινή ροή** — current Visit Plan with “Γιατί τώρα”, evidence metadata and salience.

These surfaces are generated dynamically from protected historical encounters/labs plus live current-encounter state.

**OBSERVED — the underlying clinical capture remains a six-step encounter model.**

The main clinical work is still distributed across:

1. Σύνοψη Επίσκεψης
2. Ιστορικό & Κίνδυνος
3. Εξετάσεις & Αποτελέσματα
4. Απόφαση & Πλάνο
5. Επικοινωνία
6. Τεκμηρίωση & Heidi

Step 4 contains current treatment episodes, administrations, decision, transition/sequencing, follow-up tasks and close state.

### B2. Product-type inventory

The current product is **a mixture**, not one pure category.

| Product character | R1 finding | Evidence status |
|---|---|---|
| Collection of pages | present at Cockpit/module level, but not the dominant Module-01 behavior | OBSERVED |
| Form system | strongly present through the six-step encounter shell and local working-case model | OBSERVED |
| Audit system | strongly visible in naming, pilot/baseline messaging, applicability/progress concepts and KPI heritage | OBSERVED |
| Collection of isolated tools | materially reduced at global/module boundary, but not completely eliminated from Module-01 navigation | OBSERVED |
| Decision-support workspace | substantially present through G1/G2 current-flow and evidence guidance | OBSERVED |
| Longitudinal clinical environment | partially present through protected history, projection, summary, due/task resurfacing and history-sensitive guidance | OBSERVED |
| Primary clinician-facing product model | **hybrid: encounter/audit-first shell with a genuine longitudinal decision-support layer added above and through it** | INFERRED |

### B3. Current clinician mental model implied by the UI

**INFERRED from observed composition**

The practical current sequence is approximately:

Cockpit Home
→ open Osteoporosis Module 01
→ identify/open protected patient
→ read longitudinal patient summary
→ read current “Σημερινή ροή”
→ work through whichever of the six encounter steps contain the relevant domains
→ save/Finish the encounter
→ later history is reconstructed from completed/amended encounters and dedicated lab snapshots

This is materially more longitudinal than the original Baseline Audit, but the visible centre of gravity remains **the current encounter**, not a primary patient life-course workspace.

---

## C. Product-Constitution alignment map

The Product Constitution below is supporting intent from unmerged PR #121. Alignment labels describe only the current runtime at main 2ae9f01….

| Responsibility | Current mechanism / surface | Observed behavior | Alignment | Evidence |
|---|---|---|---|---|
| Global Cockpit vs disease module | /static/cockpit/ + Module-01 route | Global Home owns modules, learning, utilities and Reception; Osteoporosis is Module 01 | **partial alignment** | OBSERVED; one module utility leak remains, see F |
| One patient context | protected Patient Registry + active patient session context | One protected patient can own multiple encounters and lab snapshots | **partial alignment** | OBSERVED; patient context is embedded inside Module 01 rather than visibly “above modules” |
| Durable patient persistence | clinical_patients | Protected server-side patient record | **alignment** for current bounded need | OBSERVED |
| Durable encounter persistence | clinical_encounters.payload_json | Draft/completed/amended encounters are stored by patient | **alignment** as existing substrate | OBSERVED |
| Durable laboratory history | clinical_lab_snapshots | Dated lab snapshots are stored separately and linked to encounter when available | **alignment** as existing substrate | OBSERVED |
| Current vs historical truth | completed/amended filters + exclusion of current UUID | Current draft is not silently treated as completed history | **alignment** | OBSERVED |
| Actual vs planned treatment | administration projection requires exact actual_date | Scheduled/planned rows do not increment actual dose count | **alignment** with “actual worldline” distinction for this mechanism | OBSERVED + regression test |
| Past changes present | G1 longitudinal projection → EncounterContext → VisitPlan | prior treatment/admin/task/due/conflict state changes current surfaced domains and “why now” | **alignment** | OBSERVED |
| Evidence changes present | G2 evidence context/contributions | patient/current/history context activates evidence-backed guidance with provenance | **alignment** for implemented rule scope | OBSERVED |
| What matters today | Σημερινή ροή + why_now + salience | visit type, new events, treatment context, explicit due state and unresolved prior items affect today's flow | **partial alignment** | OBSERVED; still projected into encounter/form domains |
| Longitudinal patient summary | G3 patient summary | course, fractures/risk, latest DXA, treatment, labs, last decision and unresolved items are summarized | **partial alignment** | OBSERVED; summary is snapshot-oriented rather than a full trajectory |
| “How did we get here?” | encounter list + summary + treatment/admin projection + DXA/FRAX history | prior facts are retrievable and partly summarized | **partial alignment** | OBSERVED; the causal/episode story is largely reconstructable rather than directly presented |
| Treatment course | Step 4 treatment episodes + G1 projection | latest reliable episode snapshot plus actual administrations influence current state | **partial alignment** | OBSERVED; sufficiency as durable life-course representation belongs to R2/R4 |
| Fracture course | encounter fracture events + G3 deduped summary | stable fracture events can be deduplicated and summarized | **partial alignment** | OBSERVED; R1 does not judge fracture/fragility semantics |
| DXA course | historical encounter DXA summary + longitudinal_review DXA history/trends | latest prior DXA is summarized; separate trend UI can hold multiple DXA points | **partial alignment** | OBSERVED; ownership/sufficiency of overlapping representations is a deeper question |
| Clinical decisions | Step 4 decision + G3 “Τελευταία απόφαση” | latest explicit decision can be carried into longitudinal summary | **partial alignment** | OBSERVED; no full decision episode/rationale history is primary UI |
| Future obligations | Step 4 tasks + explicit next-due + unresolved task projection | planned tasks and explicit due/overdue treatment timing can resurface in later guidance/summary | **partial alignment** | OBSERVED; not a first-class obligation/disposition workspace |
| Uncertainty/conflict | projection conflict records + unavailable-history states | conflicts are surfaced; unavailable history is not treated as absent history | **alignment** for implemented mechanisms | OBSERVED |
| Evidence/provenance | G2 source refs + compact evidence metadata | guidance displays source labels and distinguishes checklist-only safety from automatic clearance | **alignment** for current guidance | OBSERVED |
| Goals / desired future state | treatment plan/tasks/transition fields | plans exist, but no explicit general goal/target-state concept is a primary current surface | **absent / unclear** | OBSERVED for absence of a dedicated surface; deeper state need is for R2 |
| Clinical episode as major trajectory object | encounters + step-specific payload | facts, decisions and tasks exist, but there is no primary episode view binding before/change/decision/outcome/obligations | **absent as visible product projection** | OBSERVED |
| Two evidence clocks: “then” vs “now” | current G2 evidence-backed guidance | current evidence is visible; historical evidence-at-decision context is not a visible first-class surface | **unclear / absent in current UI** | OBSERVED for current UI; R2/R4 own deeper persistence question |
| Fifteen-year rapid life-course comprehension | summary + encounter registry + trends | some key state is rapidly visible, but a long trajectory still requires reconstruction across summaries, encounter loads and step-specific history | **partial alignment** | INFERRED from observed UI composition |

---

## D. Longitudinal capabilities already present

These are current mechanisms that should not be treated as nonexistent merely because the visible shell retains Baseline Audit heritage.

### D1. Protected patient / encounter / laboratory substrate

**OBSERVED**

clinical_data.py defines durable protected tables for:

- clinical_patients;
- clinical_encounters;
- clinical_lab_snapshots.

Encounter records carry draft/completed/amended status and payload_json. Laboratory snapshots are separately dated and can link back to a source encounter.

The server finalization semantics preserve completed/amended state: later material changes to a completed encounter become amended rather than silently reverting it to draft.

### D2. Fail-closed historical loading

**OBSERVED**

progressive-guidance-ui.js distinguishes:

- not_loaded;
- loading;
- loaded;
- unavailable.

If protected history cannot be loaded, the UI explicitly warns the clinician **not** to infer that prior visits or unresolved items are absent.

This is a materially useful longitudinal safety behavior.

### D3. Historical truth is filtered before projection

**OBSERVED**

G1 longitudinal projection:

- uses only completed/amended encounters;
- excludes the current encounter UUID;
- sorts prior encounters chronologically;
- preserves explicit conflicts rather than silently choosing between incompatible facts.

### D4. Actual administration history is clinically meaningful state, not simple row counting

**OBSERVED**

The projection:

- counts only administrations with an exact actual_date;
- deduplicates repeated representation of the same actual event;
- does not count a merely scheduled/planned dose as administered;
- tracks last actual administration by agent;
- carries explicit next-due dates where recorded;
- detects conflicting next-due values for the same actual event.

The current regression tests explicitly protect these distinctions.

### D5. History materially changes current guidance

**OBSERVED**

Prior history is not merely displayed. It feeds current behavior.

Examples already implemented:

- prior active treatment enters EncounterContext;
- explicit due/overdue state surfaces Administrations and Follow-up;
- prior planned tasks resurface as unresolved prior items;
- prior unresolved critical close state surfaces follow-up;
- treatment/admin conflicts create a current warning;
- new fracture context changes priority and breadth of the current Visit Plan;
- evidence-backed treatment milestones use prior exact actual administration dates where the rule contract allows it.

The denosumab evidence layer is a clear example: an exact prior actual denosumab date can create an **ephemeral evidence-derived due state**, while scheduled-only treatment does not become actual exposure.

### D6. Longitudinal patient summary

**OBSERVED**

G3 renders a read-only patient summary above the current flow when an active protected patient exists.

It can expose:

- course span and completed/amended encounter count;
- fracture/risk state;
- latest relevant DXA;
- treatment and latest actual administration;
- selected lab values from the latest protected snapshot;
- last explicit management decision;
- unresolved tasks/critical items/conflicts.

The current visit is explicitly labelled current/non-historical until authoritative Finish.

### D7. Longitudinal DXA / FRAX mechanisms

**OBSERVED**

The existing Module-01 UI also contains a longitudinal_review mechanism for:

- FRAX/FRAXplus snapshots and trends;
- prior DXA entries;
- BMD/T-score tables and charts;
- descriptive change only, with explicit caution against calling change significant without comparability/LSC.

prior-dxa-inline.js replaced prompt-only prior-DXA entry with an inline editor.

This is useful existing functionality, although its relationship to completed-encounter-derived longitudinal state requires deeper ownership review later.

### D8. G4 workspace ergonomics

**OBSERVED**

The patient summary and current flow can be independently collapsed/expanded. Patient-summary presentation is sticky, while the preference state lives only in sessionStorage and does not write clinical data.

### D9. Evidence/provenance surface

**OBSERVED**

G2 guidance contributions carry source references, strength/activation information and rule identity. The UI provides compact evidence metadata and explicitly labels checklist-only safety as requiring clinician confirmation rather than automatic clearance.

---

## E. Encounter/form/audit heritage still visible

These observations describe current product behavior; age alone is not treated as a defect.

### E1. Baseline Audit remains the Module-01 identity on entry

**OBSERVED**

The page title, subtitle and top banner still foreground:

- “Osteoporosis Cockpit — Baseline Audit v1”;
- “prospective encounter capture”;
- pilot case numbering;
- baseline lock / KPI language.

**Actual consequence:** the clinician is initially oriented toward a baseline encounter-capture/audit product even though the runtime now contains substantially richer longitudinal decision support.

### E2. Six-step form remains the primary work skeleton

**OBSERVED**

The patient summary and current flow sit above the same six fixed step tabs. Guidance changes which domains matter and reorders/cards highlights them, but the clinician still performs work inside the inherited Step 1–6 architecture.

**INFERRED consequence:** the product is state-aware, but the visible work model is still “navigate the encounter form” rather than “navigate the patient's trajectory/problem.”

### E3. “Case” and protected “patient encounter” coexist as two mental models

**OBSERVED**

The sidebar has New Case / Cases and a dialog for **Τοπικά drafts**, while the protected Patient Registry separately lists durable server encounters.

The working case is stored in localStorage as a cache and can be synchronized into protected encounter persistence.

**Actual consequence:** the current UI exposes both legacy local-case vocabulary and protected patient/encounter vocabulary. A clinician must understand which surface is temporary working state and which is protected history.

### E4. Privacy/pilot copy is not fully reconciled with clinical mode

**OBSERVED**

patient-registry.js rewrites the top privacy strip to say that authenticated clinical mode syncs patient data to the protected server and localStorage is a working cache.

However, the static Privacy dialog still says:

- prototype drafts remain in unencrypted localStorage;
- production use requires secure/private storage/authentication.

That dialog reflects earlier prototype-state language even though protected clinical persistence is currently mounted.

**R1 classification:** presentation/cognitive inconsistency, not evidence of a server privacy failure.

### E5. Audit-oriented navigation remains visible but inactive

**OBSERVED**

KPI Overview and Βιβλιοθήκη remain disabled sidebar entries.

**Actual consequence:** low-value/dead navigation occupies module-level orientation without providing current clinical work.

### E6. Longitudinal mechanisms were layered onto the encounter payload model

**OBSERVED**

Important longitudinal interpretation is derived from completed encounter payloads plus lab snapshots. Additional FRAX/DXA trend state exists within longitudinal_review in the working encounter payload.

**R1 boundary:** this layering is **not classified as a data-model defect here**. Whether the representation is sufficient for a 15-year clinical trajectory is explicitly an R2/R4 question.

---

## F. Navigation / information-architecture findings

### F1. Global Cockpit separation is materially correct

**OBSERVED**

The new Home correctly separates:

- Clinical Modules;
- Learning & Improvement;
- Clinic Utilities;
- Reception;
- Clinical Calendar summary.

The global Home explicitly describes general Clinic Utilities as shared rather than Osteoporosis navigation.

Top-level Heidi navigation is no longer present in the Osteoporosis sidebar; Heidi remains encounter/capture context.

### F2. Module-01 still dynamically re-injects one global Clinic Utility

**OBSERVED — current runtime fact**

static/baseline-audit/app.js still loads calendar-link.js.

calendar-link.js dynamically inserts into the Osteoporosis sidebar:

- Ημερολόγιο;
- **Παραπεμπτικό Φ/Θ** pointing to /clinical/clinic-utilities/physio-referral.

The Global Cockpit Home already has the physiotherapy referral as a global Clinic Utility.

Therefore the current composed UI still has a duplicate/global-tool entry inside Module 01 despite the newer global/module separation.

This is not inferred from old documentation; it follows directly from the current main source composition.

### F3. Current Cockpit regression coverage does not fully represent the composed Module-01 navigation

**OBSERVED**

test_cockpit_home.py checks:

- static/baseline-audit/index.html; and
- g4-workspace-ergonomics.js

for absence of global Clinic Utilities.

It does not inspect the later-loaded calendar-link.js injection that adds the physiotherapy referral.

**R1 implication:** the present navigation mismatch is a current product-surface fact. Test ownership/hardening belongs outside R1 implementation scope.

### F4. The calendar itself spans global and module contexts

**OBSERVED**

Clinical Calendar is present on the global Home and is also injected as an Osteoporosis sidebar shortcut.

R1 does not declare that shortcut wrong. It is a duplicated entry point whose product role should be distinguished from the clearly global Physio utility contamination.

### F5. Module identity still says “Osteoporosis Cockpit”

**OBSERVED**

The global product now calls Osteoporosis “Module 01”, but the module page itself still calls itself “Osteoporosis Cockpit.”

**INFERRED consequence:** the global/core vs module model is better separated structurally than it is communicated consistently inside Module 01.

### F6. Patient-first orientation is partial

**OBSERVED**

An active protected patient and longitudinal summary exist, but Module 01 opens into New Case / Baseline Audit rather than a dedicated patient trajectory surface.

The Patient Registry is an injected block inside that page, not the global Cockpit's persistent patient-level context.

---

## G. Cognitive / workflow observations

This section describes current interaction burden only; it is not an R3 redesign.

### G1. Useful information is surfaced early

**OBSERVED**

Before the step tabs, the clinician can see:

- patient longitudinal summary;
- current visit intent;
- protected-history availability;
- “Σημερινή ροή”;
- “Γιατί τώρα”;
- newly surfaced guidance;
- compact evidence metadata.

This materially reduces the need to manually inspect every step before understanding the current visit.

### G2. The screen still presents multiple overlapping product eras

**OBSERVED**

The clinician can simultaneously encounter:

- pilot/baseline audit framing;
- local Case/draft controls;
- protected Patient Registry;
- longitudinal summary;
- dynamic Visit Plan;
- six-step encounter capture;
- longitudinal DXA/FRAX charts;
- evidence-backed guidance.

**INFERRED consequence:** significant capability exists, but the clinician must understand more of the system's internal evolution/architecture than the Product Constitution's “complexity under the surface” principle would imply.

### G3. Full trajectory is visible only in fragments

**OBSERVED**

The current UI provides:

- a date/status encounter list;
- a snapshot-style patient summary;
- DXA/FRAX trend views;
- lab history;
- treatment/admin projection;
- last decision;
- unresolved items.

It does **not** provide a primary human life-course sequence showing major episodes, decisions, outcomes and future obligations as one coherent trajectory.

**R1 answer:** the trajectory is **partly visible, but still substantially reconstructable underneath**.

### G4. Current-flow items explain priority but do not themselves form a navigation layer

**OBSERVED**

The “Σημερινή ροή” list renders domain name + “Γιατί τώρα” + evidence metadata. The actual editable controls remain inside the step panels. Guidance reorders/highlights matching cards, but the summary items themselves are not a primary direct-navigation/decision surface.

**INFERRED consequence:** the clinician may still need to know which Step contains a surfaced domain.

### G5. Future obligations are visible but only partially actionable as a longitudinal concept

**OBSERVED**

- explicit due/overdue treatment timing can surface in current guidance;
- planned prior tasks can resurface;
- unresolved tasks/critical items appear in patient summary;
- Step 4 provides task editing and current close state.

There is no unified current obligation workspace with the full Product Constitution disposition vocabulary.

**R1 classification:** partial alignment/presentation + possible deeper-state question; not enough evidence in R1 to call the underlying persistence architecture defective.

### G6. Conflict and missing-history behavior are cognitively helpful

**OBSERVED**

The UI refuses to turn history-load failure into “no history,” and longitudinal conflicts are surfaced rather than silently reconciled.

This is a strong current safety/cognitive mechanism.

### G7. Existing Physio interaction mechanics are relevant reuse evidence, not a product model

**OBSERVED**

The current Knee-OA Physio UX/runtime demonstrates:

- progressive disclosure;
- direct manipulation;
- immediate downstream output updates;
- contextual dependent options;
- compact evidence on demand;
- restrained default surface;
- no mandatory Generate step.

These are proven interaction mechanics worth later reuse analysis. They do **not** establish that Osteoporosis should copy the Physio screen model or clinical logic.

---

## H. Preservation candidates

These are **R1 preservation candidates for later KEEP/ADAPT investigation**, not final R4 classifications.

### H1. Protected clinical-data substrate

**OBSERVED candidate**

- patient persistence;
- encounter persistence;
- laboratory snapshots;
- completed/amended finalization semantics;
- protected server APIs.

Reason for preservation investigation: they already support real longitudinal loading and authoritative Finish behavior.

### H2. Historical load / failure semantics

**OBSERVED candidate**

The explicit loaded/loading/unavailable distinction prevents missing history from being misread as negative history.

### H3. G1 deterministic longitudinal projection

**OBSERVED candidate**

Especially:

- completed/amended filtering;
- current-draft exclusion;
- actual administration deduplication;
- treatment snapshot projection;
- explicit conflict records;
- unresolved prior-task projection;
- explicit due-state handling.

### H4. EncounterContext → VisitPlan → “Γιατί τώρα”

**OBSERVED candidate**

This already converts historical/current state into a prioritized present workflow without automatically making the treatment decision.

### H5. G2 evidence/rule separation and provenance

**OBSERVED candidate**

Rule contributions, source refs, checklist-only safety labeling and non-automatic decision boundaries are useful reusable mechanics.

### H6. G3 longitudinal summary and salience

**OBSERVED candidate**

The read-only patient summary and “Νέο” mechanism already provide a compact current projection of history and live state.

### H7. G4 collapse/sticky mechanics

**OBSERVED candidate**

These are presentation primitives that reduce workspace density without owning clinical truth.

### H8. Laboratory history and trend mechanisms

**OBSERVED candidate**

Dedicated dated lab snapshots and Step-3 longitudinal display are useful current mechanisms.

### H9. Longitudinal DXA/FRAX analysis mechanics

**OBSERVED candidate**

Tables/charts, machine context and LSC/comparability caution provide reusable behavior, while ownership and state-source consolidation remain later questions.

### H10. Global Cockpit shell

**OBSERVED candidate**

The new Home has already separated most shared/global responsibilities from Module 01 and should be treated as existing product architecture, not ignored.

### H11. Selected Physio interaction primitives

**OBSERVED candidate**

Progressive disclosure, live dependent output and evidence-on-demand deserve explicit R4 reuse analysis as mechanics only.

---

## I. Potential deeper questions

These are **questions**, not R1-proven defects.

### I1. Encounter payloads vs durable semantic trajectory

**R2/R4 question**

Are completed/amended encounter payloads + projections sufficient for durable long-horizon concepts such as:

- treatment epochs;
- major clinical events;
- decisions;
- outcomes;
- unresolved uncertainty;
- future obligations?

R1 proves that meaningful projection exists; it does not prove whether that storage/model is sufficient for a 15-year trajectory.

### I2. Treatment episode representation

**R2 question**

The current projection uses the latest reliable treatment-episode snapshot plus actual administration history.

Is that enough to reconstruct starts/stops/switches/reasons/outcomes across many years without ambiguity?

### I3. Future obligation semantics

**R2 question**

Current tasks support planned / already_done / not_applicable, and due states can be derived/surfaced.

Does the target longitudinal model require more durable obligation identities and disposition semantics such as rescheduled, deferred, declined, superseded or alternative plan chosen?

### I4. Goals / target state

**R2 question**

Where, if anywhere, is the intended future clinical state durably represented rather than inferred from decision/tasks?

### I5. Overlapping longitudinal DXA/FRAX representations

**R2/R4 question**

How should the manually maintained longitudinal_review history relate to historical encounter-derived DXA/risk and dedicated laboratory snapshots?

R1 observes overlap; it does not choose a consolidation design.

### I6. Shared patient context ownership

**R4 question**

Should active patient identity/context remain a Module-01-injected registry/session concept or move into a reusable Core patient context for future modules?

### I7. Historical decision/evidence context

**R2/R4 question**

Current G2 exposes today's evidence context. What is required to preserve “what was known then” alongside “what is known now” for historical decision review?

### I8. Life-course human projection

**R3/R4 question**

What interaction projection can expose clinically important sequence/episodes without forcing clinicians to load multiple encounters or understand internal storage structure?

R1 only establishes that the current primary projection is summary-plus-form, not a coherent life-course view.

### I9. PR-1 / future capture integration

**R4 question**

How should provider-agnostic candidate capture later feed the same patient/trajectory/state surfaces without creating a parallel truth path?

No PR-1 change is authorized or proposed here.

---

## J. Dependency register

### DEP-R1-01 — Root PR-1 writer

**OBSERVED / authoritative boundary**

Root CURRENT_OPERATIONAL.md remains owned by PR-1 Heidi-first transcript capture.

R1 made no root operational mutation.

### DEP-R1-02 — Fracture / fragility semantics

**UNVERIFIED within OST-UI authority**

R1 observes how fracture fields/events affect current UI/projection. It does not decide whether any fracture is clinically a fragility fracture or correct clinical semantics.

Likely authoritative owner: the parallel LifeCourse/clinical semantic lifecycle identified by the governing OST-UI contract.

### DEP-R1-03 — Treatment / guideline semantics

**OBSERVED ownership boundary**

Current G2 reviewed evidence contracts own medical rule meaning. R1 audits projection/interaction only and does not redefine treatment thresholds, sequencing or medication safety.

### DEP-R1-04 — Product Constitution status

**OBSERVED**

PR #121 remains draft/unmerged. Product Constitution v0.2 is supporting intent only.

### DEP-R1-05 — Shared Core ownership

**UNVERIFIED / R4**

R1 can observe that patient context currently lives inside Module 01 and that the Global Cockpit exists. It cannot assign future Core ownership.

### DEP-R1-06 — Physio ownership

**OBSERVED**

Physio is a separate product/workstream. Only its proven interaction mechanics were inspected as reuse evidence.

---

## K. R1 disposition

### K1. Explicit answers to the twelve required R1 questions

**1. What product does the clinician actually encounter today?**

**OBSERVED / INFERRED:** A global Clinical Excellence Home leading to an Osteoporosis Module that is a **hybrid encounter/audit workspace plus protected patient registry plus genuine longitudinal decision-support layer**. The module is no longer merely a baseline form, but it is still visibly organized around that form.

**2. What parts already behave longitudinally?**

**OBSERVED:** protected patient/encounter/lab persistence; completed/amended history loading; treatment/admin projection; unresolved prior tasks; explicit due states; DXA/lab history; G3 patient summary; history-sensitive G1/G2 guidance; conflict preservation.

**3. What parts still behave primarily as encounter/form/audit surfaces?**

**OBSERVED:** module identity, pilot/baseline banners, New Case/Cases, six step tabs, step-specific capture, progress/applicability, disabled KPI/Library heritage and local working-case model.

**4. Does the current UI make the patient trajectory visible or merely reconstructable underneath?**

**OBSERVED / INFERRED:** **partly visible, substantially reconstructable underneath.** A useful snapshot summary and trends exist, but no primary clinical life-course/episode sequence presents the patient's whole trajectory.

**5. Does history materially change current guidance/state?**

**OBSERVED: yes.** Prior completed/amended facts change active treatment context, due states, unresolved tasks, conflicts, evidence rules and today's Visit Plan/why-now behavior.

**6. Are future obligations visible and actionable today?**

**OBSERVED:** partially. Prior planned tasks and explicit due states can resurface and be edited through Step 4; unresolved items are visible in summary. They are not yet a unified longitudinal obligation/disposition workspace.

**7. Which global Cockpit responsibilities have already been separated correctly from Osteoporosis?**

**OBSERVED:** Global Home, Learning & Improvement, global Clinic Utilities, Reception and global calendar summary are now structurally outside Module 01; top-level Heidi navigation is no longer in the module. The separation is not complete because calendar-link.js re-injects the Physio referral into the Module-01 sidebar.

**8. What current functionality is useful but poorly exposed?**

**OBSERVED / INFERRED:** the protected longitudinal projection, conflict semantics, actual-administration history, prior tasks, decision history and DXA/FRAX trend mechanics are useful but spread across patient registry, summaries and separate encounter steps rather than exposed as one patient trajectory.

**9. What visible surfaces appear obsolete, duplicated or weakly aligned with current product semantics?**

**OBSERVED:** Baseline Audit/pilot framing, disabled KPI Overview/Library, local Cases mental model beside protected encounters, stale prototype/privacy wording, Module-01 title as “Osteoporosis Cockpit,” and duplicate Physio navigation through calendar-link.js.

**10. Which apparent problems are only presentation problems versus possible deeper product/state problems?**

**Presentation proven by R1:** baseline/pilot wording, stale privacy dialog, dead sidebar entries, duplicate Physio link, module naming, fragmented placement of already-available capabilities.

**Potential deeper questions only:** encounter-payload sufficiency, treatment-epoch durability, obligation identity/dispositions, shared patient context, overlapping DXA/FRAX representations, historical evidence clocks. These are routed to R2/R4 and are not called defects here.

**11. What mechanisms clearly deserve preservation/reuse investigation in R4?**

**OBSERVED:** protected persistence/finalization, fail-closed history loading, G1 projection, actual-vs-scheduled administration semantics, VisitPlan/why-now, G2 evidence/provenance, G3 summary/salience, G4 workspace primitives, lab history, longitudinal trend mechanics, Global Cockpit shell and selected Physio interaction primitives.

**12. What questions cannot be answered by R1 and must be routed to R2/R3/R4?**

- R2: durable life-course adequacy, treatment epochs, obligations, goals, decision/evidence history.
- R3: best point-of-care interaction model over the proven state and journey requirements.
- R4: shared Core ownership, state architecture/reuse, projection consolidation, Live Copilot compatibility.
- Clinical owners: fracture/fragility and treatment-rule semantics.

### K2. Current-state conclusion

**INFERRED from the observed runtime**

The current Osteoporosis product is **not merely a collection of pages or a Baseline Audit form anymore**.

It already contains a meaningful longitudinal clinical substrate and history-sensitive decision-support system.

However, the clinician-facing Module-01 product still presents that substrate **through an encounter/audit-first shell**, so the Product Constitution's core idea — the patient trajectory as the primary product unit — is only **partially expressed in the visible product today**.

R1 does not decide whether this calls for incremental change, substantial restructuring or deeper reconstruction. That decision remains downstream of R2-R4 and synthesis.

### K3. Disposition

**R1 COMPLETE / R2 ELIGIBLE FOR COORDINATOR RECONCILIATION**

No coordinator REPLAN is required from R1.

R2 is **not started by this review**.
