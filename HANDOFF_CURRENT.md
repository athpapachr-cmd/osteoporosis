# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-22 18:29 Asia/Nicosia
> **Canonical repository:** `athpapachr-cmd/osteoporosis`
> **Current major phase:** prospective Osteoporosis Baseline/Audit implementation
> **Current module:** Module 01 — Osteoporosis

This file contains current operational truth only. Permanent rules belong in `AGENTS.md`; roadmap in `TODO.md`; completed history in `osteoporosis-change-log.md`.

---

## 1. Project definition

The project is a **Personal Clinical Excellence System** with a reusable Core Engine. The legacy Osteoporosis Cockpit is the point-of-care Clinical Practice / Encounter Execution layer of Module 01.

Canonical loop:

```text
STANDARD → LEARN → TEST/MASTER → APPLY → MEASURE → AUDIT
→ GAP OR STRENGTH → INTERVENE/REINFORCE → RE-MEASURE → SYSTEM LEARNS
```

Osteoporosis is the proving module. Future musculoskeletal modules remain deferred.

---

## 2. Baseline strategy

There is no reliable historical osteoporosis registry/dedicated folder, and GeSY notes may not reflect the full consultation. Main baseline remains prospective:

```text
5 consecutive pilot encounters
→ one usability/branching refinement
→ freeze form + KPI applicability
→ 30 consecutive unique scored baseline patients
→ baseline lock
→ interventions / re-audit
```

During scored baseline: no KPI coaching/red-green performance feedback; safety-critical alerts may remain active. Clinical process, formal documentation and capture quality remain separate axes.

Encounter applicability is driven independently by patient relationship and encounter archetype.

---

## 3. Implemented baseline UI

### Step 1 — encounter context
Implemented and live. Captures relationship/archetype, demographics, anthropometrics/BMI/height loss, fracture recency, parental hip fracture, GC dose/duration, falls count, secondary conditions, frailty/immobility, sarcopenia trigger and Heidi exposure metadata.

### Step 2 — fracture history and formal risk
Implemented and live. Captures structured fracture events, FRAX/FRAXplus/other formal assessment, country/surrogate model, MOF/hip probabilities, FN-BMD use, declared framework and contextual adjustment/override.

### Step 3 — examinations/results
Implemented and live. Captures DXA BMD/T-scores, machine/comparability/LSC, VFA/vertebral imaging, secondary-cause process, optional labs/BTMs, falls/frailty/function and conditional sarcopenia testing.

### Step 4 — treatment, decision and follow-up
Implemented, merged and deployed. Captures date-aware treatment episodes, scheduled/actual administration events, adherence/tolerance/response, current clinical decision and rationale, sequencing/exit/consolidation safety, patient preference, follow-up CareTask precursors and unresolved critical items.

### Longitudinal review refinement
Implemented and merged. Adds separate MOF and hip risk categories, FRAXplus adjusted outputs and modifier context, longitudinal FRAX tables/charts, DXA machine dropdown, historical DXA snapshots, BMD/T-score tables and charts, and descriptive BMD change with LSC/comparability caution.

### Step 5 — communication + immediate post-visit reflection
Implemented on active branch `feat/baseline-step5-communication`.

Schema:

```text
schemas/baseline_step5_communication_v1.yaml
```

Runtime files:

```text
static/baseline-audit/step5.js
static/baseline-audit/step5.css
static/baseline-audit/app.js   # loads Step 5 after longitudinal module
```

Step 5 captures:

- condition / fracture-risk explanation;
- results/current-status explanation;
- exercise and physical-activity discussion;
- nutrition, calcium, vitamin D and other supplements;
- medication or no-drug plan;
- treatment rationale;
- alternatives/trade-offs;
- duration/timing/review point;
- material safety messages;
- missed-dose/timing message when relevant;
- sequencing/transition explanation when relevant;
- patient questions and preferences;
- clinician impression of understanding of condition/plan/rationale;
- teach-back and detected/corrected misunderstanding;
- written/digital information provided;
- compact post-visit reflection and potential Signals.

Communication remains archetype-specific conceptually: new/initial encounters are not scored like stable follow-up, and treatment-start/change/transition visits have different expectations from routine continuation.

Clinician impression of understanding is explicitly **not** Patient Voice. The later Patient Voice instrument remains the patient's own report and is kept separate from clinician-side capture.

During baseline Step 5 shows no live KPI score, red/green completion state or missing-item verdict.

---

## 4. Migration principle from legacy Cockpit

The new interface is not a field-for-field copy:

```text
OLD COCKPIT DATA
→ classify useful / duplicate / outdated / context-specific
→ preserve useful structured data
→ add provenance, timing and applicability
→ connect to encounter archetype
→ map to KPI / audit / learning / improvement loops
```

Current encounter values remain the source of truth. Longitudinal and audit layers supplement rather than create competing records.

---

## 5. Current next action

**NEXT: Step 6 — documentation trace + final Heidi/capture-source review.**

Step 6 should finalize the separation between:

```text
what happened clinically
what is traceable in the formal GeSY record
what Heidi captured / clinician reviewed
what remains uncertain or missing from the evidence trail
```

Then:

```text
exact field→KPI calculation contract
→ run 5-case pilot
→ one refinement
→ freeze
→ 30-case scored baseline
```

Separate later instruments: Patient Voice 4-question instrument and Decision Quality 10-case review form.

---

## 6. Stop boundary

Do not yet:
- major-rewrite legacy `main.py` / `index.html`;
- create a composite Clinical Excellence score from invented data;
- display live KPI coaching during scored baseline except safety-critical alerts;
- treat GeSY as complete clinical truth;
- make Heidi mandatory before baseline;
- commit identifiable patient data, GeSY content or Heidi transcripts to the public repo;
- treat browser `localStorage` as production clinical-data storage.
