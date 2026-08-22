# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-22 18:18 Asia/Nicosia
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

Encounter applicability is driven independently by:

```text
PATIENT RELATIONSHIP
new_to_service | established_patient

ENCOUNTER ARCHETYPE
initial new/uncertain diagnosis | initial known disease | routine follow-up |
treatment start | continuation/due monitoring | change/transition |
post-fragility fracture | fracture on treatment | adverse effect/intolerance |
treatment completion/consolidation | other
```

---

## 3. Implemented baseline UI

### Step 1 — encounter context
Implemented and live. Captures relationship/archetype, demographics, anthropometrics/BMI/height loss, fracture recency, parental hip fracture, GC dose/duration, falls count, secondary conditions, frailty/immobility, sarcopenia trigger and Heidi exposure metadata.

### Step 2 — fracture history and formal risk
Implemented and live. Captures structured fracture events, FRAX/FRAXplus/other formal assessment, country/surrogate model, MOF/hip probabilities, FN-BMD use, declared framework and contextual adjustment/override.

### Step 3 — examinations/results
Implemented and live. Captures DXA BMD/T-scores, machine/comparability/LSC, VFA/vertebral imaging, secondary-cause process, optional labs/BTMs, falls/frailty/function and conditional sarcopenia testing.

### Step 4 — treatment, decision and follow-up
Implemented, merged and deployed via auto-deploy. Captures date-aware treatment episodes, scheduled/actual administration events, adherence/tolerance/response, current clinical decision and rationale, sequencing/exit/consolidation safety, patient preference, follow-up CareTask precursors and unresolved critical items.

---

## 4. Longitudinal review refinement — implemented

Merged PR #6:

```text
2387bd0ff816c499854bcb1ce152e441123923e9
```

Schema:

```text
schemas/longitudinal_risk_dxa_review_v1.yaml
```

Runtime files:

```text
static/baseline-audit/longitudinal.js
static/baseline-audit/longitudinal.css
```

### Step 2 refinements
- MOF risk category and hip risk category are now separate fields.
- Overall management category is distinct from site-specific risk categories.
- Raw FRAX MOF/hip probabilities remain visible.
- FRAXplus adjusted MOF/hip outputs can be entered separately.
- FRAXplus adjustment context includes fracture recency, higher GC exposure, TBS, falls, T2DM duration, lumbar-spine BMD, HAL, primary hyperparathyroidism, number of prior fractures and other.
- The app does **not** reproduce or stack FRAXplus adjustment algorithms locally; it records externally calculated FRAXplus outputs.
- FRAX history can be displayed in a chronological table plus raw-vs-adjusted MOF and hip trend charts.
- A read-only DXA longitudinal overview is visible from Step 2.

### Step 3 refinements
- DXA machine input is upgraded to a dropdown with common platforms plus optional local machine label.
- Prior DXA snapshots can be stored for the current patient context.
- Longitudinal DXA is shown as a table plus BMD and T-score trend charts.
- Descriptive BMD percent change is shown, but significance is not asserted without machine comparability and facility LSC.

Longitudinal data are stored under `longitudinal_review` in the same browser-local prototype case object. Current encounter values remain the source of truth; historical series supplement them.

---

## 5. Migration principle from legacy Cockpit

The new interface is not a field-for-field copy:

```text
OLD COCKPIT DATA
→ classify useful / duplicate / outdated / context-specific
→ preserve useful structured data
→ add provenance, timing and applicability
→ connect to encounter archetype
→ map to KPI / audit / learning / improvement loops
```

Examples: T-scores are retained but longitudinal DXA gains BMD/LSC/machine comparability; fall history becomes 12-month count + outpatient function; treatment duration becomes episodes + administrations + due dates; hospital-specific Morse fields are not promoted by default.

---

## 6. Current next action

**NEXT: Step 5 — Communication + immediate post-visit reflection.**

Then:

```text
Step 6 — documentation trace + final Heidi/capture-source review
→ exact field→KPI calculation contract
→ run 5-case pilot
→ one refinement
→ freeze
→ 30-case scored baseline
```

Separate later instruments: Patient Voice 4-question instrument and Decision Quality 10-case review form.

---

## 7. Stop boundary

Do not yet:
- major-rewrite legacy `main.py` / `index.html`;
- create a composite Clinical Excellence score from invented data;
- display live KPI coaching during scored baseline except safety-critical alerts;
- treat GeSY as complete clinical truth;
- make Heidi mandatory before baseline;
- commit identifiable patient data, GeSY content or Heidi transcripts to the public repo;
- treat browser `localStorage` as production clinical-data storage.
