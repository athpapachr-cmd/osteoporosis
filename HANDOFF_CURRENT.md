# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-22 18:47 Asia/Nicosia
> **Canonical repository:** `athpapachr-cmd/osteoporosis`
> **Current major phase:** prospective Osteoporosis Baseline/Audit implementation
> **Current module:** Module 01 — Osteoporosis

This file contains current operational truth only. Permanent rules belong in `AGENTS.md`; roadmap in `TODO.md`; completed history in `osteoporosis-change-log.md`.

---

## 1. Project / baseline strategy

The project is a **Personal Clinical Excellence System** with a reusable Core Engine. Osteoporosis is Module 01 and the proving module.

Prospective baseline sequence remains:

```text
5 consecutive pilot encounters
→ one usability/branching refinement
→ freeze form + KPI applicability
→ 30 consecutive unique scored baseline patients
→ baseline lock
→ interventions / re-audit
```

During scored baseline: no KPI coaching/red-green performance feedback; safety-critical alerts may remain active. Clinical process, formal documentation and capture quality are separate measurement axes.

Encounter applicability remains driven by patient relationship plus encounter archetype.

---

## 2. Baseline UI status

### Step 1 — encounter context
Implemented, merged and live.

### Step 2 — fracture history + formal risk
Implemented, merged and live. Includes FRAX/FRAXplus context, MOF/hip probabilities and explicit framework.

### Step 3 — examinations/results
Implemented, merged and live. Includes DXA BMD/T-scores, machine/comparability/LSC, VFA, secondary causes/labs, falls/frailty/function and sarcopenia.

### Step 4 — treatment, decision and follow-up
Implemented, merged and deployed. Includes treatment episodes, administrations/due dates, adherence/tolerance/response, current decision/rationale, sequencing safety, CareTask precursors and unresolved critical items.

### Longitudinal review
Implemented and merged. Includes separate MOF/hip risk categories, FRAXplus adjusted outputs/modifier context, FRAX trend tables/charts, DXA machine dropdown, historical DXA snapshots and BMD/T-score trend tables/charts.

### Step 5 — communication + immediate post-visit reflection
Implemented, merged and deployed. Captures encounter-specific communication, clinician-side understanding/teach-back, patient questions/preferences and compact reflection/potential Signals. Clinician impression remains separate from future Patient Voice.

### Step 6 — documentation trace + final Heidi/capture-source review
Implemented and merged via PR #8.

Schema:

```text
schemas/baseline_step6_documentation_v1.yaml
```

Runtime:

```text
static/baseline-audit/step6.js
static/baseline-audit/step6.css
static/baseline-audit/app.js
```

Step 6 captures:

- capture-source provenance;
- domain-level formal GeSY trace;
- domain-level Heidi trace;
- material discrepancies between evidence sources;
- formal-record completeness/missing-content domains;
- final Heidi review seeded from Step 1 without raw/corrected transcript or manual diff;
- whether a clinician-approved Heidi note exists and whether Heidi content entered the formal record;
- capture reliability, major information gaps and reasons for limited capture;
- optional completion time;
- readiness for later audit calculation.

Frozen interpretation rule:

```text
clinical process = Steps 1–5
formal documentation = separate evidence axis
Heidi = supplementary clinician-reviewed capture source
```

Absent/partial GeSY documentation must not be silently converted into a clinical omission. Heidi use itself is not a quality-success metric.

Prototype persistence remains browser `localStorage`; no identifiable clinical data belong in the public repo.

---

## 3. Migration principle

The new interface is not a field-for-field copy of the legacy Cockpit:

```text
OLD COCKPIT DATA
→ classify useful / duplicate / outdated / context-specific
→ preserve useful structured data
→ add provenance, timing and applicability
→ connect to encounter archetype
→ map to KPI / audit / learning / improvement loops
```

Current encounter values remain the source of truth; longitudinal and audit layers supplement rather than create competing records.

---

## 4. Current next action

**NEXT: define the exact form-field → KPI status calculation contract.**

Then:

```text
run 5-case pilot
→ measure completion time / friction / missing fields
→ one refinement
→ freeze Baseline Form v1 + KPI applicability
→ start 30-case scored baseline
```

Separate later instruments remain: Patient Voice 4-question instrument and Decision Quality 10-case review form.

---

## 5. Stop boundary

Do not yet:
- major-rewrite legacy `main.py` / `index.html`;
- create a composite Clinical Excellence score from invented data;
- display live KPI coaching during scored baseline except safety-critical alerts;
- treat GeSY as complete clinical truth;
- make Heidi mandatory before baseline;
- commit identifiable patient data, GeSY content or Heidi transcripts to the public repo;
- treat browser `localStorage` as production clinical-data storage.
