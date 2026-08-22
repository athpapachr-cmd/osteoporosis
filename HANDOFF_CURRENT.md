# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-22 21:46 Asia/Nicosia
> **Canonical repository:** `athpapachr-cmd/osteoporosis`
> **Current major phase:** prospective Osteoporosis Baseline/Audit pre-pilot hardening
> **Current module:** Module 01 — Osteoporosis

This file contains current operational truth only. Permanent rules belong in `AGENTS.md`; roadmap in `TODO.md`; completed history in `osteoporosis-change-log.md`.

---

## 1. Project / baseline strategy

The project is a **Personal Clinical Excellence System** with a reusable Core Engine. Osteoporosis is Module 01 and the proving module.

Prospective baseline sequence:

```text
pre-pilot data-integrity / applicability hardening
→ 5 consecutive pilot encounters
→ one usability/branching/calculation-contract refinement
→ freeze form + KPI applicability
→ 30 consecutive unique scored baseline patients
→ baseline lock
→ interventions / re-audit
```

During scored baseline: no KPI coaching/red-green performance feedback; safety-critical alerts may remain active. Clinical process, formal documentation and capture quality are separate measurement axes.

Encounter applicability is driven primarily by encounter archetype, with explicit clinician override when a normally collapsed domain is relevant to the individual encounter.

---

## 2. Baseline UI status

Steps 1–6 are implemented and merged:

1. Encounter context.
2. Fracture history + formal risk, including FRAX/FRAXplus and separate MOF/hip risk.
3. DXA/VFA, secondary causes/labs, falls/frailty/function and sarcopenia.
4. Treatment episodes, administrations, clinical decision, sequencing safety and follow-up tasks.
5. Encounter-specific communication, understanding/teach-back and immediate reflection.
6. Documentation trace, final Heidi review, capture sources and capture quality.

Longitudinal FRAX/DXA tables and charts are also implemented.

Frozen interpretation rule:

```text
clinical process = Steps 1–5
formal documentation = separate evidence axis
Heidi = supplementary clinician-reviewed capture source
```

Absent/partial GeSY documentation must not be silently converted into a clinical omission. Heidi use itself is not a quality-success metric.

Prototype persistence remains browser `localStorage`; no identifiable clinical data belong in the public repo.

---

## 3. KPI calculation contract — defined for pilot

Schema:

```text
schemas/baseline_kpi_calculation_contract_v1.yaml
```

Status model:

```text
applicability = applicable | not_applicable | uncertain
status = met | not_met | indeterminate | manual_review_required | external_pending | not_applicable
```

Rules:

- clinical-process KPIs use Steps 1–5;
- Step 6 documentation remains a separate evidence/capture axis;
- missing/uncertain required evidence never counts as met;
- KPI-12 transition safety and KPI-13 fracture-on-treatment require manual clinical review when applicable;
- KPI-14/15 remain external-pending until Patient Voice is activated;
- no KPI result is shown in the baseline UI before baseline lock except future safety-critical alert logic;
- pilot calculations are for mapping validation, not clinician feedback.

---

## 4. Pre-pilot hardening status

### Patch 2 — core save overwrite / stale-module bug
Closed after second review.

Root rule:

```text
app-core owns Steps 1–2 fields only
module-owned slices are excluded from the core save payload
```

Excluded keys: `step3`, `step4`, `step5`, `step6`, `longitudinal_review`, `pilot_completion`, `audit_evaluation_v1`.

### Patch 1 — hidden dependent values / stale data
Closed and hardened.

`data-hygiene.js` clears hidden dependent values before persistence and sanitizes legacy stale localStorage data. `longitudinal.currentDxaPoint()` independently refuses to expose a current DXA point unless `DXA used == yes`.

### Patch 4 — Step 1 ↔ Step 3 duplicate source of truth
Closed.

Step 1 `risk_context` remains canonical for falls count, CFS, cognition, immobility and basic sarcopenia screening. Step 3 displays those values as read-only projections and keeps only Step-3-specific detail editable.

### Patch 5 — DXA machine field persistence / normalization
Closed.

`dxa-machine-select.js` normalizes the current DXA machine field and persists optional `machine_label`; legacy free-text values are preserved under `other_unknown` rather than silently lost.

### Patch 7 — archetype-driven adaptive applicability
Closed and merged via PR #15.

`adaptive-applicability.js` maps encounter archetypes to `applicable`, `uncertain`, or `not_applicable` domain defaults, collapses conditional/usual-N/A cards, supports explicit `Χρήση σήμερα` override, and keeps Step 6 applicable for all pilot encounters. No KPI/performance coaching is displayed.

### Patch 3 — whole-form progress
Closed and merged via PR #16.

`whole-form-progress.js` owns the user-visible **capture-completion** percentage across Steps 1–6 after bootstrap. It excludes Patch-7-collapsed domains from the denominator until reopened and is explicitly not a KPI/performance score.

### Patch 6 — prior DXA inline entry / longitudinal safety
Implemented on branch `fix/patch6-inline-prior-dxa`.

New runtime:

```text
static/baseline-audit/prior-dxa-inline.js
```

Behavior:

- intercepts the legacy `＋ Prior DXA` action before the old prompt chain can run;
- replaces seven sequential browser prompts with one inline editor for date, machine/local label, BMD and T-scores;
- uses typed `date` / numeric inputs and a machine whitelist;
- normalizes legacy DXA-history dates to strict `YYYY-MM-DD` before the longitudinal renderer runs, preventing raw historical date strings from entering the old unescaped table path;
- normalizes unknown historical machine values to `other_unknown` while preserving the original text as `machine_label`;
- assigns stable `_id` values to historical DXA rows;
- captures removal by stable `_id`, fixing the prior sorted-display-index versus source-array-index deletion risk;
- forces the longitudinal module to reload from localStorage after add/remove so its private state cannot overwrite the fresh history.

The module is loaded before `longitudinal.js`, so legacy history is normalized before the first table/chart render.

---

## 5. Current exact next action

**NEXT: finish the last pre-pilot usability/correctness item.**

```text
Patch 8 — BMI derived/manual behavior
```

Then perform an explicit save/reload/complete smoke test including:

```text
FRAX/FRAXplus edit → Save → reload → values retained
DXA yes + values → DXA no → Save → no current DXA in trends
Step 1 shared risk values → Step 3 mirror → no divergence
DXA machine + machine_label → Save/reload → retained
archetype change → expected/conditional/N/A card state updates correctly
collapsed domain → Χρήση σήμερα → override persists after Save/reload
whole-form progress changes across Steps 1–6 and excludes collapsed domains
Prior DXA → inline add → Save/reload → retained
historical DXA delete after date sorting → correct row removed
Finish Visit → all module slices retained
```

Only then start Pilot Case 1/5.

After the 5 pilot cases:

```text
review pilot evidence once
→ make one deliberate refinement
→ freeze Baseline Form v1 + KPI calculation/applicability contract
→ start 30-case scored baseline
```

Do not start the 30-case scored baseline before that freeze.

---

## 6. Separate later instruments

Still separate from the 5-case form pilot:

- Patient Voice 4-question instrument.
- Decision Quality 10-case review form.

---

## 7. Stop boundary

Do not yet:
- major-rewrite legacy `main.py` / `index.html`;
- create a composite Clinical Excellence score from invented data;
- display live KPI coaching during scored baseline except safety-critical alerts;
- treat GeSY as complete clinical truth;
- make Heidi mandatory before baseline;
- commit identifiable patient data, GeSY content or Heidi transcripts to the public repo;
- treat browser `localStorage` as production clinical-data storage;
- start the real pilot until the identified pre-pilot data-integrity/applicability blockers are closed and smoke-tested.
