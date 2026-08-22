# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-22 21:58 Asia/Nicosia
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
→ explicit smoke test
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

### Patch 1 — hidden dependent values / stale data
Closed and hardened. `data-hygiene.js` clears hidden dependent values before persistence; `longitudinal.currentDxaPoint()` independently refuses to expose a current DXA point unless `DXA used == yes`.

### Patch 2 — core save overwrite / stale-module bug
Closed after second review. `app-core` excludes module-owned slices from its save payload: `step3`, `step4`, `step5`, `step6`, `longitudinal_review`, `pilot_completion`, `audit_evaluation_v1`.

### Patch 3 — whole-form progress
Closed and merged via PR #16. `whole-form-progress.js` owns the user-visible capture-completion percentage across Steps 1–6 and excludes Patch-7-collapsed domains from the denominator until reopened. It is not a KPI/performance score.

### Patch 4 — Step 1 ↔ Step 3 duplicate source of truth
Closed. Step 1 `risk_context` remains canonical for falls count, CFS, cognition, immobility and basic sarcopenia screening. Step 3 is a read-only projection for those shared fields plus Step-3-specific detail.

### Patch 5 — DXA machine persistence / normalization
Closed. `dxa-machine-select.js` normalizes the current DXA machine field and persists optional `machine_label`; legacy free text is preserved under `other_unknown` rather than silently lost.

### Patch 6 — prior DXA inline entry / longitudinal safety
Closed and merged via PR #17.

`prior-dxa-inline.js`:
- replaces the seven-prompt Prior DXA flow with one inline editor;
- uses typed date/numeric inputs and a machine whitelist;
- normalizes legacy historical DXA dates before longitudinal rendering;
- preserves unknown legacy machine text in `machine_label`;
- assigns stable `_id` values to historical DXA rows;
- deletes historical DXA rows by stable `_id` rather than sorted display index;
- reloads longitudinal private state after add/remove so fresh history cannot be overwritten.

### Patch 7 — archetype-driven adaptive applicability
Closed and merged via PR #15. `adaptive-applicability.js` maps encounter archetypes to applicable/conditional/usual-N/A domain defaults, supports explicit `Χρήση σήμερα` override, and keeps Step 6 applicable for all pilot encounters without KPI coaching.

### Patch 8 — BMI derived/manual behavior
Implemented on branch `fix/patch8-bmi-derived-behavior`.

New runtime:

```text
static/baseline-audit/bmi-behavior.js
```

Rule:

```text
valid weight + valid current height → BMI is derived and read-only
otherwise → BMI is editable as manual/external input
```

Behavior:
- prevents a manual BMI entry from being silently overwritten while the field still appears editable;
- marks the field `readOnly` whenever both weight and height are available, matching `app-core`'s existing calculated BMI source;
- shows a compact source note (`Αυτόματο από βάρος + ύψος` vs `Χειροκίνητο / external BMI`);
- when a previously derived BMI loses one of its source measurements, clears the stale derived value before returning the field to manual/external mode;
- preserves manual/external BMI when no valid weight-height pair exists.

---

## 5. Current exact next action

**NEXT: explicit pre-pilot smoke test. Do not add new functionality unless the smoke test identifies a defect.**

Run one synthetic/non-identifiable test case through:

```text
1. FRAX/FRAXplus edit → Save → reload → values retained
2. DXA yes + values → DXA no → Save → no current DXA in trends
3. Step 1 shared risk values → Step 3 mirror → no divergence
4. DXA machine + machine_label → Save/reload → retained
5. archetype change → expected/conditional/N/A card state updates correctly
6. collapsed domain → Χρήση σήμερα → override persists after Save/reload
7. whole-form progress changes across Steps 1–6 and excludes collapsed domains
8. Prior DXA → inline add → Save/reload → retained
9. historical DXA delete after date sorting → correct row removed
10. BMI with weight+height → read-only calculated; remove one source → derived BMI clears and manual mode returns
11. Finish Visit → all module slices retained
```

If the smoke test passes, start **Pilot Case 1/5**. Do not revise the form after each pilot case unless there is a safety-critical or data-loss defect.

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
- start the real pilot until the explicit smoke test passes.
