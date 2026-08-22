# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-22 20:56 Asia/Nicosia
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

Encounter applicability remains driven by patient relationship plus encounter archetype.

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

The first fix changed full replacement to merge-by-`internal_uuid`, but a second review correctly identified that `currentCase` could still carry stale module slices loaded earlier. Those stale slices could overwrite fresher `longitudinal_review` data during a core Save/Finish.

Root rule is now:

```text
app-core owns Steps 1–2 fields only
module-owned slices are excluded from the core save payload
```

Excluded keys:

```text
step3
step4
step5
step6
longitudinal_review
pilot_completion
audit_evaluation_v1
```

Therefore a Steps 1–2 save can no longer overwrite fresher module-owned state.

### Patch 1 — hidden dependent values / stale data
Closed and additionally hardened after second review.

`static/baseline-audit/data-hygiene.js` still clears hidden dependent values before persistence and sanitizes legacy stale localStorage data.

Additional hardening:

- `data-hygiene.js` now loads before `longitudinal.js`, so initial longitudinal render sees sanitized stored state;
- `longitudinal.currentDxaPoint()` now explicitly checks `DXA used`; when it is not `yes`, the current DXA point is empty and cannot enter charts/tables even if stale values somehow exist.

### Patch 4 — Step 1 ↔ Step 3 duplicate source of truth
Closed and verified after second review.

Canonical source remains Step 1 `risk_context` for falls count, CFS, cognition, immobility and basic sarcopenia screening; Step 3 displays those shared values as read-only projections while retaining Step-3-only functional detail.

Current implementation remains in `static/baseline-audit/shared-risk-source.js`. It is functionally correct; do not introduce a second writer. A later cleanup may move this ownership directly into Step 3, but that refactor is not required for pilot correctness unless smoke testing identifies a regression.

### Patch 5 — DXA machine field persistence / normalization
Closed and merged.

`static/baseline-audit/dxa-machine-select.js` normalizes `#s3DxaMachine` to a select before the longitudinal layer initializes and persists optional `machine_label`.

Supported normalized values:

```text
hologic_horizon
hologic_discovery
ge_lunar_idxa
ge_lunar_prodigy
norland
other_unknown
```

Legacy free-text machine values are migrated without silent loss: recognized labels map to normalized values; unrecognized text maps to `other_unknown` and is preserved in `machine_label`.

The second review confirmed this runtime implementation is active. A future owner-module cleanup may move the select directly into `step3.js`, but this is technical cleanup rather than an unresolved pilot data-integrity defect.

---

## 5. Current exact next action

**NEXT: continue pre-pilot hardening before the 5 real pilot encounters.**

Highest-priority remaining item:

```text
Patch 7 — implement real archetype-driven applicability
```

Then address the remaining usability/correctness items:

```text
Patch 3 — whole-form progress calculation
Patch 6 — inline prior-DXA entry + escaping
Patch 8 — BMI derived/manual behavior
```

After those, perform an explicit save/reload/complete smoke test including:

```text
FRAX/FRAXplus edit → Save → reload → values retained
DXA yes + values → DXA no → Save → no current DXA in trends
Step 1 shared risk values → Step 3 mirror → no divergence
DXA machine + machine_label → Save/reload → retained
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
