# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-22 20:13 Asia/Nicosia
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

### Patch 2 — core save overwrite bug
Closed and merged.

`app-core.js` now merges current Steps 1–2 state into the existing stored case by `internal_uuid` rather than replacing the complete object. The former asynchronous capture/restore workaround in `pilot-completion.js` was removed.

### Patch 1 — hidden dependent values / stale data
Closed and merged.

`static/baseline-audit/data-hygiene.js` clears hidden dependent values before persistence and sanitizes legacy stale localStorage data.

### Patch 4 — Step 1 ↔ Step 3 duplicate source of truth
Implemented on branch `fix/pilot-shared-risk-source`.

Canonical source rule:

```text
Step 1 risk_context
= source of truth for
falls count (12m)
CFS
cognitive impairment
significant immobility
sarcopenia case-finding relevance
sarcopenia screening method
SARC-F score
```

Step 3 now treats these values as a read-only projection. Its additional functional/sarcopenia fields remain editable (falls/function review, injury, aid, gait/balance, TUG, chair stand, grip, gait speed, SPPB, action, etc.).

`static/baseline-audit/shared-risk-source.js` synchronizes the Step 3 projection from Step 1 and removes duplicate copies of the shared fields from persisted `step3` storage, so later audit logic has one canonical clinical value.

---

## 5. Current exact next action

**NEXT: continue pre-pilot hardening before the 5 real pilot encounters.**

Highest-priority remaining items:

```text
Patch 5 — make DXA machine a native persistent select
Patch 7 — implement real archetype-driven applicability
```

Then address the remaining usability/correctness items (progress calculation, longitudinal prior-DXA entry/escaping, BMI override behavior), perform a save/reload/complete smoke test, and only then start Pilot Case 1/5.

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
