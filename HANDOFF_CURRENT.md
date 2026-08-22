# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-22 23:00 Asia/Nicosia
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
→ final data-quality additions
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

Absent/partial GeSY documentation must not be silently converted to clinical omission. Heidi use itself is not a quality-success metric.

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

### P1–P8
All previously identified hardening patches are closed: hidden-value hygiene; module-safe core save; whole-form progress; Step1→Step3 source-of-truth projection; DXA machine persistence; inline prior-DXA/stable deletion; archetype-driven applicability with module-owned persistence; and BMI derived/manual behavior.

### Final data-quality additions before smoke test
Implemented on branch `feat/prepilot-labs-date-step6-conflict`:

1. **Step 3 `labs_date`**
   - native HTML `type=date` calendar control;
   - stored as `step3.labs.labs_date`;
   - optional, because numeric lab entry remains optional;
   - purpose: distinguish current from historical laboratory snapshots, especially for monitoring encounters.

2. **Step 6 source-conflict clear-on-collapse**
   - `conflict_resolution` and `conflict_note` are shown only when `source_conflict_present=yes`;
   - when conflict changes away from `yes`, both dependent DOM values and persisted values are cleared;
   - `collect()` independently persists blank dependent values unless conflict is `yes`;
   - conflict note now carries a no-identifiers reminder.

Schemas updated:

```text
schemas/baseline_step3_results_v1.yaml
schemas/baseline_step6_documentation_v1.yaml
```

### External-review backlog
The full enhancement backlog from the latest Dia review is now incorporated into `CLINICAL_EXCELLENCE_PLAN.md §20`. It includes shared registries, laboratory tri-state/units discipline, FRAX reproducibility, Step-3 derived context, Step-4 safety/coherence derivations, Step-5 structured communication/Signals, Step-6 provenance/clinical-process-present logic, and cross-cutting store/accessibility work.

Do **not** implement the full backlog before the 5-case pilot. The pilot is the evidence gate for form burden and which refinements deserve the one deliberate post-pilot revision.

---

## 5. Current exact next action

**NEXT: merge/deploy the final data-quality branch and run the explicit synthetic smoke test.**

Required synthetic/non-identifiable checks:

```text
1. FRAX/FRAXplus edit → Save → reload → values retained
2. DXA yes + values → DXA no → Save → no current DXA in trends
3. Step 1 shared risk values → Step 3 mirror → no divergence
4. DXA machine + machine_label → Save/reload → retained
5. archetype change → expected/conditional/N/A card state updates correctly
6. collapsed domain → Χρήση σήμερα → Save from Step 1/2 → reload → override retained without repair shim
7. whole-form progress changes across Steps 1–6 and excludes collapsed domains
8. Prior DXA → inline add → Save/reload → retained
9. historical DXA delete after date sorting → correct row removed
10. BMI with weight+height → read-only calculated; remove one source → derived BMI clears and manual mode returns
11. Step 3 labs_date → calendar entry → Save/reload → retained
12. Step 6 conflict=yes → resolution/note visible and retained
13. Step 6 conflict yes→no/uncertain/blank → resolution/note collapse + clear → Save/reload remains clear
14. Finish Visit → all module slices retained, including applicability_review, labs_date and Step6 conflict state
```

If all checks pass, the form is cleared for **Pilot Case 1/5**. Operational target: begin real pilot encounters from Monday 2026-08-24 rather than adding more pre-pilot functionality over the weekend.

After the 5 pilot cases:

```text
review pilot evidence once
→ select material items from Plan §20
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
- implement the whole external-review enhancement backlog before the 5-case pilot;
- major-rewrite legacy `main.py` / `index.html`;
- create a composite Clinical Excellence score from invented data;
- display live KPI coaching during scored baseline except safety-critical alerts;
- treat GeSY as complete clinical truth;
- make Heidi mandatory before baseline;
- commit identifiable patient data, GeSY content or Heidi transcripts to the public repo;
- treat browser `localStorage` as production clinical-data storage;
- start the real pilot until the explicit smoke test passes.
