# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-22 17:34 Asia/Nicosia
> **Canonical repository:** `athpapachr-cmd/osteoporosis`
> **Current major phase:** prospective Osteoporosis Baseline/Audit implementation
> **Current module:** Module 01 — Osteoporosis

This file contains current operational truth only. Permanent rules belong in `AGENTS.md`; roadmap in `TODO.md`; completed history in `osteoporosis-change-log.md`.

---

## 1. Project definition

The project is a **Personal Clinical Excellence System** with a reusable Core Engine. The existing Osteoporosis Cockpit is the point-of-care Clinical Practice / Encounter Execution layer of Module 01.

Core loop:

```text
STANDARD → LEARN → TEST/MASTER → APPLY → MEASURE → AUDIT
→ GAP OR STRENGTH → INTERVENE/REINFORCE → RE-MEASURE → SYSTEM LEARNS
```

Osteoporosis is the proving module. Future musculoskeletal modules remain deferred until the reusable engine is stable.

---

## 2. Canonical control plane and active schemas

```text
AGENTS.md
TODO.md
CLINICAL_EXCELLENCE_PLAN.md
HANDOFF_CURRENT.md
osteoporosis-change-log.md
```

Supporting schemas/contracts now include:

```text
schemas/baseline_osteoporosis_audit_v1.yaml
schemas/kpi_dictionary_v1.yaml
schemas/baseline_case_form_v1.yaml
schemas/baseline_step1_risk_refinement_v1.yaml
schemas/encounter_archetypes_v1.yaml
schemas/baseline_step1_capture_v1.yaml
schemas/baseline_step2_risk_v1.yaml
schemas/baseline_step3_results_v1.yaml
schemas/baseline_step4_plan_v1.yaml
```

`README.md` is navigation only. Durable decisions must not live only in chat.

---

## 3. Baseline strategy

There is no reliable historical osteoporosis registry/dedicated folder, and GeSY notes may not reflect the entire consultation. The main baseline is therefore **prospective post-visit capture**:

```text
5 consecutive pilot encounters
→ revise usability/branching once
→ freeze form + KPI applicability
→ 30 consecutive unique scored baseline patients
→ baseline lock
→ interventions / re-audit
```

During scored baseline: no KPI coaching or red/green performance feedback; safety-critical alerts may remain active.

Clinical process, formal documentation and capture quality remain separate measurement axes.

---

## 4. Encounter-specific applicability — frozen principle

The same audit checklist must not be applied to every osteoporosis visit.

Two independent axes drive applicability:

```text
PATIENT RELATIONSHIP
new_to_service | established_patient

ENCOUNTER ARCHETYPE
initial assessment — new/uncertain diagnosis
initial assessment — known osteoporosis/osteopenia
routine stable follow-up
treatment start
treatment continuation / due monitoring
treatment change / transition
post-fragility fracture
fracture on treatment
adverse effect / intolerance
treatment completion / consolidation
other
```

Disease status is separate again. KPI denominators will be archetype-specific.

---

## 5. Steps 1–3 — implemented, merged and live

### Step 1 — encounter context
Captures patient relationship/archetype, age/sex/menopause, anthropometrics/BMI/height loss, prior fracture timing, parental hip fracture, GC dose/duration, falls count, secondary conditions, frailty/immobility, sarcopenia trigger and low-burden Heidi exposure metadata.

### Step 2 — fracture history and formal risk
Captures structured fracture events, full/interval/focused fracture review, FRAX/FRAXplus/other formal assessment, country/surrogate model, MOF/hip probabilities, FN-BMD use, FRAX context, declared framework, resulting risk category and reasoned adjustment/override.

### Step 3 — examinations/results
Captures DXA with BMD + T-score + longitudinal comparability/LSC, VFA/vertebral-imaging indication/action/result, secondary-cause process, optional mineral/renal/BTM/conditional labs, falls/frailty/function and conditional sarcopenia testing.

Step 3 merged commit:

```text
b8d2f44c0aeb118f6a4b4558ed7ffe882216eec2
```

Render deploy:

```text
dep-da4qgubl550s73d6ogj0 — live
```

Prototype storage remains browser `localStorage`; no production patient-data persistence is introduced.

---

## 6. Step 4 — implemented on active branch

Active branch:

```text
feat/baseline-step4-plan
```

Schema:

```text
schemas/baseline_step4_plan_v1.yaml
```

Implementation files:

```text
static/baseline-audit/step4.js
static/baseline-audit/step4.css
static/baseline-audit/app.js   # now loads Step 4 after Step 3
```

### Treatment timeline
- repeated treatment episodes;
- exact start/end dates when known;
- approximate duration only when exact dates are unknown;
- active/completed/stopped/holiday/planned status;
- adherence and tolerance separately;
- fracture-on-episode and response context;
- reason started and reason stopped/switched.

### Administration timeline
- repeated administration events;
- scheduled date;
- actual date;
- done/due/overdue/missed/planned status;
- next due date;
- intended for denosumab, IV bisphosphonates, romosozumab and other time-critical administrations.

### Current clinical decision
- start / continue / stop / switch / defer / no-drug-treatment / complete / consolidate / refer / uncertain;
- selected agent;
- structured reasons for the decision;
- safety/contraindication review;
- sequencing review;
- patient preference documented;
- patient accepted/declined/undecided;
- optional clinician confidence and short rationale.

### Transition / sequencing safety
- whether transition is relevant;
- denosumab exit, post-teriparatide, post-romosozumab, bisphosphonate holiday/restart or other;
- prior last-dose/end date;
- next agent and planned date;
- whether explicit transition plan exists;
- unresolved safety issue and optional note.

### Follow-up / CareTask precursor
- repeated task objects for labs, DXA, administration, visit, referral, imaging, adherence, exercise/falls, nutrition or other;
- due date or timeframe text;
- planned/already-done/N/A status.

### Encounter close
- whether plan is complete;
- whether an unresolved critical item remains;
- optional short note.

Rules remain:
- no live treatment recommendation during baseline;
- no live KPI/guideline-concordance verdict;
- reasoned clinician override is not automatically an error;
- unknown and N/A remain distinct;
- no identifiable patient data or raw Heidi transcript in the public repo.

---

## 7. Migration principle from legacy Cockpit

The new Clinical Excellence interface is **not a replacement-by-copy**. It is a migration and normalization layer:

```text
OLD COCKPIT DATA
→ classify as useful / duplicate / outdated / context-specific
→ preserve useful structured data
→ add missing provenance, timing and applicability
→ connect to encounter archetype
→ map to KPI / audit / learning / improvement loops
```

Examples:
- T-scores are retained, but longitudinal DXA gains BMD/LSC/machine comparability.
- falls/frailty fields are retained, but 12-month counts and outpatient function are emphasized.
- numeric labs are optional while audit scoring focuses on whether relevant evaluation occurred.
- treatment type/duration is retained but upgraded to exact episodes, administration dates and future CareTasks.
- hospital-specific Morse items are not automatically promoted into the outpatient osteoporosis baseline.

---

## 8. Current next action

**NEXT: Step 5 — Επικοινωνία + immediate post-visit reflection.**

Then:

```text
Step 6 — documentation trace + final Heidi/capture-source review
→ exact field→KPI calculation contract
→ 5-case pilot
→ one refinement
→ freeze
→ 30-case scored baseline
```

Separate later instruments: Patient Voice 4-question instrument and Decision Quality 10-case review form.

---

## 9. Stop boundary

Do not yet:

- major-rewrite `main.py` / legacy Cockpit `index.html`;
- create a composite Clinical Excellence score from invented data;
- display live coaching during scored baseline except safety-critical alerts;
- treat GeSY as complete clinical truth;
- make Heidi mandatory before baseline;
- expand into another musculoskeletal module;
- commit identifiable patient information, GeSY content or Heidi transcripts to the public repository;
- treat browser `localStorage` as production clinical-data storage.
