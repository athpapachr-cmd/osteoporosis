# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-22 16:50 Asia/Nicosia
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

Supporting schemas/contracts:

```text
schemas/baseline_osteoporosis_audit_v1.yaml
schemas/kpi_dictionary_v1.yaml
schemas/baseline_case_form_v1.yaml
schemas/baseline_step1_risk_refinement_v1.yaml
schemas/encounter_archetypes_v1.yaml
schemas/baseline_step1_capture_v1.yaml
schemas/baseline_step2_risk_v1.yaml
schemas/baseline_step3_results_v1.yaml
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

## 5. Step 1 — implemented, merged and live

Captures:

- patient relationship + encounter archetype;
- age/sex/menopause;
- weight/current height/height source/reference height;
- automatic BMI and derived height loss;
- prior fragility fracture with site + month/year;
- parental hip fracture;
- frailty/immobility trigger with CFS/cognition/immobility;
- systemic glucocorticoid dose + duration;
- falls count over 12 months;
- structured secondary/associated conditions;
- sarcopenia case-finding relevance + SARC-F/clinical suspicion;
- osteoporosis status and baseline sampling flag;
- low-burden Heidi exposure/review/correction metadata.

Background-only derived context includes low-BMI workflow flag, height loss >=4 cm, fracture recency, recurrent falls >=2/year, GC dose bands/exposure duration and SARC-F >=4. These are not shown as performance coaching.

Heidi rule: do not paste raw or corrected transcripts; no manual diff required.

---

## 6. Step 2 — implemented, merged and live

Captures:

### Fracture history
- review status and scope;
- interval fracture status;
- repeated structured fracture events with site, month/year, fragility classification, treatment-at-event status and vertebral detail.

### Formal fracture-risk assessment
- indicated vs performed;
- FRAX / FRAXplus / other;
- country or surrogate model;
- MOF and hip 10-year probabilities;
- femoral-neck BMD use;
- smoking, alcohol and RA context;
- declared risk framework and resulting category;
- contextual adjustment / clinician override and structured reasons.

Rules:
- no internal FRAX-like surrogate score;
- no silent guideline hybridization;
- stable follow-up can explicitly record interval-only review / formal assessment N/A;
- no live therapeutic coaching during baseline.

---

## 7. Step 3 — implemented on active branch

Schema:

```text
schemas/baseline_step3_results_v1.yaml
```

Implementation files:

```text
static/baseline-audit/app.js       # lightweight bootstrap loader
static/baseline-audit/app-core.js  # preserved Step 1 + Step 2 application core
static/baseline-audit/step3.js
static/baseline-audit/step3.css
```

Step 3 uses a **selective migration** principle from the old Cockpit: clinically useful data are retained, but not copied field-for-field.

### DXA
- current-use status;
- scan date/facility/machine metadata;
- spine / total hip / femoral neck BMD g/cm² + T-score;
- ROI/excluded-vertebra and artifact review;
- Z-score relevance;
- longitudinal comparison;
- same-machine/cross-calibration status;
- facility LSC and optional site-specific LSC;
- whether change was interpreted using BMD/LSC or explicitly declared non-comparable.

### VFA / vertebral imaging
- indication yes/no/uncertain;
- structured indication reasons;
- performed/reviewed/arranged/reasoned-not-done/missed/N/A;
- modality;
- vertebral fracture result and Genant/grade traceability.

### Secondary causes and laboratory evaluation
- separates indication/process from numeric value entry;
- numeric labs remain optional so the pilot form is not a duplicate laboratory record;
- preserves useful legacy mineral/renal labs and BTMs;
- adds optional conditional secondary-cause labs/status fields.

### Falls / frailty / function
- falls review + 12-month count;
- injury/fracture relation;
- CFS;
- cognition and immobility;
- ambulatory aid;
- gait/balance concern;
- optional TUG;
- action if material risk identified.

Default outpatient baseline does **not** copy every Morse item from the legacy Cockpit because several are hospital-context specific.

### Sarcopenia
- conditional case-finding applicability;
- SARC-F or clinical suspicion;
- optional chair stand, grip strength, gait speed, SPPB and TUG;
- probable-sarcopenia signal and optional action;
- derived SARC-F >=4 stored silently, without baseline coaching.

Step 3 seeds falls/CFS/cognition/immobility/sarcopenia fields from Step 1 where available instead of asking for duplicate entry.

Prototype storage remains browser `localStorage`; no production patient-data persistence is introduced.

---

## 8. Migration principle from legacy Cockpit

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
- falls/frailty fields are retained, but 12-month fall counts and outpatient function are emphasized.
- numeric labs are retained as optional values, while audit scoring focuses on whether relevant evaluation occurred.
- hospital-specific Morse items are not automatically promoted into the outpatient osteoporosis baseline.

---

## 9. Current next action

**NEXT: Step 4 — Απόφαση & Πλάνο.**

Implement:

1. exact current/past treatment timeline;
2. actual administrations and due dates where time-critical;
3. adherence/tolerance and treatment response context;
4. reason for start/continue/stop/switch/defer;
5. contraindications/options considered;
6. sequencing / exit / consolidation safety;
7. clinician rationale and patient preference;
8. monitoring plan: labs, DXA, administration, visit/referral due dates;
9. unresolved critical item at encounter close;
10. archetype-specific applicability and N/A handling.

Then:

```text
Step 5 — communication + post-visit reflection
Step 6 — documentation trace + final Heidi/capture-source review
→ exact field→KPI calculation contract
→ 5-case pilot
→ one refinement
→ freeze
→ 30-case scored baseline
```

Separate later instruments: Patient Voice 4-question instrument and Decision Quality 10-case review form.

---

## 10. Stop boundary

Do not yet:

- major-rewrite `main.py` / legacy Cockpit `index.html`;
- create a composite Clinical Excellence score from invented data;
- display live coaching during scored baseline except safety-critical alerts;
- treat GeSY as complete clinical truth;
- make Heidi mandatory before baseline;
- expand into another musculoskeletal module;
- commit identifiable patient information, GeSY content or Heidi transcripts to the public repository;
- treat browser `localStorage` as production clinical-data storage.
