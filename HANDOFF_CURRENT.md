# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-22 16:23 Asia/Nicosia
> **Canonical repository:** `athpapachr-cmd/osteoporosis`
> **Current major phase:** prospective Osteoporosis Baseline/Audit implementation
> **Current module:** Module 01 — Osteoporosis

This file contains only current operational truth. Permanent rules belong in `AGENTS.md`; roadmap in `TODO.md`; completed history in `osteoporosis-change-log.md`.

---

## 1. Project definition

The project is a **Personal Clinical Excellence System** with a reusable Core Engine. The existing Osteoporosis Cockpit is the point-of-care Clinical Practice / Encounter Execution layer of Module 01.

Core loop:

```text
STANDARD → LEARN → TEST/MASTER → APPLY → MEASURE → AUDIT
→ GAP OR STRENGTH → INTERVENE/REINFORCE → RE-MEASURE → SYSTEM LEARNS
```

Future musculoskeletal modules remain deferred until the reusable engine is proven with Osteoporosis.

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
```

`README.md` is navigation only. Durable decisions must not live only in chat.

---

## 3. Baseline strategy

There is no reliable historical osteoporosis registry/dedicated folder, and GeSY notes may not reflect the entire consultation. Main baseline design is therefore **prospective post-visit capture**:

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

A patient can be new to the service but already diagnosed/treated. Disease status is therefore separate again.

KPI denominators will later be archetype-specific.

---

## 5. Step 1 — implemented, merged and live

Step 1 captures:

- patient relationship + encounter archetype;
- age/sex/menopause;
- weight/current height/height source/reference height;
- automatic BMI when weight + height are present;
- derived height loss;
- prior fragility fracture with most recent site + month/year;
- parental hip fracture;
- frailty/immobility trigger with optional CFS/cognition/immobility;
- systemic glucocorticoid dose + duration;
- falls count over 12 months;
- structured secondary/associated conditions;
- sarcopenia case-finding relevance + SARC-F/clinical suspicion;
- osteoporosis status and baseline sampling flag;
- low-burden Heidi exposure/review/correction metadata.

Background-only derived context includes low-BMI workflow flag, height loss >=4 cm, fracture recency, recurrent falls >=2/year, GC dose bands/exposure duration and SARC-F >=4. These are not shown as baseline performance coaching.

Heidi rule: do not paste raw or corrected transcripts; no manual diff required. Optional one-click correction categories only.

---

## 6. Step 2 — implemented, merged and live

Merged commit:

```text
62a789a69890f6a17e290acb6b5d208dc7e406e3
```

Render deploy:

```text
dep-da4q43ojo6nc73eggicg — live
```

Step 2 now captures:

### Fracture history

- whether fracture history was reviewed;
- review scope: full / interval / focused / not reviewed;
- interval fracture status;
- repeated structured fracture events with site, month/year, low-trauma classification, on-treatment status and vertebral level/type;
- Step 1 last-fracture context can seed the first event;
- most recent event synchronizes back to Step 1 context.

### Formal fracture-risk assessment

- whether formal risk assessment was indicated;
- whether it was actually performed;
- tool: FRAX / FRAXplus / other;
- exact country or surrogate model used;
- MOF and hip 10-year probabilities;
- whether femoral-neck BMD was used;
- current smoking, alcohol >=3 units/day and RA context;
- explicitly declared risk framework;
- resulting risk category;
- contextual adjustment / override and structured reasons.

Rules:

- no internal FRAX-like surrogate score;
- no silent hybridization of guideline thresholds;
- stable follow-up can explicitly record interval-only review / formal assessment not applicable;
- no live therapeutic coaching during baseline.

Static pilot route remains:

```text
/static/baseline-audit/
```

Prototype storage remains browser `localStorage` only and is not production clinical storage.

---

## 7. Heidi AI — current role

Heidi is recent and not systematic. During pilot/baseline it is an exposure/capture variable, not a quality metric.

Capture:

```text
used?
output available?
clinician reviewed?
material correction required?
optional one-click correction category
```

Never commit raw/identifiable Heidi content to the public repository.

---

## 8. Current next action

**NEXT: Step 3 — Εξετάσεις & Αποτελέσματα.**

Step 3 should implement:

1. DXA study capture and interpretation quality;
2. BMD g/cm² + T/Z scores where relevant;
3. longitudinal DXA comparability: machine/cross-calibration, percent change, facility LSC, excluded vertebrae/artifact;
4. VFA/vertebral-imaging indication and action;
5. secondary osteoporosis history/laboratory evaluation status;
6. falls/frailty detailed review;
7. conditional sarcopenia pathway: SARC-F/clinical suspicion → grip/chair stand → DXA/BIA → gait/SPPB/TUG when clinically relevant;
8. exercise, physical activity and nutrition/supplement review when applicable;
9. archetype-specific applicability with explicit N/A;
10. no live KPI coaching during baseline.

Then:

```text
Step 4 — treatment history/safety + decision + monitoring/follow-up
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

## 9. Stop boundary

Do not yet:

- major-rewrite `main.py` / existing Cockpit `index.html`;
- create a composite Clinical Excellence score from invented data;
- display live coaching during scored baseline except safety-critical alerts;
- treat GeSY as complete clinical truth;
- make Heidi mandatory before baseline;
- expand into another musculoskeletal module;
- commit identifiable patient information, GeSY content or Heidi transcripts to the public repository;
- treat browser `localStorage` as production clinical-data storage.

Bootstrap order:

```text
AGENTS.md
HANDOFF_CURRENT.md
TODO.md — section 0/1
CLINICAL_EXCELLENCE_PLAN.md
schemas/baseline_osteoporosis_audit_v1.yaml
schemas/encounter_archetypes_v1.yaml
schemas/baseline_step1_capture_v1.yaml
schemas/baseline_step2_risk_v1.yaml
schemas/kpi_dictionary_v1.yaml
```
