# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-22 16:08 Asia/Nicosia
> **Canonical repository:** `athpapachr-cmd/osteoporosis`
> **Current major phase:** Clinical Excellence Blueprint + prospective Baseline/Audit implementation
> **Active detailed plan:** `CLINICAL_EXCELLENCE_PLAN.md`
> **Current module:** Module 01 — Osteoporosis

This file contains only current operational truth. Permanent rules belong in `AGENTS.md`; roadmap in `TODO.md`; completed history in `osteoporosis-change-log.md`.

---

## 1. Current project definition

The project is a **Personal Clinical Excellence System** with a reusable Core Engine. The existing Osteoporosis Cockpit is the point-of-care Clinical Practice / Encounter Execution layer of Module 01, not the whole improvement system.

Core loop:

```text
STANDARD
→ LEARN
→ TEST / MASTER
→ APPLY
→ MEASURE
→ AUDIT
→ GAP OR STRENGTH
→ INTERVENE / REINFORCE
→ RE-MEASURE
→ SYSTEM LEARNS
```

Future clinical modules remain deferred until the reusable engine is proven with Osteoporosis.

---

## 2. Canonical control plane and schemas

Active canonical set:

```text
AGENTS.md
TODO.md
CLINICAL_EXCELLENCE_PLAN.md
HANDOFF_CURRENT.md
osteoporosis-change-log.md
```

`README.md` is navigation only. Durable decisions must not live only in chat.

Current supporting schemas:

```text
schemas/baseline_osteoporosis_audit_v1.yaml
schemas/kpi_dictionary_v1.yaml
schemas/baseline_case_form_v1.yaml
schemas/encounter_archetypes_v1.yaml
schemas/baseline_step1_capture_v1.yaml
```

---

## 3. Baseline constraint and strategy

There is no reliable pre-existing osteoporosis patient registry/dedicated folder. GeSY visit records may be incomplete and may not reflect everything discussed, assessed or decided during the consultation. GeSY alone must not be treated as complete clinical truth.

The baseline therefore uses a neutral **prospective post-visit capture form**.

### Pilot

```text
5 consecutive eligible osteoporosis encounters
→ test usability/branching/time burden
→ revise once if needed
→ freeze form + KPI applicability rules
```

Pilot cases are not included in the locked baseline.

### Scored baseline

```text
30 consecutive unique eligible osteoporosis patients
→ one index encounter per patient
→ post-visit capture
→ no KPI coaching / red-green feedback during collection
→ safety-critical alerts remain allowed
→ lock baseline after case 30
```

This is an observed prospective baseline; it does not claim to reconstruct unobserved historical practice.

---

## 4. Separate clinical care from documentation quality

Every important audit domain must distinguish:

```text
A. CLINICAL PROCESS
What was actually assessed / discussed / decided / arranged?

B. FORMAL DOCUMENTATION
What is traceably present in GeSY or another formal record?

C. CAPTURE QUALITY
How complete/reliable is the evidence about the encounter?
```

Primary source for clinical-process measurement: clinician-validated post-visit form.

Supplementary sources: clinician-reviewed Heidi output when used, GeSY note, other declared formal record.

A missing GeSY entry can identify a documentation/system gap without falsely implying the clinical action definitely did not occur.

---

## 5. Encounter-specific audit applicability — approved

The audit must **not** apply one checklist to every osteoporosis visit.

Two independent classification axes are now mandatory:

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

The patient can be new to this service while already diagnosed or already treated. Relationship status and disease status are therefore separate.

`schemas/encounter_archetypes_v1.yaml` defines the expected domains for each archetype. KPI denominators will later become archetype-specific so that, for example, a stable follow-up is not penalized for not repeating full new-patient education or a full secondary-cause workup without indication.

---

## 6. Step 1 refined clinical context — implemented on active branch

Active branch:

```text
feat/baseline-step1-risk-refinement
```

The Step 1 UI/data model now captures:

- new-to-service vs established patient;
- encounter archetype;
- age/sex/menopause status;
- weight, current height, height source and reference/prior height;
- automatic BMI calculation when weight + height are present;
- derived height loss;
- prior fragility fracture with last site and month/year;
- parental hip fracture;
- frailty/immobility trigger with optional CFS, cognitive impairment and significant immobility;
- glucocorticoid dose + duration;
- number of falls in the last 12 months;
- structured secondary-risk/secondary-osteoporosis conditions;
- sarcopenia case-finding relevance + SARC-F/clinical-suspicion capture;
- osteoporosis status;
- baseline sampling flag.

Background-only derived context currently includes:

```text
low-BMI workflow flag (local configurable threshold, not FRAX/NOGG criterion)
height loss >=4 cm
months since last fragility fracture
fracture within 24 months
recent vertebral fracture within 24 months
recurrent falls >=2 in 12 months
glucocorticoid dose band + >=3 month exposure + >20 mg/day flag
SARC-F >=4 signal
```

These derived values are stored for later applicability logic but are not shown as performance coaching during baseline.

---

## 7. Sarcopenia — current role

Sarcopenia is now a conditional Function/Frailty/Falls domain rather than a universal mandatory diagnostic workup.

Step 1 only captures whether case-finding is relevant and whether SARC-F/clinical suspicion was used. A fuller Find → Assess → Confirm → Severity workflow belongs in Step 3 when applicable, potentially using chair stand/grip strength, DXA/BIA lean mass and physical-performance measures.

---

## 8. Heidi AI — current role

Heidi has only recently started to be used and is not systematic. During pilot/baseline it remains an exposure/capture variable, not a quality metric.

Capture now is deliberately low-burden:

```text
Heidi used?
output available?
clinician reviewed?
material correction required?
optional one-click correction category
```

Rules:

- Do **not** paste raw transcripts into the audit form.
- Do **not** require the clinician to paste a corrected transcript/note.
- Do **not** require free-text descriptions of corrections.
- Optional correction categories are omission / factual error / medication-dose / assessment-plan / context-speaker / other.
- AI output never overrides clinician validation.
- Heidi use itself is not scored as good practice.
- Raw or identifiable Heidi content must never be committed to the public repository.

Later systematic Heidi use may become an explicit Improvement Project if workflow/capture evidence supports testing it.

---

## 9. Runtime / deployment state

PR #1 for the first Baseline Audit UI slice was merged to `main` and auto-deployed by Render. The service uses the existing FastAPI `/static` mount.

Live route:

```text
/static/baseline-audit/
```

The active refinement branch changes only the static baseline-audit UI/data contract and supporting docs/schemas; it does not rewrite `main.py`, add a patient-data API or create server-side case persistence.

Prototype drafts remain in browser `localStorage` only and are explicitly not production clinical storage.

---

## 10. Current next action

**NEXT: Step 2 — Ιστορικό & Κίνδυνος.**

Step 2 must implement:

1. structured fracture-event history beyond the Step 1 last-fracture signal;
2. explicit fracture-history review status;
3. FRAX/risk-framework capture with country/surrogate model and MOF/hip values;
4. femoral-neck BMD-used status;
5. risk category under a declared framework, without silent hybridization;
6. contextual adjustments/clinician override + reason;
7. archetype-specific applicability so stable follow-up is not forced through a full new-patient pathway;
8. no live KPI score during baseline.

After Step 2:

```text
Step 3 — DXA/VFA + secondary causes + falls/frailty/sarcopenia
Step 4 — treatment history/safety + decision + monitoring/follow-up
Step 5 — communication + immediate post-visit reflection
Step 6 — documentation trace + final Heidi/capture-source review
→ exact field → KPI contract
→ 5-case pilot
→ revise once
→ freeze
→ 30-case scored baseline
```

Separate later instruments: Patient Voice 4-question instrument and Decision Quality 10-case review form.

---

## 11. Stop boundary

Do not yet:

- major-rewrite `main.py` / existing Cockpit `index.html`;
- create a composite Clinical Excellence score from invented data;
- display live coaching during scored baseline except safety-critical alerts;
- treat GeSY as complete truth;
- make Heidi mandatory before baseline;
- expand into another musculoskeletal module;
- commit identifiable patient information, GeSY content or Heidi transcripts to the public repository;
- treat browser `localStorage` as a production clinical-data store.

Bootstrap order for the next session:

```text
AGENTS.md
HANDOFF_CURRENT.md
TODO.md — section 0/1
CLINICAL_EXCELLENCE_PLAN.md
schemas/baseline_osteoporosis_audit_v1.yaml
schemas/encounter_archetypes_v1.yaml
schemas/baseline_step1_capture_v1.yaml
schemas/baseline_case_form_v1.yaml
schemas/kpi_dictionary_v1.yaml
```
