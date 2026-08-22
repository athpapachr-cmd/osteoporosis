# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-22 15:02 Asia/Nicosia
> **Canonical repository:** `athpapachr-cmd/osteoporosis`
> **Current major phase:** Clinical Excellence Blueprint + Baseline/Audit foundation
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

## 2. Canonical project control plane

Active canonical set:

```text
AGENTS.md
TODO.md
CLINICAL_EXCELLENCE_PLAN.md
HANDOFF_CURRENT.md
osteoporosis-change-log.md
```

`README.md` is navigation only. Durable decisions must not live only in chat.

Supporting machine-readable design schemas now include:

```text
schemas/baseline_osteoporosis_audit_v1.yaml
schemas/kpi_dictionary_v1.yaml
schemas/baseline_case_form_v1.yaml
```

---

## 3. Critical baseline constraint discovered

There is currently **no reliable pre-existing osteoporosis patient registry or dedicated osteoporosis folder** from which a representative retrospective baseline can be sampled.

GeSY visit records may be incomplete and may not reflect everything discussed, assessed or decided during the real consultation. Therefore GeSY alone must not be treated as complete clinical truth for baseline measurement.

Heidi AI has only recently started to be used and is **not yet used systematically**. It may improve encounter capture, but current use is an exposure variable, not a quality metric and not a reliable historical source.

This invalidates the original preference for a purely retrospective 30-case baseline as the main design.

---

## 4. Revised baseline strategy — prospective measured baseline

The baseline now uses a neutral **prospective post-visit capture form**.

### Pilot

```text
5 consecutive eligible osteoporosis encounters
→ test form usability/branching/time burden
→ revise if needed
→ freeze form + KPI applicability rules
```

Pilot cases are not included in the locked baseline.

### Scored baseline

```text
30 consecutive unique eligible osteoporosis patients
→ one index encounter per patient for core baseline
→ post-visit capture
→ no KPI coaching / red-green performance feedback during collection
→ safety-critical alerts remain active
→ lock baseline after case 30
```

This is explicitly an **observed prospective baseline**; it does not claim to reconstruct completely unobserved historical practice.

---

## 5. Separate clinical care from documentation quality

Every important audit domain must distinguish:

```text
A. CLINICAL PROCESS
What was actually assessed / discussed / decided / arranged?

B. FORMAL DOCUMENTATION
What is traceably present in GeSY or another formal clinical record?

C. CAPTURE QUALITY
How complete/reliable is our evidence about the encounter?
```

Primary baseline source for clinical-process measurement:

```text
clinician-validated post-visit form
```

Supplementary sources:

```text
clinician-reviewed Heidi output when used
GeSY note
other declared formal record
```

A missing GeSY entry can therefore identify a **documentation/system gap** without falsely implying that the clinical action definitely did not occur.

---

## 6. Heidi AI — current role

During pilot/baseline, record:

```text
Heidi used? yes/no
output available? yes/no
clinician reviewed? yes/no
material correction required? yes/no/unknown
```

Rules:

- Heidi use itself is not scored as good practice.
- AI output never overrides clinician validation.
- Do not force systematic Heidi use during the scored baseline merely to make the baseline look better.
- Later comparison of cases with versus without Heidi is observational only and does not prove causality.
- Raw transcripts or identifiable Heidi clinical content must never be committed to this public repository.

After baseline, systematic Heidi use may become an explicit Improvement Project if evidence from workflow/capture quality supports testing it.

---

## 7. Baseline/KPI design status

`schemas/baseline_osteoporosis_audit_v1.yaml` now defines:

- 5-case pilot;
- 30-case prospective baseline;
- one index encounter per unique patient;
- safety/continuity census from activation;
- 10-case decision-quality sub-audit;
- later 20-patient Patient Voice baseline;
- reliability tiers;
- baseline lock and re-audit logic;
- separate clinical/documentation measurement axes.

`schemas/kpi_dictionary_v1.yaml` contains the first 16 provisional KPIs.

`schemas/baseline_case_form_v1.yaml` defines the neutral post-visit data capture needed to calculate them while preserving clinical-process versus documentation distinctions.

---

## 8. Current next design action

**NEXT:** convert `baseline_case_form_v1.yaml` into the first usable compact adaptive form specification/UI, then create:

1. `Patient Voice` 4-question instrument;
2. `Decision Quality` 10-case review form;
3. exact KPI calculation contract mapping form fields → KPI status.

The first runtime/data-capture implementation should remain narrow: collect baseline data reliably before building the full Clinical Excellence dashboard.

---

## 9. Stop boundary

Do not yet:

- major-rewrite `main.py` / `index.html`;
- create a composite Clinical Excellence score from invented data;
- display live coaching during scored baseline collection except safety-critical alerts;
- treat GeSY as a complete record of the consultation;
- make Heidi mandatory before baseline is established;
- expand into the next musculoskeletal module;
- commit identifiable patient information, GeSY content or Heidi transcripts to the public repository.

Bootstrap order for the next session:

```text
AGENTS.md
HANDOFF_CURRENT.md
TODO.md — section 0/1
CLINICAL_EXCELLENCE_PLAN.md
schemas/baseline_osteoporosis_audit_v1.yaml
schemas/baseline_case_form_v1.yaml
schemas/kpi_dictionary_v1.yaml
```
