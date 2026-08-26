# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** ACTIVE PRE-CODE DESIGN — runtime implementation not authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **Current detailed profile under review:** `clinic_utilities/physio_profiles/shoulder_v1.md` — DESIGN CANDIDATE / NOT FROZEN.
> **Prior active slice:** PR-1 Transcript Intake + Candidate Extraction v3, intentionally paused and archived at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

CU-1 is a bounded product-owner-approved design detour. It does not cancel PR-1 and does not turn Clinic Utilities into a new clinical module.

---

# 1. Product outcome

Freeze the clinical/content and deterministic-generation contract for a substantially improved Physiotherapy Referral utility before any implementation.

Target workflow:

```text
Clinical problem
→ important findings
→ functional limitation
→ precautions/restrictions
→ goals
→ rehabilitation direction
→ structured ReferralDraft
→ short/detailed referral text
```

The utility should improve referral quality and speed while preserving clinician judgment and physiotherapist autonomy.

---

# 2. Source baseline

The supplied standalone HTML was inspected read-only.

Useful behavior to preserve:

- body-region condition groups;
- optional laterality/chronicity/session count;
- clinical findings;
- goals;
- active vs adjunct interventions;
- short/detailed outputs;
- copy/print;
- local/no-server behavior;
- consistency warnings;
- evidence/reference section.

Problems to correct:

- checkbox-catalogue flow rather than clinical referral flow;
- generic findings across unrelated conditions;
- globally preselected goals/interventions;
- insufficient condition-specific restrictions/precautions;
- direct phrase concatenation instead of a structured intermediate model;
- limited consistency/safety rules;
- incomplete common pathways;
- standalone visual identity rather than Clinical Excellence styling.

---

# 3. Frozen architectural direction

Future implementation must build a deterministic structured object before formatting prose:

```text
ReferralDraft
  patient_context
  body_region
  primary_problem
  secondary_problems[]
  laterality
  chronicity
  key_findings[]
  functional_impairments[]
  precautions[]
  explicit_restrictions[]
  goals[]
  rehab_directions[]
  adjunct_options[]
  reassessment_criteria[]
  sessions_optional
  clinician_free_text_optional
```

Then:

```text
ReferralDraft
→ ShortReferralFormatter
→ DetailedReferralFormatter
```

A condition profile may define clinically relevant candidate findings, functional impairments, precautions, goals, directions, adjuncts, consistency rules and required context.

Hard invariants:

```text
suggested != examined
suggested != selected
selected != mandatory
symptom != objective deficit
special/provocation test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

---

# 4. Body-region profile status

## 4.1 Cervical — FROZEN v1.1

Authoritative file:

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
```

Frozen semantics include strict subjective-vs-objective neurological separation, formal clinician-entered cervicogenic headache/cervical-dizziness diagnoses, directly selectable trigger-point/myofascial and referred shoulder-girdle findings, active-first rehabilitation and no routine cervical post-operative pathway.

## 4.2 Lumbar — FROZEN v1.1

Authoritative file:

```text
clinic_utilities/physio_profiles/lumbar_v1_1.md
```

Frozen pathways:

```text
L1 non-specific / mechanical low-back pain
L2 radiating/radicular-feature low-back pain
L3 lumbar stenosis / neurogenic claudication
L4 deep-gluteal / piriformis presentation
```

Lumbar preserves tri-state neurological status, cauda-equina-type reassessment semantics, directly selectable myofascial findings, no `SI dysfunction` lumbar diagnosis, optional acupuncture/dry needling with evidence/competence caveats, no routine lumbar traction and no active lumbar post-operative pathway.

## 4.3 Shoulder — ACTIVE DESIGN CANDIDATE

Current candidate:

```text
clinic_utilities/physio_profiles/shoulder_v1.md
```

Proposed primary pathways for product-owner review:

```text
S1 rotator-cuff-related shoulder pain / rotator-cuff tendinopathy
S2 confirmed full-thickness rotator-cuff tear — nonoperative rehabilitation
S3 calcific rotator-cuff tendinopathy
S4 adhesive capsulitis / frozen shoulder
S5 glenohumeral instability / dislocation rehabilitation
S6 glenohumeral osteoarthritis
S7 post-traumatic shoulder pain / rehabilitation after assessed injury
```

Proposed secondary/modifier status pending real-workflow confirmation:

```text
long-head-of-biceps-related pain/tendinopathy
AC-joint-related pain / arthropathy
scapular dyskinesis/control findings
myofascial / trigger-point findings
special-test findings
```

Key proposed shoulder rules:

- `impingement syndrome` is not the preferred top-level diagnosis;
- special tests, painful arc and scapular findings remain findings, not diagnoses;
- active and passive ROM remain distinct;
- pain-inhibited effort does not establish a tear;
- confirmed full-thickness tear requires established clinician/imaging context;
- acute trauma with marked new weakness/inability to elevate requires explicit reassessment/imaging/specialist semantics;
- adhesive capsulitis is not inferred from generic stiffness or imaging alone;
- instability is not inferred from a single apprehension/relocation test;
- active rehabilitation is the conceptual core;
- acupuncture remains an optional adjunct for rotator-cuff tendinopathy;
- dry needling is optional with competence/availability caveat;
- ESWT is exposed only for calcific rotator-cuff tendinopathy;
- routine ESWT for noncalcific rotator-cuff tendinopathy is not suggested;
- therapeutic ultrasound is excluded from the standard rotator-cuff adjunct list under the current 2025 CPG;
- calcific lavage/barbotage remains clinician/medical management, not a physiotherapist-technique option.

Shoulder remains **NOT FROZEN** until product-owner review resolves the proposed taxonomy and real-workflow questions.

---

# 5. Context-sensitive defaults

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms or changes
→ only confirmed values populate ReferralDraft
```

No global pain + ROM + strength + motor-control bundle.

---

# 6. Safety / consistency engine

The engine provides clinician-facing consistency/safety prompts, not autonomous diagnoses or treatment prohibitions.

Cross-region rules include:

```text
fracture/post-trauma + unresolved structural context
→ warning/reassessment prompt

post-op pathway + missing procedure/protocol/restrictions
→ warning

adjunct selected without active rehabilitation direction
→ warning

new/progressive objective neurological deficit
→ prominent medical reassessment prompt

material safety concern
→ require explicit clinician disposition before routine reassuring wording

not assessed neurological component
→ never generate normal wording
```

Shoulder-specific rules live in `shoulder_v1.md` and include acute traumatic weakness/possible cuff-tear concern, instability/dislocation context, calcific-specific adjunct rules and cervical/neurological overlap.

---

# 7. Output wording contract

Preferred structure:

```text
Clinical problem + important findings + functional impact.
Referral request + goals.
Rehabilitation direction + restrictions/precautions.
Optional reassessment/feedback criteria.
```

Rules:

- collaborative wording, not over-prescription of the physiotherapist;
- active rehabilitation, education/self-management and graded activity/loading where appropriate;
- technique-level interventions remain adjunctive;
- no unsupported diagnosis from tests, symptoms or incidental imaging;
- no negative neurological/red-flag/structural statements from missing or unassessed data;
- preserve explicit restrictions;
- short and detailed outputs derive from the same `ReferralDraft`.

---

# 8. Persistence / patient-linkage boundary

CU-1 does **not** freeze referral persistence yet.

Default first implementation direction:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

If persistence is later justified, reuse creates a new referral draft and never mutates historical referral truth.

---

# 9. Out of scope for CU-1

Do not yet:

- write production HTML/JS/CSS;
- integrate Clinical Excellence navigation;
- add patient persistence/history;
- use AI to generate the referral;
- implement RF workflow changes;
- alter Osteoporosis audit/PR-1 transcript runtime;
- create overlapping runtime writers.

---

# 10. Design acceptance checklist before CU-2

```text
A. body-region taxonomy
B. condition profiles and naming
C. findings vs diagnosis separation
D. high-risk/fracture/protocol context where relevant
E. goals and rehabilitation directions
F. safety/consistency warnings
G. ReferralDraft schema
H. short/detailed formatting rules
I. exact MVP fields
J. Clinical Excellence visual/navigation host
K. persistence decision for first implementation
L. final evidence check for production wording
```

---

# 11. Exact next action

```text
1. product-owner clinical review of clinic_utilities/physio_profiles/shoulder_v1.md
2. resolve stand-alone biceps pathway vs secondary modifier
3. resolve AC-joint pathway visibility
4. resolve labral/SLAP pathway need
5. resolve real-workflow post-operative shoulder need
6. confirm adjunct policy (acupuncture, dry needling, calcific ESWT, no standard therapeutic ultrasound)
7. revise shoulder candidate
8. freeze/merge only after explicit product-owner approval
9. then proceed to the next body-region profile
```

Do not write runtime code. Explicit product-owner authorization is still required before transition from CU-1 design to CU-2 implementation.
