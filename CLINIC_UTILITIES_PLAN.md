# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; cervical v1.1 and lumbar v1.1 frozen; shoulder next.

This document keeps the Clinic Utilities detour detailed without confusing operational tooling with Osteoporosis Module 01 or the reusable Clinical Excellence Core.

Permanent boundary:

```text
Clinical Excellence Core
→ reusable patient/workflow/navigation/auth/integration mechanics

Clinic Utilities
→ cross-module clinician-facing operational tools

Module 01 Osteoporosis
→ osteoporosis-specific clinical content
```

Clinic Utilities do not constitute a new clinical Module 02.

---

# 1. Detour purpose

Integrate two existing clinician-created tools into the Clinical Excellence workspace:

1. **Physiotherapy Referral Generator** — structured clinical referral text generation.
2. **Radiofrequency Request / PDF Workflow** — request creation, PDF generation, lifecycle tracking and reuse of previous request data.

The detour should improve daily clinic workflow while preserving the broader product objective: better clinical practice before, during and after the consultation.

---

# 2. Source inspection status

## 2.1 Physiotherapy source

Standalone source was inspected read-only.

Useful capabilities to preserve:

- local/no-server operation;
- body-region condition groups;
- optional laterality/chronicity/session count;
- clinical findings;
- goals;
- active vs adjunct intervention wording;
- short/detailed output modes;
- copy/print;
- consistency warnings;
- evidence/reference section.

Design weaknesses being corrected:

- checkbox catalogue instead of clinically structured flow;
- generic findings across unrelated diagnoses;
- globally preselected goals/interventions;
- under-modelled precautions/restrictions;
- direct phrase concatenation instead of structured intermediate data;
- minimal validation;
- incomplete common pathways;
- standalone styling rather than Clinical Excellence-native presentation.

## 2.2 RF source

Existing RF implementation was inspected read-only in `athpapachr-cmd/ortho-reception-backend-v2`.

RF runtime mutation is not part of CU-1 and remains governed by the separate Digital Secretary control plane.

---

# 3. Physiotherapy Referral v2 — product outcome

```text
1. Clinical problem / diagnosis
2. Important findings
3. Functional limitation
4. Precautions / restrictions
5. Rehabilitation goals
6. Rehabilitation direction
7. Final referral text
```

The application generates useful structured referral wording without replacing the physiotherapist's assessment or prescribing a complete treatment recipe.

---

# 4. Structured intermediate model

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

Hard rules:

```text
suggested != examined
suggested != selected
selected != clinically mandatory
symptom != objective deficit
provocation test != diagnosis
not assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried but not inferred
```

---

# 5. Profile status

## 5.1 Cervical — FROZEN v1.1

Authoritative file:

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
```

Key frozen pathways:

- non-specific/mechanical neck pain;
- radiating/radicular-feature neck pain;
- formal cervicogenic headache when explicitly clinician-asserted;
- cervical/cervicogenic dizziness when explicitly clinician-asserted;
- whiplash/post-traumatic neck pain.

Directly selectable cervical modifiers include myofascial/trigger-point findings and referred shoulder-girdle pain. Cervical post-operative rehabilitation is not part of the active cervical MVP.

## 5.2 Lumbar — FROZEN v1.1

Authoritative file:

```text
clinic_utilities/physio_profiles/lumbar_v1_1.md
```

Frozen pathways:

- non-specific/mechanical low-back pain;
- low-back pain with radiating leg symptoms/radicular features;
- lumbar spinal stenosis/neurogenic claudication;
- deep-gluteal/piriformis presentation with optional formal clinician diagnosis.

Frozen lumbar modifiers include ROM/movement findings, load/postural aggravation, trunk strength/endurance deficit, myofascial/trigger-point findings, referred buttock/non-radicular leg pain, and SLR/slump findings when actually examined.

Lumbar safety inherits the tri-state neurological model and adds high-priority clinician handling for cauda-equina-type concerns.

### Lumbar adjunct policy

```text
manual therapy / mobilization → optional adjunct
soft-tissue techniques → optional adjunct
neurodynamic techniques → optional when neural/radiating context exists
acupuncture → optional clinician-selected adjunct
dry needling → optional clinician-selected adjunct with competence/availability caveat
lumbar traction → excluded from MVP
```

Evidence-framework transparency is mandatory:

- NICE NG59 recommends against acupuncture for LBP/sciatica;
- WHO 2023 conditionally supports needling therapies including acupuncture/dry needling for chronic primary LBP, with low-certainty evidence;
- therefore acupuncture/dry needling must never be rendered as mandatory or universally guideline-endorsed.

### SI-region boundary

`SI dysfunction` is not a lumbar diagnosis.

Future separate SI/pelvic profile should distinguish:

```text
SI-region pain / suspected SIJ-related pain
formal clinician diagnosis of SIJ-related pain
imaging-confirmed sacroiliitis or other defined structural/inflammatory SI pathology
```

MRI can support sacroiliitis/structural pathology but does not by itself prove that a mechanically painful SI joint is the pain generator.

### Post-operative boundary

Lumbar post-operative rehabilitation is not part of the active lumbar MVP because it is not part of the product owner's current workflow. The generic shared post-operative architecture remains available for other regions only where real workflow justifies it.

## 5.3 Shoulder — NEXT DESIGN TARGET

Candidate problems for structured review include:

- rotator-cuff-related shoulder pain/tendinopathy;
- shoulder stiffness / adhesive capsulitis;
- calcific tendinopathy;
- proximal-biceps-related pain;
- instability/dislocation rehabilitation;
- post-traumatic shoulder rehabilitation;
- other common real-workflow presentations to be confirmed before freeze.

Shoulder must undergo the same strict taxonomy/findings/safety/goals/rehab/evidence review before freeze.

---

# 6. Remaining body-region sequence

After shoulder:

```text
knee / hip
→ elbow
→ wrist / hand
→ ankle / foot
→ fracture / post-immobilization
→ muscle/myotendinous injury
→ generalized deconditioning / balance / gait
```

Shared post-operative profiles are included only where the product owner's actual workflow demonstrates a need.

---

# 7. Context-sensitive goals and directions

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms/changes
→ only confirmed items enter ReferralDraft
```

No global pain + ROM + strength + motor-control bundle.

Active rehabilitation, exercise, graded activity/loading, education and self-management remain the conceptual backbone where appropriate.

---

# 8. Safety / consistency engine

The utility provides prompts, not autonomous diagnoses or treatment prohibitions.

Cross-cutting examples:

```text
fracture rehab + missing healing/weight-bearing context
→ warning

post-op pathway + missing procedure/protocol/restrictions
→ warning

adjunct selected without active rehabilitation direction
→ warning

new/progressive objective neurological deficit
→ prominent medical reassessment prompt

material safety concern
→ require clinician disposition before routine reassuring wording

not assessed neurological component
→ never generate normal wording
```

Region-specific rules live in each frozen profile.

---

# 9. Output wording rules

Preferred structure:

```text
Clinical problem + important findings + functional impact.
Referral request + goals.
Rehabilitation direction / restrictions.
Optional reassessment/communication criteria.
```

Rules:

- collaborative wording;
- active/function-oriented rehabilitation as core where appropriate;
- technique-level interventions remain adjuncts;
- no unsupported diagnosis from symptom combinations or tests;
- no normal neurological/red-flag statement from missing data;
- preserve explicit restrictions;
- short and detailed outputs derive from the same `ReferralDraft`.

---

# 10. Cockpit integration target

Future navigation candidate:

```text
Clinical Excellence Home
└── Clinical Tools
    ├── Physiotherapy Referral
    └── RF Requests
```

First physiotherapy implementation remains conceptually:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence is not frozen yet.

---

# 11. RF workflow target

Longer-term RF workflow remains conceptually:

```text
new request
→ submitted / pending decision
→ approved awaiting procedure
→ performed/completed
```

Reuse must create a new request identity and must not rewrite historical truth.

---

# 12. Implementation order

```text
CU-1  Physiotherapy Referral v2 clinical/content + structured-draft design
CU-2  Physiotherapy Referral v2 implementation + Clinical Excellence styling
CU-3  Clinical Excellence navigation integration / optional patient prefill boundary
CU-4  RF lifecycle/data-model design when separately permitted
CU-5  RF clinician UI + request registry/history/reuse
CU-6  RF PDF engine ownership/migration/integration cleanup
```

---

# 13. Current design stop point

CU-1 remains **design only**.

Cervical v1.1 and lumbar v1.1 are frozen. The next substantive body-region task is shoulder design/review.

Do not write production physiotherapy runtime code until CU-1 is sufficiently frozen and the product owner explicitly authorizes transition to CU-2.