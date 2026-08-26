# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; cervical v1.1, lumbar v1.1 and shoulder v1.1 frozen.

Clinic Utilities are cross-module clinician-facing operational tools, not a new clinical Module 02.

---

# 1. Detour purpose

Integrate two existing clinician-created tools into the Clinical Excellence workspace:

1. **Physiotherapy Referral Generator** — structured clinical referral text generation.
2. **Radiofrequency Request / PDF Workflow** — request creation, PDF generation, lifecycle tracking and reuse of previous request data.

CU-1 currently covers physiotherapy clinical/content design only.

---

# 2. Physiotherapy Referral v2 target

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

Structured intermediate model:

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
symptom != diagnosis
objective deficit != subjective symptom
provocation/special test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried but not inferred
```

---

# 3. Frozen profile status

## 3.1 Cervical — FROZEN v1.1

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
```

Key capabilities include non-specific/mechanical neck pain, radiating/radicular features, clinician-entered formal cervicogenic headache, clinician-entered cervical/cervicogenic dizziness, whiplash/post-traumatic pain, directly selectable myofascial/trigger-point findings and referred shoulder-girdle pain, with strict neurological tri-state semantics.

## 3.2 Lumbar — FROZEN v1.1

```text
clinic_utilities/physio_profiles/lumbar_v1_1.md
```

Includes non-specific/mechanical LBP, radiating/radicular features, lumbar stenosis/neurogenic claudication and deep-gluteal/piriformis presentation. Preserves explicit cauda-equina reassessment semantics, optional acupuncture/dry needling with evidence/competence caveats, no routine lumbar traction and no lumbar `SI dysfunction` diagnosis.

## 3.3 Shoulder — FROZEN v1.1

```text
clinic_utilities/physio_profiles/shoulder_v1_1.md
```

Frozen shoulder pathways:

```text
rotator-cuff-related shoulder pain / tendinopathy
confirmed full-thickness cuff tear — conservative rehabilitation
calcific rotator-cuff tendinopathy
adhesive capsulitis / frozen shoulder
glenohumeral instability / dislocation rehabilitation
glenohumeral osteoarthritis
post-traumatic assessed shoulder injury
acromioclavicular-joint disorder
sternoclavicular-joint disorder
postoperative shoulder rehabilitation
```

Frozen shoulder supporting decisions:

- long-head biceps tendinopathy is a common directly selectable secondary/coexisting diagnosis;
- `impingement syndrome` is not the preferred primary diagnosis;
- special tests remain findings;
- active and passive ROM remain separate;
- acute traumatic marked weakness/inability to elevate requires reassessment semantics;
- AC-joint pathology can stand alone as a primary referral entity;
- sternoclavicular pathology requires stronger diagnosis/context governance; suspected posterior SC dislocation and unexplained swelling/systemic concern are not routine physiotherapy pathways;
- acupuncture and dry needling remain optional adjuncts with competence/availability safeguards;
- ESWT is calcific-specific; prior barbotage/lavage is context, not an automatic sequencing rule;
- postoperative shoulder is active and requires procedure/protocol/restriction context;
- shoulder fractures route to the shared fracture/post-immobilization profile.

---

# 4. Shared fracture / post-immobilization profile

Fractures should be handled in one shared profile rather than duplicated region by region.

Required future context:

```text
bone/site
fracture date/phase
treatment
healing/stability status if known
immobilization status
weight-bearing/use status
ROM/loading restrictions
surgeon/orthopaedic instructions
```

Regional entry points may route to it, including:

```text
proximal humerus
clavicle
scapula
wrist/distal radius
hip
knee/lower-limb
ankle/foot
other
```

Unknown healing/loading context must produce a warning rather than unrestricted rehabilitation wording.

---

# 5. Context-sensitive goals and directions

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms/changes
→ only confirmed items enter ReferralDraft
```

No global pain + ROM + strength + motor-control bundle.

Active rehabilitation, exercise, graded activity/loading, education and self-management remain the conceptual backbone where appropriate.

---

# 6. Safety / consistency engine

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

material safety/red-flag concern + no clinician disposition
→ do not generate routine reassuring wording

unassessed neurological component
→ never generate normal wording
```

Region-specific rules live in each frozen profile.

---

# 7. Remaining regional design sequence

Still to freeze before CU-2 implementation:

```text
knee / hip
→ elbow
→ wrist / hand
→ ankle / foot
→ shared fracture / post-immobilization
→ muscle / myotendinous injury
→ generalized deconditioning / balance / gait
```

The exact next region is selected by the product owner after the shoulder handoff closes.

---

# 8. Output wording rules

Preferred structure:

```text
Clinical problem + important findings + functional impact.
Referral request + goals.
Rehabilitation direction / restrictions.
Optional reassessment/communication criteria.
```

Wording must be collaborative, active/function-oriented where appropriate, free of unsupported diagnoses, and must never generate reassuring negative statements from missing data.

---

# 9. Integration / persistence boundary

First implementation remains conceptually:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence is not yet frozen. If later persisted, reuse creates a new referral identity and never rewrites historical truth.

Future navigation candidate:

```text
Clinical Excellence Home
└── Clinical Tools
    ├── Physiotherapy Referral
    └── RF Requests
```

---

# 10. RF boundary

Existing RF implementation was inspected read-only in `athpapachr-cmd/ortho-reception-backend-v2`.

RF runtime mutation is not part of CU-1 and remains governed separately.

---

# 11. Implementation order

```text
CU-1  Physiotherapy Referral v2 clinical/content + structured-draft design
CU-2  Physiotherapy Referral v2 implementation + Clinical Excellence styling
CU-3  Clinical Excellence navigation integration / optional patient prefill boundary
CU-4  RF lifecycle/data-model design when separately permitted
CU-5  RF clinician UI + request registry/history/reuse
CU-6  RF PDF engine ownership/migration/integration cleanup
```

CU-1 remains **design only**. Do not write production physiotherapy runtime code until the remaining regional profiles/cross-region decisions are sufficiently frozen and the product owner explicitly authorizes CU-2.
