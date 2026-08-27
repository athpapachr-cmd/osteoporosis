# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; cervical v1.1, lumbar v1.1, shoulder v1.1, elbow v1.1, wrist/hand v1.1 and knee v1.1 frozen; Hip v1 active design candidate.

Clinic Utilities are cross-module clinician-facing operational tools, not a new clinical Module 02.

---

# 1. Physiotherapy Referral v2 target

```text
1. Clinical problem / diagnosis
2. Important findings
3. Functional limitation
4. Precautions / restrictions
5. Rehabilitation goals
6. Rehabilitation direction
7. Final referral text
```

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
brace/orthosis/taping != automatically mandatory
adjunct != core rehabilitation
clinician-entered diagnosis may be carried but not inferred
```

---

# 2. Frozen / active profile status

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN
wrist_hand_v1_1 = FROZEN
knee_v1_1 = FROZEN
hip_v1 = ACTIVE DESIGN CANDIDATE / NOT FROZEN
```

Authoritative Hip candidate:

```text
clinic_utilities/physio_profiles/hip_v1.md
```

---

# 3. Hip v1 — active candidate frame

Proposed default pathways:

```text
hip osteoarthritis
greater trochanteric pain syndrome / gluteal tendinopathy
hip-related groin pain / FAIS presentation
established acetabular labral / nonarthritic intra-articular hip pain — conservative rehabilitation
proximal hamstring tendinopathy
adductor-related groin pain
iliopsoas-related groin pain / internal snapping-hip presentation
post-traumatic hip pain/stiffness after assessed injury
postoperative hip rehabilitation — pending workflow confirmation
```

Candidate rare/advanced/context entities:

```text
established gluteus medius/minimus tendon tear — conservative pathway
symptomatic external snapping hip
acetabular dysplasia / hip instability / microinstability
known femoral-head osteonecrosis
inguinal-related / pubic-related athletic groin pain
deep-gluteal/piriformis → frozen lumbar profile
acute muscle/myotendinous injury → future shared muscle profile
fracture/stress fracture → future shared fracture profile
```

Key candidate distinctions:

```text
radiographic hip OA != automatic symptom generator
lateral hip pain != automatic GTPS/gluteal tendinopathy/trochanteric bursitis
cam/pincer morphology != FAIS
FADIR/FABER != FAIS or labral tear
MRI/MRA labral tear != automatically symptomatic pain generator
buttock/ischial pain != proximal hamstring tendinopathy
groin pain != automatically adductor or iliopsoas pathology
painless snapping != snapping-hip syndrome
postoperative rehabilitation = exact procedure/protocol governed
femoral-neck stress-fracture concern != routine tendinopathy/FAIS referral
```

Evidence-oriented core directions:

- hip OA: education/self-management plus individualized exercise; manual therapy may be impairment-specific;
- GTPS/gluteal tendinopathy: education/load-compression management plus progressive exercise as core/first line;
- nonarthritic hip pain including FAIS/labral: multimodal impairment-based rehabilitation;
- proximal hamstring/adductor/iliopsoas presentations: diagnosis-sensitive progressive loading and graded function;
- postoperative hip: exact procedure/protocol and restrictions outrank generic defaults.

Candidate adjunct questions:

```text
manual therapy / soft tissue → optional where relevant
acupuncture for hip OA → unresolved
dry needling for selected hip-OA myofascial context → unresolved; 2025 CPG supports short-term use
ESWT for GTPS/gluteal tendinopathy → unresolved
ESWT for proximal hamstring → not default; therapist-proposed context only if desired
```

Candidate pediatric/adolescent hip navigation is unresolved. If included, it is not a diagnostic umbrella: adolescent FAIS/labral/adductor/iliopsoas use ordinary pathways with age/skeletal-maturity context; apophyseal avulsion/fracture routes shared fracture; SCFE remains a medical/imaging safety route rather than a PT diagnosis.

---

# 4. Shared fracture / post-immobilization profile

Fractures remain handled once in a future shared profile rather than duplicated region by region.

Required future context:

```text
bone/site
fracture date/phase
treatment
healing/stability status
immobilization/brace/orthosis status
weight-bearing/use status
ROM/loading restrictions
surgeon/orthopaedic instructions
age/skeletal-maturity when relevant
```

Regional entry points now include shoulder, elbow, wrist/hand, knee and hip/pelvis.

Unknown healing/loading context must produce a warning rather than unrestricted rehabilitation wording.

---

# 5. Context-sensitive goals / directions

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms/changes
→ only confirmed items enter ReferralDraft
```

No global pain + ROM + strength + motor-control bundle.

Active/function-oriented rehabilitation, education, self-management and graded loading/activity remain the conceptual backbone where appropriate, subject to structural/surgical restrictions.

---

# 6. Safety / consistency engine

```text
fracture/stress-fracture concern
→ diagnostic/structural reassessment before routine rehab

post-trauma + unresolved fracture/dislocation/major avulsion context
→ warning/reassessment prompt

post-op route + missing procedure/protocol/restrictions
→ warning

hot/systemically unwell acute hip / septic-joint concern
→ medical reassessment

pediatric/adolescent SCFE concern
→ medical/imaging assessment; no routine PT diagnosis

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurological/neurovascular component
→ never generate normal wording
```

Region-specific rules live in each frozen/candidate profile.

---

# 7. Remaining regional design sequence

After Hip review/freeze, current broad sequence is:

```text
ankle / foot
→ shared fracture / post-immobilization
→ muscle / myotendinous injury
→ generalized deconditioning / balance / gait
```

The product owner may change the exact next region within CU-1.

---

# 8. Output wording rules

```text
Clinical problem + important findings + functional impact.
Referral request + goals.
Rehabilitation direction / restrictions.
Optional reassessment/communication criteria.
```

Rules:

- collaborative wording;
- active/function-oriented rehabilitation as core where appropriate;
- no unsupported diagnosis from symptoms, tests or incidental imaging;
- no normal neurological/red-flag statement from missing data;
- preserve explicit restrictions;
- short and detailed outputs derive from the same `ReferralDraft`.

---

# 9. Implementation boundary

CU-1 remains **design only**.

First implementation direction remains:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence is not frozen. Do not write production physiotherapy runtime code until CU-1 is sufficiently frozen and the product owner explicitly authorizes transition to CU-2.
