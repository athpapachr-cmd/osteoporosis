# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; cervical v1.1, lumbar v1.1, shoulder v1.1, elbow v1.1, wrist/hand v1.1, knee v1.1 frozen; Hip/Groin v1.1 frozen on docs branch pending review/merge.

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

# 2. Frozen profile status

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN
wrist_hand_v1_1 = FROZEN
knee_v1_1 = FROZEN
hip_v1_1 = FROZEN on docs branch pending exact-head review/merge
```

---

# 3. Hip / Groin v1.1 frozen design

Frozen routine pathways:

```text
lateral hip / greater-trochanteric pain pathway
nonarthritic intra-articular hip pain — FAIS / symptomatic labral
adductor-related groin pain / adductor tendinopathy
post-traumatic hip/groin pain or stiffness after assessed injury
```

Direct shared-profile gateways:

```text
proximal rectus femoris / proximal quadriceps tendon injury in athletes
→ shared muscle/myotendinous profile

pelvic apophyseal avulsion fracture in children/adolescents
→ shared fracture/post-immobilization profile
```

Workflow decisions:

```text
hip OA → medical/context only in this referral workflow
GTPS/lateral hip → visible; clinician-entered trochanteric bursitis retained
FAIS + symptomatic labral pathology → one nonarthritic intra-articular pathway
adductor-related groin pain → high visibility
proximal hamstring tendinopathy → rare/secondary
iliopsoas/internal snapping hip → rare/secondary
gluteal-abductor tendon tear → very rare/advanced
external snapping / dysplasia-instability / inguinal-pubic groin → rare/advanced
postoperative hip → not routine active pathway
no generic pediatric/adolescent Hip category
deep-gluteal/piriformis → frozen lumbar profile
```

Adjunct policy:

```text
manual therapy / soft tissue → optional where relevant
dry needling → optional clinician-selected in appropriate myofascial context
acupuncture → excluded
ESWT for GTPS/proximal hamstring → not generator-recommended; therapist-proposed use may be documented
```

Key distinctions:

```text
lateral hip pain != GTPS/gluteal tendinopathy/trochanteric bursitis automatically
cam/pincer morphology != FAIS
FADIR/FABER != FAIS or labral tear
MRI/MRA labral tear != automatically symptomatic
athletic groin pain may contain multiple entities
groin pain != automatically adductor pathology
painless snapping != snapping-hip syndrome
femoral-neck stress-fracture concern != routine tendon/FAIS referral
proximal rectus-femoris injury != ASIS avulsion
AIIS commonly relates to rectus-femoris origin
ASIS classically relates to sartorius traction
```

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

Hip/pelvis specifically adds:

```text
femoral neck / proximal femur
acetabulum / pelvic ring / rami
femoral-neck stress fracture
pelvic stress/insufficiency fracture
ASIS / AIIS / ischial-tuberosity / lesser-trochanter apophyseal avulsion
other pelvic apophyseal avulsion
```

Unknown healing/loading context must produce a warning rather than unrestricted rehabilitation wording.

---

# 5. Shared muscle / myotendinous profile

Hip/Groin now creates an explicit need for a reusable shared profile rather than region-specific duplication.

Important future entries:

```text
proximal rectus-femoris tendon/myotendinous injury
adductor strain/tear
iliopsoas/hip-flexor strain
rectus-femoris strain
hamstring strain
gastrocnemius strain from knee entry
other regional muscle/tendon injury
```

The Hip UI should be able to expose high-frequency regional gateways while the shared profile owns reusable injury grading, structural concern, loading/healing and return-to-sport semantics.

---

# 6. Context-sensitive goals / directions

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms/changes
→ only confirmed items enter ReferralDraft
```

No global pain + ROM + strength + motor-control bundle.

Active/function-oriented rehabilitation, education, self-management and graded loading/activity remain the conceptual backbone where appropriate, subject to structural restrictions.

---

# 7. Safety / consistency engine

```text
fracture/stress-fracture concern
→ diagnostic/structural reassessment before routine rehab

post-trauma + unresolved fracture/dislocation/major tendon-avulsion context
→ warning/reassessment prompt

proximal rectus-femoris major tear/avulsion concern
→ sports-medicine/structural assessment semantics

child/adolescent pelvic-apophyseal avulsion concern
→ imaging/structural pathway before unrestricted rehabilitation

hot/systemically unwell acute hip / septic-joint concern
→ medical reassessment

SCFE concern
→ medical/imaging assessment; no routine PT diagnosis

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurological/neurovascular component
→ never generate normal wording
```

Region-specific rules live in each frozen profile.

---

# 8. Remaining regional design sequence

After Hip handoff closes, current broad sequence is:

```text
ankle / foot
→ shared fracture / post-immobilization
→ muscle / myotendinous injury
→ generalized deconditioning / balance / gait
```

The product owner may change the exact next region within CU-1.

---

# 9. Output wording rules

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

# 10. Implementation boundary

CU-1 remains **design only**.

First implementation direction remains:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence is not frozen. Do not write production physiotherapy runtime code until CU-1 is sufficiently frozen and the product owner explicitly authorizes transition to CU-2.
