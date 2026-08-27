# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; cervical v1.1, lumbar v1.1, shoulder v1.1, elbow v1.1, wrist/hand v1.1 and knee v1.1 frozen.

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
```

---

# 3. Knee v1.1 frozen design

Frozen default pathways:

```text
knee osteoarthritis
degenerative meniscal lesion/tear — conservative rehabilitation
acute isolated meniscal injury — assessed nonoperative
patellofemoral pain
patellar tendinopathy
quadriceps tendinopathy
ACL injury/instability — nonoperative or preoperative rehabilitation
MCL injury — nonoperative rehabilitation
patellar instability/dislocation rehabilitation
iliotibial-band syndrome
pes-anserine region pain / established tendinobursitis
post-traumatic knee pain/stiffness after assessed injury
postoperative knee rehabilitation
```

Pediatric/adolescent group:

```text
Παιδιά / Έφηβοι — γόνατο
→ Osgood-Schlatter
→ Sinding-Larsen-Johansson
```

The pediatric/adolescent category is navigation only. Structural conditions such as PFP, meniscus, ACL, MCL and patellar instability continue to use their ordinary pathways with age/skeletal-maturity context.

Rare/advanced/context decisions:

```text
PCL/LCL/PLC/combined ligament → rare/advanced
distal hamstring insertional pathology → rare selectable secondary
Hoffa/plica → rare clinician-entered context
Baker/popliteal cyst → medical/context only
prepatellar/infrapatellar bursitis → medical/context only
osteochondral/OCD lesions → rare structural context
meniscal root/complex repair-relevant lesions → rare structural context
gastrocnemius strain → future shared muscle/myotendinous profile
inflammatory/crystal knee disease → established medical context
```

Key distinctions:

```text
radiographic OA != automatic symptom generator
degenerative MRI meniscal tear != automatic symptomatic diagnosis or surgery
clicking/catching != true locked knee
acute displaced/displacing meniscus != routine rehab
anterior knee pain != automatic PFP
tendon imaging change != automatic symptomatic tendinopathy
subjective giving-way != objective ligament instability
ACL and MCL are separate high-visibility nonoperative/prehab pathways
postoperative ACL/MCL → K13 only
first-time patellar dislocation requires osteochondral/structural context
postoperative meniscus repair != partial meniscectomy progression logic
pediatric category != diagnosis
Osgood-Schlatter != SLJ != ordinary patellar tendinopathy
posterior swelling != Baker cyst and does not exclude DVT
```

Frozen support/adjunct policy:

```text
taping / knee braces / foot orthoses → condition-sensitive supports
manual therapy / soft tissue → optional where relevant
acupuncture for selected knee OA → optional evidence-sensitive adjunct
dry needling → excluded
ESWT → not a default generator recommendation; therapist-proposed use may be documented for patellar tendinopathy
NMES → postoperative/context-specific, especially TKA; not generic OA
```

Postoperative emphasis:

- meniscus repair and partial meniscectomy are common real-workflow referrals;
- meniscectomy progression is primarily criterion/milestone based;
- meniscus repair/reconstruction requires time + criterion-based progression with lesion/repair-specific restrictions;
- ACLR, MCL repair/reconstruction, MPFL stabilization, arthroplasty, cartilage procedures and extensor-mechanism repair remain K13 procedure/protocol-governed subtypes.

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

Regional entry points now include shoulder, elbow, wrist/hand and knee fractures.

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
fracture rehab + missing healing/use context
→ warning

true locked knee / major mechanical ROM block
→ structural reassessment prompt

acute extensor-mechanism rupture concern
→ structural reassessment before routine rehab

major/multiligament instability or neurovascular deficit
→ specialist/urgent semantics

post-op route + missing procedure/protocol/restrictions
→ warning

hot swollen knee / infection / DVT concern
→ medical reassessment

atypical/high-risk pediatric presentation
→ do not hide under Osgood/SLJ label

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurological/neurovascular component
→ never generate normal wording
```

Region-specific rules live in each frozen profile.

---

# 7. Remaining regional design sequence

Current broad remaining sequence is:

```text
hip
→ ankle / foot
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
