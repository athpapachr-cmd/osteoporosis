# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** ACTIVE PRE-CODE DESIGN — runtime implementation not authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **Frozen shoulder profile:** `clinic_utilities/physio_profiles/shoulder_v1_1.md`.
> **Frozen elbow profile:** `clinic_utilities/physio_profiles/elbow_v1_1.md`.
> **Frozen wrist/hand profile:** `clinic_utilities/physio_profiles/wrist_hand_v1_1.md`.
> **Frozen knee profile:** `clinic_utilities/physio_profiles/knee_v1_1.md`.
> **Prior active slice:** PR-1 Transcript Intake + Candidate Extraction v3 remains intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

CU-1 is a bounded cross-module design detour. It does not authorize runtime implementation.

---

# 1. Frozen architecture

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

ReferralDraft
→ ShortReferralFormatter
→ DetailedReferralFormatter
```

Hard invariants:

```text
suggested != examined
suggested != selected
selected != mandatory
symptom != diagnosis
objective deficit != subjective symptom
special/provocation test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
brace/orthosis/taping != automatically mandatory
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

---

# 2. Frozen regional status

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN
wrist_hand_v1_1 = FROZEN
knee_v1_1 = FROZEN
```

---

# 3. Knee — FROZEN v1.1 design

Authoritative frozen file:

```text
clinic_utilities/physio_profiles/knee_v1_1.md
```

Frozen default pathways:

```text
K1 knee osteoarthritis
K2 degenerative meniscal lesion/tear — conservative rehabilitation
K3 acute isolated meniscal injury — assessed nonoperative
K4 patellofemoral pain
K5 patellar tendinopathy
K6 quadriceps tendinopathy
K7 ACL injury/instability — nonoperative or preoperative rehabilitation
K8 MCL injury — nonoperative rehabilitation
K9 patellar instability/dislocation rehabilitation
K10 iliotibial-band syndrome
K11 pes-anserine region pain / established tendinobursitis
K12 post-traumatic knee pain/stiffness after assessed injury
K13 postoperative knee rehabilitation
```

Frozen pediatric/adolescent navigation group:

```text
Παιδιά / Έφηβοι — γόνατο
→ Osgood-Schlatter
→ Sinding-Larsen-Johansson
```

This is a UI grouping, not a diagnostic umbrella. Pediatric/adolescent PFP, meniscal injury, ACL, MCL, patellar instability and fracture route through the same structural pathways with age/skeletal-maturity context.

Frozen workflow decisions:

- meniscus repair and partial meniscectomy are common postoperative referrals and remain separate procedure subtypes;
- degenerative and acute meniscal pathology remain separate because their structural/surgical semantics differ;
- ACL and MCL are separate top-level nonoperative/preoperative pathways;
- postoperative ACL/MCL routes exclusively through K13;
- patellar instability/dislocation, quadriceps tendinopathy, ITB syndrome and pes-anserine pathology are directly selectable;
- PCL/LCL/PLC/combined ligament injuries remain rare/advanced;
- distal hamstring insertional pathology and Hoffa/plica are rare selectable secondary/advanced entities;
- Baker cyst and prepatellar/infrapatellar bursitis are medical/context only;
- gastrocnemius strain routes to the future shared muscle/myotendinous profile;
- Osgood-Schlatter and Sinding-Larsen-Johansson are separate growth-related pathways with no rigid universal rehab protocol.

Frozen support/adjunct policy:

```text
taping / knee braces / foot orthoses → condition-sensitive supports
manual therapy / soft tissue → optional where relevant
acupuncture for selected knee OA → optional evidence-sensitive adjunct
dry needling → excluded
ESWT → not a default generator recommendation; therapist-proposed patellar-tendon use may be documented
NMES → procedure/context-specific, especially postoperative TKA; not generic OA
```

Key safety/semantic rules:

```text
radiographic OA != automatic symptom generator
degenerative MRI meniscal tear != automatic symptomatic lesion or surgical indication
clicking/catching != true locked knee
acute displaced/displacing meniscal tear with ROM block != routine rehab
anterior knee pain != automatic PFP
cartilage/chondromalacia imaging != automatic symptomatic diagnosis
tendon imaging change != automatic tendinopathy
subjective giving-way != objective ACL/MCL instability
ligament test != autonomous tear grade
postoperative ACL/MCL != K7/K8; use K13
time alone != return-to-sport clearance
first-time patellar dislocation requires structural/osteochondral context
posterior swelling != automatic Baker cyst and does not exclude DVT
pediatric anterior-knee pain != automatic Osgood/SLJ
postoperative rehabilitation = exact procedure/protocol governed
not_assessed neurovascular component != normal
```

---

# 4. Safety / consistency engine

```text
fracture/post-trauma + unresolved structural context
→ warning/reassessment prompt

true locked knee / major ROM block
→ structural reassessment prompt

acute extensor-mechanism rupture concern / loss of straight-leg raise
→ structural reassessment prompt

major/multiligament instability or neurovascular deficit
→ urgent/specialist semantics

post-op route + missing procedure/protocol/restrictions
→ warning

hot swollen knee / infection or DVT concern
→ medical reassessment; no routine reassuring wording

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurological/neurovascular component
→ never generate normal wording
```

Region-specific details live in `knee_v1_1.md`.

---

# 5. Shared fracture / post-immobilization boundary

Knee-region fractures route to the future shared profile:

```text
patella fracture
distal femur fracture
proximal tibia / tibial plateau fracture
proximal fibula fracture
other knee-region fracture
```

Unresolved healing/stability, immobilization/brace, weight-bearing, ROM or loading context prevents unrestricted rehabilitation wording.

---

# 6. Persistence / runtime boundary

Persistence is not frozen.

Default first implementation direction remains:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Do not write production HTML/JS/CSS, add patient persistence, integrate navigation or start CU-2 without explicit product-owner authorization.

---

# 7. Exact next action

```text
1. product owner selects the next remaining CU-1 regional/shared profile
2. use the same taxonomy/findings/safety/goals/rehab/evidence method
3. continue CU-1 design only
```

Runtime implementation remains unauthorized.
