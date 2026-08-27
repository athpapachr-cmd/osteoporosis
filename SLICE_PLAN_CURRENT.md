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
> **Frozen hip/groin profile on active docs branch:** `clinic_utilities/physio_profiles/hip_v1_1.md`.
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
hip_v1_1 = FROZEN on docs branch pending exact-head review/merge
```

---

# 3. Hip / Groin — FROZEN v1.1 design

Authoritative frozen file:

```text
clinic_utilities/physio_profiles/hip_v1_1.md
```

Frozen routine primary pathways:

```text
H1 lateral hip / greater-trochanteric pain pathway
H2 nonarthritic intra-articular hip pain — FAIS / symptomatic labral
H3 adductor-related groin pain / adductor tendinopathy
H4 post-traumatic hip/groin pain or stiffness after assessed injury
```

High-value direct shared-profile gateways:

```text
proximal rectus femoris / proximal quadriceps tendon injury in athletes
→ future shared muscle/myotendinous profile

pelvic apophyseal avulsion fracture in children/adolescents
→ future shared fracture/post-immobilization profile
```

Frozen workflow decisions:

- hip OA is context only because it is not routinely referred;
- lateral hip/GTPS remains directly selectable and clinician-entered trochanteric bursitis is preserved as a subtype;
- FAIS and symptomatic acetabular labral pathology are combined into one nonarthritic intra-articular pathway;
- adductor-related groin pain is high visibility because it is common in the product-owner workflow;
- proximal hamstring tendinopathy is rare/secondary;
- iliopsoas-related pain/internal snapping hip is rare/secondary;
- established gluteus medius/minimus tears are very rare/advanced;
- external snapping hip, dysplasia/instability and inguinal/pubic-related athletic groin pain remain rare/advanced;
- postoperative hip is not a routine pathway;
- there is no general pediatric/adolescent Hip navigation group;
- pelvic apophyseal avulsions remain visible via the shared-fracture gateway because the product owner sees them;
- deep-gluteal/piriformis routes to the frozen lumbar profile.

Frozen adjunct policy:

```text
manual therapy / soft tissue → optional where relevant
dry needling → optional clinician-selected adjunct in appropriate myofascial context
acupuncture → excluded
ESWT for GTPS / proximal hamstring → not generator-recommended; therapist-proposed use may be documented
```

Key safety/semantic rules:

```text
lateral hip pain != automatic GTPS / trochanteric bursitis / gluteal tendinopathy
cam/pincer morphology != FAIS
FADIR/FABER != FAIS or labral tear
MRI/MRA labral tear != automatically symptomatic pain generator
groin pain != automatically adductor-related
painless snapping != symptomatic snapping-hip syndrome
femoral-neck stress-fracture concern != routine tendon/FAIS referral
proximal rectus-femoris injury != ASIS avulsion
AIIS avulsion may be rectus-femoris-origin related
ASIS avulsion is classically sartorius-related
known apophyseal avulsion + unknown healing/loading status → warning
not_assessed neurovascular component != normal
```

---

# 4. Safety / consistency engine

```text
fracture/stress-fracture concern
→ diagnostic/structural reassessment before routine rehab

post-trauma + unresolved fracture/dislocation/major avulsion context
→ warning/reassessment prompt

acute major proximal rectus-femoris tear/avulsion concern
→ sports-medicine/structural reassessment semantics

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

Region-specific details live in `hip_v1_1.md`.

---

# 5. Shared fracture / muscle boundary

Hip/pelvic fracture entries include:

```text
femoral neck
intertrochanteric/subtrochanteric
acetabulum
pelvic ring/rami
femoral-neck stress fracture
pelvic stress/insufficiency fracture
ASIS apophyseal avulsion
AIIS apophyseal avulsion
ischial-tuberosity avulsion
lesser-trochanter avulsion
other pelvic apophyseal avulsion
other hip/pelvic fracture
```

Shared muscle/myotendinous entries include:

```text
proximal rectus-femoris tendon/myotendinous injury
adductor strain/tear
iliopsoas/hip-flexor strain
rectus-femoris muscle strain
hamstring strain
other acute hip/pelvic muscle/tendon injury
```

Unknown healing/stability/loading context prevents unrestricted rehabilitation wording.

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
1. exact branch-vs-main review of Hip v1.1 freeze
2. open docs-only Hip freeze PR if clean
3. independent exact-head review
4. merge only if exact head remains clean
5. clear canonical writer lock and record resulting main state
6. product owner selects next CU-1 regional/shared profile
```

Runtime implementation remains unauthorized.
