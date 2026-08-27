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
> **Current detailed profile under review:** `clinic_utilities/physio_profiles/hip_v1.md` — DESIGN CANDIDATE / NOT FROZEN.
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

# 2. Frozen / active regional status

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN
wrist_hand_v1_1 = FROZEN
knee_v1_1 = FROZEN
hip_v1 = DESIGN CANDIDATE / NOT FROZEN
```

---

# 3. Hip — ACTIVE DESIGN CANDIDATE

Authoritative candidate:

```text
clinic_utilities/physio_profiles/hip_v1.md
```

Proposed default primary pathways for product-owner review:

```text
H1 hip osteoarthritis
H2 greater trochanteric pain syndrome / gluteal tendinopathy
H3 hip-related groin pain / femoroacetabular impingement syndrome presentation
H4 established acetabular labral / nonarthritic intra-articular hip pain — conservative rehabilitation
H5 proximal hamstring tendinopathy
H6 adductor-related groin pain
H7 iliopsoas-related groin pain / internal snapping-hip presentation
H8 post-traumatic hip pain/stiffness after assessed injury
H9 postoperative hip rehabilitation — pending workflow confirmation
```

Candidate rare/advanced/context entities:

```text
gluteus medius/minimus established tendon tear — conservative pathway
symptomatic external snapping hip
established acetabular dysplasia / hip instability / microinstability
known femoral-head osteonecrosis
 inguinal-related / pubic-related athletic groin pain
deep-gluteal/piriformis → route to frozen lumbar profile
acute adductor/iliopsoas/rectus-femoris/hamstring strain → future shared muscle/myotendinous profile
fracture/stress fracture → future shared fracture/post-immobilization profile
```

Key candidate distinctions:

```text
radiographic hip OA != automatic symptom generator
lateral hip pain != automatically GTPS, gluteal tendinopathy or trochanteric bursitis
cam/pincer morphology != FAIS
FADIR/FABER finding != FAIS or labral tear
MRI/MRA labral tear != automatically symptomatic pain generator
buttock/ischial pain != proximal hamstring tendinopathy
athletic groin pain may contain multiple Doha entities
groin pain != automatically adductor or iliopsoas pathology
painless snapping != symptomatic snapping-hip syndrome
postoperative rehabilitation = exact procedure/protocol governed
suspected femoral-neck stress fracture != routine tendinopathy/FAIS referral
```

Candidate evidence-oriented core directions:

- hip OA: education/self-management plus individualized exercise; manual therapy may be impairment-specific;
- GTPS/gluteal tendinopathy: education/load-compression management plus progressive exercise as core/first line;
- nonarthritic hip pain including FAIS/labral: multimodal impairment-based rehabilitation with activity modification, hip/trunk/lower-limb strengthening and movement retraining where relevant;
- proximal hamstring/adductor/iliopsoas presentations: diagnosis-sensitive progressive loading and graded function;
- postoperative hip: procedure/protocol restrictions outrank generic defaults.

Candidate adjunct questions:

```text
manual therapy / soft tissue → optional where relevant
acupuncture for hip OA → unresolved product-owner decision
dry needling for selected hip-OA myofascial context → unresolved product-owner decision; 2025 CPG supports short-term use
ESWT for GTPS/gluteal tendinopathy → unresolved product-owner decision
ESWT for proximal hamstring → not default; therapist-proposed documentation only if desired
```

Candidate pediatric/adolescent navigation group is unresolved. If included, it is navigation/safety only: adolescent FAIS/labral/adductor/iliopsoas use ordinary pathways with age/skeletal-maturity context; fractures/apophyseal avulsions route shared fracture; SCFE remains urgent medical/imaging routing, not physiotherapy diagnosis.

Hip remains **NOT FROZEN** until product-owner workflow decisions in `hip_v1.md` are resolved.

---

# 4. Safety / consistency engine

```text
fracture/post-trauma + unresolved structural context
→ warning/reassessment prompt

exercise-related hip/groin pain + femoral-neck stress-fracture concern
→ diagnostic/structural reassessment before routine rehabilitation

acute inability to bear weight after trauma without assessment
→ structural reassessment

major tendon avulsion/rupture or hip dislocation concern
→ specialist/urgent semantics

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

Region-specific details live in `hip_v1.md`.

---

# 5. Shared fracture / muscle boundary

Hip-region fractures route to the future shared fracture profile:

```text
femoral neck
intertrochanteric/subtrochanteric
acetabulum
pelvic ring/rami
femoral-neck stress fracture
pelvic stress/insufficiency fracture
adolescent apophyseal avulsion fracture
other hip/pelvic fracture
```

Acute muscle/myotendinous injuries route to the future shared muscle profile rather than being duplicated in Hip v1.

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
1. product-owner clinical review of `hip_v1.md`
2. resolve H1-H9 visibility and H3-vs-H4 granularity
3. resolve gluteal-tear / external-snapping / groin-entity visibility
4. resolve acupuncture / dry-needling / ESWT policy
5. confirm pediatric/adolescent hip scope
6. revise candidate
7. freeze/merge only after explicit product-owner approval
```

Runtime implementation remains unauthorized.
