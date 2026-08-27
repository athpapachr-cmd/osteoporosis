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
> **Current detailed profile under review:** `clinic_utilities/physio_profiles/knee_v1.md` — DESIGN CANDIDATE / NOT FROZEN.
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
knee_v1 = DESIGN CANDIDATE / NOT FROZEN
```

---

# 3. Knee — ACTIVE DESIGN CANDIDATE

Authoritative candidate:

```text
clinic_utilities/physio_profiles/knee_v1.md
```

Proposed default primary pathways for product-owner review:

```text
K1 knee osteoarthritis
K2 degenerative meniscal lesion/tear — conservative rehabilitation
K3 acute isolated meniscal injury — assessed nonoperative pathway
K4 patellofemoral pain
K5 patellar tendinopathy
K6 established knee ligament injury/instability — nonoperative rehabilitation
K7 patellar instability/dislocation rehabilitation
K8 post-traumatic knee pain/stiffness after assessed injury
K9 postoperative knee rehabilitation — pending real-workflow confirmation
```

Candidate secondary/rare/context entities:

```text
quadriceps tendinopathy
iliotibial-band syndrome
pes-anserine region pain / established bursitis-tendinopathy
Baker/popliteal cyst context
prepatellar/infrapatellar bursitis context
established osteochondral/chondral lesion or osteochondritis dissecans
meniscal root tear / complex repair-relevant meniscal lesion
Hoffa fat-pad / synovial-plica context
inflammatory / crystal knee context
Osgood-Schlatter / Sinding-Larsen-Johansson if adolescent workflow requires
```

Key candidate safety/semantic decisions:

- radiographic knee OA does not automatically prove the current pain generator or functional severity;
- degenerative MRI meniscal tear does not automatically establish symptomatic meniscal pain or surgical indication;
- exercise-based physiotherapy is the preferred first-line direction for common degenerative meniscal lesions when no structural surgical indication is present;
- clicking/catching remains distinct from a true locked knee;
- an acute displaced/displacing meniscal tear restricting ROM, true locking or a repair-relevant lesion needing timely decision requires orthopaedic reassessment rather than unrestricted routine rehabilitation;
- anterior knee pain does not automatically establish patellofemoral pain, and cartilage/`chondromalacia` imaging does not automatically establish the symptomatic pain generator;
- patellofemoral-pain core treatment is education plus knee-targeted with or without hip-targeted exercise; taping, prefabricated foot orthoses, manual therapy and movement/running retraining are presentation-specific supports;
- patellar-tendon imaging abnormality alone does not establish symptomatic patellar tendinopathy;
- patellar-tendinopathy rehabilitation is progressive load-based; no single loading style is frozen as universally superior;
- subjective giving-way does not equal objective ligament instability; a positive ligament test does not autonomously establish tear grade;
- major/multiligament instability, common-peroneal deficit or vascular concern requires specialist/reassessment semantics;
- first-time patellar dislocation requires structural/osteochondral context before routine unrestricted rehabilitation;
- post-traumatic knee wording requires unresolved fracture, extensor-mechanism rupture, locked-knee, major instability and osteochondral/neurovascular concerns to be addressed;
- postoperative rehabilitation is procedure/protocol governed; procedure-specific restrictions outrank generic knee defaults;
- TKA-specific recommendations such as early postoperative NMES must not leak into generic OA or other knee pathways;
- posterior knee swelling does not automatically establish a Baker cyst or exclude DVT;
- hot swollen knee / septic, crystal or inflammatory diagnostic uncertainty is a medical reassessment issue.

Candidate support policy:

```text
knee-OA brace → condition-sensitive
patellofemoral taping/support → condition-sensitive
prefabricated foot orthosis → selected patellofemoral presentation only
ligament/postoperative brace → exact injury/protocol governed
cane/walking aid → optional when appropriate
```

Candidate adjunct policy pending product-owner decisions:

```text
manual therapy / mobilization → optional where impairment-specific
soft tissue → optional
taping → support rather than core treatment
selected thermal strategy for OA → optional
acupuncture for knee OA → unresolved; guideline frameworks conflict
dry needling → unresolved; not core
ESWT for patellar tendinopathy → not proposed as default; unresolved only if actual workflow uses it
NMES → procedure-specific, especially post-TKA; not generic OA electrotherapy
```

Knee remains **NOT FROZEN** until the product-owner workflow decisions in `knee_v1.md` are resolved.

---

# 4. Safety / consistency engine

```text
fracture/post-trauma + unresolved structural context
→ warning/reassessment prompt

true locked knee / major ROM block
→ structural reassessment prompt

acute extensor-mechanism rupture concern / new loss of straight-leg raise
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

Region-specific details live in `knee_v1.md`.

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
1. product-owner clinical review of knee_v1.md
2. resolve postoperative-knee visibility
3. confirm degenerative-vs-acute meniscus separation
4. resolve ligament menu granularity
5. confirm patellar-instability pathway visibility
6. resolve quadriceps tendon / ITB / pes-anserine visibility
7. resolve acupuncture / dry-needling / ESWT policy
8. confirm brace/taping/foot-orthosis support model
9. confirm adolescent knee scope
10. revise candidate
11. freeze/merge only after explicit product-owner approval
```

Runtime implementation remains unauthorized.
