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
> **Frozen hip/groin profile:** `clinic_utilities/physio_profiles/hip_v1_1.md`.
> **Frozen ankle/foot profile:** `clinic_utilities/physio_profiles/ankle_foot_v1_1.md`.
> **Current detailed shared profile under review:** `clinic_utilities/physio_profiles/shared_fracture_v1.md` — DESIGN CANDIDATE / NOT FROZEN.
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

# 2. Frozen / active profile status

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN
wrist_hand_v1_1 = FROZEN
knee_v1_1 = FROZEN
hip_v1_1 = FROZEN
ankle_foot_v1_1 = FROZEN
shared_fracture_v1 = DESIGN CANDIDATE / NOT FROZEN
```

---

# 3. Shared Fracture / Post-immobilization — ACTIVE DESIGN CANDIDATE

Authoritative candidate:

```text
clinic_utilities/physio_profiles/shared_fracture_v1.md
```

Architectural model:

```text
regional fracture gateway
→ fracture_rehabilitation_post_immobilization
→ fracture_site
→ treatment / phase / healing-stability
→ immobilization / use or weight-bearing / ROM / loading restrictions
→ actual findings / function
→ clinician-confirmed goals and directions
```

The shared profile owns restriction/healing semantics once. Regional profiles provide entry points and site context rather than separate fracture protocols.

Required candidate context:

```text
fracture_site
laterality
fracture date / phase when known
fracture context: traumatic / fragility-insufficiency / stress-bone-stress / pediatric physeal-apophyseal / other
management / operation if applicable
healing-stability status
immobilization/support status
lower-limb weight-bearing status when relevant
upper-limb use/loading status when relevant
ROM status / restrictions
loading-strengthening-impact restrictions
orthopaedic/surgical instructions and source
```

Hard rules:

```text
fracture != healed fracture
elapsed time != union
pain reduction != healing confirmation
cast/boot/sling removal != unrestricted loading
fixation != unrestricted loading
unknown weight-bearing/use/ROM/loading state != unrestricted
exact orthopaedic protocol > generic shared suggestion
pediatric fracture != adult timeline
fragility mechanism != software-diagnosed osteoporosis
vertebral fracture != routine nonspecific back-pain pathway
```

Candidate site registry includes:

```text
shoulder/arm: proximal humerus, humeral shaft, clavicle, scapula
elbow: distal humerus, radial head/neck, olecranon/proximal ulna, coronoid
forearm/wrist/hand: shaft radius/ulna, distal radius/ulna, scaphoid/carpal, metacarpal, phalangeal
hip/pelvis/femur: femoral neck, inter/subtrochanteric, femoral shaft, acetabular, pelvic ring/rami, sacral/pelvic insufficiency
knee/leg: distal femur, patella, tibial plateau/proximal tibia, proximal fibula, tibial/fibular shaft
ankle/foot: malleolar, fibular/Maisonneuve, talus, calcaneus, navicular, cuboid/cuneiform, metatarsal, phalangeal, Lisfranc
dedicated context groups: stress/bone-stress injuries, pediatric physeal/apophyseal fractures
candidate workflow decision: vertebral compression/fragility fracture
```

Core rehabilitation may include, only when allowed by actual restrictions:

```text
safe ROM restoration
progressive strengthening
progressive upper-limb use or lower-limb weight bearing
gait / balance / proprioception
functional task retraining
edema / scar / desensitization work
walking/endurance progression
falls-risk / balance intervention after fragility fracture when appropriate
criterion-based return to work/gym/sport
```

No universal week-based protocol is frozen.

Safety/reassessment domains:

```text
loss of reduction / reinjury / delayed union / nonunion / hardware concern
new neurovascular deficit / compartment concern
DVT/PE concern for lower-limb context
infection / wound / pin-site concern
possible CRPS without automatic diagnosis
stress-fracture impact/loading uncertainty
pediatric physeal/apophyseal restrictions
vertebral/spinal neurological or stability concern if vertebral route retained
```

Default fracture-healing adjunct recommendations are excluded:

```text
therapeutic ultrasound to accelerate union
ESWT to accelerate union
acupuncture as fracture-healing treatment
dry needling around incompletely healed fracture
bone-stimulator prescription
```

Shared Fracture remains **NOT FROZEN** until product-owner workflow decisions in `shared_fracture_v1.md` are resolved.

---

# 4. Persistence / runtime boundary

Persistence is not frozen.

Default first implementation direction remains:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Do not write production HTML/JS/CSS, add patient persistence, integrate navigation or start CU-2 without explicit product-owner authorization.

---

# 5. Exact next action

```text
1. product-owner clinical review of `shared_fracture_v1.md`
2. resolve fracture-site visibility and common-vs-advanced entries
3. resolve vertebral / fragility / pediatric scope
4. resolve restriction and adjunct policy
5. revise candidate
6. freeze/merge only after explicit product-owner approval
```

Runtime implementation remains unauthorized.
