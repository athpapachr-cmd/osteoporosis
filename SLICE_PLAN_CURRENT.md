# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** ACTIVE PRE-CODE DESIGN — runtime implementation not authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin and ankle/foot v1.1.
> **Frozen shared-fracture profile on active docs branch:** `clinic_utilities/physio_profiles/shared_fracture_v1_1.md`.
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

# 2. Frozen profile status

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN
wrist_hand_v1_1 = FROZEN
knee_v1_1 = FROZEN
hip_v1_1 = FROZEN
ankle_foot_v1_1 = FROZEN
shared_fracture_v1_1 = FROZEN on docs branch pending exact-head review/merge
```

---

# 3. Shared Fracture / Post-immobilization — FROZEN v1.1 design

Authoritative frozen file:

```text
clinic_utilities/physio_profiles/shared_fracture_v1_1.md
```

One shared route:

```text
fracture_rehabilitation_post_immobilization
```

Routing contract:

```text
regional/shared entry
→ fracture site
→ treatment / phase
→ healing/stability
→ immobilization/support
→ lower-limb weight-bearing OR upper-limb use/loading
→ ROM / strengthening / impact restrictions
→ actual findings/function
→ confirmed goals/directions
```

Restriction rules:

```text
fracture site != rehabilitation clearance
elapsed time != union
immobilization removed != unrestricted use/loading
fixation != unrestricted use/loading
unknown restriction != unrestricted
exact orthopaedic/surgical protocol > shared generic suggestion
no universal week-based timeline
manual therapy requires known stability + ROM permission
```

High-visibility entries frozen from product-owner workflow:

```text
vertebral compression / fragility fracture
proximal humerus
clavicle
distal radius
hand / finger fractures
pubic rami
patella
ankle fractures
calcaneus
anterior-process calcaneus
5th metatarsal / other metatarsal
foot / toe fractures
```

Less frequent/advanced/context:

```text
scaphoid with union-confirmation gate
elbow fractures
tibial plateau/proximal tibia
Lisfranc and other site-sensitive fractures
long-bone shaft fractures
older-adult hip fracture as context rather than routine product-owner outpatient referral
```

Fragility modifier:

```text
formal_fragility_fracture_context
known_osteoporosis_or_low_bone_strength_context
falls_risk_or_recurrent_falls_context
```

When selected, mobility/independence, strength, balance and falls-risk goals become prominent. The utility does not diagnose osteoporosis or recommend osteoporosis medication from the modifier.

### SIFK / SONK

```text
preferred structured entity = subchondral_insufficiency_fracture_of_knee
preferred current term = SIFK
SONK = legacy / clinician-entered term, not separate autonomous software diagnosis
advanced SIFK may carry osteonecrosis/osteochondral-collapse context when established
```

Hard rules:

```text
bone-marrow edema alone != SIFK
sudden knee pain alone != SIFK
SIFK + loading status unknown → no generic strengthening / impact progression
established SIFK != routine OA or meniscal pathway only
```

Pediatric/physeal/apophyseal fracture group remains active but low visibility except for pelvic apophyseal avulsions. Adult timelines are never imported automatically.

Default fracture-healing recommendations excluded:

```text
acupuncture
dry needling
ESWT
therapeutic ultrasound to accelerate union
bone-stimulator prescription
```

---

# 4. Safety / consistency engine

```text
fracture + healing/stability not stated
→ warning; no healed/stable wording

lower-limb fracture + weight-bearing status not stated
→ no progressive weight-bearing instruction

upper-limb fracture + use/loading status not stated
→ no unrestricted lifting/pushing/use instruction

ROM/loading restriction not stated
→ no unrestricted ROM/strengthening/impact instruction

new trauma / loss of reduction / delayed union / nonunion / hardware concern
→ orthopaedic reassessment semantics

infection / wound / neurovascular / compartment / DVT-PE concern
→ medical/urgent reassessment semantics

possible CRPS without established diagnosis
→ preserve concern; do not autonomously diagnose

vertebral fracture + unresolved spinal precaution / neurological concern
→ medical/specialist pathway

SIFK / bone-stress / insufficiency injury + loading status unknown
→ no generic impact progression

pediatric fracture
→ no adult timeline

material safety concern + no clinician disposition
→ no routine reassuring wording
```

---

# 5. Persistence / runtime boundary

Persistence is not frozen.

Default first implementation direction remains:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Do not write production HTML/JS/CSS, add patient persistence, integrate navigation or start CU-2 without explicit product-owner authorization.

---

# 6. Exact next action

```text
1. exact branch-vs-main review of Shared Fracture v1.1 freeze
2. open docs-only PR if clean
3. independent exact-head review
4. merge only if exact head remains clean
5. clear canonical writer lock and record resulting main state
6. product owner selects next shared CU-1 profile
```

Runtime implementation remains unauthorized.
