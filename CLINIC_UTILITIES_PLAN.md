# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; all regional v1.1 profiles frozen; Shared Fracture / Post-immobilization v1 active design candidate.

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
hip_v1_1 = FROZEN
ankle_foot_v1_1 = FROZEN
shared_fracture_v1 = ACTIVE DESIGN CANDIDATE / NOT FROZEN
```

Authoritative shared-fracture candidate:

```text
clinic_utilities/physio_profiles/shared_fracture_v1.md
```

---

# 3. Shared Fracture / Post-immobilization v1 — active candidate frame

The shared profile owns reusable fracture rehabilitation constraints once, rather than duplicating them in every body region.

Canonical design:

```text
regional gateway
→ fracture_rehabilitation_post_immobilization
→ fracture site
→ treatment / phase / healing-stability
→ immobilization / use or weight-bearing / ROM / loading restrictions
→ actual deficits / function
→ confirmed rehabilitation goals/directions
```

Minimum required shared context:

```text
fracture site
laterality
date/phase when known
treatment / surgery when applicable
healing/stability status
immobilization/support status
lower-limb weight-bearing status when relevant
upper-limb use/loading status when relevant
ROM restrictions
loading / strengthening / impact restrictions
orthopaedic/surgical instructions
age/skeletal maturity when relevant
```

Hard rules:

```text
fracture != healed fracture
elapsed time != union
cast/sling/boot removal != unrestricted loading
fixation != unrestricted use/loading
not stated != unrestricted
exact protocol > shared generic suggestion
pediatric fracture != adult timeline
fragility mechanism != osteoporosis diagnosis automatically
```

Candidate registry extends frozen regional gateways to include long-bone shaft fractures and reusable stress/bone-stress and pediatric physeal/apophyseal groups.

Candidate workflow decision:

```text
vertebral compression / fragility fracture
→ active shared route vs context-only
```

Core rehabilitation is active/function-oriented but restriction-governed:

```text
ROM restoration when allowed
progressive strengthening when allowed
upper-limb use / lower-limb weight-bearing progression when allowed
gait / balance / proprioception
functional task retraining
edema / scar / desensitization work
endurance / walking progression
falls/balance rehabilitation after fragility fracture when appropriate
criterion-based work/gym/sport progression
```

No universal week-based timetable is generated.

Safety domains include:

```text
reinjury / loss of reduction / delayed union / nonunion / malunion / hardware concern
neurovascular deficit / compartment concern
DVT/PE concern
infection/wound/pin-site concern
possible CRPS without autonomous diagnosis
stress-fracture impact uncertainty
pediatric physeal/apophyseal restrictions
spinal neurological/stability concern if vertebral route retained
```

Default fracture-healing modalities are not generated:

```text
therapeutic ultrasound to accelerate union
ESWT to accelerate union
acupuncture as fracture-healing treatment
dry needling at/around incompletely healed fracture
bone-stimulator prescription
```

---

# 4. Regional fracture gateway registry already frozen

## Shoulder / arm

```text
proximal humerus
clavicle
scapula
```

Shared candidate additionally includes humeral shaft.

## Elbow

```text
radial head/neck
olecranon/proximal ulna
distal humerus
coronoid
```

## Forearm / wrist / hand

```text
distal radius
distal ulna
scaphoid
other carpal
metacarpal
phalangeal
```

Shared candidate additionally includes radius/ulna shaft.

## Hip / pelvis

```text
femoral neck
intertrochanteric / subtrochanteric
acetabulum
pelvic ring / pubic rami
femoral-neck stress fracture
pelvic stress / insufficiency fracture
ASIS / AIIS / ischial-tuberosity / lesser-trochanter apophyseal avulsion
```

Shared candidate additionally includes femoral shaft and sacral insufficiency context.

## Knee / leg

```text
patella
distal femur
proximal tibia / tibial plateau
proximal fibula
```

Shared candidate additionally includes tibial/fibular shafts.

## Ankle / foot

```text
malleolar fractures
fibula / Maisonneuve context
talus / calcaneus / navicular
cuboid / cuneiform
5th metatarsal / other metatarsals
phalanges
Lisfranc fracture-dislocation
stress / bone-stress injury
```

Unknown healing/loading context always prevents unrestricted rehabilitation wording.

---

# 5. Shared muscle / myotendinous profile — next after fracture

Important future entries already generated by regional gateways include:

```text
proximal rectus-femoris tendon/myotendinous injury
adductor strain/tear
iliopsoas/hip-flexor strain
rectus-femoris strain
hamstring strain
gastrocnemius strain
soleus strain
calf myotendinous injury
other regional acute muscle/tendon injury
```

This work does not start until the Shared Fracture writer lock closes.

---

# 6. Context-sensitive goals / directions

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms/changes
→ only confirmed items enter ReferralDraft
```

No global pain + ROM + strength bundle.

Fracture rehabilitation is explicitly governed by healing/stability and restrictions before generic active rehabilitation suggestions become available.

---

# 7. Safety / consistency engine

```text
fracture + unknown healing/stability
→ warning; no healed/stable wording

lower-limb fracture + weight-bearing status unknown
→ no progressive weight-bearing instruction

upper-limb fracture + use/loading status unknown
→ no unrestricted lifting/pushing/use instruction

ROM/loading restriction unknown
→ no unrestricted ROM/strengthening statement

new trauma / loss-of-reduction / delayed-union / hardware concern
→ orthopaedic reassessment semantics

infection / neurovascular / compartment / DVT-PE concern
→ medical/urgent reassessment semantics

possible CRPS
→ preserve concern; do not autonomously diagnose

pediatric fracture
→ no adult timeline

material safety concern + no clinician disposition
→ no routine reassuring wording
```

---

# 8. Remaining shared design sequence

Current broad remaining sequence is:

```text
shared fracture / post-immobilization — ACTIVE CANDIDATE
→ muscle / myotendinous injury
→ generalized deconditioning / balance / gait
```

The product owner may change the exact next profile within CU-1 after the active writer lock closes.

---

# 9. Output wording rules

```text
Clinical problem + fracture/site/treatment context + actual deficits + functional impact.
Referral request + goals.
Exact restrictions / permitted progression.
Optional reassessment/communication criteria.
```

Rules:

- collaborative wording;
- no unsupported healing/stability assertion;
- no automatic timeline from elapsed weeks;
- no normal neurological/red-flag statement from missing data;
- preserve exact orthopaedic restrictions;
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
