# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; cervical v1.1, lumbar v1.1, shoulder v1.1, elbow v1.1, wrist/hand v1.1, knee v1.1 and Hip/Groin v1.1 frozen; Ankle / Foot v1.1 frozen on docs branch pending review/merge.

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
hip_v1_1 = FROZEN
ankle_foot_v1_1 = FROZEN on docs branch pending exact-head review/merge
```

---

# 3. Ankle / Foot v1.1 frozen design

Frozen routine pathways:

```text
acute/postacute lateral ankle sprain
Achilles tendinopathy — midportion/insertional
plantar heel pain / plantar fasciitis
posterior tibial tendon dysfunction / flexible PCFD
peroneal tendon disorder — conservative rehabilitation
mechanical metatarsalgia / forefoot overload
post-traumatic ankle/foot pain or stiffness after assessed injury
```

Pediatric/adolescent navigation:

```text
Παιδιά / Έφηβοι — ποδοκνημική / άκρος πόδας
→ Sever disease
→ symptomatic accessory navicular
→ symptomatic flexible flatfoot
```

Rare/advanced/context decisions:

```text
chronic ankle instability / recurrent sprain → rare/secondary
syndesmotic/high-ankle sprain → very rare/advanced
tarsal tunnel → rare neurological
heel fat-pad pain → rare/secondary
Morton neuroma → rare/context
plantar-plate / lesser-MTP instability → very rare/advanced
anterior tibial / extensor / FHL tendon disorders → rare
osteochondral talus lesion → rare/advanced
hallux rigidus / 1st-MTP OA → context only
ankle OA → context only
Charcot / neuropathic foot → medical/offloading safety context
postoperative ankle/foot → advanced only, especially occasional Achilles repair/reconstruction
```

Frozen support/adjunct policy:

```text
taping → directly visible optional support
heel lift → directly visible optional support
brace / orthosis / AFO / metatarsal pad / footwear / offloading → condition-specific context, often podiatry-coordinated
manual therapy / soft tissue → optional where relevant
dry needling → optional clinician-selected adjunct
acupuncture → excluded
ESWT plantar heel → evidence-supported optional adjunct
ESWT Achilles → evidence-conflicted optional adjunct; not routine or superior to progressive loading
```

Key distinctions:

```text
lateral ankle trauma != uncomplicated LAS automatically
subjective giving-way != objective CAI
brace/taping != stand-alone CAI rehabilitation
Achilles pain / imaging change != tendinopathy automatically
midportion Achilles != insertional protocol automatically
plantar heel pain != plantar fasciitis automatically
heel spur != automatic pain generator
central heel pain may represent fat-pad or bone pathology rather than plantar fascia
flat foot + medial pain != PCFD/PTTD automatically
lateral pain != peroneal tendinopathy automatically
peroneal snapping/subluxation != routine sprain/tendinopathy
forefoot pain != metatarsalgia / Morton / plantar-plate diagnosis automatically
pediatric heel pain != Sever automatically
accessory navicular imaging != symptomatic diagnosis automatically
asymptomatic pediatric flexible flatfoot != treatment pathway
hot swollen neuropathic foot / Charcot concern != routine PT
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

Foot/ankle adds:

```text
malleolar / fibular / Maisonneuve context
talus / calcaneus / navicular
cuboid / cuneiform
5th metatarsal / other metatarsals
phalanges
Lisfranc fracture-dislocation
stress fracture / bone-stress injury
```

Unknown healing/loading context must produce a warning rather than unrestricted rehabilitation wording.

---

# 5. Shared muscle / myotendinous profile

Foot/ankle regional gateways add:

```text
gastrocnemius strain
soleus strain
calf myotendinous injury
other acute lower-leg/foot muscle injury
```

Achilles rupture requires a structural/protocol-governed route distinct from tendinopathy.

---

# 6. Context-sensitive goals / directions

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms/changes
→ only confirmed items enter ReferralDraft
```

No global pain + ROM + strength + balance bundle.

Active/function-oriented rehabilitation, education, self-management and graded loading/activity remain the conceptual backbone where appropriate, subject to structural/healing restrictions.

---

# 7. Safety / consistency engine

```text
acute trauma + unresolved fracture concern
→ imaging/structural reassessment before routine rehab

syndesmotic / Maisonneuve concern
→ structural/specialist pathway

Lisfranc / midfoot instability concern
→ structural reassessment; no generic sprain wording

acute Achilles rupture concern
→ leave tendinopathy pathway

peroneal dislocation/subluxation / major tear concern
→ structural reassessment

bone-stress injury concern
→ diagnostic/structural pathway

hot swollen neuropathic foot / Charcot / infection / nonhealing wound
→ medical/offloading pathway

atypical pediatric rigid/painful flatfoot or focal bone concern
→ medical/structural reassessment

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurological/neurovascular component
→ never generate normal wording
```

Region-specific rules live in each frozen profile.

---

# 8. Remaining shared design sequence

After Ankle / Foot handoff closes, current broad sequence is:

```text
shared fracture / post-immobilization
→ muscle / myotendinous injury
→ generalized deconditioning / balance / gait
```

The product owner may change the exact next profile within CU-1.

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
