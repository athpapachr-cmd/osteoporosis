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
> **Frozen ankle/foot profile on active docs branch:** `clinic_utilities/physio_profiles/ankle_foot_v1_1.md`.
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
hip_v1_1 = FROZEN
ankle_foot_v1_1 = FROZEN on docs branch pending exact-head review/merge
```

---

# 3. Ankle / Foot — FROZEN v1.1 design

Authoritative frozen file:

```text
clinic_utilities/physio_profiles/ankle_foot_v1_1.md
```

Frozen routine primary pathways:

```text
AF1 acute/postacute lateral ankle sprain after structural assessment
AF2 Achilles tendinopathy — midportion / insertional
AF3 plantar heel pain / plantar fasciitis
AF4 posterior tibial tendon dysfunction / flexible PCFD
AF5 peroneal tendon disorder — conservative rehabilitation
AF6 mechanical metatarsalgia / forefoot overload
AF7 assessed post-traumatic ankle/foot pain or stiffness
```

Frozen pediatric/adolescent navigation group:

```text
Παιδιά / Έφηβοι — ποδοκνημική / άκρος πόδας
→ Sever disease / calcaneal apophysitis
→ symptomatic accessory navicular
→ symptomatic flexible flatfoot
```

This is navigation only, not a diagnostic umbrella. Asymptomatic flexible flatfoot does not generate treatment.

Frozen rare/advanced/context entities:

```text
chronic ankle instability / recurrent sprain → rare/secondary
syndesmotic/high-ankle sprain → very rare/advanced
tarsal tunnel / tibial-nerve entrapment → rare neurological
heel fat-pad pain → rare/secondary plantar-heel differential
Morton neuroma → rare/context
plantar-plate / lesser-MTP instability → very rare/advanced
anterior tibial / extensor / FHL tendon disorders → rare
osteochondral talus lesion → rare/advanced
hallux rigidus / 1st-MTP OA → context only
ankle OA → context only
Charcot / neuropathic foot → medical/offloading safety context
postoperative ankle/foot → advanced only; occasional Achilles repair/reconstruction
```

Frozen support/adjunct policy:

```text
taping → directly visible optional support
heel lift → directly visible optional support
brace / orthosis / AFO / metatarsal offloading / footwear → condition-specific context, often podiatry-coordinated
manual therapy / soft tissue → optional where relevant
dry needling → optional clinician-selected adjunct
acupuncture → excluded
ESWT plantar heel → evidence-supported optional adjunct
ESWT Achilles → evidence-conflicted optional adjunct; not routine and not superior to progressive loading
```

Key semantic and safety rules:

```text
inversion + lateral pain != uncomplicated lateral ankle sprain
subjective giving-way != objective CAI
Achilles pain / imaging != symptomatic tendinopathy automatically
acute Achilles rupture concern != tendinopathy pathway
midportion Achilles != insertional protocol automatically
plantar heel pain != plantar fasciitis automatically
heel spur != automatic pain generator
flat foot + medial pain != PCFD/PTTD automatically
lateral pain != peroneal tendinopathy automatically
peroneal snapping/subluxation != routine sprain/tendinopathy
forefoot pain != exact structural diagnosis
pediatric heel pain != Sever automatically
accessory navicular imaging != symptomatic diagnosis automatically
asymptomatic flexible flatfoot != treatment pathway
tarsal-tunnel symptoms/test != diagnosis
hot swollen neuropathic foot / Charcot concern != routine PT
```

---

# 4. Safety / consistency engine

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
→ structural/medical reassessment

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurological/neurovascular component
→ never generate normal wording
```

Region-specific details live in `ankle_foot_v1_1.md`.

---

# 5. Shared fracture / muscle boundary

Foot/ankle fracture, Lisfranc fracture-dislocation and stress/bone-stress injuries route to the future shared fracture profile. Acute gastrocnemius/soleus/calf injuries route to the future shared muscle/myotendinous profile. Achilles rupture remains a structural/protocol-governed gateway distinct from tendinopathy.

Unknown healing/stability, weight-bearing, immobilization, ROM or loading context prevents unrestricted rehabilitation wording.

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
1. exact branch-vs-main review of Ankle / Foot v1.1 freeze
2. open docs-only PR if clean
3. independent exact-head review
4. merge only if exact head remains clean
5. clear canonical writer lock and record resulting main state
6. product owner selects next shared CU-1 profile
```

Runtime implementation remains unauthorized.
