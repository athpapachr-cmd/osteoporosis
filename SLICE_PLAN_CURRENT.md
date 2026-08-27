# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** ACTIVE PRE-CODE DESIGN — runtime implementation not authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin and ankle/foot v1.1.
> **Frozen shared-fracture profile:** `clinic_utilities/physio_profiles/shared_fracture_v1_1.md`.
> **Current detailed shared profile under review:** `clinic_utilities/physio_profiles/shared_muscle_myotendinous_v1.md` — DESIGN CANDIDATE / NOT FROZEN.
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
shared_fracture_v1_1 = FROZEN
shared_muscle_myotendinous_v1 = DESIGN CANDIDATE / NOT FROZEN
```

---

# 3. Muscle / Myotendinous Injury — ACTIVE DESIGN CANDIDATE

Authoritative candidate:

```text
clinic_utilities/physio_profiles/shared_muscle_myotendinous_v1.md
```

Proposed shared route:

```text
acute_muscle_myotendinous_injury_rehabilitation
```

Routing contract:

```text
regional/shared entry
→ muscle group / specific muscle
→ injury type / phase / tissue location if established
→ conservative vs specialist/postoperative context
→ actual restrictions / findings / functional demand
→ clinician-confirmed goals and rehabilitation directions
```

Candidate core muscle groups:

```text
hamstring
quadriceps / rectus femoris
adductor
a cute hip-flexor / iliopsoas
calf — gastrocnemius / soleus
other lower-limb muscle injuries as secondary/advanced
upper-limb/trunk acute muscle injuries pending product-owner workflow decision
```

[Candidate hygiene note: normalize any machine-key spelling/ASCII issues before v1.1 freeze.]

Hard boundaries:

```text
acute strain != chronic tendinopathy automatically
muscle pain != structural tear automatically
MRI grade != rehabilitation clearance
elapsed time != tissue readiness
pain reduction != restored load capacity
bony avulsion → shared fracture profile
complete free-tendon rupture / major tendon avulsion → structural/specialist route
postoperative repair protocol > generic shared muscle suggestion
Achilles rupture/tendinopathy, regional tendon tears and digital tendon repairs remain in their frozen dedicated routes
```

Return-to-function principle:

```text
criterion-based where relevant
→ symptoms + ROM/length tolerance + strength/capacity + progressive task exposure + sport/work demands + confidence
→ no calendar-only clearance
```

Evidence caution:

```text
hamstring rehabilitation / running / RTS evidence = comparatively developed
adductor / quadriceps / calf exact RTS thresholds = less certain
→ do not encode false precision
```

Safety domains:

```text
major tear / tendon avulsion / palpable defect / major acute functional loss
new trauma or reinjury
large/expanding haematoma
calf DVT differential
compartment / vascular / neurological concern
postoperative repair with missing protocol
atypical failure to progress / myositis-ossificans concern after significant contusion
```

Candidate adjunct/support decisions remain for product-owner review.

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
1. product-owner clinical review of `shared_muscle_myotendinous_v1.md`
2. resolve high-visibility muscle groups vs rare/advanced entries
3. resolve major tear/avulsion/postoperative boundaries
4. resolve adjunct/support policy
5. resolve return-to-running/sport/work semantics without fixed timelines
6. revise candidate
7. freeze/merge only after explicit product-owner approval
```

Runtime implementation remains unauthorized.
