# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** ACTIVE PRE-CODE DESIGN — runtime implementation not authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin and ankle/foot v1.1.
> **Frozen shared profiles:** `shared_fracture_v1_1.md`; `shared_muscle_myotendinous_v1_1.md` on active docs branch pending exact-head review/merge.
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

Hard invariants remain:

```text
suggested != examined
suggested != selected
selected != mandatory
symptom != diagnosis
objective deficit != subjective symptom
special/provocation test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
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
shared_fracture_v1_1 = FROZEN
shared_muscle_myotendinous_v1_1 = FROZEN on docs branch pending exact-head review/merge
```

---

# 3. Shared Muscle / Myotendinous Injury — FROZEN v1.1 design

Authoritative frozen file:

```text
clinic_utilities/physio_profiles/shared_muscle_myotendinous_v1_1.md
```

Shared route:

```text
acute_muscle_myotendinous_injury_rehabilitation
```

Routing contract:

```text
regional/shared entry
→ muscle group / specific muscle
→ injury type / phase / tissue location when established
→ conservative vs specialist/postoperative context
→ actual restrictions / findings / functional demand
→ clinician-confirmed goals and rehabilitation directions
```

High-visibility groups frozen from product-owner workflow:

```text
quadriceps / rectus femoris
acute adductor strain/tear
gastrocnemius / soleus / calf
hamstring strain / partial tear
```

Visible but less frequent:

```text
pectoralis-major muscle/myotendinous injury
biceps muscle-belly/myotendinous injury
abdominal-wall muscle injury
```

Rare/secondary includes iliopsoas/hip-flexor and tibialis-anterior muscle injury among other uncommon groups.

Structural/retraction rules:

```text
bony avulsion → Shared Fracture
complete/major tendon-avulsion concern without disposition → specialist structural pathway
postoperative repair → exact protocol governs
<2 cm retraction may support established conservative/PT workflow but is NOT autonomous clearance
>=2 cm / multi-tendon complete avulsion / major weakness-deformity / high-demand unresolved case
→ prominent specialist-disposition check, not automatic surgery recommendation
```

Return-to-running/sport/work:

```text
criterion-based where possible
elapsed time alone != clearance
MRI appearance/grade alone != clearance
strength symmetry alone != universal clearance
pain-free jogging != sprint readiness
```

Adjunct policy:

```text
acupuncture → optional clinician-selected adjunct, no healing claim
dry needling → excluded
ESWT / therapeutic ultrasound → excluded as default acute-muscle healing recommendations
compression / taping → treating-physiotherapist discretion
```

Safety domains include major tear/avulsion, reinjury, expanding haematoma, calf DVT differential, compartment/vascular/neurological concern, missing postoperative protocol and myositis-ossificans concern after significant contusion.

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
1. exact branch-vs-main review of Shared Muscle / Myotendinous v1.1 freeze
2. docs-only PR if clean
3. independent exact-head review
4. merge only if exact head remains clean
5. clear writer lock / reconcile main
6. next remaining shared profile = generalized deconditioning / balance / gait
```

Runtime implementation remains unauthorized.
