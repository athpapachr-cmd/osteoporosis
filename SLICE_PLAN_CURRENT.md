# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** ACTIVE PRE-CODE DESIGN — runtime implementation not authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin and ankle/foot v1.1.
> **Frozen shared profiles:** `shared_fracture_v1_1.md`; `shared_muscle_myotendinous_v1_1.md`.
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
shared_muscle_myotendinous_v1_1 = FROZEN
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

High-visibility workflow groups:

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

Key frozen boundaries:

```text
bony avulsion → Shared Fracture
major free-tendon rupture/avulsion without disposition → specialist structural route
postoperative repair → exact protocol governs
<2 cm retraction may support established conservative/PT workflow but is NOT autonomous clearance
>=2 cm / multi-tendon complete avulsion / major weakness-deformity / high-demand unresolved case
→ prominent specialist-disposition check, not automatic surgery recommendation
```

Return-to-running/sport/work is criterion-based. Elapsed time alone, MRI appearance/grade alone and strength symmetry alone never generate clearance.

Adjunct policy:

```text
acupuncture → optional clinician-selected adjunct, no healing claim
dry needling → excluded
ESWT / therapeutic ultrasound → excluded as default acute-muscle healing recommendations
compression / taping → treating-physiotherapist discretion
```

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
1. design the final currently planned shared CU-1 profile: generalized deconditioning / balance / gait
2. perform product-owner review and freeze it separately
3. only after all design profiles are frozen, decide whether CU-1 is sufficiently complete for a separately authorized implementation step
```

Runtime implementation remains unauthorized.
