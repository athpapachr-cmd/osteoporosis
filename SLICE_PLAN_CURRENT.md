# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** ACTIVE PRE-CODE DESIGN — runtime implementation not authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin and ankle/foot v1.1.
> **Frozen shared profiles:** `shared_fracture_v1_1.md`; `shared_muscle_myotendinous_v1_1.md`.
> **Current detailed shared profile under review:** `clinic_utilities/physio_profiles/shared_deconditioning_balance_gait_v1.md` — DESIGN CANDIDATE / NOT FROZEN.
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
shared_muscle_myotendinous_v1_1 = FROZEN
shared_deconditioning_balance_gait_v1 = DESIGN CANDIDATE / NOT FROZEN
```

---

# 3. Generalized Deconditioning / Balance / Gait — ACTIVE DESIGN CANDIDATE

Authoritative candidate:

```text
clinic_utilities/physio_profiles/shared_deconditioning_balance_gait_v1.md
```

Proposed shared route:

```text
functional_deconditioning_balance_gait_rehabilitation
```

Candidate presentation families:

```text
D1 generalized deconditioning / functional decline
D2 balance impairment / falls-risk rehabilitation
D3 gait / mobility impairment rehabilitation
D4 post-illness / post-hospital deconditioning
D5 frailty-associated functional decline — clinician-established/context only
```

Core semantic boundaries:

```text
deconditioning != frailty automatically
one fall != recurrent-falls syndrome
fear of falling != objective balance impairment
abnormal TUG/5xSTS/gait speed/SPPB/Berg/Mini-BESTest != autonomous diagnosis
unexplained new gait disorder != generic deconditioning
assistive device != automatically mandatory
not assessed != normal
```

Candidate rehabilitation direction is individualized and progressive, emphasizing actual functional deficits:

```text
strength / resistance
balance / coordination / stepping
power where appropriate
gait and transfer practice
walking/endurance/aerobic conditioning where medically appropriate
stairs / obstacle / community mobility
activity and sedentary-behaviour reduction
falls-prevention exercise when indicated
```

Falls-management context remains multifactorial when non-physical risk factors are present; the utility does not silently substitute generic exercise for medical, medication, vision, neurological, vestibular, podiatry or home-environment management.

Safety domains include acute neurological/gait change, syncope/LOC, unstable cardiopulmonary symptoms, acute vestibular syndrome, fracture/restriction uncertainty, DVT/vascular concern, infection/systemic deterioration and acute cognitive change.

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
1. product-owner clinical review of `shared_deconditioning_balance_gait_v1.md`
2. resolve presentation visibility / frailty / falls / gait / assistive-device / home-hazard workflow decisions
3. revise candidate
4. freeze/merge only after explicit product-owner approval
5. after this final shared profile is frozen, perform CU-1 design-completeness review
6. runtime implementation requires a separate explicit authorization decision
```
