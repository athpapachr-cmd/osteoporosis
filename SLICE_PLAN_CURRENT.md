# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** ACTIVE PRE-CODE DESIGN — all currently planned clinical/content profiles frozen; runtime implementation not authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin and ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **Prior active slice:** PR-1 remains intentionally paused.

CU-1 remains design-only.

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

Hard invariants remain: suggested/examined/selected/mandatory are distinct; symptoms/tests/imaging do not autonomously create diagnoses; not-assessed does not mean normal; adjuncts do not replace core rehabilitation; clinician-entered diagnoses may be carried but not inferred.

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
shared_deconditioning_balance_gait_v1_1 = FROZEN
```

The final shared functional profile retains generalized deconditioning and clinician-established frailty-associated decline as direct routes; balance/gait/post-hospital presentations are context/findings in this workflow. SPPB is preferred optional, not diagnostic. Falls history, fear of falling, muscular weakness, poor coordination and walking-aid assessment/training are explicitly represented. Shared Fracture restrictions remain authoritative when routed here.

---

# 3. Persistence / runtime boundary

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence remains unfrozen. No production HTML/JS/CSS or runtime implementation is authorized.

---

# 4. Exact next action — CU-1 design-completeness review

```text
1. inspect all frozen regional/shared profile contracts together
2. validate cross-profile routing and ownership
3. validate structured keys / schema consistency / tri-state semantics
4. validate safety and reassessment invariants across profiles
5. validate goal / rehab-direction / adjunct semantics
6. validate ShortReferralFormatter / DetailedReferralFormatter requirements
7. inspect current runtime seams only to determine implementation fit — do not code
8. classify outcome as DESIGN-COMPLETE or BLOCK with exact blockers
```

Runtime implementation requires a separate explicit product-owner authorization after this review.
