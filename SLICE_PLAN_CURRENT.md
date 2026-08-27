# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** ACTIVE PRE-CODE DESIGN — runtime implementation not authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin and ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1 on docs branch pending exact-head review/merge.
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
shared_deconditioning_balance_gait_v1_1 = FROZEN on docs branch pending exact-head review/merge
```

---

# 3. Final Shared Deconditioning / Balance / Gait v1.1 design

Authoritative frozen file:

```text
clinic_utilities/physio_profiles/shared_deconditioning_balance_gait_v1_1.md
```

Shared route:

```text
functional_deconditioning_balance_gait_rehabilitation
```

Routine direct presentation families:

```text
D1 generalized deconditioning / functional decline
D2 frailty-associated functional decline — clinician-established only
```

Balance-only, gait-only and post-hospital presentations are retained as context/findings rather than routine top-level routes in this product-owner workflow.

Direct findings include:

```text
generalized muscular weakness
lower-limb weakness
poor coordination
balance/postural-control deficit when actually assessed
walking/mobility limitation
fear of falling
falls history
walking-aid context
```

Preferred optional functional battery:

```text
SPPB
→ total score + standing-balance component + 4-m gait component + 5-chair-rise component
```

No performance-test threshold autonomously creates frailty, falls-risk or neurological diagnosis.

Falls history preserves single/recurrent/injurious/unable-to-rise/loss-of-consciousness states. Walking-aid assessment/training is directly selectable. Home-hazard assessment is not a routine local option. Neurological disease-specific referral pathways are not added to CU-1.

Shared Fracture may gateway to this profile for strength/balance/falls/independence goals; fracture restrictions remain authoritative.

Generic aerobic/endurance conditioning is not a routine generator direction. Acupuncture, dry needling, ESWT and therapeutic ultrasound are excluded.

Safety boundaries include acute neurological/gait change, LOC/syncope, cardiopulmonary instability, acute vestibular syndrome, fracture-restriction uncertainty, DVT/vascular concern, infection/systemic deterioration, acute cognitive change and progressive unexplained coordination loss.

---

# 4. Persistence / runtime boundary

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence remains unfrozen. No production HTML/JS/CSS or runtime implementation is authorized.

---

# 5. Exact next action

```text
1. exact branch-vs-main review of the final shared-profile freeze
2. docs-only PR if clean
3. independent exact-head review
4. merge only if exact head remains clean
5. clear writer lock / reconcile main
6. perform CU-1 design-completeness review across all frozen profiles and shared architecture
7. implementation requires a separate explicit product-owner authorization
```
