# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; all regional v1.1 profiles plus Shared Fracture, Shared Muscle/Myotendinous and Shared Deconditioning/Balance/Gait v1.1 are frozen, with the final shared profile pending exact-head review/merge.

Clinic Utilities are cross-module clinician-facing operational tools, not a new clinical Module 02.

---

# 1. Physiotherapy Referral v2 target

```text
Clinical problem / diagnosis
→ important findings
→ functional limitation
→ precautions / restrictions
→ rehabilitation goals
→ rehabilitation direction
→ final referral text
```

Structured intermediate model remains `ReferralDraft → ShortReferralFormatter / DetailedReferralFormatter` with the existing CU-1 hard invariants.

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
shared_deconditioning_balance_gait_v1_1 = FROZEN on docs branch pending review/merge
```

---

# 3. Final shared functional profile

Authoritative file:

```text
clinic_utilities/physio_profiles/shared_deconditioning_balance_gait_v1_1.md
```

Frozen workflow:

```text
D1 generalized deconditioning / functional decline — direct
D2 frailty-associated functional decline — direct only when frailty already established
balance-only / gait-only / post-hospital presentations — context/findings, not routine top-level routes
```

Direct findings and referral directions include generalized muscular weakness, poor coordination, fear of falling, falls history and walking-aid assessment/training.

Preferred optional functional battery:

```text
SPPB
→ total + standing balance + 4-m gait + 5 chair rises
```

No SPPB/TUG/gait-speed/5xSTS result autonomously creates frailty, falls-risk or neurological diagnosis.

Falls history preserves single/recurrent/injurious/unable-to-rise/LOC states. Shared Fracture may route here for strength/balance/falls/independence goals, but its restrictions remain authoritative.

Home-hazard assessment is not a routine local option; neurological disease-specific pathways are not added; generic aerobic conditioning is not a routine generator direction; acupuncture, dry needling, ESWT and therapeutic ultrasound are excluded.

---

# 4. Safety / consistency

```text
acute unexplained gait/coordination change
new focal neurological deficit
syncope / unexplained LOC
unstable cardiopulmonary symptoms
acute vestibular syndrome
fracture/restriction uncertainty
DVT / vascular concern
infection/systemic deterioration
acute cognitive change
→ reassessment / appropriate specialty route
```

No reassuring negative statement is generated from missing assessment.

---

# 5. CU-1 design-completeness gate

This is the final currently planned clinical/content profile for CU-1.

After merge/handoff close:

```text
all regional/shared profiles frozen
→ perform CU-1 design-completeness review
→ inspect cross-profile routing, schema consistency, safety semantics, formatter requirements and implementation seams
→ decide separately whether implementation should be authorized
```

Design completion **does not authorize runtime implementation**.

---

# 6. Implementation boundary

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence is not frozen. No production physiotherapy runtime code until explicit product-owner authorization after design-completeness review.
