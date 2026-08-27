# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; all currently planned regional and shared clinical/content profiles are frozen and merged. Next step is design-completeness review, not implementation.

Clinic Utilities are cross-module clinician-facing operational tools, not a new clinical Module 02.

---

# 1. CU-1 target

```text
Clinical problem / diagnosis
→ important findings
→ functional limitation
→ precautions / restrictions
→ rehabilitation goals
→ rehabilitation direction
→ final referral text
```

Structured intermediate model remains `ReferralDraft → ShortReferralFormatter / DetailedReferralFormatter` under the frozen CU-1 invariants.

---

# 2. Frozen profile set

```text
cervical_v1_1
lumbar_v1_1
shoulder_v1_1
elbow_v1_1
wrist_hand_v1_1
knee_v1_1
hip_v1_1
ankle_foot_v1_1
shared_fracture_v1_1
shared_muscle_myotendinous_v1_1
shared_deconditioning_balance_gait_v1_1
```

All are FROZEN clinical/content design.

Final shared functional-profile decisions include direct generalized deconditioning and clinician-established frailty-associated decline; SPPB as preferred optional multidomain functional battery; explicit falls history/fear-of-falling/walking-aid semantics; and no autonomous diagnostic inference from performance testing.

---

# 3. Design-completeness gate

Before implementation authorization, review the frozen set as one system:

```text
cross-profile ownership / routing
structured key consistency
tri-state / not-assessed semantics
safety / reassessment invariants
fracture / tendon / postoperative restriction precedence
goal / rehab-direction semantics
adjunct ownership / evidence labels
short vs detailed formatter requirements
current runtime/schema seams
privacy / persistence boundary
```

Required outcome:

```text
DESIGN-COMPLETE
or
BLOCK with exact unresolved design blockers
```

Design completeness does **not** itself authorize coding.

---

# 4. Implementation boundary

Current first implementation direction remains:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence is not frozen. No production physiotherapy runtime code until a separate explicit product-owner authorization after the design-completeness review.
