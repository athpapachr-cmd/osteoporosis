# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2; clinical/content profiles remain frozen and B1–B6 machine-contract hardening is now frozen pending exact review/merge and repeat design-completeness review.

Clinic Utilities are cross-module clinician-facing operational tools, not a new clinical Module 02.

---

# 1. Frozen clinical/content set

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

The broad clinical taxonomy remains frozen.

---

# 2. Frozen cross-profile contract set

```text
clinic_utilities/contracts/CU1_CORE_CONTRACT_V1.md
clinic_utilities/contracts/cu1_registry_v1.yaml
clinic_utilities/contracts/cu1_design_fixtures_v1.yaml
```

These artifacts resolve the prior B1–B6 design gaps by freezing:

```text
typed ReferralDraftV1 + ProblemSelection/shared context homes
canonical lowercase snake_case machine namespace + aliases
exact regional→shared gateway targets
route ownership/precedence
common SafetyResult severity/blocking/disposition behavior
ShortReferralFormatter / DetailedReferralFormatter semantics
common tri-state/enums
synthetic semantic fixtures
```

---

# 3. Completeness gate

Prior review:

```text
clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW.md
→ BLOCK before hardening
```

Current hardening state:

```text
B1–B6 = design-resolved / FROZEN PENDING INDEPENDENT REVIEW
```

Required next gate:

```text
exact review + merge hardening
→ fresh repeat CU-1 design-completeness review
→ DESIGN-COMPLETE or remaining BLOCK
```

Design completeness does not itself authorize runtime coding.

---

# 4. First implementation direction — unchanged and still unauthorized

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence remains unfrozen and is not required for first implementation. No CU-1 runtime exists today. Runtime coding starts only after a repeat review reaches `DESIGN-COMPLETE` and the product owner separately authorizes implementation.
