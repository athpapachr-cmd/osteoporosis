# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2; all planned clinical/content profiles are frozen, but design-completeness review is **BLOCK** pending bounded cross-profile machine-contract hardening.

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

# 2. Completeness gate

Review report:

```text
clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW.md
```

Current classification:

```text
BLOCK for implementation authorization
```

The block is not broad clinical-content incompleteness. It is unresolved implementation-contract design:

```text
B1 typed core/problem context
B2 canonical machine registry + exact shared gateways
B3 route ownership/precedence
B4 safety warning/blocking/disposition model
B5 formatter contract
B6 normalized common enums/tri-states/key semantics
```

---

# 3. Required bounded hardening pass

Produce/freeze before implementation:

```text
CU-1 core typed contract v1
canonical profile/route/key registry v1
regional→shared gateway mapping table
route precedence/ownership table
common SafetyResult / warning-disposition model
ShortReferralFormatter / DetailedReferralFormatter specification
common enum/tri-state definitions
synthetic design-fixture matrix
```

Do not reopen regional clinical taxonomies unless a specific blocker requires it.

---

# 4. First implementation direction — unchanged but not yet authorized

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence remains unfrozen and is not required for the first implementation.

Current repository runtime is FastAPI/Pydantic-capable, but no CU-1 runtime exists today. Runtime coding begins only after repeat review reaches `DESIGN-COMPLETE` and the product owner separately authorizes implementation.
