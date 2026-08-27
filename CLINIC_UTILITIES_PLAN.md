# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2; clinical/content profiles and machine contract v1 are frozen, but repeat design-completeness review remains **BLOCK** on two narrow declarative gaps R1–R2.

Clinic Utilities are cross-module clinician-facing operational tools, not a new clinical Module 02.

---

# 1. Frozen design set

Clinical profiles:

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

Machine contract entrypoint:

```text
clinic_utilities/contracts/cu1_contract_manifest_v1.yaml
```

The broad clinical taxonomy and merged machine-contract v1 remain frozen.

---

# 2. Repeat completeness gate

Review history:

```text
CU1_DESIGN_COMPLETENESS_REVIEW.md
→ initial BLOCK B1–B6

PR #50 machine-contract hardening
→ B1–B6 substantially resolved

CU1_DESIGN_COMPLETENESS_REVIEW_V2.md
→ remaining BLOCK R1–R2
```

Passed areas:

```text
typed draft/context homes
canonical route/key namespace
shared gateway mapping
long-tail route/site IDs
primary route precedence
common enums and aliases
safety severity/blocking/disposition behavior after trigger
formatter contract
structured-option boundary
synthetic semantic fixtures
```

Remaining gaps:

```text
R1 declarative safety/consistency trigger expressions
R2 declarative route-specific required/conditional field validation
```

---

# 3. Only remaining hardening scope

If CU-1 design continues, create only:

```text
cu1_rule_catalog_v1.yaml
cu1_route_requirements_v1.yaml
```

These must mechanically connect existing canonical fields/IDs to the already frozen safety and route semantics. They must not reopen clinical taxonomy or invent new clinical recommendations.

Then repeat design-completeness review.

---

# 4. First implementation direction — unchanged and still unauthorized

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence remains unfrozen. Runtime coding begins only after a later review reaches `DESIGN-COMPLETE` and the product owner separately authorizes implementation.
