# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2; clinical/content profiles and machine contract are frozen, including the bounded declarative R1–R2 hardening, pending exact review/merge and final repeat completeness review.

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

The manifest now includes declarative R1–R2 authorities:

```text
cu1_rule_catalog_v1.yaml
cu1_route_requirements_v1.yaml
cu1_r1_r2_design_fixtures_v1.yaml
```

The broad clinical taxonomy remains frozen.

---

# 2. Completeness history

```text
CU1_DESIGN_COMPLETENESS_REVIEW.md
→ initial BLOCK B1–B6

PR #50 machine-contract hardening
→ B1–B6 substantially resolved

CU1_DESIGN_COMPLETENESS_REVIEW_V2.md
→ remaining BLOCK R1–R2

current bounded hardening
→ R1 safety/consistency triggers machine-declarative
→ R2 route required/conditional validation machine-declarative
→ frozen pending independent review
```

---

# 3. R1–R2 hardening boundaries

R1 is resolved by a closed safety-input namespace plus declarative trigger expressions. Non-specific symptoms do not autonomously create unresolved DVT/rupture/infection/Charcot/etc. concerns unless a frozen rule explicitly says so.

R2 is resolved by deterministic base, wording-mode, subtype, postoperative, route-specific and shared-context requirements. Validation must be driven by the catalog rather than profile prose.

The hardening does not add new clinical recommendations, reopen taxonomy, authorize persistence or authorize runtime code.

---

# 4. Required gate

```text
exact branch-vs-main review
→ docs/schema-only PR
→ independent exact-head review
→ merge if clean
→ fresh final CU-1 design-completeness review
→ DESIGN-COMPLETE or remaining BLOCK
```

Even `DESIGN-COMPLETE` does not authorize implementation; product-owner runtime authorization remains separate.

---

# 5. First implementation direction — unchanged and still unauthorized

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence remains unfrozen.
