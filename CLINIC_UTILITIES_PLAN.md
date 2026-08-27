# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 pre-code design is `DESIGN-COMPLETE`; runtime remains separately unauthorized.

Clinic Utilities are cross-module clinician-facing operational tools, not a new clinical Module 02.

---

# 1. CU-1 frozen design set

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

Final completeness review:

```text
clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW_V3.md
→ DESIGN-COMPLETE
```

The broad clinical taxonomy and machine contract are frozen unless a later implementation review proves a contradiction.

---

# 2. Completeness history

```text
CU1_DESIGN_COMPLETENESS_REVIEW.md
→ initial BLOCK B1–B6

PR #50 machine-contract hardening
→ B1–B6 substantially resolved

CU1_DESIGN_COMPLETENESS_REVIEW_V2.md
→ remaining BLOCK R1–R2

PR #52 declarative hardening
→ R1/R2 artifacts frozen

PR #53 exact R2 transcription correction
→ optional shared-muscle imaging semantics restored
→ canonical major-avulsion safety flag reference corrected

CU1_DESIGN_COMPLETENESS_REVIEW_V3.md
→ DESIGN-COMPLETE
```

Resolved areas include:

```text
typed draft/context homes
canonical route/key namespace
shared gateway mapping
long-tail route/site IDs
primary route precedence
common enums and aliases
closed context value sets
machine-declarative safety triggers
machine-declarative route validation
validation-error behavior
formatter contract
structured-option boundary
synthetic semantic fixtures
```

---

# 3. Frozen first implementation direction

```text
ephemeral ReferralDraftV1
→ deterministic validation/rule evaluation
→ generated short/detailed referral text
→ copy / print
```

Persistence remains outside the first implementation scope.

Runtime implementation has **not** been authorized or started.

---

# 4. Future implementation gate

If the product owner later authorizes CU-1 implementation:

```text
fresh six-canonical bootstrap
→ inspect current Clinical Excellence runtime/navigation seams
→ create a fresh CU-1 runtime implementation slice/branch
→ implement against cu1_contract_manifest_v1.yaml
→ executable tests derived from frozen fixtures
→ focused evidence + independent exact-head review
→ merge/deploy only under the newly authorized slice
```

Do not treat `DESIGN-COMPLETE` as `IMPLEMENTED`.

---

# 5. CU-2 remains separate

Radiofrequency treatment request / PDF workflow remains a separate future Clinic Utilities slice. It must not be folded into CU-1 and must not begin merely because CU-1 design is complete.
