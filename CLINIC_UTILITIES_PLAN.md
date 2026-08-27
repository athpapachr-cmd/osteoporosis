# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 is implemented, tested, merged and deployed; CU-2 remains separately gated and unauthorized.

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

Final pre-code completeness review:

```text
clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW_V3.md
→ DESIGN-COMPLETE
```

The broad clinical taxonomy and machine contract remain frozen unless a later maintenance review proves a concrete contradiction.

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

PR #56 runtime implementation
→ protected ephemeral CU-1 utility
→ manifest-driven validation/rule/formatter engine
→ exact frozen gateway trust boundary
→ fail-closed safety-state boundary
→ 29/29 focused tests PASS
→ squash merge `c1da07f581cf8ccf1159d18bb63c23b674cbe9bd`
→ Render deploy `dep-da8afeuk1f9s73f5sr6g` LIVE
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
runtime trust-boundary validation
protected Clinical Excellence integration
ephemeral copy/print workflow
```

---

# 3. CU-1 delivered implementation

```text
ephemeral ReferralDraftV1
→ deterministic normalization
→ frozen gateway/ownership resolution
→ route validation
→ declarative safety/consistency evaluation
→ acknowledgement/disposition gate
→ generated short/detailed referral text
→ copy / print
```

Runtime entrypoints:

```text
/clinical/clinic-utilities/physio-referral
/clinical/clinic-utilities/physio-referral/api/contract
/clinical/clinic-utilities/physio-referral/api/validate
/clinical/clinic-utilities/physio-referral/api/generate
```

Persistence remains outside CU-1 v1. No referral draft or generated referral is saved in PostgreSQL, localStorage or sessionStorage.

External route-level HTTP smoke from the assistant sandbox was not executable because of DNS resolution failure; Render build/start/live status at the exact merge commit is proven. This limitation does not authorize additional scope by itself.

---

# 4. CU-1 closure rule

CU-1 is closed at runtime v1 after control-plane closeout and writer-lock release.

A future CU-1 maintenance slice requires a concrete defect/contradiction and fresh authorization. Design completion or deployment does not authorize persistence, taxonomy expansion or new physiotherapy recommendations.

---

# 5. CU-2 remains separate

Radiofrequency treatment request / PDF workflow remains a separate future Clinic Utilities slice. It must not be folded into CU-1 and must not begin merely because CU-1 is complete.

Before CU-2 implementation:

```text
explicit product-owner selection
→ fresh six-canonical bootstrap
→ read-only inspection of the real current workflow/runtime seams
→ dedicated CU-2 slice design
→ explicit writer lock
```

CU-2 is currently **NOT AUTHORIZED**.