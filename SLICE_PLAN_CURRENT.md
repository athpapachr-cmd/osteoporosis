# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** PRE-CODE DESIGN — clinical/content profiles and machine contract v1 frozen; repeat design-completeness review = **BLOCK** on two narrow declarative gaps R1–R2.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin and ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **Frozen machine contract:** `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml` and normative artifacts.
> **Prior review:** `clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW.md`.
> **Repeat review:** `clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW_V2.md`.
> **Prior active slice:** PR-1 remains intentionally paused.

CU-1 remains design-only. Runtime implementation is not authorized.

---

# 1. What is frozen and passed

```text
11 regional/shared clinical profiles = FROZEN
ReferralDraftV1 typed homes = PASS
canonical route/key/gateway registry = PASS
long-tail route/site/muscle identities = PASS
route ownership / precedence = PASS
common enums / ID normalization = PASS
SafetyResult severity/blocking/disposition model = PASS after rule trigger
ShortReferralFormatter / DetailedReferralFormatter contract = PASS
structured-option scope boundary = PASS
synthetic semantic fixtures = PASS as design oracles
```

The broad clinical taxonomy must not be reopened for the remaining work.

---

# 2. Remaining blockers

```text
R1 — machine-declarative safety/consistency trigger catalog is missing
R2 — machine-declarative route requirements / conditional validation catalog is missing
```

R1 means runtime would still need to interpret profile prose to decide exactly when a canonical safety rule fires.

R2 means runtime would still need to interpret profile prose to decide route-specific required/conditional context, assertion policy, subtype requirements and applicability of restrictions.

---

# 3. Exact bounded next design scope

If the product owner chooses to continue CU-1 design hardening, produce only:

```text
clinic_utilities/contracts/cu1_rule_catalog_v1.yaml
clinic_utilities/contracts/cu1_route_requirements_v1.yaml
```

They must reference the existing frozen contract/registry and must not alter clinical meaning.

Then repeat design-completeness review.

---

# 4. Persistence / runtime boundary

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence remains unfrozen and out of first implementation scope.

---

# 5. Stop rule

```text
current result = BLOCK
runtime = NOT AUTHORIZED
```

Do not write production HTML/JS/CSS, FastAPI CU-1 endpoints, persistence or patient-data storage. Runtime implementation requires a future `DESIGN-COMPLETE` review result plus separate explicit product-owner authorization.
