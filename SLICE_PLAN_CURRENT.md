# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** PRE-CODE DESIGN — clinical/content profiles + machine contract v1 + declarative R1–R2 catalogs FROZEN on active docs/schema branch pending exact review/merge and repeat completeness review.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin and ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **Frozen machine contract entrypoint:** `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml`.
> **Prior review:** `clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW.md`.
> **Repeat review:** `clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW_V2.md`.
> **Prior active slice:** PR-1 remains intentionally paused.

CU-1 remains design-only. Runtime implementation is not authorized.

---

# 1. Frozen design set

The 11 clinical profiles remain unchanged and frozen.

The machine contract now includes the original B1–B6 artifacts plus:

```text
clinic_utilities/contracts/cu1_rule_catalog_v1.yaml
clinic_utilities/contracts/cu1_route_requirements_v1.yaml
clinic_utilities/contracts/cu1_r1_r2_design_fixtures_v1.yaml
```

`cu1_typed_supplement_v1.yaml` now explicitly types `safety.input_flags[]` and points trigger/validation resolution to the two declarative catalogs.

---

# 2. R1 resolution claim — frozen pending independent review

`cu1_rule_catalog_v1.yaml` freezes:

```text
closed safety_input_flags[] namespace
declarative all/any/not/eq/in/contains/missing/empty/validation_error DSL
route/profile applicability per rule
mechanical trigger expression per v1 safety/consistency rule
severity references to existing cu1_option_catalog_v1
no symptom-only autonomous inference for unresolved concerns unless explicitly encoded
runtime prohibition on inventing rules/flags
```

Examples now deterministic without profile-prose interpretation:

```text
DVT concern → explicit dvt_concern_unresolved flag
adjunct + no core rehab → mechanical consistency rule
SIFK + loading status missing/not_stated → sifk_loading_status_not_stated
lower-limb fracture + missing WB status → fracture_weight_bearing_not_stated_when_required
```

---

# 3. R2 resolution claim — frozen pending independent review

`cu1_route_requirements_v1.yaml` freezes:

```text
base required/optional fields
allowed wording-mode source
formal diagnosis assertion policy
context-based frailty assertion exception
subtype policies
postoperative required/conditional context
route-specific structural/nonoperative overrides
ACL/MCL postoperative exclusivity validation
wrist/hand dedicated structural-vs-generic postoperative validation
shared fracture site groups and WB/upper-limb-use applicability
shared muscle required/conditional context
shared deconditioning functional-route requirements
canonical validation-error IDs
closed allowed-context-key derivation
```

All registry routes are covered by base + wording-mode policy, with additive deterministic route/owner/shared overrides. Runtime must not read profile Markdown to invent validation logic.

---

# 4. Persistence / runtime boundary

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence remains unfrozen and out of first implementation scope.

---

# 5. Exact next action

```text
1. exact branch-vs-main review of R1–R2 hardening
2. verify no frozen clinical profile/runtime mutation
3. docs/schema-only PR + independent exact-head review
4. merge only if deterministic and internally consistent
5. fresh bootstrap from merged main
6. repeat CU-1 design-completeness review
7. STOP at DESIGN-COMPLETE or remaining BLOCK
8. runtime implementation requires separate explicit product-owner authorization even after DESIGN-COMPLETE
```
