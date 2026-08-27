# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** PRE-CODE DESIGN — clinical/content profiles frozen; B1–B6 cross-profile machine contract now FROZEN on active docs/schema branch pending exact review/merge and repeat completeness review.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin and ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **Prior completeness review:** `clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW.md` = BLOCK before hardening.
> **Frozen hardening contract:** `clinic_utilities/contracts/CU1_CORE_CONTRACT_V1.md` + `cu1_registry_v1.yaml` + `cu1_design_fixtures_v1.yaml`.
> **Prior active slice:** PR-1 remains intentionally paused.

CU-1 remains design-only. No runtime implementation is authorized.

---

# 1. Frozen clinical architecture

The 11 regional/shared v1.1 clinical/content profiles remain unchanged and frozen.

Hard invariants remain: suggested/examined/selected/mandatory are distinct; symptoms/tests/imaging do not autonomously create diagnoses; not-assessed does not mean normal; adjuncts do not replace core rehabilitation; clinician-entered diagnoses may be carried but not inferred.

---

# 2. Frozen cross-profile machine contract v1

Normative artifacts:

```text
clinic_utilities/contracts/CU1_CORE_CONTRACT_V1.md
clinic_utilities/contracts/cu1_registry_v1.yaml
clinic_utilities/contracts/cu1_design_fixtures_v1.yaml
```

They freeze:

```text
ReferralDraftV1 typed nested model
ProblemSelection / SharedTarget / FindingSelection / RestrictionSelection / MeasurementSelection
AssertionState / AssessmentState / PresenceState / Laterality / Visibility
SafetyScreenState / SafetySeverity / ClinicianDisposition / SafetyResult
canonical lowercase snake_case registry + aliases
exact regional→shared gateway targets
route ownership / precedence
ShortReferralFormatter / DetailedReferralFormatter interface and omission rules
synthetic semantic test fixtures
```

---

# 3. Prior blocker disposition

```text
B1 typed homes for route/profile-specific state = resolved by core contract
B2 canonical machine registry/gateway map = resolved by registry v1
B3 route ownership/precedence = resolved by global + route-specific precedence
B4 safety severity/blocking/disposition = resolved by SafetyResult v1
B5 formatter interface/output/omission = resolved by formatter contract
B6 normalized tri-state/enums/key namespace = resolved by common enum + alias policy
```

These are design claims pending independent repeat completeness review; they are not implementation evidence.

---

# 4. Persistence / runtime boundary

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence remains unfrozen and out of first implementation scope. No production HTML/JS/CSS, FastAPI CU-1 endpoints or database changes are authorized.

---

# 5. Exact next action

```text
1. exact branch-vs-main review of frozen hardening artifacts
2. docs/schema-only PR + independent exact-head review
3. merge only if contract is internally deterministic and no runtime/profile mutation exists
4. fresh bootstrap from merged main
5. repeat CU-1 design-completeness review against frozen contract and fixtures
6. STOP at DESIGN-COMPLETE or remaining BLOCK
7. runtime implementation requires separate explicit product-owner authorization even if DESIGN-COMPLETE
```

---

# 6. Explicitly out of scope

```text
production HTML/JS/CSS
FastAPI CU-1 runtime endpoints
physiotherapy persistence
patient-data storage
broad clinical-taxonomy expansion
PR-1 runtime implementation
```
