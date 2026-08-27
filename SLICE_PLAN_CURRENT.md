# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** PRE-CODE DESIGN — clinical/content profiles frozen; design-completeness review = **BLOCK** pending bounded cross-profile machine-contract hardening.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin and ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **Completeness review:** `clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW.md`.
> **Prior active slice:** PR-1 remains intentionally paused.

CU-1 remains design-only. No runtime implementation is authorized.

---

# 1. Frozen clinical architecture

```text
ReferralDraft
  patient_context
  body_region
  primary_problem
  secondary_problems[]
  laterality
  chronicity
  key_findings[]
  functional_impairments[]
  precautions[]
  explicit_restrictions[]
  goals[]
  rehab_directions[]
  adjunct_options[]
  reassessment_criteria[]
  sessions_optional
  clinician_free_text_optional

ReferralDraft
→ ShortReferralFormatter
→ DetailedReferralFormatter
```

Clinical invariants remain frozen: suggested/examined/selected/mandatory are distinct; symptoms/tests/imaging do not autonomously create diagnoses; not-assessed does not mean normal; adjuncts do not replace core rehabilitation; clinician-entered diagnoses may be carried but not inferred.

No broad regional taxonomy is reopened by the completeness review.

---

# 2. Completeness-review result

```text
clinical/content taxonomy = preserve / substantially complete
implementation-contract completeness = BLOCK
runtime implementation = NOT READY
```

Blocking items:

```text
B1 — frozen ReferralDraft lacks typed homes for route/profile-specific structured state
B2 — no canonical machine-readable profile/route/key registry or exact regional→shared gateway map
B3 — selected postoperative/structural scenarios have unresolved primary-route ownership/precedence
B4 — no common warning/safety severity + blocking/disposition contract
B5 — ShortReferralFormatter / DetailedReferralFormatter interface/output/omission rules are not frozen
B6 — common tri-state/enumeration/key semantics are not normalized/versioned
```

Authoritative details and examples are in `clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW.md`.

---

# 3. Bounded design-hardening scope

Do not reopen broad clinical-content design unless required to resolve a blocker.

Required pre-implementation design artifacts:

```text
1. CU-1 core typed contract v1
2. canonical profile/route/key registry v1
3. exact regional→shared gateway mapping table
4. route ownership / precedence table
5. common safety-warning-disposition contract
6. ShortReferralFormatter / DetailedReferralFormatter specification
7. normalized common enum / tri-state definitions
8. focused synthetic design-fixture matrix
```

Persistence remains unfrozen and is not required for first implementation.

---

# 4. Exact next action

```text
1. complete one docs/schema-only CU-1 design-hardening pass for B1–B6
2. exact review of the resulting cross-profile machine contract
3. repeat CU-1 design-completeness review
4. STOP at DESIGN-COMPLETE or remaining BLOCK
5. runtime implementation requires a separate explicit product-owner authorization after DESIGN-COMPLETE
```

---

# 5. Explicitly out of scope now

```text
production HTML/JS/CSS
FastAPI CU-1 runtime endpoints
physiotherapy persistence
patient-data storage
broad clinical-taxonomy expansion
PR-1 runtime implementation
```
