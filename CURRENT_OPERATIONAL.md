# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified base main for this Shared Fracture design:** `0673dc24d1e34d7a6d562103bab84830da8f585e`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **Frozen shoulder profile:** `clinic_utilities/physio_profiles/shoulder_v1_1.md`.
> **Frozen elbow profile:** `clinic_utilities/physio_profiles/elbow_v1_1.md`.
> **Frozen wrist/hand profile:** `clinic_utilities/physio_profiles/wrist_hand_v1_1.md`.
> **Frozen knee profile:** `clinic_utilities/physio_profiles/knee_v1_1.md`.
> **Frozen hip/groin profile:** `clinic_utilities/physio_profiles/hip_v1_1.md`.
> **Frozen ankle/foot profile:** `clinic_utilities/physio_profiles/ankle_foot_v1_1.md`.
> **CURRENT SHARED DESIGN TARGET:** fracture / post-immobilization.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-shared-fracture-v1-design-2026-08-27` for Shared Fracture / Post-immobilization v1 CU-1 clinical/content design.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE in this repository.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED for CU-1; design only.
> **PR-1 TRANSCRIPT SLICE:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

This file is the sole owner of operational NOW for the active branch. Do not infer mutation authority from chat history.

---

# 1. Product boundary

```text
Clinical Excellence Core
→ reusable platform/workspace mechanics

Clinic Utilities / Clinical Operations
→ cross-module clinician workflow tools

Osteoporosis Module 01
→ osteoporosis-specific clinical standards/audit/workflows
```

CU-1 remains a bounded cross-module design detour. No runtime authority is implied.

---

# 2. Frozen regional state

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN
wrist_hand_v1_1 = FROZEN
knee_v1_1 = FROZEN
hip_v1_1 = FROZEN
ankle_foot_v1_1 = FROZEN
```

---

# 3. Shared Fracture / Post-immobilization — ACTIVE DESIGN CANDIDATE WORK

Authorized scope is clinical/content design only:

```text
shared fracture taxonomy across upper limb / lower limb / pelvis
fracture date / phase / treatment context
healing / stability state
immobilization / brace / orthosis state
weight-bearing / use status
ROM / loading restrictions
surgeon / orthopaedic instructions
post-immobilization stiffness / weakness / function
pediatric skeletal-maturity / physeal / apophyseal context
fragility / stress / insufficiency fracture context where relevant
return-to-function / return-to-sport semantics
safety / nonunion / displacement / infection / neurovascular / CRPS reassessment semantics
```

Target candidate:

```text
clinic_utilities/physio_profiles/shared_fracture_v1.md
```

It remains **DESIGN CANDIDATE / NOT FROZEN** until explicit product-owner review and approval.

---

# 4. Existing frozen shared-fracture contract

Inherited minimum required context:

```text
bone/site
fracture date/phase
treatment
healing/stability status
immobilization/brace/orthosis status
weight-bearing/use status
ROM/loading restrictions
surgeon/orthopaedic instructions
age/skeletal-maturity when relevant
```

Hard safety rule:

```text
unknown healing / stability / weight-bearing / use / loading context
→ warning
→ no unrestricted rehabilitation wording
```

Fracture logic should be owned once in this shared profile rather than duplicated across frozen regional profiles.

---

# 5. Exact next action

```text
1. inspect all frozen regional fracture gateways
2. perform current evidence/safety review for fracture rehabilitation and post-immobilization semantics
3. create `shared_fracture_v1.md` as DESIGN CANDIDATE
4. align `SLICE_PLAN_CURRENT.md` and `CLINIC_UTILITIES_PLAN.md` to active shared-fracture candidate
5. present taxonomy / visibility / restrictions / pediatric / fragility / postoperative decisions to product owner
6. revise after real-workflow feedback
7. freeze/merge only after explicit product-owner approval
```

---

# 6. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
FREEZE or merge Shared Fracture without product-owner approval
START muscle/myotendinous design before this writer lock is released
CREATE overlapping runtime writers
```

---

# 7. Resolved prior operational incident

During Ankle/Foot PR preparation, temporary `_noop` files were created and fully removed/discarded before PR #45. Exact comparison showed zero net file diff on `main`, none exists in the merged tree, and no clinical/runtime content changed. The incident remains closed.

---

# 8. Handoff completeness

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
cervical = frozen v1.1
lumbar = frozen v1.1
shoulder = frozen v1.1
elbow = frozen v1.1
wrist/hand = frozen v1.1
knee = frozen v1.1
hip/groin = frozen v1.1
ankle/foot = frozen v1.1
shared fracture = active design candidate work
canonical writer = docs/cu1-shared-fracture-v1-design-2026-08-27
runtime writer = none
runtime implementation = unauthorized
```
