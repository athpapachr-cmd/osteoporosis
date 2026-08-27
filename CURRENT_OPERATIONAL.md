# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified base main for this knee design:** `a22253d6065abdf4624d17d7639dcfe297b5dc25`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **Frozen shoulder profile:** `clinic_utilities/physio_profiles/shoulder_v1_1.md`.
> **Frozen elbow profile:** `clinic_utilities/physio_profiles/elbow_v1_1.md`.
> **Frozen wrist/hand profile:** `clinic_utilities/physio_profiles/wrist_hand_v1_1.md`.
> **CURRENT BODY-REGION DESIGN TARGET:** knee.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-knee-v1-design-2026-08-27` for Knee v1 CU-1 clinical/content design.
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
```

---

# 3. Knee — ACTIVE DESIGN CANDIDATE WORK

Authorized scope is clinical/content design only:

```text
primary knee pathway taxonomy
findings vs diagnosis separation
OA / meniscal / patellofemoral / tendon / ligament semantics
post-traumatic and postoperative safety/restriction semantics
functional limitations
condition-sensitive goals and rehabilitation directions
brace/taping/orthosis/support visibility
adjunct visibility
generated wording
current evidence review
```

Target candidate:

```text
clinic_utilities/physio_profiles/knee_v1.md
```

It remains **DESIGN CANDIDATE / NOT FROZEN** until explicit product-owner review and approval.

Initial evidence frame includes current knee-OA rehabilitation guidance, acute meniscal pathology guidance, patellofemoral-pain best-practice evidence, tendon-loading evidence and current ligament/postoperative rehabilitation consensus/guidelines.

---

# 4. Shared fracture / structural boundary

Knee-region fractures remain routed to the future shared fracture/post-immobilization profile rather than duplicated inside this regional profile.

Examples include:

```text
patella fracture
distal femur fracture
proximal tibia / tibial plateau fracture
proximal fibula fracture
other knee-region fracture
```

Unresolved fracture, extensor-mechanism rupture, locked-knee structural concern, major instability or postoperative restriction context must prevent routine unrestricted rehabilitation wording.

---

# 5. Exact next action

```text
1. create knee_v1.md as DESIGN CANDIDATE
2. perform strict current-evidence + safety review
3. present taxonomy/adjunct/open workflow decisions to product owner
4. revise after real-workflow feedback
5. freeze/merge only after explicit product-owner approval
```

---

# 6. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
FREEZE or merge knee without product-owner approval
CREATE overlapping runtime writers
```

---

# 7. Handoff completeness

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
cervical = frozen v1.1
lumbar = frozen v1.1
shoulder = frozen v1.1
elbow = frozen v1.1
wrist/hand = frozen v1.1
knee = active design candidate work
canonical writer = docs/cu1-knee-v1-design-2026-08-27
runtime writer = none
runtime implementation = unauthorized
```
