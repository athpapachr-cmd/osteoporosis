# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-26 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified base main for this wrist/hand design:** `7ce2b408ce5a940d9839289e5a3deab3fa6defc2`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **Frozen shoulder profile:** `clinic_utilities/physio_profiles/shoulder_v1_1.md`.
> **Frozen elbow profile:** `clinic_utilities/physio_profiles/elbow_v1_1.md`.
> **CURRENT BODY-REGION DESIGN TARGET:** wrist / hand.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-wrist-hand-v1-design-2026-08-26` for wrist/hand CU-1 clinical/content design.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE in this repository.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED for CU-1; design only.
> **PR-1 TRANSCRIPT SLICE:** intentionally paused, preserved at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

This file is the sole owner of operational **NOW** for the active branch. Do not infer mutation authority from chat history.

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

CU-1 is a bounded cross-module Clinic Utility design detour. PR-1 remains paused intact, not abandoned.

---

# 2. Physiotherapy v2 architecture

```text
clinical problem
→ important findings
→ functional limitation
→ precautions/restrictions
→ goals
→ rehabilitation direction
→ structured ReferralDraft
→ short/detailed formatter
```

Hard rules:

```text
suggested finding != examined finding
selected goal != mandatory goal
condition profile != automatic diagnosis
subjective symptom != objective deficit
special/provocation test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
adjunct technique != default primary treatment
clinician-entered diagnosis may be carried but not inferred
```

---

# 3. Frozen regional profiles

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN
```

---

# 4. Wrist / Hand — ACTIVE DESIGN CANDIDATE WORK

Authorized scope is clinical/content design only:

```text
primary pathway taxonomy
findings vs diagnosis separation
median/ulnar/radial neurological semantics
tendon/ligament/TFCC safety semantics
functional limitations and dexterity
condition-sensitive goals
hand-therapy/active rehabilitation directions
orthosis visibility
adjunct visibility
generated wording
current evidence review
```

Target candidate:

```text
clinic_utilities/physio_profiles/wrist_hand_v1.md
```

It remains **DESIGN CANDIDATE / NOT FROZEN** until explicit product-owner review and approval.

Current evidence frame includes the 2024 AAOS carpal-tunnel CPG, current De Quervain systematic review/network meta-analysis, current thumb-CMC OA rehabilitation systematic reviews/trials, current TFCC nonoperative evidence, trigger-digit orthosis evidence, wrist-tendinopathy literature and CRPS rehabilitation guidance.

---

# 5. Shared fracture / tendon-repair boundaries

Wrist/hand fractures remain routed to the shared fracture/post-immobilization profile rather than duplicated inside this regional profile.

Examples:

```text
distal radius / ulna
scaphoid / other carpal fracture
metacarpal fracture
phalangeal fracture
```

Unresolved healing/loading/ROM context must prevent unrestricted routine rehabilitation wording.

Established tendon laceration/repair or complex hand surgery requires procedure/protocol/restriction context and must not be converted into a generic tendinopathy pathway.

---

# 6. Exact next action

```text
1. create wrist_hand_v1.md as DESIGN CANDIDATE
2. perform strict current-evidence + safety review
3. present taxonomy/adjunct/open workflow decisions to product owner
4. revise after real-workflow feedback
5. freeze/merge only after explicit product-owner approval
```

---

# 7. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
FREEZE or merge wrist/hand without product-owner approval
CREATE overlapping runtime writers
```

---

# 8. Handoff completeness

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
cervical = frozen v1.1
lumbar = frozen v1.1
shoulder = frozen v1.1
elbow = frozen v1.1
wrist / hand = active design candidate work
canonical writer = docs/cu1-wrist-hand-v1-design-2026-08-26
runtime writer = none
runtime implementation = unauthorized
```
