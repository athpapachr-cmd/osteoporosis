# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified base main for this design:** `8c9a1cdd9692fe827827b90e30db9a70d828eb22`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin, ankle/foot v1.1.
> **Frozen shared profiles:** `shared_fracture_v1_1.md`; `shared_muscle_myotendinous_v1_1.md`.
> **CURRENT SHARED DESIGN TARGET:** Generalized Deconditioning / Balance / Gait v1.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-deconditioning-balance-gait-v1-design-2026-08-27` for CU-1 clinical/content design.
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

# 2. Frozen profile state

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN
wrist_hand_v1_1 = FROZEN
knee_v1_1 = FROZEN
hip_v1_1 = FROZEN
ankle_foot_v1_1 = FROZEN
shared_fracture_v1_1 = FROZEN
shared_muscle_myotendinous_v1_1 = FROZEN
```

---

# 3. Generalized Deconditioning / Balance / Gait — ACTIVE DESIGN CANDIDATE WORK

Authorized scope is clinical/content design only.

Primary design questions:

```text
generalized deconditioning vs frailty vs focal neurological/orthopaedic impairment
balance impairment / falls-risk context without autonomous diagnosis
gait and mobility impairment
post-illness / post-hospital / inactivity deconditioning
older-adult functional decline
strength/endurance/mobility/balance findings
assistive-device context
falls history and fear-of-falling context
home/environmental safety coordination
exercise progression and supervision level
reassessment / escalation criteria
```

Hard initial boundaries:

```text
one fall != automatic falls syndrome/frailty diagnosis
abnormal TUG/5xSTS/gait speed != autonomous diagnosis
frailty score != rehabilitation prescription by itself
unexplained new gait disorder != generic deconditioning
new focal neurological deficit != routine balance referral
syncope/presyncope/acute vestibular/cardiopulmonary instability != generic gait pathway
assistive device != automatically mandatory
not assessed != normal
```

---

# 4. Exact next action

```text
1. review current falls/frailty/balance/gait evidence and safety semantics
2. create `clinic_utilities/physio_profiles/shared_deconditioning_balance_gait_v1.md` as DESIGN CANDIDATE
3. align `SLICE_PLAN_CURRENT.md` and `CLINIC_UTILITIES_PLAN.md`
4. present product-owner workflow/taxonomy decisions
5. revise and freeze only after explicit product-owner approval
6. after this final shared profile closes, decide separately whether CU-1 design is complete enough for implementation authorization
```

---

# 5. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
FREEZE or merge this profile without product-owner approval
START CU-1 runtime implementation before a separate authorization decision
CREATE overlapping runtime writers
```
