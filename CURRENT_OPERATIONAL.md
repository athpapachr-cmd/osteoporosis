# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified main immediately before this operational handoff write:** `52c5bb39bd400eeb95c2e719d6e138c5cc16c03d`; this handoff commit advances `main` once written.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **Frozen shoulder profile:** `clinic_utilities/physio_profiles/shoulder_v1_1.md`.
> **Frozen elbow profile:** `clinic_utilities/physio_profiles/elbow_v1_1.md`.
> **Frozen wrist/hand profile:** `clinic_utilities/physio_profiles/wrist_hand_v1_1.md`.
> **WRIST/HAND FREEZE PR:** PR #42 squash-merged as `f6e0a3126a6a93f48c140a7738a3c9d4bbe60563`.
> **POST-MERGE CANONICAL ALIGNMENT:** `3dda1752f59ea61bdc159d62f23746be41cf6dd3` → `52c5bb39bd400eeb95c2e719d6e138c5cc16c03d` before this handoff write.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE in this repository.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED for CU-1; design only.
> **PR-1 TRANSCRIPT SLICE:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

This file is the sole owner of operational NOW. Do not infer mutation authority from chat history.

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

# 3. Wrist / Hand v1.1 — closed freeze

Frozen default pathways:

```text
WH1 De Quervain / first dorsal compartment disorder
WH2 thumb CMC-1 osteoarthritis / rhizarthrosis
WH3 interphalangeal / generalized hand osteoarthritis
WH4 median neuropathy at wrist / carpal tunnel syndrome
WH5 ulnar-sided wrist / TFCC-related presentation
WH6 intersection syndrome
WH7 thumb MCP collateral-ligament injury — UCL or RCL
WH8 sagittal-band injury / extensor tendon instability at MCP
WH9 digital tendon injury / deformity-specific rehabilitation
WH10 post-traumatic wrist/hand pain or stiffness after assessed injury
WH11 postoperative wrist/hand rehabilitation
```

Workflow decisions:

- trigger finger/thumb is context only, not a routine physiotherapy referral in the local Cyprus workflow;
- Guyon's canal and scapholunate/lunotriquetral instability remain rare/advanced;
- postoperative wrist/hand is active;
- UCL and RCL are both included in the thumb-MCP collateral-ligament pathway;
- UCL Stener concern and major RCL instability/subluxation prevent unrestricted rehabilitation wording;
- CRPS is an established-diagnosis advanced pathway only;
- mallet, central-slip/boutonniere and flexor/extensor tendon injuries are directly selectable and protocol/zone governed;
- intersection syndrome is directly selectable and distinct from De Quervain;
- TFCC is the canonical structured term; `TFCL` is not used;
- dedicated `hand therapist` availability is not assumed; referrals use physiotherapy/wrist-hand rehabilitation plus competence/protocol requirements;
- wrist/hand fractures route to the shared fracture/post-immobilization profile.

Adjunct decisions:

```text
manual therapy / soft tissue / taping → optional where relevant
selected thermal strategy for OA → optional
acupuncture → excluded
dry needling → excluded
ESWT → excluded
```

Orthosis/splint support is condition-sensitive and protocol governed.

No runtime behavior changed.

---

# 4. Exact next action

```text
1. product owner selects the next remaining CU-1 regional profile
2. use the same taxonomy/findings/safety/goals/rehab/evidence method
3. continue CU-1 design only
```

---

# 5. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
CREATE overlapping runtime writers
```

---

# 6. Handoff completeness

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
cervical = frozen v1.1
lumbar = frozen v1.1
shoulder = frozen v1.1
elbow = frozen v1.1
wrist/hand = frozen v1.1
wrist/hand PR = #42 merged
canonical writer = none
runtime writer = none
runtime implementation = unauthorized
next action = product owner selects next regional profile
```