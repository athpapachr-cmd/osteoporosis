# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified base main for this wrist/hand design:** `7ce2b408ce5a940d9839289e5a3deab3fa6defc2`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **Frozen shoulder profile:** `clinic_utilities/physio_profiles/shoulder_v1_1.md`.
> **Frozen elbow profile:** `clinic_utilities/physio_profiles/elbow_v1_1.md`.
> **Frozen wrist/hand profile on active docs branch:** `clinic_utilities/physio_profiles/wrist_hand_v1_1.md`.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-wrist-hand-v1-design-2026-08-26` until exact-head review/PR/merge/handoff close.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE in this repository.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED for CU-1; design only.
> **PR-1 TRANSCRIPT SLICE:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

This file is the sole owner of operational NOW for the active branch.

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
wrist_hand_v1_1 = FROZEN on docs branch pending review/merge
```

---

# 3. Wrist / Hand v1.1 — product-owner-approved design

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
1. exact branch-vs-main review of wrist/hand freeze
2. open docs-only PR if clean
3. independent exact-head review
4. merge only if exact head remains clean
5. clear canonical writer lock and reconcile main
6. product owner selects next CU-1 region
```

---

# 5. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
START next regional mutation before wrist/hand handoff closes
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
wrist/hand = frozen v1.1 on docs branch pending review/merge
canonical writer = docs/cu1-wrist-hand-v1-design-2026-08-26
runtime writer = none
runtime implementation = unauthorized
```