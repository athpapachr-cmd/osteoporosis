# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified base main for this hip design:** `546ff428cd1b8a4c2fadc65adbe740ed16483ab1`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **Frozen shoulder profile:** `clinic_utilities/physio_profiles/shoulder_v1_1.md`.
> **Frozen elbow profile:** `clinic_utilities/physio_profiles/elbow_v1_1.md`.
> **Frozen wrist/hand profile:** `clinic_utilities/physio_profiles/wrist_hand_v1_1.md`.
> **Frozen knee profile:** `clinic_utilities/physio_profiles/knee_v1_1.md`.
> **Frozen hip/groin profile on active docs branch:** `clinic_utilities/physio_profiles/hip_v1_1.md`.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-hip-v1-design-2026-08-27` until exact-head review/PR/merge/handoff close.
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
hip_v1_1 = FROZEN on docs branch pending exact-head review/merge
```

---

# 3. Hip / Groin v1.1 — product-owner-approved design

Frozen routine pathways:

```text
H1 lateral hip / greater-trochanteric pain pathway
H2 nonarthritic intra-articular hip pain — FAIS / symptomatic labral
H3 adductor-related groin pain / adductor tendinopathy
H4 post-traumatic hip/groin pain or stiffness after assessed injury
```

High-value direct shared-profile gateways:

```text
proximal rectus femoris / proximal quadriceps tendon injury in athletes
→ shared muscle/myotendinous profile

pelvic apophyseal avulsion fracture, especially ASIS/AIIS
→ shared fracture/post-immobilization profile
```

Workflow decisions:

- hip OA is context only because it is not routinely referred;
- lateral hip/GTPS remains visible, with clinician-entered trochanteric bursitis directly selectable;
- FAIS and symptomatic labral pathology are combined into one nonarthritic intra-articular pathway;
- adductor-related groin pain is high visibility because it is seen/referred frequently;
- proximal hamstring tendinopathy and iliopsoas/internal snapping hip are rare/secondary;
- gluteus medius/minimus tears, external snapping, dysplasia/instability and inguinal/pubic-related groin pain remain rare/advanced;
- postoperative hip is excluded from the routine menu;
- there is no general pediatric/adolescent Hip navigation group;
- pediatric pelvic apophyseal avulsions remain visible through the shared fracture gateway because they are encountered clinically;
- deep-gluteal/piriformis continues to route to the frozen lumbar profile.

Adjunct decisions:

```text
manual therapy / soft tissue → optional where relevant
dry needling → optional clinician-selected adjunct in appropriate myofascial context
acupuncture → excluded
ESWT for GTPS / proximal hamstring → not generator-recommended; therapist-proposed use may be documented
```

Anatomical safeguard:

```text
proximal rectus femoris origin → AIIS / supra-acetabular region
ASIS avulsion → classically sartorius-related traction
ASIS avulsion != proximal rectus femoris injury by default
```

No runtime behavior changed.

---

# 4. Exact next action

```text
1. exact branch-vs-main review of Hip v1.1 freeze
2. open docs-only PR if clean
3. independent exact-head review
4. merge only if exact head remains clean
5. clear canonical writer lock and reconcile main
6. product owner selects next CU-1 regional/shared profile
```

---

# 5. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
START next overlapping regional mutation before hip handoff closes
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
knee = frozen v1.1
hip = frozen v1.1 on docs branch pending review/merge
canonical writer = docs/cu1-hip-v1-design-2026-08-27
runtime writer = none
runtime implementation = unauthorized
```
