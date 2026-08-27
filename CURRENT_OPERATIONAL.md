# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **KNEE FREEZE PR:** PR #43 squash-merged as `9c7089f08ec10c21a7e72a915e72a29351a9a2ff`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **Frozen shoulder profile:** `clinic_utilities/physio_profiles/shoulder_v1_1.md`.
> **Frozen elbow profile:** `clinic_utilities/physio_profiles/elbow_v1_1.md`.
> **Frozen wrist/hand profile:** `clinic_utilities/physio_profiles/wrist_hand_v1_1.md`.
> **Frozen knee profile:** `clinic_utilities/physio_profiles/knee_v1_1.md`.
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
knee_v1_1 = FROZEN
```

---

# 3. Knee v1.1 — closed freeze

Frozen default pathways:

```text
K1 knee osteoarthritis
K2 degenerative meniscal lesion/tear — conservative rehabilitation
K3 acute isolated meniscal injury — assessed nonoperative
K4 patellofemoral pain
K5 patellar tendinopathy
K6 quadriceps tendinopathy
K7 ACL injury/instability — nonoperative or preoperative rehabilitation
K8 MCL injury — nonoperative rehabilitation
K9 patellar instability/dislocation rehabilitation
K10 iliotibial-band syndrome
K11 pes-anserine region pain / established tendinobursitis
K12 post-traumatic knee pain/stiffness after assessed injury
K13 postoperative knee rehabilitation
```

Pediatric/adolescent navigation group:

```text
Παιδιά / Έφηβοι — γόνατο
→ Osgood-Schlatter
→ Sinding-Larsen-Johansson
→ ordinary PFP/meniscus/ACL/MCL/patellar-instability structural pathways when those are the real problem
```

Workflow decisions:

- postoperative knee is active; meniscal repair and partial meniscectomy are the most commonly seen postoperative referrals;
- degenerative and acute traumatic meniscus remain separate;
- ACL and MCL are separate top-level nonoperative/preoperative pathways;
- all postoperative ACL/MCL care routes through K13 to avoid duplicate primary pathways;
- patellar instability/dislocation, quadriceps tendinopathy, ITB syndrome and pes-anserine pathology are directly selectable;
- PCL/LCL/PLC/combined ligament injuries remain rare/advanced;
- distal hamstring insertional pathology and Hoffa/plica remain rare selectable context;
- Baker cyst and prepatellar bursitis are medical/context only in this workflow;
- gastrocnemius strain routes to the future shared muscle/myotendinous profile;
- Osgood-Schlatter and Sinding-Larsen-Johansson are dedicated pediatric/adolescent growth-related pathways, not a generic pediatric diagnosis.

Adjunct/support decisions:

```text
acupuncture for selected knee OA → optional evidence-sensitive adjunct
dry needling → excluded
ESWT → not a default generator recommendation; therapist-proposed patellar-tendon use may be documented
taping / knee braces / foot orthoses → condition-sensitive supports
NMES → procedure/context-specific, especially post-TKA; not generic OA
```

No runtime behavior changed.

---

# 4. Exact next action

```text
1. product owner selects the next remaining CU-1 regional/shared profile
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
knee = frozen v1.1
knee PR = #43 merged
canonical writer = none
runtime writer = none
runtime implementation = unauthorized
next action = product owner selects next regional/shared profile
```
