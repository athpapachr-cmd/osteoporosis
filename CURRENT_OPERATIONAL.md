# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **ANKLE/FOOT FREEZE PR:** PR #45 squash-merged as `64b79d571e57a457480cd5a7814001c0566a9e4b`.
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
hip_v1_1 = FROZEN
ankle_foot_v1_1 = FROZEN
```

---

# 3. Ankle / Foot v1.1 — closed freeze

Frozen routine pathways:

```text
AF1 lateral ankle sprain rehabilitation
AF2 Achilles tendinopathy — midportion / insertional
AF3 plantar heel pain / plantar fasciitis
AF4 posterior tibial tendon dysfunction / flexible PCFD
AF5 peroneal tendon disorder — conservative rehabilitation
AF6 mechanical metatarsalgia / forefoot overload
AF7 post-traumatic ankle/foot pain or stiffness after assessed injury
```

Frozen pediatric/adolescent navigation:

```text
Παιδιά / Έφηβοι — ποδοκνημική / άκρος πόδας
→ Sever disease / calcaneal apophysitis
→ symptomatic accessory navicular
→ symptomatic flexible flatfoot
```

Rare/advanced/context decisions:

```text
chronic ankle instability / recurrent sprain → rare/secondary
syndesmotic/high-ankle sprain → very rare/advanced
tarsal tunnel → rare neurological
heel fat-pad pain → rare/secondary
Morton neuroma → rare/context
plantar-plate / lesser-MTP instability → very rare/advanced
anterior tibial / extensor / FHL tendon disorder → rare
osteochondral talus lesion → rare/advanced
hallux rigidus / 1st-MTP OA → context only
ankle OA → context only
Charcot / neuropathic hot swollen foot → high-priority medical/offloading context, not PT pathway
postoperative ankle/foot → advanced only; occasional Achilles repair/reconstruction
```

Support/adjunct policy:

```text
taping → directly visible optional support
heel lift → directly visible optional support
brace / orthosis / AFO / metatarsal pad / footwear / offloading → condition-specific context, often podiatry-coordinated
manual therapy / soft tissue → optional where relevant
dry needling → optional clinician-selected adjunct
acupuncture → excluded
ESWT plantar heel → evidence-supported optional adjunct
ESWT Achilles → evidence-conflicted optional adjunct; not routine or superior to progressive loading
```

Safety emphasis:

```text
fracture / bone-stress / Lisfranc concern
syndesmotic / Maisonneuve concern
Achilles rupture concern
peroneal dislocation/subluxation or major tear
hot swollen neuropathic foot / Charcot
infection / nonhealing wound / vascular-neurological deficit
unknown healing / weight-bearing / postoperative restrictions
```

No runtime behavior changed.

---

# 4. Resolved operational incident

During PR preparation, `_noop` was accidentally created on `main` and immediately removed. Exact comparison of the pre-incident and repaired main states showed zero net file diff. Temporary branch-only `_noop2`–`_noop8` files created during the rebase attempt were discarded before PR #45; none exists in the merged tree.

The incident changed no clinical/runtime content and is closed.

---

# 5. Exact next action

```text
1. product owner selects the next remaining shared CU-1 profile
2. current preferred sequence: shared fracture / post-immobilization → muscle / myotendinous injury → generalized deconditioning / balance / gait
3. continue CU-1 design only until explicit runtime authorization
```

---

# 6. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
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
knee = frozen v1.1
hip/groin = frozen v1.1
ankle/foot = frozen v1.1
ankle/foot PR = #45 merged
canonical writer = none
runtime writer = none
runtime implementation = unauthorized
next action = product owner selects next shared profile
```
