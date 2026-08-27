# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified base main for this Shared Fracture design:** `0673dc24d1e34d7a6d562103bab84830da8f585e`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin, ankle/foot v1.1.
> **Frozen shared-fracture profile on active docs branch:** `clinic_utilities/physio_profiles/shared_fracture_v1_1.md`.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-shared-fracture-v1-design-2026-08-27` until exact-head review/PR/merge/handoff close.
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
shared_fracture_v1_1 = FROZEN on docs branch pending exact-head review/merge
```

---

# 3. Shared Fracture / Post-immobilization v1.1 — product-owner-approved design

Authoritative frozen file:

```text
clinic_utilities/physio_profiles/shared_fracture_v1_1.md
```

Architecture:

```text
regional/shared fracture entry
→ fracture_rehabilitation_post_immobilization
→ fracture site
→ treatment / phase / healing-stability
→ immobilization / lower-limb weight-bearing OR upper-limb use/loading
→ ROM / strengthening / impact restrictions
→ actual deficits / function
→ confirmed rehabilitation goals/directions
```

Hard rules:

```text
elapsed time != union
cast/sling/boot removal != unrestricted loading
fixation != unrestricted loading
not stated != unrestricted
exact orthopaedic protocol > generic shared suggestion
no fixed week-based ROM / weight-bearing / strengthening timeline without explicit instruction
manual therapy requires known stability + ROM permission
pediatric fracture != adult timeline
fragility fracture != software diagnosis of osteoporosis
```

High-visibility workflow entries include:

```text
vertebral compression / fragility fracture
proximal humerus
clavicle
distal radius
hand / finger fractures
pubic rami
patella
ankle fractures
calcaneus including anterior-process calcaneus
5th metatarsal / other metatarsal
foot / toe fractures
```

Less frequent / advanced include scaphoid with strong union gate, elbow fractures, tibial plateau, Lisfranc and other site-sensitive fractures. Long-bone shaft fractures and older-adult hip fracture are not routine high-visibility outpatient-referral entries in this workflow.

Fragility modifier:

```text
formal_fragility_fracture_context
known_osteoporosis_or_low_bone_strength_context
falls_risk_or_recurrent_falls_context
```

When selected, balance/falls/strength/functional-independence goals become prominent without creating osteoporosis diagnosis or medication advice.

SIFK / legacy SONK decision:

```text
preferred structured entity = subchondral_insufficiency_fracture_of_knee
SIFK = preferred current terminology
SONK = legacy / clinician-entered wording, not an autonomous second software diagnosis
advanced SIFK may carry osteonecrosis / osteochondral-collapse context when established
bone-marrow edema alone != SIFK
SIFK + loading status unknown → no generic strengthening / impact progression
```

Pediatric group remains low visibility except for pelvic apophyseal avulsions; buckle fractures are not routine referrals in this workflow.

Excluded as default fracture-healing recommendations:

```text
acupuncture
dry needling
ESWT
therapeutic ultrasound to accelerate union
bone-stimulator prescription
```

No runtime behavior changed.

---

# 4. Safety emphasis

```text
unknown healing/stability/use/loading state
loss of reduction / reinjury / delayed union / nonunion / hardware concern
infection / wound / pin-site concern
new neurovascular deficit / compartment concern
DVT/PE concern in relevant lower-limb context
possible CRPS without autonomous diagnosis
vertebral neurological/spinal-precaution concern
stress/insufficiency fracture with unknown impact/loading status
pediatric physeal/apophyseal restrictions
```

Material concern without clinician disposition prevents routine reassuring referral wording.

---

# 5. Exact next action

```text
1. exact branch-vs-main review of Shared Fracture v1.1 freeze
2. open docs-only PR if clean
3. independent exact-head review
4. merge only if exact head remains clean
5. clear canonical writer lock and reconcile main
6. product owner selects next shared CU-1 profile
```

---

# 6. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
START muscle/myotendinous design before shared-fracture handoff closes
CREATE overlapping runtime writers
```

---

# 7. Prior resolved operational incident

The Ankle/Foot `_noop` incident remains closed: all temporary files were removed/discarded before PR #45 and the main-tree net file diff was zero.

---

# 8. Handoff completeness

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
all regional profiles = frozen v1.1
shared fracture = frozen v1.1 on docs branch pending review/merge
canonical writer = docs/cu1-shared-fracture-v1-design-2026-08-27
runtime writer = none
runtime implementation = unauthorized
```
