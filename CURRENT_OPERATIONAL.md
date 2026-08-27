# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified base main for this Muscle / Myotendinous design:** `6ca1b9178a9f70755d7caf342ac5ff85282ebfeb`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin, ankle/foot v1.1.
> **Frozen shared-fracture profile:** `clinic_utilities/physio_profiles/shared_fracture_v1_1.md`.
> **Frozen shared-muscle profile on active docs branch:** `clinic_utilities/physio_profiles/shared_muscle_myotendinous_v1_1.md`.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-muscle-myotendinous-v1-design-2026-08-27` until exact-head review/PR/merge/handoff close.
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
shared_muscle_myotendinous_v1_1 = FROZEN on docs branch pending exact-head review/merge
```

---

# 3. Shared Muscle / Myotendinous Injury v1.1 — product-owner-approved design

Authoritative frozen file:

```text
clinic_utilities/physio_profiles/shared_muscle_myotendinous_v1_1.md
```

Architecture:

```text
regional/shared entry
→ acute_muscle_myotendinous_injury_rehabilitation
→ muscle/site + injury type/phase + structural context
→ conservative/specialist/postoperative disposition
→ actual restrictions/findings/function
→ confirmed rehabilitation directions
```

High-visibility workflow groups:

```text
quadriceps / rectus femoris
acute adductor injury
gastrocnemius / soleus / calf
hamstring strain / partial tear
```

Visible but less frequent:

```text
pectoralis major muscle/myotendinous injury
biceps muscle-belly/myotendinous injury
abdominal-wall muscle injury
```

Rare/secondary includes iliopsoas/hip-flexor, tibialis anterior, gluteal, popliteus and other less frequent muscle injuries.

Structural boundary:

```text
bony avulsion → Shared Fracture
complete/major free-tendon rupture or avulsion without disposition → specialist structural route
postoperative repair → exact protocol owns progression
```

Product-owner retraction workflow:

```text
<2 cm retraction may support routine conservative/PT workflow
ONLY after conservative/nonoperative management is established
2 cm is not an autonomous software threshold
>=2 cm / multi-tendon complete avulsion / major weakness-deformity / high-demand unresolved case
→ prominent specialist-disposition check
→ no automatic surgery recommendation
```

Return-to-sport/work remains criterion-based and never calendar-only or MRI-only.

Adjunct policy:

```text
acupuncture → clinician-selectable optional adjunct; no tissue-healing efficacy claim
dry needling → excluded
ESWT / therapeutic ultrasound → excluded as default acute-muscle healing recommendations
compression / taping → treating-physiotherapist discretion, not generator defaults
```

No runtime behavior changed.

---

# 4. Safety emphasis

```text
major weakness / palpable defect / tendon-avulsion concern
new trauma / reinjury
large or expanding haematoma
calf DVT concern
compartment / vascular / neurological concern
postoperative repair with missing protocol
atypical failure to progress / myositis-ossificans concern after significant contusion
```

Material concern without clinician disposition prevents routine reassuring referral wording.

---

# 5. Exact next action

```text
1. exact branch-vs-main review of Shared Muscle / Myotendinous v1.1 freeze
2. open docs-only PR if clean
3. independent exact-head review
4. merge only if exact head remains clean
5. clear canonical writer lock and reconcile main
6. next remaining shared CU-1 profile = generalized deconditioning / balance / gait
```

---

# 6. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
START generalized deconditioning / balance / gait design before this handoff closes
CREATE overlapping runtime writers
```

---

# 7. Handoff completeness

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
all regional profiles = frozen v1.1
shared fracture = frozen v1.1
shared muscle/myotendinous = frozen v1.1 on docs branch pending review/merge
canonical writer = docs/cu1-muscle-myotendinous-v1-design-2026-08-27
runtime writer = none
runtime implementation = unauthorized
```
