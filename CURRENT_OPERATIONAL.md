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
> **CURRENT SHARED DESIGN TARGET:** Muscle / Myotendinous Injury v1.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-muscle-myotendinous-v1-design-2026-08-27` for Muscle / Myotendinous Injury v1 CU-1 clinical/content design.
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
```

---

# 3. Muscle / Myotendinous Injury — ACTIVE DESIGN CANDIDATE WORK

Authorized scope is clinical/content design only.

Primary reusable design questions:

```text
acute muscle strain / myotendinous injury taxonomy
partial tear vs complete tear / avulsion boundary
muscle-belly vs myotendinous vs intratendinous / free-tendon context when established
injury mechanism / date / phase
imaging/classification as context, never autonomous clearance
loading / stretching / running / sprinting restrictions
progressive strength / lengthening / functional loading
return-to-running / return-to-sport / return-to-work semantics
reinjury / haematoma / compartment / DVT / neurovascular safety
postoperative or specialist-protocol gateway for major tears/avulsions
regional quick-entry ownership vs shared rehabilitation logic
```

Existing frozen regional gateways already point here for:

```text
proximal rectus-femoris tendon/myotendinous injury
adductor strain/tear
iliopsoas / hip-flexor strain
rectus-femoris strain
hamstring strain
gastrocnemius strain
soleus strain
calf myotendinous injury
other acute regional muscle injury
```

Candidate should also determine whether commonly useful upper-limb/trunk muscle injuries belong here or remain in regional profiles.

The shared profile must not duplicate already-frozen dedicated tendon/pathology routes such as rotator cuff, distal biceps/triceps, digital flexor/extensor tendon repair, Achilles tendinopathy/rupture structural routes or fracture/apophyseal-avulsion logic.

---

# 4. Initial evidence / safety direction

```text
elapsed time != tissue recovery / return-to-sport clearance
MRI grade != autonomous rehabilitation prescription
pain improvement != full load capacity
strength symmetry alone != complete readiness
full-thickness proximal tendon tear / major avulsion != routine strain pathway
bony apophyseal avulsion != muscle strain → shared fracture profile
progressive loading is individualized by tissue, severity, symptoms and task demands
return to sport/work is criterion-based where relevant, not calendar-only
```

Hamstring evidence provides the strongest current criterion-based rehabilitation/RTS framework; evidence for adductor, quadriceps/rectus-femoris and calf RTS criteria is less certain and should not be presented with false precision.

---

# 5. Exact next action

```text
1. inspect frozen regional gateways and existing dedicated tendon boundaries
2. perform current evidence/safety review for acute muscle/myotendinous rehabilitation
3. create `clinic_utilities/physio_profiles/shared_muscle_myotendinous_v1.md` as DESIGN CANDIDATE
4. align `SLICE_PLAN_CURRENT.md` and `CLINIC_UTILITIES_PLAN.md` to active candidate
5. present taxonomy / visibility / tear-avulsion / postoperative / adjunct / RTS decisions to product owner
6. revise after real-workflow feedback
7. freeze/merge only after explicit product-owner approval
```

---

# 6. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
FREEZE or merge Muscle / Myotendinous Injury without product-owner approval
START generalized deconditioning / balance / gait design before this writer lock closes
CREATE overlapping runtime writers
```

---

# 7. Handoff completeness

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
all regional profiles = frozen v1.1
shared fracture = frozen v1.1
shared muscle/myotendinous = active design candidate work
canonical writer = docs/cu1-muscle-myotendinous-v1-design-2026-08-27
runtime writer = none
runtime implementation = unauthorized
```
