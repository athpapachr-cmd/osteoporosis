# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified base main for this design:** `8c9a1cdd9692fe827827b90e30db9a70d828eb22`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice:** CU-1 Physiotherapy Referral v2 clinical/content design.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin, ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1.
> **Frozen final shared profile on active docs branch:** `clinic_utilities/physio_profiles/shared_deconditioning_balance_gait_v1_1.md`.
> **ACTIVE CANONICAL WRITER/LOCK:** `docs/cu1-deconditioning-balance-gait-v1-design-2026-08-27` until exact-head review/PR/merge/handoff close.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED.
> **PR-1:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. Frozen final shared profile

Authoritative file:

```text
clinic_utilities/physio_profiles/shared_deconditioning_balance_gait_v1_1.md
```

Routine direct routes:

```text
D1 generalized deconditioning / functional decline
D2 frailty-associated functional decline — clinician-established only
```

Non-routine context/findings rather than first-line pathways:

```text
balance impairment / falls-risk context
gait / mobility impairment context
post-illness / post-hospital deconditioning context
```

Direct findings/goals include generalized muscular weakness, poor coordination, fear of falling, falls history and walking-aid assessment/training.

Preferred optional functional battery:

```text
SPPB
→ total + standing-balance component + 4-m gait component + 5-chair-rise component
```

Hard rule:

```text
SPPB/TUG/gait-speed/5xSTS result != autonomous frailty/falls/neurological diagnosis
```

Falls history preserves single/recurrent/injurious/unable-to-rise/loss-of-consciousness states. Fear of falling remains distinct from objective balance deficit.

Shared Fracture may gateway here for strength/balance/falls/independence goals, but fracture restrictions always remain authoritative.

Walking-aid assessment/training is a direct referral direction. Home-hazard assessment is not exposed as a routine local option. Parkinson disease, stroke, peripheral neuropathy and similar neurological diagnoses are not routine CU-1 pathways in this product-owner workflow.

Generic aerobic/endurance conditioning is not a routine generator direction. Acupuncture, dry needling, ESWT and therapeutic ultrasound are excluded from this shared profile.

---

# 2. Safety boundaries

```text
new focal neurological deficit
acute / rapidly progressive unexplained gait change
new inability to stand/walk without explained stable cause
syncope / presyncope / unexplained LOC
unstable cardiopulmonary symptoms / marked unexplained breathlessness
acute vestibular syndrome
new fracture / unresolved fracture restrictions
DVT / vascular concern
acute infection/systemic deterioration
new delirium/confusion
progressive unexplained coordination loss
→ medical/specialist/regional reassessment as appropriate
```

No reassuring negative statement is generated from missing assessment.

---

# 3. Exact next action

```text
1. exact branch-vs-main review of final Shared Deconditioning/Balance/Gait v1.1 freeze
2. docs-only PR if clean
3. independent exact-head review
4. merge only if exact head remains clean
5. clear canonical writer lock / reconcile main
6. perform CU-1 design-completeness review
7. runtime implementation only after a separate explicit product-owner authorization
```

---

# 4. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
COMMIT identifiable patient data
START implementation before design-completeness review + explicit authorization
CREATE overlapping runtime writers
```
