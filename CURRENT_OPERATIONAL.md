# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **FINAL SHARED PROFILE FREEZE PR:** PR #48 squash-merged as `50c304cbb8bf68cba4fde981942b5fc26065afee`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice:** CU-1 Physiotherapy Referral v2 clinical/content design.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin, ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED.
> **PR-1:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. CU-1 clinical/content design state

All currently planned CU-1 regional/shared clinical/content profiles are frozen and merged.

Final shared functional profile:

```text
clinic_utilities/physio_profiles/shared_deconditioning_balance_gait_v1_1.md
```

Key final decisions:

```text
D1 generalized deconditioning / functional decline — direct
D2 frailty-associated functional decline — direct only when frailty is established
balance-only / gait-only / post-hospital presentations — context/findings, not routine top-level routes
SPPB — preferred optional functional battery, never a diagnostic gate
falls history — single/recurrent/injurious/unable-to-rise/LOC preserved
walking-aid assessment/training — direct referral direction
fear of falling — distinct from objective balance deficit
Shared Fracture restrictions override this profile
```

No neurological disease-specific pathways were added; home-hazard assessment is not a routine local option; generic aerobic conditioning is not a routine generator direction; acupuncture, dry needling, ESWT and therapeutic ultrasound are excluded.

---

# 2. Safety boundaries

```text
acute unexplained gait/coordination change
new focal neurological deficit
syncope / unexplained LOC
unstable cardiopulmonary symptoms
acute vestibular syndrome
fracture/restriction uncertainty
DVT / vascular concern
infection/systemic deterioration
acute cognitive change
→ reassessment / appropriate specialty route
```

No reassuring negative statement is generated from missing assessment.

---

# 3. Exact next action

```text
1. perform CU-1 design-completeness review across all frozen regional/shared profiles
2. inspect cross-profile routing, schema consistency, safety invariants, formatter requirements and implementation seams
3. identify any blocking contradictions or missing design contracts
4. STOP at DESIGN-COMPLETE or BLOCK
5. runtime implementation requires a separate explicit product-owner authorization after that review
```

---

# 4. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
COMMIT identifiable patient data
START CU-1 implementation before design-completeness review + explicit authorization
CREATE overlapping runtime writers
```
