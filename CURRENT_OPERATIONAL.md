# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **CU-1 COMPLETENESS REVIEW PR:** PR #49 squash-merged as `48cd6947b8a3201cad1283cf558a3f979243ec27`.
> **Verified base main:** `aed3787188e681e6d57a2ac237a1cf8310099a95`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice:** CU-1 Physiotherapy Referral v2 — bounded B1–B6 design hardening.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin, ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **ACTIVE CANONICAL WRITER/LOCK:** docs/schema-only branch `docs/cu1-contract-hardening-v1-2026-08-27`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED.
> **PR-1:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. Current authorized work

Resolve only the design-completeness blockers B1–B6 identified in:

```text
clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW.md
```

Required artifacts:

```text
CU-1 typed core contract v1
canonical machine registry v1
exact regional→shared gateway map
route ownership / precedence rules
common safety/warning/disposition contract
ShortReferralFormatter / DetailedReferralFormatter specification
normalized common enums / tri-states
synthetic design-fixture matrix
```

Broad clinical taxonomy remains frozen.

---

# 2. Hard constraints

```text
NO production HTML/JS/CSS
NO FastAPI CU-1 runtime endpoints
NO persistence for physiotherapy referrals
NO identifiable patient data
NO broad regional clinical redesign unless an exact blocker forces it
NO PR-1 runtime work
```

---

# 3. Exact next action

```text
1. freeze B1–B6 machine/design contract on this branch
2. exact branch-vs-main review
3. docs/schema-only PR + independent exact-head review
4. merge only if clean
5. repeat CU-1 design-completeness review against the frozen contract
6. STOP at DESIGN-COMPLETE or remaining BLOCK
7. runtime implementation still requires separate explicit product-owner authorization
```
