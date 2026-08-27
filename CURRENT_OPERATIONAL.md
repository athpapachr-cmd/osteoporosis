# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified base main for this review:** `4b12932cb007994ca5d998f47719ff706191d2e9`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice:** CU-1 Physiotherapy Referral v2 design-completeness review.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin, ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-design-completeness-review-2026-08-27` for review/canonical documentation only.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED.
> **PR-1:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. Review outcome

```text
CU-1 clinical/content profile set = FROZEN
CU-1 implementation-contract completeness = BLOCK
runtime implementation authorization = NOT READY
```

The clinical taxonomy/safety content is broadly coherent. The block is caused by unresolved cross-profile and machine-contract design that would otherwise have to be invented during implementation.

Authoritative review report on this branch:

```text
clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW.md
```

---

# 2. Blocking design gaps

```text
B1 — ReferralDraft cannot deterministically carry profile-specific structured context
B2 — no frozen machine-readable profile/route/key registry or exact regional→shared gateway mapping
B3 — unresolved route ownership/precedence in selected postoperative/structural overlaps
B4 — no shared warning/safety severity + blocking/disposition contract
B5 — ShortReferralFormatter / DetailedReferralFormatter interface and omission/output rules are not frozen
B6 — tri-state/enumeration/key naming is not normalized/versioned across profiles
```

No frozen clinical pathway is revoked by this review.

---

# 3. Exact next action

```text
1. complete one bounded CU-1 design-hardening pass resolving B1–B6
2. do not reopen broad regional clinical taxonomy unless a blocker requires it
3. freeze the cross-profile machine contract + routing/precedence + formatter contract
4. repeat exact design-completeness review
5. STOP at DESIGN-COMPLETE or remaining BLOCK
6. only after DESIGN-COMPLETE may the product owner separately authorize runtime implementation
```

---

# 4. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
START UI implementation
ADD persistence for physiotherapy referrals
INVENT schema/routing/formatter semantics inside implementation
COMMIT identifiable patient data
RESTART PR-1 runtime work
CREATE overlapping runtime writers
```
