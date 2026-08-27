# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **CU-1 COMPLETENESS REVIEW PR:** PR #49 squash-merged as `48cd6947b8a3201cad1283cf558a3f979243ec27`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice:** CU-1 Physiotherapy Referral v2 — bounded design hardening required after completeness review.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin, ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED.
> **PR-1:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. Review outcome — CLOSED

Authoritative review:

```text
clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW.md
```

Outcome:

```text
CU-1 clinical/content profile set = FROZEN / PRESERVE
CU-1 implementation-contract completeness = BLOCK
runtime implementation authorization = NOT READY
```

The clinical taxonomy/safety content is broadly coherent. No broad regional clinical redesign is indicated.

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

No frozen clinical pathway is revoked by the review.

---

# 3. Exact next action

```text
1. product owner may authorize the bounded docs/schema-only CU-1 design-hardening pass
2. resolve B1–B6 without reopening broad clinical taxonomy unless specifically required
3. freeze cross-profile typed contract + registry/gateways + precedence + safety + formatter + common enums
4. repeat exact design-completeness review
5. STOP at DESIGN-COMPLETE or remaining BLOCK
6. runtime implementation requires a separate explicit product-owner authorization only after DESIGN-COMPLETE
```

---

# 4. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
START CU-1 UI implementation
ADD persistence for physiotherapy referrals
INVENT schema/routing/formatter semantics inside implementation
COMMIT identifiable patient data
RESTART PR-1 runtime work
CREATE overlapping runtime writers
```
