# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **CU-1 HARDENING PR:** PR #50 squash-merged as `5cd3cdd9cd735b7ac55a1a162bae5a9daee08c1f`.
> **CU-1 REPEAT COMPLETENESS REVIEW PR:** PR #51 squash-merged as `9e00d231f14a119a6078564d09c7f557060e7f71`.
> **Verified base main for R1–R2 hardening:** `643d4e2443fcb03c23b39a2acf2bd4c57412b2c9`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice:** CU-1 Physiotherapy Referral v2 — bounded declarative hardening for remaining completeness gaps R1–R2.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin, ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **Frozen machine contract:** `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml` and its normative artifacts.
> **ACTIVE CANONICAL WRITER/LOCK:** docs/schema-only branch `docs/cu1-r1-r2-declarative-hardening-2026-08-27`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED.
> **PR-1:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. Current authorized work

Resolve only the two remaining completeness blockers from `clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW_V2.md`:

```text
R1 — machine-declarative safety/consistency trigger catalog
R2 — machine-declarative route requirements / conditional validation catalog
```

Required artifacts:

```text
clinic_utilities/contracts/cu1_rule_catalog_v1.yaml
clinic_utilities/contracts/cu1_route_requirements_v1.yaml
```

The artifacts must reference the already frozen clinical profiles and machine contract and must not change clinical meaning.

---

# 2. Hard constraints

```text
NO production HTML/JS/CSS
NO FastAPI CU-1 runtime endpoints
NO persistence for physiotherapy referrals
NO identifiable patient data
NO broad regional clinical redesign
NO mutation of frozen clinical profiles unless a contradiction is proven
NO PR-1 runtime work
```

---

# 3. Exact next action

```text
1. freeze declarative R1–R2 catalogs
2. update contract manifest + focused semantic fixtures as required
3. exact branch-vs-main review
4. docs/schema-only PR + independent exact-head review
5. merge only if clean
6. fresh repeat CU-1 design-completeness review
7. STOP at DESIGN-COMPLETE or remaining BLOCK
8. runtime implementation still requires separate explicit product-owner authorization
```
