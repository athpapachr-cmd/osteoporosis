# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **CU-1 HARDENING PR:** PR #50 squash-merged as `5cd3cdd9cd735b7ac55a1a162bae5a9daee08c1f`.
> **CU-1 REPEAT COMPLETENESS REVIEW PR:** PR #51 squash-merged as `9e00d231f14a119a6078564d09c7f557060e7f71`.
> **Verified base main for R1–R2 hardening:** `643d4e2443fcb03c23b39a2acf2bd4c57412b2c9`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice:** CU-1 Physiotherapy Referral v2 — R1–R2 declarative hardening FROZEN pending exact review/merge and final repeat completeness review.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin, ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **Frozen machine contract entrypoint:** `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml`.
> **ACTIVE CANONICAL WRITER/LOCK:** docs/schema-only branch `docs/cu1-r1-r2-declarative-hardening-2026-08-27`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED.
> **PR-1:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. R1–R2 frozen artifacts

```text
clinic_utilities/contracts/cu1_rule_catalog_v1.yaml
clinic_utilities/contracts/cu1_route_requirements_v1.yaml
clinic_utilities/contracts/cu1_r1_r2_design_fixtures_v1.yaml
```

`cu1_typed_supplement_v1.yaml` types `safety.input_flags[]` and the manifest makes both new catalogs normative.

R1 claim:

```text
safety/consistency trigger conditions = machine-declarative
runtime symptom-to-concern invention = forbidden
```

R2 claim:

```text
route required/conditional fields + assertion/subtype/context applicability = machine-declarative
runtime profile-prose validation interpretation = forbidden
```

These claims still require independent exact-head review and final repeat completeness review.

---

# 2. Hard constraints

```text
NO production HTML/JS/CSS
NO FastAPI CU-1 runtime endpoints
NO physiotherapy persistence
NO identifiable patient data
NO broad clinical taxonomy reopening
NO runtime invention of safety triggers or validation rules
NO PR-1 runtime work
```

---

# 3. Exact next action

```text
1. exact branch-vs-main review
2. verify only docs/schema/control-plane changes
3. validate R1–R2 against existing + focused synthetic fixtures
4. docs/schema-only PR + independent exact-head review
5. merge only if clean
6. fresh bootstrap from merged main
7. final repeat CU-1 design-completeness review
8. STOP at DESIGN-COMPLETE or remaining BLOCK
9. runtime implementation still requires separate explicit product-owner authorization
```
