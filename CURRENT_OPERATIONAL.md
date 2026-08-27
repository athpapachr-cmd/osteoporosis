# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **CU-1 HARDENING PR:** PR #50 squash-merged as `5cd3cdd9cd735b7ac55a1a162bae5a9daee08c1f`.
> **CU-1 REPEAT COMPLETENESS REVIEW PR:** PR #51 squash-merged as `9e00d231f14a119a6078564d09c7f557060e7f71`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice:** CU-1 Physiotherapy Referral v2 — repeat completeness review CLOSED at remaining BLOCK R1–R2.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin, ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **Frozen machine contract:** `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml` and its normative artifacts.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED.
> **PR-1:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. Repeat review outcome — CLOSED

Authoritative review:

```text
clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW_V2.md
```

Result:

```text
original B1–B6 = substantially resolved / PASS at their defined level
remaining completeness result = BLOCK
```

Remaining blockers only:

```text
R1 — safety/consistency trigger conditions are not yet machine-declarative
R2 — route-specific required/conditional context and formal-assertion validation are not yet machine-declarative
```

Clinical/content profiles and machine contract v1 remain frozen.

---

# 2. Exact next authorized design action

If the product owner chooses to continue CU-1 design hardening:

```text
1. create a bounded docs/schema-only pass for R1–R2
2. add declarative `cu1_rule_catalog_v1.yaml`
3. add declarative `cu1_route_requirements_v1.yaml`
4. do not reopen clinical taxonomy
5. repeat design-completeness review
6. STOP at DESIGN-COMPLETE or remaining BLOCK
```

---

# 3. Explicitly forbidden now

```text
WRITE CU-1 runtime code
START CU-1 UI implementation
ADD physiotherapy persistence
INVENT safety triggers or route validation semantics in runtime
COMMIT identifiable patient data
RESTART PR-1 runtime work
CREATE overlapping runtime writers
```

Runtime implementation requires a future `DESIGN-COMPLETE` result plus separate explicit product-owner authorization.
