# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **CU-1 HARDENING PR:** PR #50 squash-merged as `5cd3cdd9cd735b7ac55a1a162bae5a9daee08c1f`.
> **Verified base main for repeat review:** `5cd3cdd9cd735b7ac55a1a162bae5a9daee08c1f`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice:** CU-1 Physiotherapy Referral v2 — repeat design-completeness review after machine-contract hardening.
> **Frozen regional profiles:** cervical, lumbar, shoulder, elbow, wrist/hand, knee, hip/groin, ankle/foot v1.1.
> **Frozen shared profiles:** Shared Fracture v1.1; Shared Muscle/Myotendinous v1.1; Shared Deconditioning/Balance/Gait v1.1.
> **Frozen machine contract:** `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml` and its normative artifacts.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-design-completeness-review-v2-2026-08-27` for repeat review documentation only.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED.
> **PR-1:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. Current review state

The B1–B6 hardening contract is merged and frozen. Repeat review is evaluating the original design-completeness exit criteria against that exact merged contract.

No clinical profile or machine contract mutation is authorized on this review branch.

---

# 2. Exact repeat-review findings

The original B1–B6 are substantially resolved, but two remaining implementation-semantic gaps were identified:

```text
R1 — safety-rule trigger conditions are not yet machine-declarative
R2 — route-specific required/conditional context and formal-assertion validation are not yet machine-declarative
```

Details belong in `clinic_utilities/CU1_DESIGN_COMPLETENESS_REVIEW_V2.md`.

---

# 3. Exact next action

```text
1. record repeat review as BLOCK with only R1–R2
2. exact docs-only review/merge of the repeat-review result
3. release writer lock to NONE
4. next authorized design action, if product owner chooses to continue, is a very small declarative-rule hardening pass for R1–R2
5. repeat completeness review after that pass
6. runtime implementation remains forbidden until DESIGN-COMPLETE + separate explicit product-owner authorization
```

---

# 4. Explicitly forbidden now

```text
WRITE CU-1 runtime code
START UI implementation
ADD persistence
MUTATE frozen clinical profiles
MUTATE the frozen v1 machine contract during this review
INVENT safety triggers or required-field semantics in runtime
RESTART PR-1 runtime work
```
