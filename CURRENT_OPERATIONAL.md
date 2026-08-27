# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **CU-1 R1–R2 HARDENING PR:** PR #52 squash-merged as `f0a31b335848a1799a0b0b116a3bbe29a75fa7b3`.
> **Verified base main for final correction:** `f0a31b335848a1799a0b0b116a3bbe29a75fa7b3`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice:** CU-1 Physiotherapy Referral v2 — exact R2 transcription correction found by final post-merge completeness review.
> **Frozen clinical profiles:** all planned CU-1 regional/shared v1.1 profiles remain frozen.
> **Frozen machine contract entrypoint:** `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml`.
> **ACTIVE CANONICAL WRITER/LOCK:** docs/schema-only branch `docs/cu1-r2-contract-correction-2026-08-27`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED.
> **PR-1:** intentionally paused.

---

# 1. Exact correction scope

Final review found two transcription defects in `cu1_route_requirements_v1.yaml` shared-muscle validation:

```text
1. MRI/ultrasound confirmation had been made conditionally required although the frozen shared-muscle profile defines imaging context as optional.
2. major-avulsion safety boundary referenced an invalid artifact path instead of the canonical safety-input flag id.
```

Correction authority:

```text
clinic_utilities/contracts/cu1_route_requirements_correction_v1.yaml
```

The manifest applies that correction before route validation. No clinical recommendation changes.

---

# 2. Exact next action

```text
1. exact docs/schema-only review of the correction
2. PR + exact-head review
3. merge only if clean
4. fresh final CU-1 design-completeness review
5. STOP at DESIGN-COMPLETE or remaining BLOCK
6. runtime implementation remains separately unauthorized
```
