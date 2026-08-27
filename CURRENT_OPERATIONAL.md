# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-27 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified implementation base main:** `7c49c2c6ad5ad9c710a6c02fe1ec4df467b4bab2`.
> **CU-1 DESIGN-COMPLETE CLOSEOUT:** PR #54 merged as `e064e1fe86a49dcf5026b4346d9e5f3fedfd3d92`; writer-lock cleanup PR #55 merged as `7c49c2c6ad5ad9c710a6c02fe1ec4df467b4bab2`.
> **Current major phase:** Personal Clinical Excellence foundation with an explicitly authorized CU-1 runtime implementation slice.
> **Active slice:** CU-1 Physiotherapy Referral v2 — runtime implementation v1.
> **Design authority:** `SLICE_PLAN_CURRENT.md` + `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml` + frozen v1.1 clinical profiles.
> **ACTIVE CANONICAL WRITER/LOCK:** `feat/cu1-physio-referral-runtime-v1-2026-08-27`.
> **ACTIVE RUNTIME WRITER/LOCK:** `feat/cu1-physio-referral-runtime-v1-2026-08-27`.
> **RUNTIME IMPLEMENTATION:** AUTHORIZED for this bounded CU-1 slice only.
> **CU-2:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. Authorized implementation boundary

Implement the frozen CU-1 first-runtime direction only:

```text
protected Clinical Excellence utility page
→ ephemeral ReferralDraftV1 in browser memory only
→ canonical normalization
→ route/gateway/ownership resolution
→ route requirement validation
→ declarative safety/consistency rule evaluation
→ ShortReferralFormatter / DetailedReferralFormatter
→ generated referral text
→ copy / print
```

No referral draft or generated referral text is to be persisted server-side or in browser storage in this slice.

---

# 2. Runtime/integration seams

Frozen integration direction after fresh inspection of current `main`:

```text
main.py
→ include dedicated CU-1 router

/clinical/clinic-utilities/physio-referral
→ protected HTML entrypoint

/clinical/clinic-utilities/physio-referral/api/*
→ protected ephemeral contract/validation/generation API

static/clinic-utilities/physio-referral/*
→ presentation assets only; no persisted clinical state
```

The backend must load/compose the frozen machine contract through `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml`. Runtime must not parse clinical-profile Markdown to invent routing, safety or validation semantics.

---

# 3. Hard constraints

```text
NO PostgreSQL/localStorage/sessionStorage referral persistence
NO identifiable patient fixtures/data in the public repository
NO clinical taxonomy reopening without REPLAN
NO runtime-invented route IDs, context enums, safety flags or validation rules
NO profile-Markdown interpretation for machine semantics
NO CU-2 work
NO PR-1 work
```

A contradiction between implementation reality and frozen contract/profile meaning is a REPLAN trigger, not permission to patch semantics ad hoc.

---

# 4. Acceptance evidence required before merge

```text
1. deterministic contract loader/composition tests
2. alias/context normalization tests
3. gateway/route ownership tests
4. route-required/conditional validation tests
5. declarative safety/consistency rule tests
6. formatter tests for short + detailed output
7. explicit negative tests for forbidden inference / not-assessed semantics
8. no-persistence browser/runtime evidence
9. protected-route smoke
10. exact branch-vs-main review
11. independent exact-head review
```

---

# 5. Exact next action

```text
1. freeze runtime implementation slice in SLICE_PLAN_CURRENT.md
2. implement the bounded backend contract engine + protected utility shell
3. add executable focused tests derived from frozen fixtures
4. run focused evidence
5. STOP at MERGE-READY or BLOCK for independent exact-head review
```

Merge/deploy is not implied merely by implementation completion; it requires the focused evidence/review gate above.
