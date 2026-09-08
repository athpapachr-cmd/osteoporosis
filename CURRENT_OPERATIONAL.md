# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1 — INDEPENDENT REVIEW REMEDIATION ACTIVE / PR #82 RELEASE HOLD
> **Updated:** 2026-09-08 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified base / current main:** `5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6`.
> **Production Render at activation:** `dep-dafg47mq1p3s73dv8k30` — LIVE at the same `main`.
> **Active branch:** `feat/clinical-learning-l1-challenge-foundation-mvp-2026-09-07`.
> **Current slice:** `CORE-LEARNING-HUB-L1-CHALLENGE-FOUNDATION-MVP-2026-09-07`.
> **Exact slice owner:** `SLICE_PLAN_CURRENT.md`.
> **Frozen contract owners:** `schemas/clinical_learning_core_v1.yaml`, `schemas/clinical_learning_l1_boundary_v1.yaml`, `schemas/clinical_learning_contract_manifest_v1.yaml`, `schemas/clinical_learning_design_fixtures_v1.yaml`, `schemas/osteoporosis_foundation_map_v1.yaml`.
> **ACTIVE RUNTIME WRITER/LOCK:** L-1 independent-review remediation only — bounded to findings from review `5144979821`.
> **PHYSIOTHERAPY/CU-1 STATUS:** PAUSED by product owner; remains separate/read-only through the L-1 release decision.
> **L-1 implementation authority:** GRANTED / EXERCISED within frozen scope; remediation explicitly authorized 2026-09-08.
> **PR:** #82 OPEN / DRAFT / RELEASE HOLD.
> **Merge/deploy authority:** NONE until separate explicit product-owner release decision.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **Signal/DailyCase/PracticeReview authority:** NONE.
> **Production config/secret authority:** NONE.
> **RF authority:** NONE.

---

# 1. Production truth

L-0 is frozen/complete and merged. PR #80 carried the independently reviewed contracts; PR #81 reconciled the post-merge canonicals. Fresh `main` remained unchanged throughout the L-1 final review and PR opening:

```text
5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6
```

Render production is still on the existing main-line state; L-1 has **not** been merged or deployed.

```text
dep-dafg47mq1p3s73dv8k30
LIVE at activation/main baseline
```

---

# 2. Frozen L-1 scope delivered

Protected route family:

```text
/clinical/learning
```

Runtime/UI owners:

```text
clinical_learning/
static/clinical-learning/
```

Delivered L-1 capabilities:

```text
protected Challenge JSON validation / preview
recursive unknown-field rejection + deterministic PHI guard
sanitized privacy/schema errors
Fact Ledger + progressive-disclosure provenance
clinician Accept / Modify / Dismiss observations
immutable Challenge revisions + deterministic duplicate/conflict/idempotency semantics
Challenge History with immutable revision inspection and clinician new-revision workflow
content purge + non-content tombstone delete
reference-verification external overlay
14-node Osteoporosis Foundation Map
explicit structured clinician-reviewed Foundation assessment
Foundation state persistence with stale/concurrent overwrite protection
explicit next-review / due-state materialization
occurrence-based learning due state without adaptive cadence
JSON / Markdown learning export
browser-session + X-Clinical-Key protected access
clinician-facing four-view Learning Hub MVP
```

Four MVP views:

1. Challenge Import
2. Challenge History
3. Foundation Map
4. Due / Learning Actions

---

# 3. Frozen exclusions — unchanged

```text
NO patient encounter/lab reads or writes
NO raw transcript intake/storage
NO PR-1/PR-2 transcript semantics
NO Practice Review AI
NO Daily Case Review runtime/UI/table
NO Signal promotion/backlink authority
NO external bearer learning API
NO adaptive spaced-repetition algorithm
NO composite knowledge/excellence score
NO production config/secrets changes
NO RF mutation
NO physiotherapy/CU-1 mutation
```

Hard invariant remains:

```text
LEARNING RECORD != PATIENT RECORD
LearningFactV1.authoritative_for_patient = false
```

Any future requirement to change a frozen L-0 semantic owner remains a `STOP → REPLAN` trigger.

---

# 4. Physiotherapy / adjacent-owner isolation

The separate productization branch remains:

```text
design/physio-referral-product-ux-v1-knee-oa-2026-09-07
```

It is read-only during this remediation. L-1 must not mutate:

```text
clinic_utilities/physio_referral_*
clinic_utilities/contracts/cu1_*
static/clinic-utilities/physio-referral/*
clinic_utilities/rf/*
static/clinic-utilities/rf/*
```

The L-1 regression gate enforces this boundary.

---

# 5. Independent reviewer reopening — 2026-09-08

A fresh read-only review of exact PR head `777647a27fa34a3afdb7ad4d4195be942e720d` returned **CHANGES REQUIRED**. The prior `FINAL REVIEW PASS` is therefore superseded until this bounded remediation closes and a new exact-head gate passes.

Bounded findings to close:

1. Enforce frozen nested `unique_items` invariants that were not yet runtime-validated.
2. Complete frozen Challenge History UI contract: date/review-state filters and visible linked Foundation nodes/due state.
3. Prevent save from using a stale server preview after the Challenge JSON textarea has changed.
4. Make same-revision duplicate preview comparison authority-neutral so server-reset clinician/reference authority does not create a false conflict.
5. Keep Foundation browser attempt payload stable across preview/save/network retry by reusing assessed timestamp and evidence IDs within the open assessment.

No frozen schema change is required. **REPLAN: NO.**

Required closure evidence:

```text
bounded code corrections
+ regression tests for every finding
+ full exact-head Clinical Learning L1 regression gate
+ fresh focused closure review
+ canonical reconciliation
→ RELEASE HOLD remains until separate product-owner merge authority
```

---

# 6. Prior review evidence retained for history

Prior substantive code/review head:

```text
ada16afb573609cd555b99c1cc62a4a160d4215f
run 34158892560 — SUCCESS
```

Prior canonical HOLD head:

```text
777647a27fa34a3afabdb7ad4d4195be942e720d
run 34159589677 — L1 SUCCESS
```

The separate L-0 workflow on that head passed `Validate Clinical Learning L0 contracts` and failed only its intentionally design-only `Confirm design-only scope` guard. That closed L-0 guard is not modified by this remediation.

---

# 7. Lifecycle

```text
L-0 CONTRACT                         FROZEN / COMPLETE / MERGED
L-1 PRODUCT-OWNER AUTHORITY          GRANTED
L-1 CORE IMPLEMENTED                 YES
L-1 INDEPENDENT REVIEW               CHANGES REQUIRED
L-1 REMEDIATION                      ACTIVE
L-1 PHYSIO/RF ISOLATION              REQUIRED / ENFORCED
L-1 REPLAN REQUIRED                  NO
L-1 PR                               #82 OPEN / DRAFT / RELEASE HOLD
L-1 MERGED                           NO
L-1 DEPLOYED                         NO
L-1 PRODUCTION-SMOKE-VERIFIED        NO
```

---

# 8. Exact next action / HOLD

```text
close only independent-review findings 1–5
→ add regression coverage
→ run complete exact-head L-1 gate
→ focused closure review
→ reconcile canonicals and release writer lock
→ STOP in RELEASE HOLD
```

Do **not** merge, deploy, change production configuration/secrets, start PR-1/PR-2/Daily Case Review/Signal work, or resume physiotherapy/CU-1/RF during this remediation.