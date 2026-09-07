# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1 — IMPLEMENTED / TESTED / FINAL EXACT-HEAD REVIEW PASS / PR-READY
> **Updated:** 2026-09-07 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified base / current main:** `5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6`.
> **Production Render at activation:** `dep-dafg47mq1p3s73dv8k30` — LIVE at the same `main`.
> **Active branch:** `feat/clinical-learning-l1-challenge-foundation-mvp-2026-09-07`.
> **Current slice:** `CORE-LEARNING-HUB-L1-CHALLENGE-FOUNDATION-MVP-2026-09-07`.
> **Exact slice owner:** `SLICE_PLAN_CURRENT.md`.
> **Frozen contract owners:** `schemas/clinical_learning_core_v1.yaml`, `schemas/clinical_learning_l1_boundary_v1.yaml`, `schemas/clinical_learning_contract_manifest_v1.yaml`, `schemas/clinical_learning_design_fixtures_v1.yaml`, `schemas/osteoporosis_foundation_map_v1.yaml`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE — final review is closed; only canonical closeout, exact-head verification and PR opening are authorized before RELEASE HOLD.
> **PHYSIOTHERAPY/CU-1 STATUS:** PAUSED by product owner; remains separate/read-only through the L-1 PR checkpoint.
> **L-1 implementation authority:** GRANTED / EXERCISED within frozen scope.
> **PR authority:** AUTHORIZED after exact final docs-head regression PASS.
> **Merge/deploy authority:** NONE until separate explicit product-owner release decision.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **Signal/DailyCase/PracticeReview authority:** NONE.
> **Production config/secret authority:** NONE.
> **RF authority:** NONE.

---

# 1. Production truth

L-0 is frozen/complete and merged. PR #80 carried the independently reviewed contracts; PR #81 reconciled the post-merge canonicals. Fresh `main` remained unchanged throughout the L-1 final review:

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

It was treated as read-only. Full L-1 branch-vs-main inspection contains no mutation under:

```text
clinic_utilities/physio_referral_*
clinic_utilities/contracts/cu1_*
static/clinic-utilities/physio-referral/*
clinic_utilities/rf/*
static/clinic-utilities/rf/*
```

The L-1 regression gate also enforces this boundary.

---

# 5. Final exact-head review findings and bounded corrections

The final review covered the full branch vs fresh `main`, not only the handoff commit.

## Clinically meaningful UX findings — corrected

1. Challenge History rendered a `Νέα revision` control without a functioning revision workflow.
2. Foundation assessment HTML exposed structured controls while JavaScript referenced a nonexistent `assessmentJson`, making the clinician-facing workflow unusable.
3. Foundation evidence controls included values outside the frozen `AssessmentMethod` enum and did not expose explicit per-evidence result/note capture.
4. Prior immutable Challenge revisions were not meaningfully inspectable from History.
5. Fact Ledger provenance was insufficiently visible for progressive-disclosure review.

Corrections remain inside frozen L-1 semantics:

- History can inspect specific immutable revisions.
- `Νέα revision` clones only the latest accepted payload into a **candidate**, resets external/server authority, requires fresh server preview + clinician dispositions + explicit confirmation, then uses `PUT` to append the next server-owned revision.
- Foundation assessment is structured around only the frozen methods/states/results, explicit proposed/final state, clinician-reviewed evidence, notes and optional next-review date.
- FORMAL_SOLID requirements are made explicit in the UI while backend validation remains authoritative.
- Fact Ledger exposes source / introduced-via / stage provenance.
- UI-contract regression coverage now catches missing DOM bindings and invalid Foundation method values.

## Data-integrity / operational findings — corrected

1. Concurrent Challenge revise/reference/delete operations could otherwise cross transaction boundaries; PK constraints prevented silent duplicate rows but residual races could surface as raw `IntegrityError`, and a delete/reference race could leave an orphan mutable overlay.
2. Concurrent Foundation assessments could both read older materialized state and allow the older assessment to commit last.

Corrections:

- authoritative Challenge revision rows are row-locked for mutation on the production Postgres path;
- delete locks Challenge revisions before overlay/due/content purge + tombstone commit;
- residual DB integrity races fail closed as sanitized HTTP `409 learning_write_conflict_retry`;
- existing Foundation node state is row-locked before materialization;
- a distinct Foundation attempt must have `assessed_at` strictly newer than current state; equal timestamps fail closed because they have no deterministic ordering;
- exact same Foundation `attempt_id` remains idempotent.

No finding required frozen L-0 contract/schema mutation. No REPLAN trigger fired.

---

# 6. Reviewed substantive head and regression evidence

Final substantive code/review head:

```text
ada16afb573609cd555b99c1cc62a4a160d4215f
```

Complete Clinical Learning L1 regression gate:

```text
run 34158892560
SUCCESS
```

The run passed all required steps:

```text
Python syntax
Browser JavaScript syntax
L1 runtime + hardening + clinician UI contract tests
Inherited L0 contract regression
Scope / adjacent-owner guard
Diff hygiene
```

Fresh branch comparison at review close:

```text
base / merge-base   5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6
ahead / behind      35 / 0
physio/CU1/RF leak  NONE
frozen-schema diff  NONE
```

The canonical closeout commits after this reviewed substantive head are docs-only. The **final PR head must still pass the complete regression gate exactly at that final head** before the PR is opened.

---

# 7. Lifecycle at canonical closeout

```text
L-0 CONTRACT                         FROZEN / COMPLETE / MERGED
L-1 PRODUCT-OWNER AUTHORITY          GRANTED
L-1 CORE IMPLEMENTED                 YES
L-1 CLINICIAN-FACING UX HARDENING    IMPLEMENTED
L-1 TESTED                           YES
L-1 FINAL EXACT-HEAD REVIEW          PASS
L-1 PHYSIO/RF ISOLATION              PASS
L-1 REPLAN REQUIRED                  NO
L-1 PR                               NONE — NEXT AFTER FINAL DOCS-HEAD PASS
L-1 MERGED                           NO
L-1 DEPLOYED                         NO
L-1 PRODUCTION-SMOKE-VERIFIED        NO
```

---

# 8. Exact next action / release hold boundary

```text
finish canonical closeout
→ verify final branch-vs-main diff and fresh main
→ require complete Clinical Learning L1 regression gate SUCCESS on the exact final docs head
→ verify no existing L-1 PR
→ open one bounded PR to main
→ RELEASE HOLD
→ STOP for explicit product-owner squash-merge/release authority
```

Do **not** merge, deploy, change production configuration/secrets, start PR-1/PR-2/Daily Case Review/Signal work, or resume physiotherapy/CU-1/RF before that separate product-owner decision.
