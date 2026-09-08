# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1 — MERGED / DEPLOYED / AUTHENTICATED PRODUCTION SMOKE PENDING
> **Updated:** 2026-09-08 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Runtime squash merge SHA:** `66900c2d50f98446184a92539baab89e9d49514c` — PR #82.
> **Last production-verified deploy commit:** `1b7c0fa9443c4fe3376f80e6bc3819473de7e95b` — docs-only descendant containing the same reviewed runtime tree.
> **Merged PR:** `#82 Clinical Learning L1: Challenge + Foundation MVP`.
> **Reviewed source head:** `e73dc7862ecd1084de8e2912f45ae758110a7447`.
> **Exact pre-merge L-1 gate:** run `34264097408` — SUCCESS.
> **Independent pre-merge review:** `5144979821` — CHANGES REQUIRED, fully remediated.
> **Focused closure review:** `5145618117` — PASS.
> **Render service:** `srv-d5qfk31r0fns73di596g` (`osteoporosis`).
> **Verified live deploy:** `dep-dag5ld0ae00c738g0370` — LIVE on `1b7c0fa9443c4fe3376f80e6bc3819473de7e95b`, trigger `new_commit`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **ACTIVE DESIGN/CANONICAL WRITER:** NONE after deployment reconciliation.
> **Merge authority:** EXERCISED / COMPLETE.
> **Deployment authority:** normal Render auto-deploy only; no manual duplicate deploy used.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **Signal/DailyCase/PracticeReview authority:** NONE.
> **Production config/secret authority:** NONE.
> **RF authority:** NONE.

---

# 1. Production / release truth

PR #82 was squash-merged by explicit product-owner authority.

```text
source reviewed head      e73dc7862ecd1084de8e2912f45ae758110a7447
runtime merge commit      66900c2d50f98446184a92539baab89e9d49514c
verified live descendant  1b7c0fa9443c4fe3376f80e6bc3819473de7e95b
base parent               5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6
```

GitHub confirms PR #82 is closed and merged. The reviewed runtime entered `main` at `66900c2d50f98446184a92539baab89e9d49514c`. Subsequent canonical reconciliation commits are documentation-only descendants and do not change the reviewed runtime tree.

Render auto-deploy is verified against the first post-merge docs-only descendant:

```text
service     srv-d5qfk31r0fns73di596g
name        osteoporosis
branch      main
autoDeploy  yes / commit
url         https://ortho-reception-backend.onrender.com
deploy      dep-dag5ld0ae00c738g0370
commit      1b7c0fa9443c4fe3376f80e6bc3819473de7e95b
status      live
trigger     new_commit
finished    2026-09-08T18:59:43.990625Z
```

No manual duplicate deploy was triggered. Later docs-only canonical descendants may advance `main` and trigger further automatic deployments without changing runtime code; they do not invalidate the verified L-1 runtime deployment above.

The reviewed runtime tree delivered:

```text
protected /clinical/learning route family
Challenge import / preview / clinician review / immutable revision persistence
recursive unknown-field rejection + deterministic PHI guard
Fact Ledger and progressive-disclosure provenance
reference-verification external overlay
Challenge History / revision inspection / delete tombstone
14-node Osteoporosis Foundation Map
explicit clinician-reviewed Foundation assessments and state
occurrence-based due-state materialization
JSON / Markdown learning export
browser session + X-Clinical-Key protection
no-store/no-cache for Clinical Learning assets
```

---

# 2. Independent review closure

A fresh independent pre-merge review of prior HOLD head `777647a27fa34a3afdb7ad4d4195be942e720d` returned CHANGES REQUIRED.

All findings were corrected inside frozen L-1 runtime/UI owners without changing frozen L-0 schemas:

1. nested `unique_items` enforcement aligned with frozen contracts;
2. Challenge History gained frozen date/review-state filters and Foundation/due visibility;
3. stale Challenge textarea after preview now fails closed before save;
4. exact duplicate preview uses authority-neutral comparison without changing immutable content-hash semantics;
5. Foundation browser retry keeps stable attempt/evidence identity and timestamp for one open assessment.

Regression owner:

```text
test_clinical_learning_l1_independent_review.py
```

Substantive remediation head and gate:

```text
b4b0438e44b8077ce5a9f469b5e4669d083886f0
run 34263571488 — SUCCESS
```

Final canonical source head and gate:

```text
e73dc7862ecd1084de8e2912f45ae758110a7447
run 34264097408 — SUCCESS
```

The final exact-head L-1 gate passed Python syntax, browser JavaScript syntax, all L-1 runtime/hardening/independent-review tests, inherited L-0 regression, scope/adjacent-owner guard and diff hygiene.

The separate closed L-0 design-only workflow still passes its actual L-0 contract validation and then intentionally fails its `Confirm design-only scope` guard on this runtime PR. That guard was not weakened.

---

# 3. Frozen exclusions remain unchanged

```text
LEARNING RECORD != PATIENT RECORD
LearningFactV1.authoritative_for_patient = false
NO patient encounter/lab reads or writes
NO raw transcript intake/storage
NO PR-1/PR-2 transcript semantics
NO Daily Case Review runtime/UI/table
NO Practice Review AI
NO Signal promotion/backlink authority
NO external bearer learning API
NO adaptive spaced-repetition algorithm
NO composite knowledge/excellence score
NO production config/secrets changes
NO RF mutation
NO physiotherapy/CU-1 mutation
```

No frozen schema/semantic owner changed during remediation. `REPLAN: NO`.

---

# 4. Physiotherapy / adjacent-owner state

The separate physiotherapy productization workstream remained read-only throughout L-1 review/remediation/release.

No L-1 diff touched:

```text
clinic_utilities/physio_referral_*
clinic_utilities/contracts/cu1_*
static/clinic-utilities/physio-referral/*
clinic_utilities/rf/*
static/clinic-utilities/rf/*
```

The prior pause tied specifically to the L-1 release decision is no longer a release blocker. Resuming another workstream still requires an explicit next-work decision; this closeout does not silently create a new writer.

---

# 5. Lifecycle

```text
L-0 CONTRACT                         FROZEN / COMPLETE / MERGED
L-1 IMPLEMENTED                      YES
L-1 TESTED                           YES
L-1 INDEPENDENT REVIEW               CHANGES REQUIRED → REMEDIATED
L-1 FOCUSED CLOSURE REVIEW           PASS
L-1 FINAL EXACT-HEAD GATE            PASS
L-1 PR #82                           MERGED
L-1 RUNTIME MERGE SHA                66900c2d50f98446184a92539baab89e9d49514c
L-1 VERIFIED LIVE DEPLOY SHA         1b7c0fa9443c4fe3376f80e6bc3819473de7e95b
L-1 DEPLOYED                         YES — dep-dag5ld0ae00c738g0370 LIVE
L-1 PRODUCTION-SMOKE-VERIFIED        NO — AUTHENTICATED SMOKE PENDING
ACTIVE WRITER                        NONE
```

---

# 6. Exact next action

Do not make another L-1 code change unless production smoke demonstrates a material defect.

Next release verification sequence:

```text
authenticated production smoke of /clinical/learning
→ verify page loads through existing clinical session / X-Clinical-Key protection
→ verify Foundation registry and Challenge/Due APIs respond normally
→ if PASS, mark L-1 PRODUCTION-SMOKE-VERIFIED and close the slice completely
```

No manual duplicate deploy, production configuration change, patient-data mutation, or adjacent-workstream mutation is authorized by this closeout.
