# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1 — INDEPENDENT-REVIEW REMEDIATION CLOSED / FOCUSED REVIEW PASS / PR #82 RELEASE HOLD
> **Updated:** 2026-09-08 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified base / current main:** `5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6`.
> **Production Render baseline:** `dep-dafg47mq1p3s73dv8k30` — existing `main`; L-1 is not deployed.
> **Active branch:** `feat/clinical-learning-l1-challenge-foundation-mvp-2026-09-07`.
> **Current slice:** `CORE-LEARNING-HUB-L1-CHALLENGE-FOUNDATION-MVP-2026-09-07`.
> **Exact slice owner:** `SLICE_PLAN_CURRENT.md`.
> **Frozen contract owners:** `schemas/clinical_learning_core_v1.yaml`, `schemas/clinical_learning_l1_boundary_v1.yaml`, `schemas/clinical_learning_contract_manifest_v1.yaml`, `schemas/clinical_learning_design_fixtures_v1.yaml`, `schemas/osteoporosis_foundation_map_v1.yaml`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE — remediation is closed; final canonical HOLD head must satisfy the full L-1 gate.
> **ACTIVE DESIGN/CANONICAL WRITER:** NONE after this closeout commit.
> **PHYSIOTHERAPY/CU-1 STATUS:** PAUSED / separate / read-only through the L-1 release decision.
> **PR:** #82 OPEN / DRAFT / RELEASE HOLD.
> **Merge/deploy authority:** NONE until separate explicit product-owner release decision.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **Signal/DailyCase/PracticeReview authority:** NONE.
> **Production config/secret authority:** NONE.
> **RF authority:** NONE.

---

# 1. Production truth

L-0 remains frozen/complete/merged. Current `main` / merge-base remains:

```text
5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6
```

PR #82 is still unmerged and undeployed. Production remains on the existing main-line state.

---

# 2. L-1 delivered boundary

Protected route family:

```text
/clinical/learning
```

Delivered within frozen L-1 scope:

```text
Challenge JSON validation / privacy preview
Fact Ledger / progressive-disclosure provenance
clinician observation disposition
immutable Challenge revisions/history/delete+tombstone
reference-verification external overlay
14-node Osteoporosis Foundation Map
explicit clinician-reviewed Foundation assessment/state
occurrence-based due-state materialization
four-view clinician workspace
JSON/Markdown learning export
browser session + X-Clinical-Key protection
no-store/no-cache learning assets
```

Hard invariants remain:

```text
LEARNING RECORD != PATIENT RECORD
LearningFactV1.authoritative_for_patient = false
```

Frozen exclusions remain unchanged:

```text
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

---

# 3. Independent review superseded the prior release-ready conclusion

Fresh independent review of prior exact HOLD head:

```text
777647a27fa34a3afdb7ad4d4195be942e720d
```

returned **CHANGES REQUIRED** and therefore superseded the earlier final-pass conclusion.

User explicitly authorized bounded remediation on 2026-09-08. No frozen schema/semantic owner needed modification. **REPLAN: NO.**

Five findings were closed:

1. **Frozen nested `unique_items` enforcement**
   - explicit runtime rejection now covers disclosure Fact IDs, observation linked IDs/gap classes, action reference/Foundation IDs and Challenge Foundation/gap/Signal snapshot IDs;
   - focused regression coverage added.

2. **Challenge History frozen UI contract**
   - date and review-state filters are now wired;
   - linked Foundation nodes and due status/date are visible in History rows.

3. **Stale Challenge preview/save integrity**
   - browser fingerprints the exact artifact sent to server preview;
   - any textarea change before Save invalidates that preview and requires fresh server validation.

4. **Authority-neutral exact-duplicate preview**
   - same-revision comparison now projects the stored accepted payload through the same import-authority reset used for the candidate;
   - accepted immutable `content_hash` semantics were not changed.

5. **Foundation browser retry stability**
   - one open assessment retains stable `attempt_id`, `assessed_at`, and per-method `evidence_id` values across preview/save/retry.

New focused regression owner:

```text
test_clinical_learning_l1_independent_review.py
```

---

# 4. Exact remediation evidence

Substantive remediation head:

```text
b4b0438e44b8077ce5a9f469b5e4669d083886f0
```

Full L-1 regression gate:

```text
run 34263571488
SUCCESS
```

Passed on that exact head:

```text
Python syntax
Browser JavaScript syntax
all L-1 runtime / hardening / independent-review tests
Inherited L-0 contract regression
Scope and adjacent-owner guard
Diff hygiene
```

Focused closure review:

```text
PR review 5145618117
PASS for all five remediation findings
```

The separate closed L-0 design-only workflow on the same substantive head again behaved as expected:

```text
Validate Clinical Learning L0 contracts — SUCCESS
Confirm design-only scope — EXPECTED FAILURE because PR #82 contains L-1 runtime code
```

The L-0 design-only guard is not changed or weakened.

---

# 5. Adjacent-owner isolation

No remediation change touched:

```text
clinic_utilities/physio_referral_*
clinic_utilities/contracts/cu1_*
static/clinic-utilities/physio-referral/*
clinic_utilities/rf/*
static/clinic-utilities/rf/*
```

Paused physiotherapy branch remains:

```text
design/physio-referral-product-ux-v1-knee-oa-2026-09-07
```

---

# 6. Lifecycle

```text
L-0 CONTRACT                         FROZEN / COMPLETE / MERGED
L-1 IMPLEMENTATION AUTHORITY         GRANTED / EXERCISED
L-1 CORE IMPLEMENTED                 YES
L-1 INDEPENDENT REVIEW               CHANGES REQUIRED → REMEDIATED
L-1 REMEDIATION TESTED               YES
L-1 FOCUSED CLOSURE REVIEW           PASS
L-1 REPLAN REQUIRED                  NO
L-1 ACTIVE WRITER                    NONE
L-1 PHYSIO/RF ISOLATION              PASS
L-1 PR                               #82 OPEN / DRAFT / RELEASE HOLD
L-1 MERGED                           NO
L-1 DEPLOYED                         NO
L-1 PRODUCTION-SMOKE-VERIFIED        NO
```

---

# 7. Exact next action / RELEASE HOLD

This canonical closeout commit becomes the final candidate HOLD head. It must pass the complete L-1 regression gate exactly as committed.

```text
this exact final canonical HOLD head
→ full Clinical Learning L1 regression gate SUCCESS
→ STOP
→ keep writer lock NONE
→ keep PR #82 OPEN / DRAFT / RELEASE HOLD
→ await separate explicit product-owner squash-merge/release authority
```

If later explicitly authorized:

```text
verify fresh main + exact PR head + required checks
→ squash merge PR #82 using exact expected head
→ allow normal Render auto-deploy from main
→ no manual duplicate deploy
→ authenticated production smoke /clinical/learning
→ post-merge/deploy canonical reconciliation
```

Do **not** merge, deploy, mutate production configuration/secrets, start PR-1/PR-2/Daily Case Review/Signal work, or resume physiotherapy/CU-1/RF under the remediation authority that is now closed.