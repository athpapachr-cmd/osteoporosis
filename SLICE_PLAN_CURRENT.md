# SLICE_PLAN_CURRENT.md — Clinical Learning Hub L-1 Challenge + Foundation MVP

> **STATUS:** IMPLEMENTED / TESTED / INDEPENDENT-REVIEW REMEDIATION CLOSED / FOCUSED REVIEW PASS — PR #82 OPEN / RELEASE HOLD
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CORE-LEARNING-HUB-L1-CHALLENGE-FOUNDATION-MVP-2026-09-07`.
> **Fresh implementation base:** `5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6`.
> **Implementation branch:** `feat/clinical-learning-l1-challenge-foundation-mvp-2026-09-07`.
> **Release PR:** `#82` — OPEN / DRAFT / RELEASE HOLD.
> **Frozen L-0 contract owner:** `schemas/clinical_learning_contract_manifest_v1.yaml`.
> **L-1 boundary owner:** `schemas/clinical_learning_l1_boundary_v1.yaml`.
> **Foundation registry:** `schemas/osteoporosis_foundation_map_v1.yaml`.
> **Product design:** `CLINICAL_LEARNING_HUB_DESIGN_V1.md`.
> **Remediation substantive head:** `b4b0438e44b8077ce5a9f469b5e4669d083886f0`.
> **Remediation full gate:** `34263571488` — SUCCESS.
> **Focused closure review:** PR review `5145618117` — PASS for the five independent-review findings.
> **Merge/deploy authority:** NONE; separate product-owner release decision required.
> **Patient-data mutation authority:** NONE.
> **Production config/secret authority:** NONE.

---

# 1. Objective

Implement the first protected Clinical Learning runtime without reopening frozen L-0 semantics:

```text
Challenge JSON import / preview / clinician review / save
+ immutable Challenge history / revision / delete
+ reusable reference-verification overlay
+ Osteoporosis Foundation Map + explicit clinician-reviewed assessments
+ deterministic learning due state
+ clinician-facing four-view MVP
```

This slice is learning infrastructure. It is not patient documentation.

---

# 2. Non-negotiable inherited invariants

```text
LEARNING RECORD != PATIENT RECORD
EVERY LearningFactV1.authoritative_for_patient = false
RAW TRANSCRIPT != L-1 STORAGE
IMPORTED JSON != CLINICIAN-REVIEW AUTHORITY
REFERENCE VERIFICATION != IMMUTABLE CHALLENGE PAYLOAD
FOUNDATION STATE != SELF-RATING
DUE STATE != COMPETENCE EVIDENCE
```

Additional boundaries:

- no direct patient identifier may be persisted in L-1 learning content;
- unknown imported fields are rejected recursively;
- privacy/schema failures expose bounded codes/paths only;
- accepted Challenge revisions are immutable;
- Challenge deletion purges content and leaves only the frozen non-content tombstone;
- Foundation state changes only through a valid explicit clinician-reviewed assessment;
- no adaptive spaced-repetition algorithm;
- no composite Clinical Knowledge / Clinical Excellence score;
- no autonomous Signal authority.

---

# 3. Frozen implementation owners

Allowed L-1 mutation owners:

```text
clinical_learning/__init__.py
clinical_learning/models.py
clinical_learning/contracts.py
clinical_learning/privacy.py
clinical_learning/persistence.py
clinical_learning/service.py
clinical_learning/api.py
static/clinical-learning/index.html
static/clinical-learning/styles.css
static/clinical-learning/app.js
main.py
test_clinical_learning_l1_*.py
.github/workflows/clinical-learning-l1-tests.yml
SLICE_PLAN_CURRENT.md
CURRENT_OPERATIONAL.md
TODO.md
osteoporosis-change-log.md
```

Frozen semantic/schema owners remain read-only:

```text
schemas/clinical_learning_core_v1.yaml
schemas/clinical_learning_l1_boundary_v1.yaml
schemas/clinical_learning_contract_manifest_v1.yaml
schemas/clinical_learning_design_fixtures_v1.yaml
schemas/osteoporosis_foundation_map_v1.yaml
```

No remediation finding required a frozen-contract change. **REPLAN: NO.**

---

# 4. Explicit adjacent-owner isolation

Physiotherapy/CU-1/RF remain outside this slice and read-only:

```text
clinic_utilities/physio_referral_*
clinic_utilities/contracts/cu1_*
static/clinic-utilities/physio-referral/*
clinic_utilities/rf/*
static/clinic-utilities/rf/*
```

The paused physiotherapy productization branch remains separate:

```text
design/physio-referral-product-ux-v1-knee-oa-2026-09-07
```

The L-1 CI scope guard enforces this isolation.

---

# 5. Delivered L-1 runtime

Protected base:

```text
/clinical/learning
```

Delivered capabilities:

- Challenge JSON schema/privacy/reference-integrity preview;
- recursive unknown-field rejection and deterministic PHI guard;
- server-authority normalization for review, Signal links and reference verification;
- Fact Ledger with provenance and progressive-disclosure visibility;
- clinician Accept / Modify / Dismiss observation workflow;
- immutable Challenge revisions and tombstone deletion semantics;
- mutable reference-verification overlay outside immutable Challenge payload/hash;
- 14-node Osteoporosis Foundation Map;
- explicit structured clinician-reviewed Foundation assessment/state;
- stale/equal-timestamp Foundation overwrite protection;
- explicit next-review and occurrence-based due-state materialization;
- JSON/Markdown selected-revision export;
- protected browser session / `X-Clinical-Key` access;
- no-store/no-cache Clinical Learning assets.

Four clinician-facing views:

1. Challenge Import
2. Challenge History
3. Foundation Map
4. Due / Learning Actions

---

# 6. Independent review — findings and closure

A fresh independent review of exact prior HOLD head
`777647a27fa34a3afdb7ad4d4195be942e720d`
returned **CHANGES REQUIRED**. User explicitly authorized bounded remediation.

## Finding 1 — frozen nested `unique_items`

Closed by explicit runtime validation and regression tests for:

- progressive disclosure `fact_ids`;
- observation linked Fact/reference IDs and gap classes;
- learning-action reference/Foundation IDs;
- Challenge Foundation IDs, gap classes and linked Signal snapshot IDs.

Topic tags retain their separately frozen trim/casefold/collapse semantics rather than being turned into a new ontology.

## Finding 2 — incomplete Challenge History UI contract

Closed:

- date filter added;
- review-state filter added;
- Foundation node links are visible in History rows;
- Challenge repetition due status/date is visible in History rows.

Backend date/review filtering already existed and is now wired through the UI.

## Finding 3 — stale Challenge preview could be saved

Closed:

- the browser stores a stable semantic fingerprint of the exact textarea artifact sent to server preview;
- Save reparses the current textarea;
- any mismatch invalidates the preview and requires a fresh `Validate & Preview`;
- syntax-invalid post-preview edits also fail closed.

The clinician can no longer unknowingly save a prior preview while looking at changed JSON.

## Finding 4 — false same-revision conflict after authority reset

Closed without changing immutable accepted content hashes:

- imported candidate is still normalized to untrusted/pending authority for preview;
- persisted accepted payload is projected through the same import-normalization boundary solely for duplicate classification;
- those two authority-neutral projections are compared for `exact_idempotent_duplicate` vs `same_revision_conflict`;
- create/revise immutable `content_hash` behavior remains unchanged.

## Finding 5 — Foundation browser retry instability

Closed:

- one open assessment owns one stable `attempt_id`;
- `assessed_at` is generated once when the assessment opens;
- each selected assessment method gets one stable `evidence_id` for that open assessment;
- preview/save/network retry therefore resends the same assessment identity/content unless the clinician actually edits content.

---

# 7. Remediation regression evidence

New focused coverage:

```text
test_clinical_learning_l1_independent_review.py
```

It proves:

```text
nested frozen unique-items rejection
reviewed Challenge reimport → exact_idempotent_duplicate
same-revision changed content → same_revision_conflict
History date/review filters + Foundation/due visibility
stale Challenge preview save guard
stable Foundation attempt/evidence/timestamp browser identity
```

Exact substantive remediation head:

```text
b4b0438e44b8077ce5a9f469b5e4669d083886f0
```

Exact full L-1 regression gate:

```text
run 34263571488
SUCCESS
```

Successful steps include:

```text
Python syntax
Browser JavaScript syntax
all L-1 runtime / hardening / independent-review regressions
Inherited L-0 contract regression
Scope and adjacent-owner guard
Diff hygiene
```

The separate closed L-0 design-only workflow on the same head again passed:

```text
Validate Clinical Learning L0 contracts — SUCCESS
```

and then failed only:

```text
Confirm design-only scope — EXPECTED FAILURE on an L-1 runtime PR
```

That guard is not weakened or changed.

Focused closure review `5145618117` found no residual material issue in the five-item remediation set.

---

# 8. Explicit exclusions preserved

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

# 9. Lifecycle

```text
L-0 CONTRACT                         FROZEN / COMPLETE / MERGED
L-1 PRODUCT-OWNER IMPLEMENTATION     AUTHORIZED / EXERCISED
L-1 CORE IMPLEMENTED                 YES
L-1 INITIAL FINAL REVIEW             SUPERSEDED BY INDEPENDENT REVIEW
L-1 INDEPENDENT REVIEW               CHANGES REQUIRED → REMEDIATED
L-1 REMEDIATION TESTED               YES
L-1 FOCUSED CLOSURE REVIEW           PASS
L-1 REPLAN REQUIRED                  NO
L-1 PHYSIO/RF ISOLATION              PASS
L-1 PR                               #82 OPEN / DRAFT / RELEASE HOLD
L-1 MERGED                           NO
L-1 DEPLOYED                         NO
L-1 PRODUCTION-SMOKE-VERIFIED        NO
```

---

# 10. Exact next action / RELEASE HOLD

The final canonical HOLD commit itself must pass the complete Clinical Learning L-1 regression gate.

```text
final canonical HOLD head
→ complete L-1 gate SUCCESS
→ STOP with writer lock NONE
→ remain PR #82 OPEN / DRAFT / RELEASE HOLD
→ await separate explicit product-owner squash-merge/release authority
```

If release is later explicitly authorized:

```text
fresh main/head/check verification
→ squash merge PR #82 using exact expected head
→ normal Render auto-deploy from main
→ no manual duplicate deploy
→ authenticated production smoke /clinical/learning
→ post-merge/deploy canonical reconciliation
```

No merge, deploy, production configuration/secret mutation, PR-1/PR-2/Daily Case/Signal work, or physiotherapy/CU-1/RF resumption is authorized by this remediation.