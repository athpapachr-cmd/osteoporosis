# SLICE_PLAN_CURRENT.md — Clinical Learning Hub L-1 Challenge + Foundation MVP

> **STATUS:** MERGED / POST-MERGE DEPLOY VERIFICATION PENDING
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CORE-LEARNING-HUB-L1-CHALLENGE-FOUNDATION-MVP-2026-09-07`.
> **Implementation branch:** `feat/clinical-learning-l1-challenge-foundation-mvp-2026-09-07`.
> **Release PR:** `#82` — MERGED.
> **Reviewed source head:** `e73dc7862ecd1084de8e2912f45ae758110a7447`.
> **Squash merge SHA:** `66900c2d50f98446184a92539baab89e9d49514c`.
> **Final exact-head L-1 gate:** run `34264097408` — SUCCESS.
> **Independent pre-merge review:** `5144979821` — findings remediated.
> **Focused closure review:** `5145618117` — PASS.
> **Merge/deploy authority:** merge exercised; deployment limited to normal Render auto-deploy.
> **Patient-data mutation authority:** NONE.
> **Production config/secret authority:** NONE.

---

# 1. Objective — complete

The L-1 slice implemented the first protected Clinical Learning runtime against frozen L-0 contracts:

```text
Challenge JSON import / preview / review / save
+ immutable Challenge history / revision / delete
+ reusable reference-verification overlay
+ Osteoporosis Foundation Map + explicit assessments
+ deterministic learning due state
+ clinician-facing MVP UI
```

This remains learning infrastructure, not patient documentation.

---

# 2. Frozen invariants preserved

```text
LEARNING RECORD != PATIENT RECORD
EVERY LearningFactV1.authoritative_for_patient = false
RAW TRANSCRIPT != L-1 STORAGE
IMPORTED JSON != CLINICIAN-REVIEW AUTHORITY
REFERENCE VERIFICATION != IMMUTABLE CHALLENGE PAYLOAD
FOUNDATION STATE != SELF-RATING
DUE STATE != COMPETENCE EVIDENCE
```

No composite Clinical Knowledge / Clinical Excellence score was introduced.

No patient encounter/lab read/write path, Signal promotion, Daily Case Review, Practice Review AI, transcript intake, external bearer learning API, adaptive spaced-repetition algorithm, RF mutation or physiotherapy/CU-1 mutation was added.

---

# 3. Delivered runtime owners

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
L-1-focused tests/workflow
```

Frozen schema owners remained read-only:

```text
schemas/clinical_learning_core_v1.yaml
schemas/clinical_learning_l1_boundary_v1.yaml
schemas/clinical_learning_contract_manifest_v1.yaml
schemas/clinical_learning_design_fixtures_v1.yaml
schemas/osteoporosis_foundation_map_v1.yaml
```

`REPLAN: NO`.

---

# 4. Independent-review remediation — complete

The second independent reviewer found five material pre-merge issues. All were corrected within L-1 owners:

1. frozen nested `unique_items` enforcement;
2. missing frozen Challenge History date/review-state filters and Foundation/due visibility;
3. stale browser Challenge preview could otherwise save an older previewed artifact;
4. exact-duplicate preview classification incorrectly mixed imported authority reset with persisted reviewed authority;
5. Foundation browser retry regenerated attempt evidence identity/timestamp.

Focused regression file:

```text
test_clinical_learning_l1_independent_review.py
```

Remediation evidence:

```text
substantive remediation head  b4b0438e44b8077ce5a9f469b5e4669d083886f0
remediation gate             34263571488 — SUCCESS
focused closure review       5145618117 — PASS
```

Final reviewed source evidence:

```text
exact source head            e73dc7862ecd1084de8e2912f45ae758110a7447
exact source gate            34264097408 — SUCCESS
```

The final gate passed:

```text
Python syntax
Browser JavaScript syntax
L-1 runtime/hardening/independent-review regressions
Inherited L-0 contract regression
Scope / adjacent-owner guard
Diff hygiene
```

The separate L-0 design-only workflow still intentionally fails only its runtime-scope guard after passing L-0 contract validation.

---

# 5. Merge — complete

PR #82 was marked ready only after exact-head verification and then squash-merged with an `expected_head_sha` guard against:

```text
e73dc7862ecd1084de8e2912f45ae758110a7447
```

GitHub returned:

```text
merged = true
merge SHA = 66900c2d50f98446184a92539baab89e9d49514c
```

Fresh verification confirmed:

```text
PR #82 state = closed / merged
main = 66900c2d50f98446184a92539baab89e9d49514c
```

---

# 6. Adjacent-owner isolation

The separate physiotherapy/CU-1/RF workstreams remained read-only for the whole L-1 release path.

No diff occurred under:

```text
clinic_utilities/physio_referral_*
clinic_utilities/contracts/cu1_*
static/clinic-utilities/physio-referral/*
clinic_utilities/rf/*
static/clinic-utilities/rf/*
```

The L-1 release decision is now complete, so that earlier pause is no longer itself a blocker. Starting or resuming another workstream still requires an explicit next-work selection.

---

# 7. Lifecycle state

```text
IMPLEMENTED                     YES
TESTED                          YES
INDEPENDENT REVIEW              CHANGES REQUIRED → REMEDIATED
FOCUSED CLOSURE REVIEW          PASS
FINAL EXACT-HEAD REVIEW         PASS
PR #82                          MERGED
MAIN MERGE SHA                  66900c2d50f98446184a92539baab89e9d49514c
DEPLOYED                        PENDING VERIFICATION
PRODUCTION-SMOKE-VERIFIED       NO
ACTIVE WRITER                   NONE
```

---

# 8. Exact next action

```text
verify normal Render auto-deploy from main
→ authenticated production smoke /clinical/learning
→ if PASS, reconcile lifecycle to DEPLOYED / PRODUCTION-SMOKE-VERIFIED
→ close L-1 completely
```

Do not manually trigger a duplicate deploy, change production configuration/secrets, or reopen frozen L-0 semantics unless production evidence demonstrates a material defect.
