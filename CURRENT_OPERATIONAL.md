# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1 — PRODUCT-OWNER AUTHORIZED / IMPLEMENTATION ACTIVE
> **Updated:** 2026-09-07 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified implementation base:** `5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6`.
> **Production Render at activation:** `dep-dafg47mq1p3s73dv8k30` — LIVE at the same `main`.
> **Active branch:** `feat/clinical-learning-l1-challenge-foundation-mvp-2026-09-07`.
> **Current slice:** `CORE-LEARNING-HUB-L1-CHALLENGE-FOUNDATION-MVP-2026-09-07`.
> **Exact slice owner:** `SLICE_PLAN_CURRENT.md`.
> **Frozen contract owners:** `schemas/clinical_learning_core_v1.yaml`, `schemas/clinical_learning_l1_boundary_v1.yaml`, `schemas/osteoporosis_foundation_map_v1.yaml`.
> **ACTIVE RUNTIME WRITER/LOCK:** ChatGPT — L-1 `clinical_learning/`, `static/clinical-learning/`, bounded `main.py` composition, L-1 tests/workflow and canonical lifecycle updates only.
> **ACTIVE PHYSIOTHERAPY/CU-1 WRITER:** NONE for this session; product owner reports the other conversation is paused until this L-1 step is complete.
> **L-1 implementation authority:** GRANTED by product owner.
> **PR/merge/deploy authority:** NONE — separate release decision required.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **Signal/DailyCase/PracticeReview authority:** NONE.
> **Production config/secret authority:** NONE.
> **RF authority:** NONE.

---

# 1. Production truth at activation

L-0 is frozen/complete and merged. PR #80 carried the independently reviewed contracts; PR #81 reconciled post-merge canonicals. Current fresh `main` is:

```text
5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6
```

Render auto-deploy for that docs-only closeout:

```text
dep-dafg47mq1p3s73dv8k30
LIVE
```

No L-1 runtime exists in production at this point.

---

# 2. Active L-1 objective

Implement only:

```text
protected Challenge JSON preview/import
immutable Challenge history/revisions/delete
reference-verification overlay
Osteoporosis Foundation Map registry/state
explicit clinician-reviewed Foundation assessment
learning due-state materialization/read
protected Learning Hub MVP UI
```

Protected route family:

```text
/clinical/learning
```

Future runtime package/UI owners:

```text
clinical_learning/
static/clinical-learning/
```

---

# 3. Frozen exclusions

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
NO production config/secrets
NO RF mutation
NO physiotherapy/CU-1 mutation
```

The active physiotherapy productization branch observed read-only during bootstrap is:

```text
design/physio-referral-product-ux-v1-knee-oa-2026-09-07
```

L-1 must not modify its owned paths or attempt to merge/rebase that work.

---

# 4. L-1 implementation decisions frozen for this slice

- Pydantic/import models reject unknown fields recursively.
- Validation/privacy failures return bounded error code + field path only; no rejected input echo.
- Preview normalizes imported review/reference/Signal authority exactly as frozen by L-0.
- For MVP finalization, every imported observation is conservatively treated as material and must be `accepted`, `modified` or `dismissed`; no new materiality threshold is invented.
- Server owns canonical Challenge hash and revision sequence.
- Reference verification is stored only in the reusable external overlay.
- Challenge delete is one transaction: overlay + target/source due rows + revisions → non-content tombstone.
- Foundation registry is loaded from the frozen Module-01 map.
- Foundation state is changed only by a validated explicit clinician-reviewed assessment attempt.
- Optional explicit `next_review_due` may schedule Foundation reassessment; no adaptive cadence is inferred.
- Due occurrence semantics follow the frozen L-0 contract.
- Learning auth is local to the learning package and reuses existing `X-Clinical-Key` / browser-session behavior; it must not import from physiotherapy/CU-1.
- Existing baseline/CU navigation is not touched in L-1, to avoid overlap with the paused physiotherapy productization branch. Initial access is direct protected `/clinical/learning`.

---

# 5. Exact implementation paths

Allowed:

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

Frozen schemas are read-only normative inputs. Any material need to change them is a REPLAN trigger.

---

# 6. Current lifecycle

```text
L-0 CONTRACT                         FROZEN / COMPLETE / MERGED
L-1 PRODUCT-OWNER AUTHORITY          GRANTED
L-1 SLICE                            ACTIVE
L-1 IMPLEMENTED                      NO
L-1 TESTED                           NO
L-1 EXACT-HEAD REVIEW                NO
L-1 PR                               NONE
L-1 MERGED                           NO
L-1 DEPLOYED                         NO
L-1 PRODUCTION-SMOKE-VERIFIED        NO
```

---

# 7. Exact next action

```text
implement the bounded L-1 runtime + UI
→ synthetic/focused tests
→ inherited L-0 contract regression
→ scope/privacy/security/diff review
→ canonical closeout
→ HOLD for separate product-owner release decision
```

If a frozen contract cannot be implemented without changing semantic ownership, STOP and REPLAN rather than patching the contract during coding.
