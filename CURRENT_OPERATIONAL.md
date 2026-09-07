# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1 — IMPLEMENTED / TESTED EXACT-HEAD / FINAL REVIEW ACTIVE
> **Updated:** 2026-09-07 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified implementation base:** `5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6`.
> **Production Render at activation:** `dep-dafg47mq1p3s73dv8k30` — LIVE at the same `main`.
> **Active branch:** `feat/clinical-learning-l1-challenge-foundation-mvp-2026-09-07`.
> **Current slice:** `CORE-LEARNING-HUB-L1-CHALLENGE-FOUNDATION-MVP-2026-09-07`.
> **Exact slice owner:** `SLICE_PLAN_CURRENT.md`.
> **Frozen contract owners:** `schemas/clinical_learning_core_v1.yaml`, `schemas/clinical_learning_l1_boundary_v1.yaml`, `schemas/osteoporosis_foundation_map_v1.yaml`.
> **ACTIVE RUNTIME WRITER/LOCK:** CLAIMED — bounded L-1 final exact-head code/product/security/privacy review and justified in-scope corrections only; no physio/CU-1/RF, frozen L-0 contract, patient-data, transcript, Signal/DailyCase/PracticeReview, production config or deployment authority.
> **PHYSIOTHERAPY/CU-1 STATUS:** PAUSED by product owner until this L-1 checkpoint is completed; L-1 must continue to avoid all physio/CU-1/RF owners.
> **L-1 implementation authority:** GRANTED by product owner.
> **PR/merge/deploy authority:** PR opening is authorized only after clean closeout; merge/deploy remain separate product-owner decisions.
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
L-1 CORE IMPLEMENTED                 YES
L-1 CLINICIAN-FACING UX HARDENING    IMPLEMENTED
L-1 EXACT RUNTIME/UX HEAD            b53a39f12363e08a7be952e1d7a25fbb5511bca8
L-1 CLEAN HANDOFF HEAD               a6064d42e8dddf3de5316f032df02a2c98f92aee
L-1 CLEAN-HANDOFF REGRESSION GATE    PASS — run 34157346466
L-1 PHYSIO/RF ISOLATION              PASS
L-1 FINAL EXACT-HEAD REVIEW          ACTIVE
L-1 PR                               NONE
L-1 MERGED                           NO
L-1 DEPLOYED                         NO
L-1 PRODUCTION-SMOKE-VERIFIED        NO
```

---

# 7. Exact next action / active review

```text
review the full branch vs fresh main, not only the latest commit
→ architecture/runtime/privacy/integrity/concurrency review
→ clinician-facing Challenge revision + Foundation assessment + Import + Due UX review
→ classify findings as safety/data-integrity, clinically meaningful, operational refinement or cosmetic
→ correct only bounded categories 1–3 when justified
→ if frozen L-0 semantics would need change: STOP / REPLAN
→ rerun complete Clinical Learning L1 regression gate after any runtime/UI correction
→ if clean: canonical closeout to IMPLEMENTED / TESTED / REVIEWED
→ verify exact final head + full gate + scope/diff hygiene
→ open bounded L-1 PR to main
→ RELEASE HOLD for separate product-owner merge authority
```

Verified pre-review evidence:

```text
main/base                 5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6
merge-base                5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6
clean handoff head        a6064d42e8dddf3de5316f032df02a2c98f92aee
ahead / behind            26 / 0
clean-head regression     34157346466 — SUCCESS
existing L-1 PR           NONE
physio product branch     separate / paused
physio-CU1-RF diff leak   NONE
```

Do not start PR-1, PR-2, Daily Case Review, Signal work, physiotherapy/CU-1 or RF changes during this closeout.
