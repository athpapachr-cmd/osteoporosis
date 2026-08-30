# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-1 WHY-NOW SMOKE CORRECTION IMPLEMENTED / TESTED / RELEASE GATE.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main` at correction start:** `21c188fad41743a8a3b82b0954e471d2c07bdcc8`.
> **Runtime release ancestry:** PR #64 → `a6ba9ef1719a18a48a1756bf08bbd157d448a63e`.
> **Correction branch:** `fix/module01-g1-why-now-production-smoke-2026-08-30`.
> **Tested runtime/CI head before this closeout commit:** `b78f8856b3e101bef3e048f0bd4a3999e3f09f32`.
> **ACTIVE CANONICAL WRITER/LOCK:** correction branch — release closeout only.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE — bounded implementation complete.

---

# 1. Production smoke finding

Authenticated production smoke by the product owner confirmed that G-1 loaded and the existing `Τύπος σημερινής επίσκεψης` control was usable, but the explicit WHY-NOW explanation was not discoverable.

Root cause inspection showed:

```text
core VisitPlan item.why_now                 PRESENT
explicit `Γιατί τώρα:` inside destination cards PRESENT
summary `Σημερινή ροή` reason text          PRESENT but unlabeled/small
clinician-visible WHY-NOW discoverability   INSUFFICIENT
```

This was a presentation defect, not a guidance-rule failure.

---

# 2. Bounded correction

Changed only the summary presentation so every surfaced item now renders:

```text
Γιατί τώρα: <existing deterministic item.why_now>
```

No guidance reason, priority, clinical rule, treatment logic, taxonomy, storage or schema changed.

A focused regression was added to prevent loss of the explicit summary label while preserving the existing in-card label.

---

# 3. Exact test evidence

Tested head:

```text
b78f8856b3e101bef3e048f0bd4a3999e3f09f32
```

GitHub Actions:

```text
workflow: G1 progressive guidance foundation
run:      33333473030
job:      g1-guidance
result:   SUCCESS
```

The workflow passed:

- JavaScript syntax checks;
- progressive guidance core regressions;
- wiring/ownership regression;
- R1/R2 UI-state regressions;
- new explicit WHY-NOW presentation regression;
- inherited authoritative Finish browser regression;
- inherited server finalization lifecycle regression.

---

# 4. Exact scope review

Compare from `main` `21c188fad41743a8a3b82b0954e471d2c07bdcc8` to tested head changed only:

```text
static/baseline-audit/progressive-guidance-ui.js      1 line changed
test_progressive_guidance_why_now_ui.js               new focused regression
.github/workflows/g1-progressive-guidance-tests.yml   run new regression
CURRENT_OPERATIONAL.md                                 operational state only
```

No clinical-content or unrelated parked scope is included.

---

# 5. Release authority / next action

The product owner instructed continuation of the production smoke and supplied the defect report. The bounded smoke-fix acceptance path already authorizes release after PASS.

Exact next action:

```text
fresh main verification
→ PR
→ exact PR-head CI
→ exact scope/mergeability review
→ squash merge
→ normal Render auto-deploy
→ product-owner production re-smoke of `Σημερινή ροή` / `Γιατί τώρα:`
```

Do not mark full `PRODUCTION-SMOKE-VERIFIED` until the corrected WHY-NOW path is directly confirmed in production.

---

# 6. Status matrix

```text
C1 MERGED / DEPLOYED                       YES
G-1 MERGED / DEPLOYED                      YES
WHY-NOW CORE GENERATION                    YES
WHY-NOW SUMMARY CORRECTION IMPLEMENTED     YES
WHY-NOW SUMMARY CORRECTION TESTED          YES
CORRECTION MERGED / DEPLOYED               NO
PRODUCTION-SMOKE-VERIFIED                  NO
PR-1 / PR-2                                NOT IMPLEMENTED
REAL 5-CASE PILOT                          NOT STARTED
MODULE 01 CLOSED                           NO
```
