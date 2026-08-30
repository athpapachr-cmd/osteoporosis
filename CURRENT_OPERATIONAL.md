# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-1 WHY-NOW CORRECTION MERGED / DEPLOYED / PRODUCT-OWNER RE-SMOKE PENDING.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main` after correction release:** `d9423f4dcf6bebd056e83407132c6ce3e25d2280`.
> **Correction PR:** `#66` — SQUASH-MERGED.
> **Correction merge SHA:** `d9423f4dcf6bebd056e83407132c6ce3e25d2280`.
> **Render deploy:** `dep-daa93ljncjis739ssef0` — LIVE at exact correction merge SHA.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE — release closeout complete.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.

---

# 1. Production-smoke finding

The product owner performed the authenticated production smoke and reported that the smoke worked except for one concrete G-1 presentation problem:

```text
existing `Τύπος σημερινής επίσκεψης` completed
→ G-1 flow available
→ explicit WHY NOW not discoverable
```

Code inspection showed that the deterministic `item.why_now` already existed and the explicit `Γιατί τώρα:` label already appeared inside destination cards, but the top `Σημερινή ροή` summary displayed the reason only as unlabeled secondary text.

Classification:

```text
core guidance calculation                 functional / already tested
visit-type input                          functional in production smoke
WHY-NOW summary discoverability           production defect
clinical guidance rules                   unchanged
```

---

# 2. Bounded correction released

PR #66 changed only the summary presentation so every surfaced `Σημερινή ροή` item now renders:

```text
Γιατί τώρα: <existing deterministic item.why_now>
```

No guidance reason, priority, treatment rule, clinical recommendation, taxonomy, persistence, database/schema or KPI semantics changed.

A focused regression was added to require the explicit summary `Γιατί τώρα:` label while preserving the existing in-card label.

---

# 3. Exact test / release evidence

Final PR head:

```text
e2960454cfa1acf6fa4e2c0735a2e7ba3c267f48
```

Both exact-head G-1 runs completed successfully:

```text
33333512964  SUCCESS
33333526378  SUCCESS
```

The workflow includes:

- JavaScript syntax checks;
- progressive-guidance core regressions;
- wiring/ownership regression;
- R1/R2 UI-state regressions;
- explicit WHY-NOW presentation regression;
- inherited authoritative Finish browser regression;
- inherited server finalization lifecycle regression.

Exact PR file scope:

```text
static/baseline-audit/progressive-guidance-ui.js      1 runtime line changed
test_progressive_guidance_why_now_ui.js               focused regression
.github/workflows/g1-progressive-guidance-tests.yml   executes regression
CURRENT_OPERATIONAL.md                                 operational state only
```

PR #66 was squash-merged to:

```text
d9423f4dcf6bebd056e83407132c6ce3e25d2280
```

Fresh verification confirmed that exact SHA as `main` immediately after merge.

Render auto-deploy occurred normally without manual duplication:

```text
deploy:  dep-daa93ljncjis739ssef0
commit:  d9423f4dcf6bebd056e83407132c6ce3e25d2280
trigger: new_commit
status:  LIVE
```

---

# 4. Production-smoke boundary

The original authenticated smoke was reported by the product owner as working apart from WHY-NOW discoverability.

Therefore the only correction-specific re-smoke still required is:

```text
select/use the existing `Τύπος σημερινής επίσκεψης`
→ inspect the top `Σημερινή ροή`
→ confirm surfaced items literally display `Γιατί τώρα: ...`
```

Until that direct production confirmation is received:

```text
C1/G-1 MERGED                         YES
C1/G-1 DEPLOYED                       YES
WHY-NOW DEFECT FIX MERGED             YES
WHY-NOW DEFECT FIX DEPLOYED           YES
WHY-NOW PRODUCTION RE-SMOKE           PENDING
PRODUCTION-SMOKE-VERIFIED             NO
```

Do not repeat already successful smoke steps unless the product owner reports another defect.

---

# 5. Current status matrix

```text
C1 IMPLEMENTED / TESTED / MERGED / DEPLOYED        YES
G-1 IMPLEMENTED / TESTED / MERGED / DEPLOYED       YES
G1-R1 / G1-R2                                      CLOSED / TESTED / DEPLOYED
WHY-NOW SUMMARY DISCOVERABILITY FIX                 MERGED / DEPLOYED
WHY-NOW PRODUCTION RE-SMOKE                         PENDING
PRODUCTION-SMOKE-VERIFIED                           NO
PR-1 HEIDI                                          NOT IMPLEMENTED
PR-2 REVIEW/POPULATION                              NOT IMPLEMENTED
REAL 5-CASE SYSTEM-ASSISTED PILOT                   NOT STARTED
30-CASE SYSTEM-ASSISTED BASELINE                    NOT STARTED
MODULE 01 CLOSED                                    NO
```

---

# 6. Exact next action

STOP runtime mutation.

Product owner re-smokes only the corrected production path:

```text
`Τύπος σημερινής επίσκεψης`
→ `Σημερινή ροή`
→ visible literal `Γιατί τώρα: ...`
```

If PASS:

```text
record PRODUCTION-SMOKE-VERIFIED
→ append final correction/smoke evidence to changelog
→ release the G-1 production-readiness gate
→ fresh-bootstrap before the next authorized Module-01 slice
```

If FAIL, reopen only the exact observed presentation seam. PR-1/PR-2, new medication-specific milestone content, physiotherapy/RF mutation and real pilot collection are not authorized by this closeout alone.
