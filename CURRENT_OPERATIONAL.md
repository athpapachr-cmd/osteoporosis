# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-1 PRODUCTION SMOKE PARTIAL FAIL / BOUNDED WHY-NOW UI CORRECTION ACTIVE.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `21c188fad41743a8a3b82b0954e471d2c07bdcc8`.
> **Runtime release ancestry:** PR #64 → `a6ba9ef1719a18a48a1756bf08bbd157d448a63e`.
> **Current correction branch:** `fix/module01-g1-why-now-production-smoke-2026-08-30`.
> **ACTIVE CANONICAL WRITER/LOCK:** `fix/module01-g1-why-now-production-smoke-2026-08-30` — bounded smoke-defect closeout only.
> **ACTIVE RUNTIME WRITER/LOCK:** `fix/module01-g1-why-now-production-smoke-2026-08-30` — G-1 WHY-NOW presentation seam only.

---

# 1. Production smoke evidence from product owner

Authenticated production use confirmed that the released G-1 flow is present and the existing `Τύπος σημερινής επίσκεψης` control can be completed.

The product owner reported one concrete smoke defect:

```text
visit type selected
→ G-1 flow present
→ explicit WHY NOW not discoverable to the clinician
```

This means the production smoke is not a full PASS.

Classification:

```text
core guidance calculation          appears functional
visit-type input                   functional
WHY-NOW presentation/discoverability DEFECT
production-smoke-verified          NO
```

---

# 2. Fresh code inspection finding

The current core produces `why_now` text for surfaced cards.

The UI renderer currently:

- renders an explicit `Γιατί τώρα:` label only inside individual destination cards;
- renders the top `Σημερινή ροή` summary reasons as small secondary text without the `Γιατί τώρα:` label;
- therefore requires the clinician to infer that the grey summary text is the WHY-NOW explanation or navigate into later cards to see the explicit label.

The existing UI-state regression suite tests history availability and live-state precedence but does not assert visible WHY-NOW discoverability in the summary.

This is a bounded UX/runtime defect against the frozen G-1 contract; it does not require new guidance rules or clinical content.

---

# 3. Authorized bounded correction

The product owner instructed continuation of the smoke after reporting the defect.

Allowed mutation scope:

```text
static/baseline-audit/progressive-guidance-ui.js
static/baseline-audit/progressive-guidance.css       only if needed for legibility
test_progressive_guidance_ui_state.js                or one focused equivalent regression
.github/workflows/g1-progressive-guidance-tests.yml  only if needed to run the new regression
CURRENT_OPERATIONAL.md
osteoporosis-change-log.md                           only after validated completion/release
```

Required behavior:

```text
Σημερινή ροή
→ each surfaced summary item visibly says `Γιατί τώρα:`
→ reason text remains the deterministic existing `why_now`
→ no new clinical rule or treatment recommendation
→ existing in-card `Γιατί τώρα:` remains intact
```

---

# 4. Explicitly out of scope

```text
new archetypes
new medication-specific milestone rules
change to guidance priority/content semantics
PR-1 Heidi
PR-2 provisional population
physiotherapy/RF
KPI/audit methodology
real pilot data
```

---

# 5. Acceptance gate

Before release:

```text
1. add explicit summary WHY-NOW affordance
2. add focused regression proving visible `Γιατί τώρα:` summary labeling after a valid visit type creates cards
3. rerun full G-1 + inherited C1 regression workflow
4. exact scope review
5. if PASS, PR → squash merge → normal Render auto-deploy
6. product-owner re-smoke only the corrected WHY-NOW path plus already-completed smoke checks as needed
```

Do not mark `PRODUCTION-SMOKE-VERIFIED` until the product owner can directly see/use the explicit WHY-NOW path in production and the remaining agreed smoke behavior is confirmed.

---

# 6. Current status matrix

```text
C1 MERGED / DEPLOYED                       YES
G-1 MERGED / DEPLOYED                      YES
G-1 CORE WHY-NOW GENERATION                IMPLEMENTED / TESTED
G-1 WHY-NOW SUMMARY DISCOVERABILITY        DEFECT CONFIRMED
BOUNDED CORRECTION                         ACTIVE
PRODUCTION-SMOKE-VERIFIED                  NO
PR-1 / PR-2                                NOT IMPLEMENTED
REAL 5-CASE PILOT                          NOT STARTED
MODULE 01 CLOSED                           NO
```

---

# 7. Exact next action

Implement and test only the explicit WHY-NOW summary presentation correction. If the focused and full regression gates pass, release that bounded correction through the normal PR/auto-deploy path.