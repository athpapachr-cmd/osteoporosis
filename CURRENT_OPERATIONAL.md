# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-3 MERGED / DEPLOYED / PRODUCTION SMOKE FAILED; VISIBILITY HOTFIX ACTIVE.
> **Updated:** 2026-09-01 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `ef17367c7b8959f51e05b80909226804951d1bc7`.
> **G-3 release PR:** `#70` — squash merged.
> **G-3 merge/runtime SHA:** `ef17367c7b8959f51e05b80909226804951d1bc7`.
> **G-3 Render deploy:** `dep-dabj38s9v7es73fmq800` — `live`, exact commit `ef17367c...`.
> **Production smoke result:** FAIL — product owner reports neither the `Νέο` salience nor `Σύνοψη ασθενούς` is visible.
> **Hotfix branch:** `fix/module01-g3-production-visibility-cache-2026-09-01`.
> **ACTIVE CANONICAL WRITER/LOCK:** this session — G-3 production-smoke visibility correction only.
> **ACTIVE RUNTIME WRITER/LOCK:** this session — baseline static cache/bootstrap visibility + focused regressions only.

---

# 1. Proven state before defect

C1 / G-1 / G-2 remain implemented, tested, merged, deployed and production-smoke-verified.

G-3 state is deliberately separated:

```text
G-3 IMPLEMENTED = YES
G-3 TESTED = YES
G-3 MERGED = YES
G-3 DEPLOYED = YES
G-3 PRODUCTION-SMOKE-VERIFIED = NO
G-3 PRODUCTION SMOKE = FAILED
```

Do not proceed to C2 release while the G-3 production acceptance criteria are not visible.

---

# 2. Failed acceptance criteria

Product-owner production observation on exact deployed G-3 ancestry:

1. newly applicable guidance does not visibly show the explicit `Νέο` salience;
2. `Σύνοψη ασθενούς` is not visible.

Repository inspection confirms the merged/deployed source contains both G-3 implementations and the expected bootstrap ordering. Therefore this is treated as a production integration/delivery defect rather than a missing merge.

---

# 3. Current root-cause hypothesis / bounded correction

The baseline workspace currently uses Starlette `StaticFiles` with stable unversioned JS/CSS paths. The G-3 source-presence tests did not prove deployed browser bundle freshness or end-to-end authenticated visibility.

Bounded hotfix scope:

```text
baseline-audit static responses → explicit no-store/no-cache policy
+ focused server regression proving cache headers on HTML/JS
+ preserve G3 bootstrap/load-order regression
+ rerun full inherited G3/G2/G1/C1 gates
```

No G-2 rule/evidence semantics, patient data model, C2 workflow, transcript PR-1/PR-2, DB migration or treatment logic is in scope.

---

# 4. Safety / release rules

```text
PRODUCTION SMOKE FAILURE != PRODUCTION-SMOKE-VERIFIED
HOTFIX != C2 RELEASE
NO MANUAL RENDER DEPLOY AFTER NORMAL MERGE
NO CLINICAL RULE CHANGE
NO REAL PATIENT DATA IN REPOSITORY
```

The browser may require one hard refresh after the hotfix deployment to discard an already-resident pre-hotfix bundle; after that, the explicit no-store policy must prevent this deployment-coherence failure from recurring on baseline workspace assets.

---

# 5. Exact next action

```text
implement cache-coherency hotfix
→ focused regression + inherited full gate
→ exact-head review
→ open bounded hotfix PR only if clean
→ STOP before merge unless product-owner merge authority is explicit
```

C2 remains implemented/tested but release-blocked until G-3 production smoke passes.
