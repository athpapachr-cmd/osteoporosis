# SLICE_PLAN_CURRENT.md — G-3 Production Visibility Hotfix v1

> **STATUS:** IMPLEMENTED / TESTED — RELEASE PR ALLOWED; MERGE HOLD.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** `M01-G3-PRODUCTION-VISIBILITY-HOTFIX-v1`.
> **Production base:** `ef17367c7b8959f51e05b80909226804951d1bc7`.
> **Failed production deploy:** `dep-dabj38s9v7es73fmq800` — live at exact G-3 merge SHA.
> **Hotfix branch:** `fix/module01-g3-production-visibility-cache-2026-09-01`.
> **Exact tested runtime head:** `3287e511cc6e9023552442283c0cc4b9117aaa4f`.
> **Test workflow:** `G3 guidance salience longitudinal summary` run `33556354517` — SUCCESS.
> **Runtime writer:** NONE after hotfix implementation/test closeout.

---

# 1. Production-smoke failure

Product-owner smoke on the deployed G-3 ancestry found both new acceptance criteria absent from the visible UI:

```text
explicit `Νέο` salience = NOT VISIBLE
`Σύνοψη ασθενούς` = NOT VISIBLE
```

Therefore G-3 remains merged/deployed but **not production-smoke-verified**.

---

# 2. Root causes found

## A. Deployment/cache coherency

The baseline workspace used stable unversioned JS/CSS paths under Starlette `StaticFiles` without an explicit no-store policy. A browser could therefore continue running a pre-G3 asset bundle after the server itself had deployed the new commit.

Hotfix:

```text
/static/baseline-audit/*
→ Cache-Control: no-store, no-cache, must-revalidate, max-age=0
→ Pragma: no-cache
→ Expires: 0
```

The root redirect is also non-cacheable.

## B. Summary discoverability contract

The G-3 renderer hid the whole patient-summary root when no protected patient was active. This contradicted the product-owner requirement that the summary area remain visible.

Hotfix behavior:

```text
no protected patient selected
→ `Σύνοψη ασθενούς` remains visible
→ explicit instruction to open a protected patient

protected patient selected
→ existing deterministic longitudinal summary remains authoritative/read-only
```

## C. `Νέο` semantics on an already-visible card

The original browser salience implementation compared only `card_id` presence. If VFA was already present as base-flow content and R02 evidence later activated on that same card, no new card ID appeared and the UI failed to mark it `Νέο`.

Corrected semantics track **material trigger tokens**, not only cards:

```text
E|<card_id>|<evidence_rule_id>
R|<card_id>|<high-value reason code>
```

Thus:

```text
VFA already visible as VISIT_TYPE_CORE
+ new OST_G2_R02_VFA_STRUCTURED_TRIGGER
→ new evidence token
→ VFA marked `Νέο`
```

Initial render still establishes a baseline and does not mark everything new. Base-flow-only changes remain non-salient noise. A marker clears when its material trigger ceases to apply.

---

# 3. Runtime boundary

Hotfix files:

```text
main.py
static/baseline-audit/app.js
static/baseline-audit/g3-salience-token-core.js
static/baseline-audit/g3-production-visibility-guard.js
test_g3_salience_token_core.js
test_g3_production_visibility_cache.py
.github/workflows/g3-guidance-summary-tests.yml
```

No G-2 evidence/rule/threshold, treatment decision, patient-data schema, DB migration, C2 persistence logic, PR-1/PR-2 or utility code is changed.

The compatibility visibility guard is a bounded production hotfix layer. It does not write patient data or own clinical rule evaluation. A later cleanup may consolidate presentation ownership after production verification, but no speculative refactor is required before correcting the failed smoke.

---

# 4. Test evidence

Exact tested runtime head:

```text
3287e511cc6e9023552442283c0cc4b9117aaa4f
```

Workflow:

```text
G3 guidance salience longitudinal summary
run 33556354517
SUCCESS
```

The complete gate passed:

1. JavaScript syntax;
2. original G-3 summary/salience regressions;
3. G-3 wiring/ownership regressions;
4. **new same-card/new-evidence salience regression**;
5. **production static cache/served-bundle visibility regression**;
6. frozen G-2 evidence contract;
7. G-2 core/live-state/wiring;
8. G-1 core/wiring/UI-state/WHY-NOW;
9. authoritative Finish browser regression;
10. server finalization lifecycle regression.

Exact production fixture added:

```text
prior: vfa card + VISIT_TYPE_CORE only
next:  same vfa card + OST_G2_R02_VFA_STRUCTURED_TRIGGER
expect: newly_surfaced_domains contains `vfa`
```

---

# 5. State / stop gate

```text
G-3 MERGED                         YES
G-3 DEPLOYED                       YES
G-3 ORIGINAL PRODUCTION SMOKE      FAILED
HOTFIX IMPLEMENTED                 YES
HOTFIX TESTED                      YES
HOTFIX EXACT-HEAD REVIEW           PASS
HOTFIX PR                          PENDING
HOTFIX MERGED                      NO
HOTFIX DEPLOYED                    NO
G-3 PRODUCTION-SMOKE-VERIFIED      NO
C2 RELEASE                         BLOCKED
```

Next allowed action:

```text
open bounded hotfix PR
→ verify exact PR-head checks
→ STOP before merge unless product owner explicitly authorizes merge
```
