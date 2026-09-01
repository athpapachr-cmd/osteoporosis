# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-3 MERGED / DEPLOYED / PRODUCTION SMOKE FAILED; HOTFIX IMPLEMENTED / TESTED / PR NEXT.
> **Updated:** 2026-09-01 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `ef17367c7b8959f51e05b80909226804951d1bc7`.
> **G-3 release PR:** `#70` — squash merged.
> **G-3 Render deploy:** `dep-dabj38s9v7es73fmq800` — `live`, exact commit `ef17367c...`.
> **G-3 production smoke:** FAILED — neither `Νέο` nor `Σύνοψη ασθενούς` visible.
> **Hotfix branch:** `fix/module01-g3-production-visibility-cache-2026-09-01`.
> **Exact tested hotfix runtime head:** `3287e511cc6e9023552442283c0cc4b9117aaa4f`.
> **Hotfix workflow:** `G3 guidance salience longitudinal summary` run `33556354517` — SUCCESS.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE after hotfix closeout.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.

---

# 1. Production state

```text
C1 / G-1 / G-2                  PRODUCTION-SMOKE-VERIFIED
G-3 IMPLEMENTED                 YES
G-3 TESTED                      YES
G-3 MERGED                      YES
G-3 DEPLOYED                    YES
G-3 PRODUCTION-SMOKE-VERIFIED   NO
G-3 PRODUCTION SMOKE            FAILED
```

C2 remains implemented/tested but must not release until the G-3 production defect is corrected and re-smoked.

---

# 2. Production failure and confirmed defects

The failed smoke exposed three delivery/presentation defects, not a G-2 clinical-rule defect:

1. baseline workspace static assets had no explicit release-coherency cache policy;
2. `Σύνοψη ασθενούς` was hidden entirely before an active protected patient existed, conflicting with the requested always-visible summary area;
3. browser `Νέο` logic compared only card IDs, so a new R02 evidence trigger on an already-visible VFA card was not treated as newly surfaced.

No evidence threshold or clinical treatment semantics were changed.

---

# 3. Hotfix implemented/tested

Implemented:

- explicit `no-store/no-cache` headers for `/static/baseline-audit/*` and the workspace root redirect;
- visible `Σύνοψη ασθενούς` placeholder when no protected patient is selected;
- pure material-trigger salience token core;
- compatibility visibility layer that marks a domain new when a new evidence/high-value reason token appears even if the card was already visible;
- exact regression for `VFA base card → R02 appears → VFA = Νέο`;
- served-bundle/cache coherency regression.

Exact tested runtime head:

```text
3287e511cc6e9023552442283c0cc4b9117aaa4f
```

Workflow `33556354517` passed the new hotfix regressions and the full inherited G3/G2/G1/C1 gate.

---

# 4. Release discipline

The hotfix delta is based directly on deployed G-3 `main` and contains only bounded delivery/presentation/test/canonical changes. C2, PR-1/PR-2, clinical evidence rules, treatment thresholds and DB schema are excluded.

A normal merge of a hotfix PR will trigger Render auto-deploy. Do not trigger a second manual deploy.

---

# 5. Exact next action / authority

```text
open bounded G-3 production-visibility hotfix PR
→ verify exact PR-head checks
→ STOP before merge
```

A separate explicit product-owner merge instruction is required before the hotfix may be squash-merged.

After a successful merge/auto-deploy, production smoke must specifically verify:

1. `Σύνοψη ασθενούς` is visible even before patient selection, then populates when a protected patient is opened;
2. VFA already present in the visit flow + a transition from height loss <4 cm to >=4 cm causes explicit `Νέο` salience;
3. existing G-2 `Γιατί τώρα`/provenance remains intact.

Only after that may G-3 be marked production-smoke-verified and C2 release revalidation resume.
