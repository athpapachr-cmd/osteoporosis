# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-3 PRODUCTION-SMOKE-VERIFIED / G-4 RELEASE REVIEW PASS / PR NEXT.
> **Updated:** 2026-09-02 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `ab94c6286bdc49cb8304b072e557c5eb0a96b0c6`.
> **G-3 hotfix PR:** `#71` — squash merged.
> **G-3 Render deploy:** `dep-dabolap5efls739s9am0` — `live`, exact commit `ab94c628...`.
> **G-3 product-owner production re-smoke:** PASS.
> **G-4 branch:** `feat/module01-g4-collapsible-sticky-summary-rf-utility-2026-09-02`.
> **G-4 exact tested runtime head:** `942d4e06944ebd6de97891cb8e2739c88ba85a38`.
> **G-4 workflow:** `G3 guidance salience longitudinal summary` run `33599860151` — SUCCESS.
> **G-4 release-readiness review:** PASS after canonical reconciliation and exact-head verification.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.

---

# 1. Production state

```text
C1 / G-1 / G-2                    PRODUCTION-SMOKE-VERIFIED
G-3 IMPLEMENTED                   YES
G-3 TESTED                        YES
G-3 MERGED                        YES
G-3 DEPLOYED                      YES
G-3 PRODUCTION-SMOKE-VERIFIED     YES
G-3 PILOT-VALIDATED               NO
```

The G-3 hotfix deployment at exact `ab94c628...` was product-owner re-smoked successfully: `Νέο` and `Σύνοψη ασθενούς` are visible and working well. This remains production smoke, not pilot validation.

---

# 2. G-4 implemented boundary

G-4 is presentation/navigation only.

## Workspace ergonomics

- `Σύνοψη ασθενούς` has a native accessible `Σύμπτυξη / Ανάπτυξη` control;
- `Σημερινή ροή` has the same independent control;
- collapse state uses `sessionStorage` only as per-browser UI preference and is never clinical data;
- patient summary is sticky at the top of the encounter scroll context;
- no second summary renderer or guidance owner was introduced;
- dynamic G-3 re-renders are tolerated by a small decoration helper that reapplies controls when needed.

## Clinic Utilities navigation

A `Clinic Utilities` group is injected into the existing Cockpit sidebar with:

```text
Φυσιοθεραπεία
→ /clinical/clinic-utilities/physio-referral

Ραδιοκύματα — PDF
→ https://ortho-reception-backend-v2.onrender.com/rf
→ opens in a new tab with noopener/noreferrer
```

The RF PDF generator remains owned by the existing clinic reception backend. G-4 does not copy RF templates, coordinates, request persistence, authentication or patient history into the osteoporosis repository or encounter payload.

---

# 3. G-4 test/review evidence

Exact tested runtime head:

```text
942d4e06944ebd6de97891cb8e2739c88ba85a38
```

Workflow:

```text
G3 guidance salience longitudinal summary
run 33599860151
SUCCESS
```

Passed:

1. JavaScript syntax;
2. G-4 collapsible/sticky/RF utility integration regression;
3. original G-3 salience/summary regressions;
4. G-3 production visibility/cache regressions;
5. frozen G-2 evidence contract and G-2 runtime regressions;
6. G-1 core/wiring/UI/WHY-NOW regressions;
7. C1 authoritative Finish browser and server-finalization regressions.

Release-readiness review confirmed:

```text
runtime/code review                 PASS
clinical/data-integrity review      PASS
scope/leakage review                PASS
canonical reconciliation            PASS
post-tested-runtime drift            NONE
```

The RF external target is preserved as a production-smoke criterion because assistant-side independent HTTP probing of that separate service was not available during review.

---

# 4. State matrix

```text
G-4 DESIGN                         COMPLETE
G-4 IMPLEMENTED                    YES
G-4 TESTED                         YES
G-4 EXACT-HEAD REVIEW              PASS
G-4 RELEASE-READINESS REVIEW       PASS
G-4 PR                             NONE
G-4 MERGED                         NO
G-4 DEPLOYED                       NO
G-4 PRODUCTION-SMOKE-VERIFIED      NO
ACTIVE WRITER                      NONE
```

C2 remains separately implemented/tested and unreleased. Its revalidation/release sequencing is not part of G-4.

---

# 5. Exact next action / STOP gate

```text
open bounded G-4 release PR after product-owner release progression authority
→ verify exact PR-head checks
→ STOP before merge/deploy without separate explicit merge authority
```

No manual Render deploy.
