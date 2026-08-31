# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — C1 + G-1 PRODUCTION-SMOKE-VERIFIED / G-1 PRODUCTION-READINESS GATE CLOSED.
> **Updated:** 2026-08-31 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main` before closeout:** `de32e7625e87ace8038223934ee88d64c9cdd2e5`.
> **Original C1 + G-1 release PR:** `#64`.
> **WHY-NOW correction PR:** `#66`.
> **WHY-NOW correction runtime merge SHA:** `d9423f4dcf6bebd056e83407132c6ce3e25d2280`.
> **Correction Render deploy:** `dep-daa93ljncjis739ssef0` — LIVE at exact runtime correction SHA.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE — smoke closeout complete.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.

---

# 1. Production smoke — CLOSED

The product owner completed the authenticated production smoke across the released C1/G-1 ancestry.

Initial production smoke showed that the workflow worked but the deterministic WHY-NOW reason was not sufficiently discoverable in the top `Σημερινή ροή` summary.

That bounded presentation defect was corrected by PR #66 so each surfaced summary item renders:

```text
Γιατί τώρα: <existing deterministic item.why_now>
```

No guidance reason, priority, treatment rule, clinical recommendation, taxonomy, persistence, database/schema or KPI semantics changed in that correction.

---

# 2. Exact release evidence retained

WHY-NOW correction final PR head:

```text
e2960454cfa1acf6fa4e2c0735a2e7ba3c267f48
```

Exact-head GitHub Actions:

```text
33333512964  SUCCESS
33333526378  SUCCESS
```

PR #66 squash merge:

```text
d9423f4dcf6bebd056e83407132c6ce3e25d2280
```

Render auto-deploy:

```text
deploy:  dep-daa93ljncjis739ssef0
commit:  d9423f4dcf6bebd056e83407132c6ce3e25d2280
trigger: new_commit
status:  LIVE
```

The focused workflow preserved:

- JavaScript syntax checks;
- progressive-guidance core regressions;
- wiring/ownership regression;
- G1-R1/G1-R2 UI-state regressions;
- explicit WHY-NOW presentation regression;
- inherited authoritative Finish browser regression;
- inherited server finalization lifecycle regression.

---

# 3. Product-owner production re-smoke evidence

On 2026-08-31 Asia/Nicosia, the product owner directly confirmed in production that:

```text
existing `Τύπος σημερινής επίσκεψης`
→ top `Σημερινή ροή` is present
→ literal `Γιατί τώρα:` is visible
→ the surfaced guidance changes dynamically with the selected/current visit context
→ the resulting content is experienced as informative / guiding
```

Interpretation:

```text
WHY-NOW discoverability                    PASS
G-1 dynamic interaction in production      PASS
G-1 clinician-facing guidance usefulness   positive product-owner observation
```

The usefulness observation is **not** equivalent to real-clinic pilot validation and must not be represented as such.

---

# 4. Status matrix

```text
C1 IMPLEMENTED                              YES
C1 TESTED                                   YES
C1 MERGED                                   YES
C1 DEPLOYED                                 YES
G-1 IMPLEMENTED                             YES
G-1 TESTED                                  YES
G-1 MERGED                                  YES
G-1 DEPLOYED                                YES
G1-R1 / G1-R2                               CLOSED / TESTED / DEPLOYED
WHY-NOW SUMMARY DISCOVERABILITY FIX         MERGED / DEPLOYED
WHY-NOW PRODUCTION RE-SMOKE                 PASS
PRODUCTION-SMOKE-VERIFIED                   YES
PILOT-VALIDATED                             NO
PR-1 HEIDI                                  NOT IMPLEMENTED
PR-2 REVIEW/POPULATION                      NOT IMPLEMENTED
REAL 5-CASE SYSTEM-ASSISTED PILOT           NOT STARTED
30-CASE SYSTEM-ASSISTED BASELINE            NOT STARTED
MODULE 01 CLOSED                            NO
```

---

# 5. Closed gate / next action

The G-1 production-readiness gate is closed.

Do not reopen G-1 runtime or taxonomy merely because later product expansion is planned. Future G-1 refinement should be driven by evidence-backed guidance content or real-use evidence and handled as a separately authorized slice.

No new runtime mutation is authorized by this smoke closeout alone.

Before the next substantial Module-01 task:

```text
fresh six-canonical bootstrap
→ confirm next product-owner-authorized slice
→ claim writer/lock only for that bounded scope
```

Broad remaining order remains:

```text
evidence-backed minimum osteoporosis guidance content
→ PR-1 transcript extraction
→ PR-2 inline provisional population
→ 5-case real system-assisted pilot
→ one deliberate refinement
→ later scored baseline / Practice Review / improvement loop
```

Parked physiotherapy/RF work remains outside this closeout unless separately authorized.
