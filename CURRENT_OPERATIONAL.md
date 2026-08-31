# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-2 EVIDENCE-BACKED GUIDANCE RUNTIME IMPLEMENTED / TESTED; RELEASE REVIEW REQUIRED.
> **Updated:** 2026-08-31 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main` before implementation closeout:** `5182d250e244b2ed9e086138cb3b2edcdb967e25`.
> **Design-complete ancestry:** `design/module01-g2-evidence-backed-guidance-2026-08-31 @ 0395e52ed75f835d49713504df3df4ce51183edf`.
> **Implementation branch:** `feat/module01-g2-evidence-guidance-runtime-2026-08-31`.
> **Exact tested runtime head:** `e0657ba5924db87b38a0e05514613fbadf45bcd9`.
> **Runtime CI:** `G2 evidence guidance runtime` run `33403182604` — SUCCESS.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE after G-2 implementation/test closeout.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE after G-2 implementation/test closeout.

---

# 1. Production base retained

C1 authoritative Finish and G-1 progressive guidance remain:

```text
IMPLEMENTED
TESTED
MERGED
DEPLOYED
PRODUCTION-SMOKE-VERIFIED
```

G-2 has not changed production yet.

---

# 2. G-2 current state

Slice:

```text
M01-G2-EVIDENCE-GUIDANCE-RUNTIME-v1
```

State:

```text
DESIGN-COMPLETE = YES
IMPLEMENTED = YES
TESTED = YES
PRODUCT-OWNER RELEASE REVIEW = NO
PR = NONE
MERGED = NO
DEPLOYED = NO
PRODUCTION-SMOKE-VERIFIED = NO
PILOT-VALIDATED = NO
```

Do not collapse these states.

---

# 3. Implemented runtime boundary

The tested runtime preserves the generic G-1 architecture:

```text
G-1 longitudinal projection
+ live current encounter snapshot
→ G-2 osteoporosis evidence context
→ pure deterministic evidence evaluator
→ evidence contributions
→ deterministic merge with G-1 Visit Plan
→ existing Σημερινή ροή / Γιατί τώρα UI
```

Primary new runtime:

```text
static/baseline-audit/osteoporosis-evidence-guidance-core.js
```

`progressive-guidance-ui.js` remains the single guidance render/order owner.

---

# 4. Frozen clinical/runtime safeguards preserved

```text
GUIDANCE != AUTOMATIC TREATMENT DECISION
CHECKLIST GUIDANCE != SAFETY CLEARANCE
MISSING / UNKNOWN != NEGATIVE
SCHEDULED / PLANNED DOSE != ACTUAL DOSE
ADMINISTRATION COUNT != ELAPSED EXPOSURE
LIVE CURRENT CONTROL, INCLUDING BLANK > PERSISTED BROWSER CACHE
```

Tested behavior includes:

- R01 only with explicit formal-risk indication and NOGG scope/framework guard;
- VFA from supported structured triggers such as ≥4 cm height loss;
- new-fragility-fracture and fracture-on-treatment event overrides;
- no automatic failure/switch after fracture on treatment;
- denosumab exact 6-month evidence due from reliable actual administration date only;
- scheduled-only denosumab does not count as actual;
- denosumab >7-month NOGG escalation only after ≥2 reliable actual doses;
- conflicting denosumab history suppresses exact milestone derivation;
- denosumab exit guidance never writes a selected agent;
- oral-BP 12–16-week / ≥5-year and zoledronate ≥3-year milestones require exact reliable exposure;
- medication safety rules render as checklists requiring clinical confirmation, not clearance;
- concise NOGG/EMA provenance is visible with evidence-backed guidance.

---

# 5. Explicitly blocked / forbidden

The following remain inactive:

```text
OST_G2_R15_DENOSUMAB_EXIT_CTX_FOLLOWUP
OST_G2_R16_DENOSUMAB_EXIT_NO_CTX_OPTION
```

The runtime still has no:

```text
automatic CTX 280/300 second-zoledronate command
mandatory CTX-at-3-month retreatment command
generic Prolia 4th/8th/10th milestone
automatic treatment failure/switch label
automatic selected-agent mutation
automatic romosozumab cardiology/vascular referral without approved clinic policy
```

---

# 6. Test / review evidence

Exact tested runtime head:

```text
e0657ba5924db87b38a0e05514613fbadf45bcd9
```

CI:

```text
G2 evidence guidance runtime
run 33403182604
COMPLETED / SUCCESS
```

All job steps passed:

- JS syntax;
- frozen G-2 contract validation;
- G-2 core regressions;
- G-2 live-state regressions;
- G-2 wiring/ownership regressions;
- inherited G-1 core/wiring/UI-state/WHY-NOW regressions;
- authoritative Finish browser regression;
- server finalization lifecycle regression.

Exact-head compare/review:

```text
main 5182d250… → e0657ba5…
status ahead
behind 0
merge base exactly main
```

Design head `0395e52e…` → tested head contains only expected G-2 runtime/test/canonical changes. No PR-1/PR-2 or parked utility leakage.

---

# 7. Status matrix

```text
G-2 EVIDENCE REGISTRY                    DESIGN-COMPLETE
G-2 RULE REGISTRY                        DESIGN-COMPLETE
G-2 VISIT PROFILES                       DESIGN-COMPLETE
G-2 THERAPY MILESTONES                   DESIGN-COMPLETE
G-2 MACHINE CONTRACT CI                  PASS
G-2 HUMAN DESIGN REVIEW                  COMPLETE
G-2 RUNTIME IMPLEMENTED                  YES
G-2 RUNTIME TESTED                       YES
G-2 RELEASE REVIEW                       NOT YET PRODUCT-OWNER AUTHORIZED
G-2 PR                                   NONE
G-2 MERGED                               NO
G-2 DEPLOYED                             NO
G-2 PRODUCTION-SMOKE-VERIFIED            NO
PR-1 HEIDI                               NOT IMPLEMENTED
PR-2 INLINE REVIEW/POPULATION             NOT IMPLEMENTED
REAL 5-CASE SYSTEM-ASSISTED PILOT         NOT STARTED
MODULE 01 CLOSED                         NO
```

---

# 8. Exact next action

The bounded runtime implementation authority is exhausted and the writer lock is released.

The next action, **only after a new explicit product-owner instruction to release/proceed with release**, is:

```text
fresh verify remote main
→ fresh six-canonical bootstrap
→ exact-head release-readiness review of the tested G-2 ancestry
→ confirm main compatibility / no new blockers
→ if explicitly authorized, open reviewed release PR
→ STOP again before merge unless merge authority is separately explicit
```

Do not infer release authority from implementation completion.

No PR, merge, deploy or production smoke is currently authorized.

Parked physiotherapy/RF work and PR-1/PR-2 remain outside the closed G-2 implementation slice.
