# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-2 RELEASE REVIEW PASS / PR #69 OPEN / MERGE HOLD.
> **Updated:** 2026-08-31 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main` for release review:** `5182d250e244b2ed9e086138cb3b2edcdb967e25`.
> **Design-complete ancestry:** `design/module01-g2-evidence-backed-guidance-2026-08-31 @ 0395e52ed75f835d49713504df3df4ce51183edf`.
> **Implementation branch:** `feat/module01-g2-evidence-guidance-runtime-2026-08-31`.
> **Exact tested runtime head:** `e0657ba5924db87b38a0e05514613fbadf45bcd9`.
> **Release-review / PR-open head before this docs-only operational closeout:** `4c9ecad3535d795bcf85b4687ce7db44187e68a2`.
> **Release PR:** `#69` — OPEN, non-draft, base `main`, explicit MERGE HOLD.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE after release-review/PR operational closeout.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.

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

G-2 has **not** been merged or deployed.

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
RELEASE-READINESS REVIEW = PASS
RELEASE PR = #69 OPEN
MERGE AUTHORITY = NO
MERGED = NO
DEPLOYED = NO
PRODUCTION-SMOKE-VERIFIED = NO
PILOT-VALIDATED = NO
```

Do not collapse these states.

The product owner explicitly authorized the fresh release-readiness review and opening of the release PR. That authorization did **not** authorize merge or deploy.

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

# 6. Implementation/test evidence

Exact tested runtime head:

```text
e0657ba5924db87b38a0e05514613fbadf45bcd9
```

Implementation workflow:

```text
G2 evidence guidance runtime
run 33403182604
COMPLETED / SUCCESS
```

That exact runtime head passed:

- JavaScript syntax;
- frozen G-2 contract validation;
- G-2 evidence-core regressions;
- G-2 live-state regressions;
- G-2 wiring/ownership regressions;
- inherited G-1 core/wiring/UI-state/WHY-NOW regressions;
- authoritative Finish browser regression;
- server finalization lifecycle regression.

No runtime file changed after `e0657ba5…` before release PR creation; the four subsequent closeout commits affected only:

```text
CURRENT_OPERATIONAL.md
SLICE_PLAN_CURRENT.md
TODO.md
osteoporosis-change-log.md
```

---

# 7. Fresh release-readiness review

Fresh remote main remained:

```text
5182d250e244b2ed9e086138cb3b2edcdb967e25
```

Release-review head before the present docs-only operational closeout:

```text
4c9ecad3535d795bcf85b4687ce7db44187e68a2
```

Compare `main → 4c9ecad3…`:

```text
status: ahead
ahead_by: 39
behind_by: 0
merge_base: exactly current main
changed_files: 20
```

The changed-file set contains only expected G-2 contracts, evidence/design review, runtime integration, tests/workflows and canonicals. No PR-1/PR-2, physiotherapy or RF leakage was found.

Release PR opened:

```text
PR #69
feat: release G-2 evidence-backed osteoporosis guidance runtime
base: main
head: feat/module01-g2-evidence-guidance-runtime-2026-08-31
state: OPEN
mergeable: YES at reviewed head
MERGE HOLD: YES
```

Exact PR-head checks at `4c9ecad3…` all passed:

```text
G2 evidence guidance contract       run 33405228779  SUCCESS
G2 evidence guidance runtime        run 33405228669  SUCCESS
G1 progressive guidance foundation  run 33405228587  SUCCESS
Baseline finalization integrity     run 33405228589  SUCCESS
```

This establishes a clean release-review gate. The present update is documentation-only and does not change runtime semantics.

---

# 8. Status matrix

```text
G-2 EVIDENCE REGISTRY                    DESIGN-COMPLETE
G-2 RULE REGISTRY                        DESIGN-COMPLETE
G-2 VISIT PROFILES                       DESIGN-COMPLETE
G-2 THERAPY MILESTONES                   DESIGN-COMPLETE
G-2 MACHINE CONTRACT CI                  PASS
G-2 HUMAN DESIGN REVIEW                  COMPLETE
G-2 RUNTIME IMPLEMENTED                  YES
G-2 RUNTIME TESTED                       YES
G-2 RELEASE REVIEW                       PASS
G-2 RELEASE PR                           #69 OPEN
G-2 MERGE AUTHORITY                      NO
G-2 MERGED                               NO
G-2 DEPLOYED                             NO
G-2 PRODUCTION-SMOKE-VERIFIED            NO
PR-1 HEIDI                               NOT IMPLEMENTED
PR-2 INLINE REVIEW/POPULATION             NOT IMPLEMENTED
REAL 5-CASE SYSTEM-ASSISTED PILOT         NOT STARTED
MODULE 01 CLOSED                         NO
```

---

# 9. Exact next action / STOP gate

The authorized release-review/PR-opening action is complete.

**STOP before merge.**

Only after a separate explicit product-owner merge instruction may a future session perform:

```text
fresh verify remote main
→ fresh six-canonical bootstrap
→ fetch exact current PR #69 head
→ confirm no branch/base drift
→ confirm exact-head required checks/review remain green
→ inspect any new review comments/threads
→ if still clean, squash-merge using expected exact head SHA
→ allow normal Render auto-deploy from main
→ verify deploy identity/state
→ perform/coordinate production smoke
→ canonical release closeout
```

No merge, deploy or production smoke is authorized now.

Parked physiotherapy/RF work and PR-1/PR-2 remain outside this release PR.
