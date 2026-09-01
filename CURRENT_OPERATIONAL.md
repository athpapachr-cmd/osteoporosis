# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-2 PRODUCTION-SMOKE-VERIFIED / G-3 IMPLEMENTED + TESTED / RELEASE HOLD.
> **Updated:** 2026-09-01 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `9cfad82d1258a44e71080e0aa4d6d644e581cfbf`.
> **G-2 release PR:** `#69` — squash merged.
> **G-2 merge/runtime SHA:** `9cfad82d1258a44e71080e0aa4d6d644e581cfbf`.
> **G-2 Render deploy:** `dep-daaph5vlk1mc73940g60` — `live`, trigger `new_commit`, exact commit `9cfad82d...`.
> **G-3 branch:** `feat/module01-g3-guidance-salience-longitudinal-summary-2026-09-01`.
> **G-3 exact tested runtime head:** `dab45baf9f80632ee6e58f03fa4d5005c68e0ac5`.
> **G-3 workflow:** `G3 guidance salience longitudinal summary` run `33486322905` — SUCCESS.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE after G-3 implementation/test closeout.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.

---

# 1. G-2 release state — CLOSED

The product owner completed the requested authenticated synthetic production smoke on the deployed G-2 ancestry and reported the smokes passed.

Exact state:

```text
G-2 DESIGN-COMPLETE = YES
G-2 IMPLEMENTED = YES
G-2 TESTED = YES
G-2 RELEASE REVIEW = PASS
G-2 MERGED = YES
G-2 DEPLOYED = YES
G-2 PRODUCTION-SMOKE-VERIFIED = YES
G-2 PILOT-VALIDATED = NO
```

Do not relabel product-owner production smoke as real-clinic pilot validation.

The G-2 clinical safeguards remain frozen, including inactive R15/R16 and no automatic CTX threshold retreatment command, ordinal Prolia milestones, automatic treatment failure/switch, selected-agent mutation, or automatic named-specialty referral.

---

# 2. G-3 trigger — evidence from production use

After G-2 production smoke, two clinician-facing UX needs were identified:

1. newly applicable guidance should become materially more visible at the moment it appears; example: VFA/vertebral-imaging guidance after live derived height loss reaches at least 4 cm;
2. a concise longitudinal patient summary should always be visible and describe authoritative patient course from first completed/amended encounter through the latest reliable state.

These were treated as clinically meaningful workflow refinements rather than cosmetic-only requests.

---

# 3. G-3 implemented boundary

Slice:

```text
M01-G3-GUIDANCE-SALIENCE-LONGITUDINAL-SUMMARY-v1
```

Implemented runtime:

```text
static/baseline-audit/osteoporosis-longitudinal-summary-core.js
static/baseline-audit/progressive-guidance-ui.js
static/baseline-audit/progressive-guidance.css
static/baseline-audit/app.js
```

## Newly surfaced guidance

The current Visit Plan is compared with the previous stable plan for the same active patient/case.

Rules:

```text
initial plan = baseline, not NEW
new evidence/event/unresolved/due/treatment-context item = eligible for NEW
base-flow-only VISIT_TYPE_CORE addition = not NEW noise
new item = explicit `Νέο` text badge + stronger visual emphasis
color alone is never the only signal
new state is ephemeral/in-memory only
item removed from plan = new state cleared
patient/case switch = baseline reset
```

The product-owner VFA example is covered explicitly:

```text
height loss 3.9 cm
→ no R02 evidence trigger
→ height loss 4.0 cm
→ OST_G2_R02_VFA_STRUCTURED_TRIGGER activates
→ VFA enters newly-surfaced set
→ top guidance item + destination card receive `Νέο` emphasis
```

## Longitudinal patient summary

A protected-patient `Σύνοψη ασθενούς` is rendered above `Σημερινή ροή`.

The summary is deterministic/read-only and derives from:

```text
completed/amended protected encounters
+ protected lab snapshots
+ existing G-1 LongitudinalGuidanceProjectionV1
+ current visit explicitly marked current/non-historical
```

Visible summary domains:

```text
Πορεία
Κατάγματα / κίνδυνος
DXA
Θεραπεία / actual administrations
Εργαστηριακά
Τελευταία explicit management απόφαση
Εκκρεμότητες / conflicts
```

Truth rules preserved:

```text
LATEST BLANK != ERASE PRIOR AUTHORITATIVE FACT
MISSING != NEGATIVE
CONFLICT != SILENTLY PICK LATEST
SCHEDULED DOSE != ACTUAL DOSE
DISCUSSION / OPTION != FINAL DECISION
CURRENT DRAFT != COMPLETED HISTORICAL FACT
```

No AI narrative summary, DB migration, new clinical write or treatment decision automation was introduced.

---

# 4. Test evidence

Exact tested runtime head:

```text
dab45baf9f80632ee6e58f03fa4d5005c68e0ac5
```

Workflow:

```text
G3 guidance salience longitudinal summary
run 33486322905
job 99787125415
COMPLETED / SUCCESS
```

Passed together:

- JavaScript syntax checks;
- G-3 salience + longitudinal-summary deterministic regressions;
- G-3 wiring/ownership regressions;
- frozen G-2 evidence contract;
- inherited G-2 evidence-core/live-state/wiring regressions;
- inherited G-1 core/wiring/UI-state/WHY-NOW regressions;
- authoritative Finish browser regression;
- server finalization lifecycle regression.

---

# 5. Exact-head review

Compare:

```text
base main: 9cfad82d1258a44e71080e0aa4d6d644e581cfbf
head:      dab45baf9f80632ee6e58f03fa4d5005c68e0ac5
status:    ahead
behind:    0
merge base: exactly current main
```

The delta contains only expected G-3 runtime/test/canonical files. No PR-1/PR-2, physiotherapy or RF leakage was found.

Source/ownership review confirmed:

- `osteoporosis-longitudinal-summary-core.js` is pure deterministic logic with no network/storage/DOM ownership;
- `progressive-guidance-ui.js` remains the single protected history/top-guidance render owner;
- encounters and labs are loaded through the same protected history owner with independent unavailable states;
- G-2 rule thresholds and treatment semantics were not reopened;
- no authoritative patient write was added;
- C1 Finish ownership remains unchanged.

A small duplicate implementation of salience state transition remains between the pure helper and UI integration. It is behaviorally locked by the same regression semantics and is classified as non-blocking maintainability debt, not a current ownership or safety defect. It should not trigger speculative refactoring before release unless review demonstrates divergence.

---

# 6. Status matrix

```text
G-2 PRODUCTION-SMOKE-VERIFIED                YES
G-3 DESIGN                                   COMPLETE
G-3 IMPLEMENTED                              YES
G-3 TESTED                                   YES
G-3 EXACT-HEAD REVIEW                        PASS
G-3 PRODUCT-OWNER RELEASE REVIEW             NO
G-3 PR                                       NONE
G-3 MERGED                                   NO
G-3 DEPLOYED                                 NO
G-3 PRODUCTION-SMOKE-VERIFIED                NO
PR-1 HEIDI                                   NOT IMPLEMENTED
PR-2 INLINE REVIEW/POPULATION                NOT IMPLEMENTED
REAL 5-CASE SYSTEM-ASSISTED PILOT            NOT STARTED
MODULE 01 CLOSED                             NO
```

---

# 7. Exact next action / STOP gate

The bounded G-3 implementation/test action is complete.

**STOP before release PR / merge / deploy.**

A future release action requires fresh bootstrap + explicit product-owner authorization, then:

```text
fresh verify main
→ release-readiness review of exact G-3 branch head
→ inspect exact checks/delta
→ open PR only if clean
→ STOP before merge unless separately authorized
```

PR-1/PR-2 and parked physiotherapy/RF work remain outside this slice.
