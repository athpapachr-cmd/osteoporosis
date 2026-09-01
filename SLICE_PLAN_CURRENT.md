# SLICE_PLAN_CURRENT.md — G-3 Guidance Salience + Longitudinal Patient Summary v1

> **STATUS:** IMPLEMENTED / TESTED — RELEASE REVIEW REQUIRED BEFORE PR/MERGE/DEPLOY.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** `M01-G3-GUIDANCE-SALIENCE-LONGITUDINAL-SUMMARY-v1`.
> **Fresh base main:** `9cfad82d1258a44e71080e0aa4d6d644e581cfbf`.
> **Implementation branch:** `feat/module01-g3-guidance-salience-longitudinal-summary-2026-09-01`.
> **Exact tested runtime head:** `dab45baf9f80632ee6e58f03fa4d5005c68e0ac5`.
> **Test workflow:** `G3 guidance salience longitudinal summary` run `33486322905` — SUCCESS.
> **Runtime writer:** NONE after implementation/test closeout.

---

# 1. Trigger / evidence from use

Product-owner authenticated G-2 production smoke passed on the deployed exact runtime ancestry.

During that smoke the product owner identified two concrete workflow needs:

1. newly applicable guidance should become visibly more salient at the moment it appears; example: VFA/vertebral-imaging guidance after current height loss reaches the evidence trigger of at least 4 cm;
2. a concise patient-level longitudinal summary should remain visible throughout the encounter and summarize authoritative history from first completed/amended encounter through the latest reliable state.

This is a product-UX/read-only longitudinal slice. It does not reopen G-2 evidence content.

---

# 2. Implemented G3-A — newly surfaced guidance salience

The runtime now maintains a per-active-patient/case Visit Plan baseline and detects canonical card/domain additions after current-state/history changes.

Implemented semantics:

```text
initial Visit Plan
→ establishes baseline
→ no blanket NEW labels

previous stable card/domain absent
+ current plan adds evidence/event/unresolved/due/treatment-context item
→ newly surfaced
→ textual `Νέο` badge
→ stronger card + top-flow visual emphasis

item ceases to apply
→ newly-surfaced state cleared

patient/case changes
→ baseline reset
```

Pure base-flow `VISIT_TYPE_CORE` additions are not treated as new-item noise unless a higher-value trigger/evidence state is also present.

The salience state is ephemeral/in-memory only and does not create authoritative patient persistence.

Primary product-owner example is explicitly regression-tested:

```text
height loss = 3.9 cm
→ OST_G2_R02_VFA_STRUCTURED_TRIGGER absent

height loss = 4.0 cm
→ OST_G2_R02_VFA_STRUCTURED_TRIGGER present
→ VFA enters newly-surfaced set
→ top summary item and destination VFA card receive `Νέο`
```

Color is not the only cue: `Νέο` is rendered textually.

---

# 3. Implemented G3-B — always-visible longitudinal patient summary

A compact `Σύνοψη ασθενούς` is rendered above `Σημερινή ροή` whenever a protected patient is active.

Derived inputs:

```text
completed/amended protected encounters
+ protected lab snapshots
+ existing LongitudinalGuidanceProjectionV1
+ current visit snapshot clearly marked current/non-historical
```

Visible domains:

1. **Πορεία** — first completed/amended encounter date, latest date, encounter count.
2. **Κατάγματα / κίνδυνος** — deduplicated stable fracture events/most recent reliable event plus latest explicit formal risk state where available.
3. **DXA** — latest reliable DXA date and key T-scores; no significance claim without comparability/LSC.
4. **Θεραπεία** — active/latest reliable treatment episode plus actual-administration timeline/count from existing G-1 semantics.
5. **Εργαστηριακά** — latest protected laboratory snapshot and selected concise key values.
6. **Τελευταία απόφαση** — latest explicit Step-4 final management decision/selected agent when present.
7. **Εκκρεμότητες / conflicts** — unresolved tasks, unresolved-critical close state and longitudinal conflicts.

The current visit is visually described as current context and does not become completed longitudinal fact until authoritative Finish.

---

# 4. Truth / conflict rules preserved

```text
READ-ONLY SUMMARY != NEW SOURCE OF CLINICAL TRUTH
LATEST BLANK != ERASE PRIOR AUTHORITATIVE FACT
MISSING != NEGATIVE
CONFLICT != CHOOSE LATEST SILENTLY
SCHEDULED DOSE != ACTUAL DOSE
DISCUSSION/OPTION != FINAL DECISION
CURRENT DRAFT != HISTORICAL COMPLETED FACT
```

Additional behavior:

- history unavailable is an explicit unavailable state, never `0 prior visits`;
- lab availability is tracked separately from encounter-history availability;
- repeated fracture snapshots are deduplicated by stable event identity where possible;
- actual administration semantics reuse the existing G-1 projection rather than implementing a second dosing timeline;
- treatment/admin conflicts are displayed as conflicts, not silently resolved;
- no AI free-text summary is used in v1.

---

# 5. Runtime ownership / API boundary

New pure module:

```text
static/baseline-audit/osteoporosis-longitudinal-summary-core.js
```

It receives structured data and performs deterministic summary/salience calculations. It does **not** own:

```text
network fetch
DOM rendering
localStorage/sessionStorage
patient writes
Finish
```

Existing owner retained:

```text
static/baseline-audit/progressive-guidance-ui.js
```

It remains the single protected encounter/lab history and top guidance rendering owner.

Existing protected endpoints are reused:

```text
GET /clinical/patient/{patient_id}/encounters
GET /clinical/patient/{patient_id}/labs
```

No DB migration, new API route, authoritative write or alternate history store was introduced.

---

# 6. UI contract implemented

Top order:

```text
Patient Registry / active patient
→ Σύνοψη ασθενούς
→ Σημερινή ροή
→ step tabs / clinical cards
```

Summary states explicitly distinguish:

```text
τεκμηριωμένο
δεν έχει τεκμηριωθεί
μη διαθέσιμο
ασυμφωνία / attention
```

New guidance salience uses:

```text
is-newly-surfaced
+ explicit `Νέο` badge
+ stronger border/background/outline
```

No animation is required and no clinical meaning depends only on color.

---

# 7. Acceptance evidence

New focused tests:

```text
test_g3_guidance_summary_node.js
test_g3_guidance_summary_wiring.js
```

Workflow:

```text
G3 guidance salience longitudinal summary
run:      33486322905
job:      99787125415
head:     dab45baf9f80632ee6e58f03fa4d5005c68e0ac5
result:   SUCCESS
```

Passed:

### Salience
- initial render baseline produces no blanket new state;
- `<4 cm → >=4 cm` transition activates R02/VFA and marks VFA new;
- unchanged plan does not create a second transition;
- new state persists while the newly surfaced item remains applicable;
- item removal clears salience;
- base-flow-only content does not create salience noise.

### Longitudinal summary
- chronological first/latest encounter dates and count;
- current `internal_uuid` excluded from historical completed course;
- history unavailable != zero visits;
- later blank does not erase prior DXA/risk state;
- repeated stable fracture event is not double-counted;
- scheduled-only administration is not counted as actual;
- treatment/admin conflict is surfaced;
- latest protected lab snapshot selected deterministically;
- latest explicit management decision selected deterministically;
- current visit remains non-historical.

### Inherited gates
- frozen G-2 evidence contract PASS;
- G-2 evidence-core/live-state/wiring PASS;
- G-1 core/wiring/UI-state/WHY-NOW PASS;
- C1 authoritative Finish browser PASS;
- server finalization lifecycle PASS.

---

# 8. Exact-head review

At exact tested runtime head `dab45baf…`:

```text
base main: 9cfad82d1258a44e71080e0aa4d6d644e581cfbf
status: ahead
behind: 0
merge base: exactly base main
```

Only expected G-3 runtime/test/canonical files changed. No PR-1, PR-2, physiotherapy or RF leakage was found.

No REPLAN trigger occurred.

A small duplicated salience state-transition implementation remains between the pure helper and browser integration. Exact regression semantics currently match and this is classified as non-blocking maintainability debt, not a safety/data-integrity or ownership blocker. Avoid speculative refactoring unless later review demonstrates divergence.

---

# 9. Out of scope retained

- new G-2 clinical rules or thresholds;
- AI-generated narrative summary;
- patient-level canonical treatment DB table;
- transcript extraction / PR-1;
- inline transcript population / PR-2;
- Practice Review scoring;
- real 5-case pilot;
- physiotherapy/RF work.

---

# 10. Completion matrix

```text
G-3 design                                  COMPLETE
G-3 implementation                         YES
G-3 focused regressions                    PASS
G-2 inherited contract/runtime gates       PASS
G-1 inherited regressions                  PASS
C1 inherited finalization gates            PASS
Exact-head source/delta review             PASS
Product-owner release review               NO
PR opened                                  NO
Merged                                     NO
Deployed                                   NO
Production-smoke-verified                  NO
Pilot-validated                            NO
```

---

# 11. Stop gate

This bounded implementation slice is closed at `IMPLEMENTED / TESTED`.

Next possible release action requires a separate fresh release-readiness bootstrap/review and explicit product-owner release authority.

Until then:

```text
NO ACTIVE WRITER
NO RELEASE PR
NO MERGE
NO DEPLOY
NO PRODUCTION-SMOKE CLAIM FOR G-3
```
