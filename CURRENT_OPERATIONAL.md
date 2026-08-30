# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-1 R1/R2 IMPLEMENTED / TESTED / RELEASE GATE.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Implementation/correction branch:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30`.
> **Release-review block head:** `56267d08dc5d68b8c5e4208f2ae3761fa15156b5`.
> **R1/R2 runtime+test head:** `3294deebb97cf3f0a0d8fa2848ac4af7a04b01de`.
> **Inherited tested C1 head:** `a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871`.
> **ACTIVE CANONICAL WRITER/LOCK:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30` — release-gate canonical closeout only.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE — bounded R1/R2 correction complete.
> **PR/merge/deploy/production smoke:** NOT DONE / NOT AUTHORIZED.

---

# 1. Product-owner authorization completed

Authorized scope was exactly:

```text
fix G1-R1 + G1-R2
+ new focused regressions
+ full G-1 + C1 CI
```

No PR, merge, deploy, production smoke, PR-1/PR-2, taxonomy expansion, medication-specific milestone rule, physiotherapy or RF mutation was authorized or performed.

---

# 2. G1-R1 closed — history availability is explicit

Prior release blocker:

```text
protected history fetch failure
→ []
→ possible false "0 previous encounters" presentation
```

Current invariant:

```text
HISTORY UNAVAILABLE != NO HISTORY
AUTH/NETWORK/SERVER FAILURE != ZERO PRIOR ENCOUNTERS
```

Runtime behavior now:

- explicit `not_loaded / loading / loaded / unavailable` state;
- failed protected history fetch displays unavailable/incomplete longitudinal context rather than zero history;
- successful loaded empty history may legitimately display zero visits;
- beginning a new patient/history load clears previous in-memory historical rows immediately;
- a stale response for a no-longer-active patient cannot overwrite current patient history state;
- current local visit guidance remains usable when longitudinal history is loading/unavailable.

---

# 3. G1-R2 closed — live UI state outranks persisted cache

Current invariant:

```text
IF A LIVE CONTROL EXISTS
→ its present value, including blank/empty, owns today's in-memory guidance snapshot

persisted cache
→ fallback only when corresponding live control/root is absent
```

Covered fields:

- encounter archetype;
- encounter date;
- quick notes;
- interval fracture status;
- rendered fracture-event collection.

Deleting all currently rendered fracture events now projects an empty event list rather than resurrecting persisted events before Save.

---

# 4. Exact test evidence

R1/R2 runtime+test head:

```text
3294deebb97cf3f0a0d8fa2848ac4af7a04b01de
```

GitHub Actions:

```text
workflow: G1 progressive guidance foundation
run:      33329341340
job:      g1-guidance
result:   SUCCESS
```

Passed at that exact head:

- JavaScript syntax;
- progressive guidance core regressions;
- guidance wiring/ownership regression;
- new progressive guidance UI-state regressions for R1/R2;
- inherited authoritative Finish browser regression;
- inherited server finalization lifecycle regression.

Focused new regression proof includes:

```text
failed history load → unavailable
unavailable summary != zero-history claim
successful empty history → loaded + zero
new-patient load clears old patient's history immediately
live blank quick_notes > persisted nonblank
live blank archetype/date/fracture status > persisted value
live empty fracture-event list > persisted events
persisted fallback only if live control/root absent
```

---

# 5. Scope review

Exact compare from block head `56267d08dc5d68b8c5e4208f2ae3761fa15156b5` to runtime/test head `3294deebb97cf3f0a0d8fa2848ac4af7a04b01de` changed only:

```text
static/baseline-audit/progressive-guidance-ui.js
test_progressive_guidance_ui_state.js
.github/workflows/g1-progressive-guidance-tests.yml
CURRENT_OPERATIONAL.md
```

No core clinical guidance rules, database/schema/KPI semantics, physiotherapy/RF code or transcript implementation were changed.

---

# 6. C1 preservation

The branch still directly descends from exact tested C1 head:

```text
a4005dc88140d8f988fcac2b4f4bd9f9bb0c3871
```

The full R1/R2 workflow re-ran and passed both C1 browser Finish ownership and server finalization lifecycle regressions.

C1 remains NOT MERGED / NOT DEPLOYED / NOT PRODUCTION-SMOKED.

---

# 7. Current status matrix

```text
C1 IMPLEMENTED / TESTED                    YES
C1 MERGED / DEPLOYED / PROD-SMOKED         NO
G-1 BASE IMPLEMENTED / TESTED              YES
G1-R1                                      CLOSED / TESTED
G1-R2                                      CLOSED / TESTED
G-1 RELEASE-READY AT CODE/CI LEVEL          YES
G-1 MERGED / DEPLOYED / PROD-SMOKED        NO
PR-1 HEIDI                                 NOT IMPLEMENTED
PR-2 INLINE REVIEW/POPULATION              NOT IMPLEMENTED
5-CASE SYSTEM-ASSISTED PILOT               NOT STARTED
30-CASE SYSTEM-ASSISTED BASELINE            NOT STARTED
MODULE 01 CLOSED                           NO
```

`RELEASE-READY AT CODE/CI LEVEL` does not mean production-validated.

---

# 8. Exact next action

STOP runtime mutation at the release gate.

A separate product-owner release decision is required before:

```text
fresh six-canonical bootstrap
→ fresh main verification
→ exact full compare/review
→ PR
→ merge
→ normal Render auto-deploy
→ production synthetic smoke:
   C1 authoritative Finish
   + G-1 guidance/history availability
   + dropdown/context/WHY NOW
→ canonical release evidence closeout
```

No further runtime work is authorized by the completed R1/R2 request.