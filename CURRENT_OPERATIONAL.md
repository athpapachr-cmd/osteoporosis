# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 CLOSURE PROGRAM — C1 AUTHORITATIVE FINISH IMPLEMENTED / TESTED / MERGE GATE.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Parent closure branch:** `design/module01-closure-program-2026-08-30` @ `804c5cd3db9d8089efc127c0cf1866768fa4140d`.
> **Current major phase:** close Osteoporosis Module 01 against explicit exit evidence, then generalize later.
> **ACTIVE CANONICAL WRITER/LOCK:** `fix/module01-c1-authoritative-finish-2026-08-30`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE — bounded C1 implementation is complete.
> **Merge/deploy/preview:** NOT AUTHORIZED / NOT DONE.

---

# 1. Product-owner authorization

The product owner explicitly authorized the bounded C1 correction:

```text
make one authoritative Finish fix
```

The authorized scope was limited to the pilot finalization-integrity seam. Clinical questions, KPI semantics, pilot target N, baseline methodology, PR-1/PR-2 and physiotherapy were not changed.

---

# 2. Preserved project state

Physiotherapy remains PARKED/PRESERVED.

The later product-reviewed rich-referral enhancement remains at:

```text
branch: feat/cu1-rich-referral-global-evidence-2026-08-29
head:   bdd23b83a8252405f5aa5a0c0b67f303ccfcef5f
state:  IMPLEMENTED / TESTED / PRODUCT-OWNER REVIEWED
PR:     NONE OPEN
MERGED: NO
DEPLOYED: NO
```

Do not mutate, merge or deploy it in Module 01 closure work without separate authorization.

---

# 3. C1 defect closed in code

Prior wiring allowed two competing Finish owners:

```text
pilot-completion.js capture Finish
→ stopped propagation
→ triggered ordinary Save
→ local pilot_completion=complete

patient-registry.js
Save → draft sync
Finish → intended completed sync
```

This could yield local `complete` with server `draft`.

The implementation now establishes one authoritative end-to-end Finish path.

---

# 4. Implemented invariant

Current implementation on this branch:

```text
Step-6 Finish
→ acquire BaselineFinalizationCoordinator guard
→ suppress ordinary Save→draft server synchronization
→ click local Save so Steps/modules persist their current state
→ flush setTimeout(0) module persistence
→ mark local pilot_completion=complete
→ call ClinicalRegistry.finalizeActiveEncounter()
→ strict protected server sync with requested status=completed using the final local payload
→ await returned server row
→ show protected-completion success only after confirmed server response
```

If protected context is missing or synchronization fails:

```text
local data remain available
protected completion is NOT claimed
failure is shown explicitly
ordinary draft fallback is suppressed during Finish
retry remains possible
```

The server-side `resolve_encounter_status()` semantics were not changed.

---

# 5. Implementation evidence

Runtime/test implementation head before canonical closeout:

```text
a26e2a7415cff5b1409400ddfddce4ba01e6b6b7
```

Focused GitHub Actions evidence:

```text
workflow: Baseline finalization integrity
run:      33323066983
head:     a26e2a7415cff5b1409400ddfddce4ba01e6b6b7
result:   SUCCESS
```

Passed steps:

- JavaScript syntax checks for coordinator, registry, pilot completion, bootstrap and browser regression;
- dynamic Node browser event-order/ownership regression;
- FastAPI/SQLite server finalization lifecycle regression.

The browser regression proves:

- Save→draft sync is suppressed while authoritative Finish is active;
- server finalization is called exactly once;
- module state persisted through the local Save is present before server finalization;
- `pilot_completion.status=complete` is included in the final server payload;
- failed protected finalization cannot display false protected-completion success;
- local data remain available after failure;
- the coordinator guard is released after success or failure.

The API lifecycle regression proves in one synthetic end-to-end server test:

```text
draft
→ completed with final payload
→ reload remains completed with same payload
→ no-op Save requesting draft remains completed
→ material payload edit + Save becomes amended
```

Historical `test_encounter_finalization.py` remains the pre-existing unit-level state-machine evidence; the new workflow's Python `unittest` discovery executed the new lifecycle test, while the dynamic Node regression covers the newly fixed browser seam.

---

# 6. Files changed in bounded C1

```text
static/baseline-audit/finalization-coordinator.js   NEW
static/baseline-audit/app.js
static/baseline-audit/pilot-completion.js
static/baseline-audit/patient-registry.js
test_baseline_finish_browser.js                    NEW
test_baseline_finalization_api.py                  NEW
.github/workflows/baseline-finalization-tests.yml  NEW
CURRENT_OPERATIONAL.md
SLICE_PLAN_CURRENT.md
```

No clinical form/schema/KPI content was changed.

---

# 7. Status matrix

```text
C1 DESIGNED                            YES
C1 IMPLEMENTED                         YES
C1 TESTED                              YES
C1 MERGED                              NO
C1 DEPLOYED                            NO
C1 PRODUCTION-SMOKE-VERIFIED           NO
C1 CODE-LEVEL PILOT BLOCKER CLOSED     YES
PRODUCTION PILOT READINESS             BLOCKED PENDING MERGE/DEPLOY + SMOKE
5-CASE REAL PILOT                      NOT STARTED
MODULE 01 CLOSED                       NO
```

---

# 8. Exact next action

```text
STOP runtime mutation.
Obtain separate merge/deploy decision.
If authorized:
→ fresh main verification
→ PR/review/merge using exact tested head ancestry
→ allow normal Render auto-deploy from main
→ production synthetic Finish smoke:
   local completion + protected server completed + reload
→ if PASS, mark C1 production-ready and release the real-pilot gate.
```

Do not start real pilot collection against production until the fix is merged/deployed and the authoritative Finish path is smoke-verified there.
