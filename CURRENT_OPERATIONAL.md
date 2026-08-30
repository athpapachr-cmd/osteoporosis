# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 CLOSURE PROGRAM — C1 AUTHORITATIVE FINISH IMPLEMENTATION ACTIVE.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Parent closure branch:** `design/module01-closure-program-2026-08-30` @ `804c5cd3db9d8089efc127c0cf1866768fa4140d`.
> **Current major phase:** close Osteoporosis Module 01 against explicit exit evidence, then generalize later.
> **ACTIVE CANONICAL WRITER/LOCK:** `fix/module01-c1-authoritative-finish-2026-08-30`.
> **ACTIVE RUNTIME WRITER/LOCK:** `fix/module01-c1-authoritative-finish-2026-08-30`.
> **Runtime mutation:** AUTHORIZED ONLY for the bounded C1 Finish/local/server finalization seam.
> **Merge/deploy/preview:** NOT YET AUTHORIZED / NOT DONE.

---

# 1. Product-owner authorization

The product owner explicitly authorized the bounded C1 correction:

```text
make one authoritative Finish fix
```

This authorization applies only to the pilot finalization-integrity blocker previously frozen in the closure plan. It does not authorize clinical-form redesign, KPI changes, PR-1/PR-2 work, physiotherapy work or unrelated persistence refactors.

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

Do not mutate, merge or deploy it in this slice.

---

# 3. C1 defect being corrected

Current browser wiring has competing owners:

```text
pilot-completion.js
Finish capture listener
→ stopImmediatePropagation()
→ trigger Save
→ local pilot_completion=complete

patient-registry.js
Save listener → server sync as draft
Finish listener → intended server sync as completed
```

The capture listener can suppress the later Finish listener, allowing:

```text
local pilot_completion=complete
server clinical_encounter.status=draft
```

The existing server-side finalization state machine remains valid and must be preserved.

---

# 4. Authorized implementation invariant

There must be one authoritative Finish operation:

```text
Step-6 Finish
→ persist complete current browser/module state
→ mark pilot completion locally
→ synchronize the same final payload to protected server storage with requested status=completed
→ await server success
→ only then show successful protected completion
```

If there is no active protected patient/server context, the UI must not falsely imply protected-server completion. The local-only state, if preserved, must be explicitly distinguishable from successful protected completion.

---

# 5. Acceptance evidence required

At minimum prove:

```text
A. draft encounter + Step-6 Finish
   → local pilot_completion=complete
   → server status=completed
   → final Steps 1–6 payload present server-side

B. reload/reopen
   → same encounter loadable and completed

C. no-op Save after completion
   → remains completed

D. material edit + Save
   → becomes amended

E. no active protected patient/server context
   → no false protected-completion success state
```

An integrated browser/JavaScript regression is required in addition to the existing Python server-transition unit test.

---

# 6. Explicitly out of scope

Do not in C1:

- alter clinical questions/fields;
- change pilot eligibility or target N;
- change KPI definitions/applicability;
- change 30-case baseline methodology;
- add transcript extraction or Practice Review;
- refactor unrelated patient-registry behavior;
- resume physiotherapy work;
- merge/deploy without a separately recorded decision.

---

# 7. Status matrix

```text
C1 DESIGN                              FROZEN
C1 IMPLEMENTATION                      ACTIVE
C1 TESTED                              NO
C1 MERGED                              NO
C1 DEPLOYED                            NO
C1 PRODUCTION-SMOKE-VERIFIED           NO
5-CASE REAL PILOT                      NOT STARTED
MODULE 01 CLOSED                       NO
```

---

# 8. Exact next authorized action

```text
1. inspect exact Finish/save/server-sync seams on this branch;
2. implement one authoritative finalization path only;
3. add integrated regression coverage;
4. run focused tests;
5. update CURRENT_OPERATIONAL with exact implementation/test evidence;
6. STOP before merge/deploy unless separately authorized.
```
