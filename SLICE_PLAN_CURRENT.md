# SLICE_PLAN_CURRENT.md — C1 Authoritative Finish / Pilot Finalization Integrity v1

> **STATUS:** IMPLEMENTATION AUTHORIZED / ACTIVE.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** M01-C1-FINISH-v1.
> **Verified remote main:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Parent closure branch:** `design/module01-closure-program-2026-08-30` @ `804c5cd3db9d8089efc127c0cf1866768fa4140d`.
> **Writer/runtime writer:** `fix/module01-c1-authoritative-finish-2026-08-30`.
> **Merge/deploy/preview:** NOT AUTHORIZED in this slice without a separate decision.

---

# 1. Problem

The Baseline Audit browser currently has two competing finalization owners:

```text
pilot-completion.js capture listener
→ intercepts Finish
→ stops propagation
→ triggers ordinary Save
→ marks local pilot completion

patient-registry.js listeners
→ Save syncs draft
→ Finish intends to sync completed
```

Because the capture listener can suppress the later Finish listener, local state can say complete while protected server state remains draft.

This is a pilot-blocking persistence/finalization integrity defect.

---

# 2. Required invariant

One operation must own finalization end-to-end:

```text
Finish
→ snapshot/persist latest Steps 1–6/module state
→ establish local pilot_completion=complete
→ send that same final payload to protected server with requested status=completed
→ await successful server response
→ report success
```

The server-side `resolve_encounter_status()` semantics remain unchanged:

```text
draft + completed → completed
completed + no-op draft save → completed
completed + material edit → amended
amended → amended
```

---

# 3. Ownership design

The preferred correction is to remove competing independent Finish click ownership and expose an explicit finalization function/interface that the Finish handler invokes once.

Allowed implementation options include:

- `pilot-completion.js` owns the single user Finish event and invokes an exported protected-sync finalization function from `patient-registry.js`; or
- a small shared finalization coordinator owns Finish and calls local completion + protected sync in deterministic sequence.

Do not retain two independent listeners whose ordering/propagation determines correctness.

Ordinary Save remains draft synchronization for draft encounters and continues to rely on the server state machine to preserve completed/amended states.

---

# 4. Failure semantics

Protected completion and local pilot completion are not interchangeable.

If no active protected patient exists, no server link can be created, authentication fails, or final completed synchronization fails:

- do not display a message claiming protected encounter completion;
- surface an explicit failure/not-synced state;
- preserve locally entered data rather than discarding it;
- allow retry after the protected context is restored.

No silent fallback from `completed` to `draft` is allowed during Finish.

---

# 5. Scope

Files may be changed only as needed around:

```text
static/baseline-audit/pilot-completion.js
static/baseline-audit/patient-registry.js
static/baseline-audit/app.js              (only if load/ownership wiring requires it)
focused regression test(s)
CURRENT_OPERATIONAL.md / SLICE_PLAN_CURRENT.md
```

Do not change clinical fields, schemas, KPI definitions, pilot N, baseline methodology or unrelated Clinic Utilities code.

---

# 6. Acceptance tests

Required integrated cases:

1. **Successful Finish**
   - draft encounter exists or is created;
   - final module state is saved;
   - local `pilot_completion.status=complete`;
   - protected server receives requested `completed` with final payload;
   - resulting server status is `completed`.

2. **Reload**
   - reopen same server encounter;
   - final payload reloads;
   - status remains `completed`.

3. **No-op Save**
   - ordinary Save after completion;
   - server remains `completed`.

4. **Material amendment**
   - edit material payload after completion and Save;
   - server becomes `amended`.

5. **Missing protected context / sync failure**
   - Finish must not claim protected completion;
   - local data remains available;
   - failure is explicit and retryable.

The browser-side test must prove event ownership/order rather than only string presence. Existing Python `test_encounter_finalization.py` remains the server-transition regression.

---

# 7. REPLAN triggers

Stop and replan if:

- one authoritative Finish requires changing clinical-form semantics;
- the defect is actually caused by broader unsound persistence ownership beyond Finish/Save;
- final payload cannot be obtained deterministically before server sync;
- fixing this seam requires changing server encounter status semantics;
- protected completion cannot be distinguished from local-only completion without a larger UI redesign.

---

# 8. Completion gate

This slice is complete when:

```text
single authoritative Finish owner          YES
final payload synchronized as completed    TESTED
reload completed                           TESTED
no-op Save preservation                    TESTED
material edit → amended                    TESTED
missing/sync-failure state explicit         TESTED
clinical form/KPI semantics changed         NO
```

Then update `CURRENT_OPERATIONAL.md` and stop before merge/deploy unless separately authorized.
