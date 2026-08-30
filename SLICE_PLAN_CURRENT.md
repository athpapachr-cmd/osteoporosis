# SLICE_PLAN_CURRENT.md — C1 Authoritative Finish / Pilot Finalization Integrity v1

> **STATUS:** IMPLEMENTED / TESTED / MERGE GATE.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** M01-C1-FINISH-v1.
> **Verified remote main:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Parent closure branch:** `design/module01-closure-program-2026-08-30` @ `804c5cd3db9d8089efc127c0cf1866768fa4140d`.
> **Implementation branch:** `fix/module01-c1-authoritative-finish-2026-08-30`.
> **Runtime writer:** NONE — implementation complete.
> **Merge/deploy/preview:** NOT AUTHORIZED / NOT DONE.

---

# 1. Problem

The Baseline Audit browser had two competing finalization owners:

```text
pilot-completion.js capture listener
→ intercept Finish
→ stop propagation
→ trigger ordinary Save
→ local pilot completion

patient-registry.js
→ Save sync draft
→ separate Finish sync completed
```

This allowed the finalization order to depend on browser event propagation and could create local `complete` with protected server `draft`.

---

# 2. Frozen invariant

One operation now owns finalization end-to-end:

```text
Finish
→ suppress ordinary Save→draft server sync
→ persist latest local/module state
→ mark pilot completion locally
→ synchronize the same final payload to protected server with requested status=completed
→ await successful server response
→ report protected completion only after confirmation
```

Server status semantics remain unchanged:

```text
draft + completed → completed
completed + no-op draft save → completed
completed + material edit → amended
amended → amended
```

---

# 3. Implemented ownership

Implementation uses:

```text
BaselineFinalizationCoordinator
→ owns finalization-in-progress guard

ClinicalRegistry.finalizeActiveEncounter()
→ strict completed server synchronization

pilot-completion.js
→ sole Finish click owner
```

`patient-registry.js` no longer owns a second Finish click listener.

Ordinary Save still performs draft synchronization when no authoritative Finish is active. During Finish, the coordinator suppresses that draft sync so no race is scheduled.

The bootstrap order is explicitly:

```text
finalization coordinator
→ patient registry / exported strict finalization API
→ pilot completion / sole Finish owner
```

---

# 4. Final payload sequencing

Existing Steps 3–6 persist module state from Save through `setTimeout(..., 0)` handlers.

The authoritative Finish therefore:

1. acquires the coordinator guard;
2. clicks local Save;
3. waits one event-loop flush so module persistence completes;
4. marks `pilot_completion.status=complete` locally;
5. reads/sends that final local case through `ClinicalRegistry.finalizeActiveEncounter()`;
6. waits for the protected server response before success UI.

No Steps 3–6 clinical capture logic was changed.

---

# 5. Failure semantics implemented

If protected patient context is absent, encounter payload is absent, date is invalid, authentication fails or server sync fails:

- strict finalization throws/returns failure to the sole Finish owner;
- local data remain available;
- no protected-completion success text is shown;
- ordinary draft fallback remains suppressed during the failed Finish attempt;
- the coordinator is released and Finish can be retried.

Local preservation does not equal protected-server completion.

---

# 6. Acceptance evidence

Runtime/test head:

```text
a26e2a7415cff5b1409400ddfddce4ba01e6b6b7
```

GitHub Actions:

```text
workflow: Baseline finalization integrity
run:      33323066983
result:   SUCCESS
```

### Browser/event regression

Dynamic Node regression proves:

- one Finish owner;
- no second registry Finish listener;
- coordinator → registry → pilot load order;
- Save draft sync suppressed during Finish;
- final server payload sees module state persisted by local Save;
- final server payload includes `pilot_completion.status=complete`;
- server finalization called exactly once;
- protected-sync failure is explicit and local data remain retryable.

### Server lifecycle regression

Synthetic FastAPI + SQLite lifecycle test proves:

```text
draft encounter
→ Finish payload requested completed
→ server completed
→ GET/reload completed with same final payload
→ no-op Save requesting draft stays completed
→ material edit + Save becomes amended
```

JavaScript syntax checks also passed.

---

# 7. Scope verification

Changed only the finalization seam, focused tests/CI and active canonicals.

Not changed:

- clinical questions/fields;
- baseline schemas;
- KPI definitions/applicability;
- pilot target N;
- 30-case baseline methodology;
- transcript extraction;
- Practice Review;
- physiotherapy code.

---

# 8. Completion matrix

```text
single authoritative Finish owner          YES / TESTED
final payload synchronized as completed    YES / TESTED
reload completed                           YES / TESTED
no-op Save preservation                    YES / TESTED
material edit → amended                    YES / TESTED
missing/sync-failure state explicit         YES / TESTED
clinical form/KPI semantics changed         NO
merged                                      NO
deployed                                    NO
production smoke                            NO
```

The code-level C1 blocker is closed. Real pilot readiness remains blocked until this tested fix is merged, auto-deployed and smoke-verified in production.

---

# 9. Exact next action

STOP at merge gate.

A separate authorization is required for PR/merge/deploy. After production deployment, run one synthetic authoritative-Finish smoke and reload check. Only after that PASS should the real-pilot gate be released.
