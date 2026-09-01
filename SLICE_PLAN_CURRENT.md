# SLICE_PLAN_CURRENT.md — C2 Server-authoritative Patient / Encounter Workspace v1

> **STATUS:** IMPLEMENTED / TESTED — RELEASE REVIEW REQUIRED BEFORE PR/MERGE/DEPLOY.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** `M01-C2-SERVER-AUTHORITATIVE-PATIENT-WORKSPACE-v1`.
> **Fresh production main:** `9cfad82d1258a44e71080e0aa4d6d644e581cfbf`.
> **Required predecessor ancestry:** G-3 PR #70 head at branch point `f12b32d06b2c48e4566c2653c437c86ac6f55e7f`.
> **Implementation branch:** `feat/module01-c2-server-authoritative-patient-workspace-2026-09-01`.
> **Exact tested runtime head:** `27d5248e51daaa61bf29c70d3bebf5d73b93d6a2`.
> **Test workflow:** `C2 server-authoritative patient workspace` run `33544223692` — SUCCESS.
> **Runtime writer:** NONE after implementation/test closeout.

---

# 1. Trigger / result

Product-owner workflow requirements:

1. work occurs on three computers; a saved/updated patient encounter on one device must be available from the protected server on the others;
2. the clinician-facing `PILOT CASE` / local `Cases` shell must be replaced by real protected patients with their own server-backed visits.

C2 implements the bounded Core workspace/data-integrity layer required for those requirements without reopening osteoporosis clinical guidance.

Cross-device continuity in v1 means:

```text
Device A edit
→ debounced protected server autosave
→ server is authoritative
→ Device B/C opening or reloading that encounter hydrates latest server state
```

It is **not** realtime push/websocket mirroring between already-open browser tabs.

---

# 2. Core-vs-module ownership

This is reusable **Core clinical-workspace behavior**, not osteoporosis-specific evidence content.

Core owns:

```text
patient + encounter identity
protected encounter create/load/update
optimistic concurrency
cross-device draft continuity
browser cache role
autosave serialization
conflict UX
completed/amended lifecycle
```

Module 01 retains its osteoporosis payload/cards/guidance. No G-2 rule, threshold or evidence contract changed.

---

# 3. Authoritative identity implemented

After a protected server encounter exists:

```text
patient_id
+ encounter_id
+ server updated_at version token
= authoritative working encounter identity/version
```

`internal_uuid` remains only as a legacy browser-payload/cache compatibility identity and retry-de-duplication key. It is not authoritative cross-device identity.

`localStorage` remains physically present because legacy form modules use it as a working cache, but it is explicitly disposable and non-authoritative. A different device hydrates from the protected server and does not need another device's local link map.

---

# 4. Server concurrency contract implemented

`EncounterUpdate` now accepts:

```text
expected_updated_at: datetime | null
```

C2 client writes to an existing encounter always provide the last server `updated_at` received from create/load/update.

Server behavior:

```text
expected_updated_at absent
→ legacy-compatible update path remains reachable

expected_updated_at matches current row.updated_at
→ apply existing update/finalization semantics
→ return fresh updated_at token

expected_updated_at differs
→ HTTP 409
→ no payload/date/status mutation
→ no updated_at mutation
```

Timezone-aware request timestamps are normalized to the existing naive-UTC DB representation before comparison.

The browser never implements last-writer-wins retry after HTTP 409.

---

# 5. Cross-device draft workflow implemented

## New visit

```text
active protected patient required
→ existing form creates fresh working payload
→ immediate Save pipeline
→ protected server draft POST
→ encounter_id + updated_at stored in disposable browser link cache
```

If no protected patient is active, `Νέα επίσκεψη` is blocked. No orphan local clinical visit is intentionally created through the C2 navigation path.

To reduce duplicate-visit risk if a server POST succeeds but its response is lost, a later retry first searches that patient's server encounters for the same payload `internal_uuid` before creating another row.

## Open existing visit

```text
GET server encounter
→ hydrate full payload into browser working cache
→ store patient_id + encounter_id + updated_at
→ render current visit
```

## Meaningful current-visit changes

```text
live UI change
→ 900 ms debounce
→ existing complete local/module Save pipeline
→ one serialized server draft-sync queue
→ successful response refreshes version token
```

Only one encounter sync executes at a time in one browser; later changes queue after the active sync.

## Device conflict

```text
Device A writes v12 → v13
Device B still holds v12 and writes
→ HTTP 409
→ Device-B local edits remain in its working cache
→ autosave pauses for that encounter
→ explicit conflict banner
→ explicit clinician action: Φόρτωση server έκδοσης
```

Server reload requires confirmation because it replaces unsynced local working edits. No automatic field merge exists in v1.

---

# 6. Laboratory boundary retained

C2 does not create an independent lab-concurrency subsystem.

Existing encounter-linked lab sync remains in the encounter save flow. Identical existing source-encounter/date/value snapshots are not unnecessarily PUT again.

C2 does **not** claim safe simultaneous independent editing of the same lab snapshot from multiple devices. That remains a future bounded extension only if real use requires it.

---

# 7. Pilot-shell retirement implemented

Ordinary clinician-facing workflow is now presented as:

```text
Ασθενείς & Επισκέψεις
→ active protected patient
→ Επισκέψεις
   → Νέα επίσκεψη
   → open existing draft/completed/amended visit
```

Normal presentation is rewritten before the legacy shell is revealed so the user does not see an initial pilot flash.

Retired from normal clinical presentation:

```text
PILOT CASE n/5
PILOT-nnn as displayed encounter identity
Νέο Case
Cases
pilot-only privacy copy
pilot-specific Finish messages
manual first-baseline-sample question as a progress requirement
legacy local-case cancel action
```

A dedicated early `clinical-workspace-shell.js` rewrites the legacy shell before reveal and explicitly describes browser storage as a temporary/non-authoritative cache.

Methodological future pilot/baseline cohort metadata remains separate from ordinary visit identity. Historical payloads are not deleted or retroactively relabelled.

The compatibility filename `pilot-completion.js` remains in v1 solely to preserve the production-proven single authoritative Finish ownership/load-order seam. Its clinician-facing behavior is now generic encounter completion and new clinical completion metadata uses `encounter_completion` rather than creating new `pilot_completion` state.

---

# 8. Finish / lifecycle invariants preserved

The existing one-owner Finish architecture remains:

```text
finalization-coordinator
→ patient-registry
→ single completion/Finish owner
```

No second `#finishVisitBtn` listener was added by the patient registry.

Server lifecycle remains:

```text
draft → completed
completed + later content change → amended
amended → remains amended
completed/amended never silently regress to draft
```

---

# 9. Safety / privacy boundaries

```text
SERVER ENCOUNTER != BROWSER CACHE
STALE DEVICE WRITE != SILENT OVERWRITE
CONFLICT != AUTO-MERGE
COMPLETED != REGRESS TO DRAFT
POST-COMPLETION CONTENT CHANGE = AMENDED
SCHEDULED DOSE != ACTUAL DOSE
CURRENT DRAFT != COMPLETED HISTORICAL FACT
```

C2 introduces no whole-service GDPR/privacy-compliance claim. Broader authentication/authorization review, audit trail, retention/data-minimization and privacy work remain explicit production concerns.

No identifiable patient data or real patient fixture was committed to the public repository.

---

# 10. Test evidence

Focused tests:

```text
test_c2_server_authoritative_workspace.py
test_c2_workspace_wiring.js
```

Inherited authoritative Finish regression was updated only to express generic encounter completion semantics while retaining its original integrity guarantees.

Exact tested runtime head:

```text
27d5248e51daaa61bf29c70d3bebf5d73b93d6a2
```

Workflow:

```text
C2 server-authoritative patient workspace
run 33544223692
job 99977647943
COMPLETED / SUCCESS
```

The exact-head job passed:

1. syntax checks;
2. C2 server optimistic-concurrency regressions;
3. C2 browser workspace/ownership regressions;
4. authoritative Finish browser regression;
5. server finalization lifecycle regression;
6. G-3 salience/longitudinal-summary regressions;
7. G-3 wiring/ownership regressions;
8. frozen G-2 evidence contract;
9. G-2 evidence-core/live-state/wiring regressions;
10. G-1 core/wiring/UI-state/WHY-NOW regressions.

Representative two-device test proves:

```text
v1
→ Device A update succeeds and creates v2
→ Device B stale v1 update returns 409
→ server still contains Device A v2 payload/version
→ later GET returns that latest server state
```

---

# 11. Exact-head review

Compared required predecessor head:

```text
base: f12b32d06b2c48e4566c2653c437c86ac6f55e7f
head: 27d5248e51daaa61bf29c70d3bebf5d73b93d6a2
status: ahead
behind: 0
merge base: exactly predecessor head
```

The delta contains only expected C2 files:

```text
clinical_data.py
static/baseline-audit/app.js
static/baseline-audit/clinical-workspace-shell.js
static/baseline-audit/patient-registry.js
static/baseline-audit/pilot-completion.js
static/baseline-audit/whole-form-progress.js
test_baseline_finish_browser.js
test_c2_server_authoritative_workspace.py
test_c2_workspace_wiring.js
.github/workflows/c2-server-authoritative-workspace-tests.yml
CURRENT_OPERATIONAL.md
SLICE_PLAN_CURRENT.md
```

No PR-1/PR-2, physiotherapy/RF or osteoporosis evidence-rule leakage was found. No DB migration was introduced.

No REPLAN trigger occurred.

---

# 12. Completion matrix

```text
C2 DESIGN                                   COMPLETE
C2 IMPLEMENTED                              YES
C2 TESTED                                   YES
C2 EXACT-HEAD REVIEW                        PASS
C2 PRODUCT-OWNER RELEASE REVIEW             NO
C2 PR                                       NONE
C2 MERGED                                   NO
C2 DEPLOYED                                 NO
C2 PRODUCTION-SMOKE-VERIFIED                NO
G-3 PR #70                                  OPEN / MERGE HOLD
REAL 5-CASE SYSTEM-ASSISTED PILOT           NOT STARTED
```

`IMPLEMENTED / TESTED` does not mean released or production-proven.

---

# 13. Out of scope retained

- realtime push/websocket synchronization across already-open devices;
- independent simultaneous lab-edit conflict resolution;
- offline-first multi-device merge;
- full identity-provider redesign;
- G-2 rule/evidence changes;
- PR-1 transcript extraction;
- PR-2 inline candidate review;
- Practice Review;
- formal 5-case validation collection;
- retention/GDPR certification claims;
- physiotherapy/RF work.

---

# 14. Stop gate

The bounded C2 implementation/test slice is closed at `IMPLEMENTED / TESTED`.

C2 has a hard predecessor dependency on G-3. It must not release ahead of G-3 PR #70.

A future C2 release action requires separate explicit product-owner authority and a fresh bootstrap, then:

```text
fresh verify main + six canonicals
→ verify G-3 predecessor release state / ancestry
→ exact-head C2 release-readiness review
→ open C2 release PR only if clean and explicitly authorized
→ STOP before merge unless merge authority is separately explicit
```

Until then:

```text
NO ACTIVE C2 WRITER
NO C2 RELEASE PR
NO C2 MERGE
NO C2 DEPLOY
NO C2 PRODUCTION-SMOKE CLAIM
```
