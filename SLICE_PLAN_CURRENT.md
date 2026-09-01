# SLICE_PLAN_CURRENT.md — C2 Server-authoritative Patient / Encounter Workspace v1

> **STATUS:** ACTIVE DESIGN / IMPLEMENTATION.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** `M01-C2-SERVER-AUTHORITATIVE-PATIENT-WORKSPACE-v1`.
> **Fresh production main:** `9cfad82d1258a44e71080e0aa4d6d644e581cfbf`.
> **Required predecessor ancestry:** G-3 PR #70 head `f12b32d06b2c48e4566c2653c437c86ac6f55e7f`.
> **Implementation branch:** `feat/module01-c2-server-authoritative-patient-workspace-2026-09-01`.
> **Runtime writer:** this session, bounded to C2.

---

# 1. Trigger / problem

Product-owner requirements from real intended workflow:

1. work occurs on three computers; a saved/updated patient encounter on one device must be available from the protected server on the others;
2. the clinician-facing `PILOT CASE` / local `Cases` shell must be replaced by real protected patients with their own server-backed visits.

Current runtime already has protected PostgreSQL patients/encounters/labs, but browser working identity remains local-first and encounter updates are last-writer-wins.

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

Module 01 retains its osteoporosis payload/cards/guidance. C2 must not change G-2 evidence rules.

---

# 3. Authoritative identity contract

After a protected server encounter exists:

```text
patient_id
+ encounter_id
+ server updated_at version token
= authoritative working encounter identity/version
```

`internal_uuid` may remain inside the legacy browser payload/cache for compatibility, but it is not authoritative cross-device identity.

`localStorage` is a disposable working cache only. A different device must be able to hydrate the encounter from the server without possessing another device's local link map.

---

# 4. Server concurrency contract

Extend `EncounterUpdate` with optional:

```text
expected_updated_at: datetime | null
```

Client C2 writes to an existing encounter always provide the last server `updated_at` received from create/load/update.

Server behavior:

```text
expected_updated_at absent
→ legacy-compatible update path remains reachable

expected_updated_at present and equals current row.updated_at
→ apply existing update/finalization semantics
→ return new updated_at token

expected_updated_at present and differs
→ HTTP 409
→ no payload/date/status mutation
→ no updated_at mutation
```

Datetime comparison must normalize timezone-aware request values to the existing naive-UTC DB representation before equality testing.

The client must not implement last-writer-wins retry after 409.

---

# 5. Cross-device draft workflow

## New visit

```text
active protected patient required
→ create fresh local working payload
→ immediately POST server draft
→ store encounter_id + updated_at in local cache link
→ render visit as server-backed draft
```

If no protected patient is active, `Νέα επίσκεψη` is blocked; do not create an orphan local clinical case.

## Open existing visit

```text
GET server encounter
→ hydrate full payload into disposable local cache
→ store patient_id + encounter_id + updated_at
→ render current visit
```

## Meaningful current-visit changes

```text
live UI change
→ debounce
→ run existing complete local/module Save pipeline
→ serialized server draft sync
→ successful response updates version token
```

Only one encounter sync may execute at a time in one browser. New changes occurring during a sync must queue a later save rather than race the earlier request.

## Device conflict

```text
Device A writes v12 → v13
Device B still holds v12 and writes
→ 409
→ local Device-B edits preserved
→ autosave paused for that encounter
→ visible conflict banner
→ explicit clinician action reloads server v13
```

No automatic field merge in v1.

---

# 6. Laboratory boundary

C2 does not create a second independent lab-concurrency model.

Existing encounter-linked lab sync remains under the encounter save transaction flow at the client boundary. Avoid unnecessary lab PUTs when the same source-encounter/date/values are already present.

C2 does **not** claim safe simultaneous independent editing of the same lab snapshot from multiple devices; that can be a later bounded extension if real use requires it.

---

# 7. Pilot-shell retirement contract

Ordinary clinician-facing workflow becomes:

```text
Patient Registry
→ active protected patient
→ Επισκέψεις
   → Νέα επίσκεψη
   → open existing draft/completed/amended visit
```

Remove from normal presentation:

```text
PILOT CASE n/5
PILOT-nnn as displayed encounter identity
Νέο Case
Cases
pilot-only privacy copy
pilot-specific Finish messages
manual first-baseline-sample question as required progress item
```

Methodological pilot/baseline cohort metadata is separate from normal visit identity. Retiring visible pilot semantics does not delete historical payloads or retroactively relabel prior research/audit history.

The existing `pilot-completion.js` file may remain as a compatibility filename in v1, but it must continue to be the **single authoritative Finish listener** and its clinician-facing behavior becomes generic encounter completion.

---

# 8. UI contract

Top workflow should make three states obvious:

```text
active patient
active visit date/status
sync state: saved / saving / conflict / unavailable
```

Required conflict presentation:

- explicit text; not color-only;
- explains that the encounter changed on another device;
- states automatic overwrite was blocked;
- offers explicit server reload;
- does not silently discard local changes before the clinician chooses reload.

No claim of whole-service GDPR compliance is introduced.

---

# 9. Acceptance tests

## Server/API

1. create draft returns version token;
2. matching expected token permits update;
3. stale expected token returns 409;
4. stale write leaves newer payload/status intact;
5. completed/amended lifecycle remains unchanged;
6. legacy no-token request remains backward-compatible unless later deliberately removed.

## Browser/workspace

7. new visit without active protected patient is blocked;
8. new visit with active patient creates server draft immediately;
9. load encounter hydrates from server and stores current version token;
10. autosave sends expected token and refreshes it after success;
11. same-browser writes are serialized;
12. 409 surfaces explicit conflict and suppresses further autosave;
13. explicit reload hydrates server state and clears conflict;
14. visible shell has no `PILOT CASE`, `PILOT-nnn`, `Νέο Case`, local `Cases` workflow or pilot Finish messaging;
15. authoritative Finish still runs exactly once and produces completed/amended server state.

## Inherited gates

- G-3 salience/summary regressions;
- G-2 frozen contract/runtime regressions;
- G-1 progressive guidance regressions;
- C1 browser Finish/server finalization regressions.

---

# 10. Out of scope

- G-2 rule/evidence changes;
- PR-1 transcript extraction;
- PR-2 inline candidate review;
- Practice Review;
- formal 5-case validation collection;
- independent simultaneous lab-edit merge/conflict resolution;
- offline-first multi-device merge;
- full identity-provider redesign;
- retention/GDPR certification claims;
- physiotherapy/RF work.

---

# 11. REPLAN triggers

Stop/replan if:

- complete encounter state cannot be persisted safely through the existing protected encounter payload;
- existing module Save listeners cannot be composed before autosave without losing module-owned slices;
- optimistic concurrency requires a DB migration rather than the existing `updated_at` token;
- C2 would require a second Finish owner;
- G-3 current-visit/history separation breaks under server-authoritative identity.

Current seam inspection has found none of these blockers.

---

# 12. Stop gate

Authorized work may proceed through:

```text
IMPLEMENTED
→ TESTED
→ exact-head review
```

STOP before C2 PR/merge/deploy unless separately authorized by the product owner. G-3 PR #70 remains independently merge-held.
