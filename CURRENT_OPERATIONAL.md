# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — C2 SERVER-AUTHORITATIVE PATIENT WORKSPACE ACTIVE.
> **Updated:** 2026-09-01 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `9cfad82d1258a44e71080e0aa4d6d644e581cfbf`.
> **Required predecessor:** G-3 PR #70 — OPEN / MERGE HOLD, head at C2 branch point `f12b32d06b2c48e4566c2653c437c86ac6f55e7f`.
> **C2 branch:** `feat/module01-c2-server-authoritative-patient-workspace-2026-09-01`.
> **ACTIVE CANONICAL WRITER/LOCK:** this session — C2 only.
> **ACTIVE RUNTIME WRITER/LOCK:** this session — C2 patient/encounter persistence, workspace shell, finalization integration and focused tests only.

---

# 1. Proven production base

C1 / G-1 / G-2 remain production-smoke-verified. G-3 is implemented/tested and has release PR #70 open; C2 is intentionally based on the exact G-3 PR head so it cannot release ahead of G-3.

No C2 merge/deploy authority is implied by implementation authority.

---

# 2. C2 product-owner requirements

The product owner works on three computers and requires:

```text
SAVE / UPDATE ON DEVICE A
→ SAME PROTECTED PATIENT + ENCOUNTER STATE AVAILABLE ON DEVICE B/C
```

The product owner also requires the clinician-facing pilot shell to be retired:

```text
PILOT CASE / local Cases
→ Protected Patient
→ server-backed Επισκέψεις
```

The formal future 5-case validation cohort may remain methodological background metadata, but it must not define ordinary encounter identity or normal clinician-facing workflow.

---

# 3. Current defects confirmed from runtime

Existing protected PostgreSQL tables already provide patient, encounter and lab persistence, but:

- browser `localStorage` still owns working-case identity/cache;
- draft server sync is primarily Save-triggered;
- a new visit is initially local and becomes server-backed only after later sync;
- encounter PUT has no optimistic-concurrency token, so stale-device last-writer-wins is possible;
- visible UI still contains `PILOT CASE`, `PILOT-nnn`, `Νέο Case`, `Cases`, pilot/baseline sampling copy and pilot Finish messages.

---

# 4. Active C2 boundary

Implement/test only:

1. `patient_id + server encounter_id` as authoritative clinical identity after server creation/load.
2. `updated_at`-based optimistic concurrency; stale update returns HTTP 409 and never overwrites silently.
3. server draft creation immediately when starting a visit for an active protected patient.
4. server hydration when opening an encounter.
5. debounced serialized autosave of meaningful encounter changes; `localStorage` remains disposable working cache only.
6. explicit cross-device conflict state + clinician-triggered reload from server; no silent discard of local edits.
7. retire visible pilot-case/local-case language in ordinary clinical workflow.
8. preserve one authoritative Finish listener and completed/amended semantics.
9. preserve G-3/G-2/G-1/C1 regressions.

No new clinical guidance rules, transcript/PR-1/PR-2, Practice Review, DB migration, physiotherapy/RF, manual deploy, or real-patient data enters the public repository.

---

# 5. Safety/data-integrity invariants

```text
SERVER ENCOUNTER != BROWSER CACHE
STALE DEVICE WRITE != SILENT OVERWRITE
CONFLICT != AUTO-MERGE
COMPLETED != REGRESS TO DRAFT
POST-COMPLETION CONTENT CHANGE = AMENDED
SCHEDULED DOSE != ACTUAL DOSE
CURRENT DRAFT != COMPLETED HISTORICAL FACT
```

If a stale-version conflict occurs, preserve local working state, block further automatic writes for that encounter, surface the conflict, and require explicit server reload/review before resuming.

---

# 6. Exact next action

```text
freeze C2 design in SLICE_PLAN_CURRENT
→ implement server concurrency contract
→ implement patient/visit workspace + autosave/conflict handling
→ genericize Finish/pilot presentation without adding a second Finish owner
→ focused C2 tests + inherited G3/G2/G1/C1 gates
→ exact-head review
→ STOP before PR/merge/deploy
```

G-3 PR #70 remains independently on MERGE HOLD.
