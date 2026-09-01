# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-3 PR #70 OPEN / MERGE HOLD; C2 IMPLEMENTED / TESTED / RELEASE REVIEW IN PROGRESS.
> **Updated:** 2026-09-01 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `9cfad82d1258a44e71080e0aa4d6d644e581cfbf`.
> **G-3 predecessor:** PR #70 — OPEN, mergeable, unmerged, base `main`, head `f12b32d06b2c48e4566c2653c437c86ac6f55e7f`, explicit MERGE HOLD.
> **C2 branch:** `feat/module01-c2-server-authoritative-patient-workspace-2026-09-01`.
> **C2 exact tested runtime head:** `27d5248e51daaa61bf29c70d3bebf5d73b93d6a2`.
> **C2 test workflow:** `C2 server-authoritative patient workspace` run `33544223692` — SUCCESS.
> **ACTIVE CANONICAL WRITER/LOCK:** this session — C2 closeout/release-readiness docs only.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.

---

# 1. Proven production base

C1 / G-1 / G-2 remain:

```text
IMPLEMENTED
TESTED
MERGED
DEPLOYED
PRODUCTION-SMOKE-VERIFIED
```

Fresh production `main` remains the G-2 release SHA:

```text
9cfad82d1258a44e71080e0aa4d6d644e581cfbf
```

G-3 is not production state. PR #70 is open and mergeable but remains explicitly merge-held.

---

# 2. G-3 predecessor state

```text
G-3 DESIGN                           COMPLETE
G-3 IMPLEMENTED                      YES
G-3 TESTED                           YES
G-3 RELEASE-READINESS REVIEW         PASS
G-3 PR                               #70 OPEN
G-3 MERGED                           NO
G-3 DEPLOYED                         NO
G-3 PRODUCTION-SMOKE-VERIFIED        NO
```

C2 intentionally starts from the exact G-3 PR head `f12b32d...`; therefore C2 must not be released ahead of G-3.

---

# 3. C2 proven implementation state

Slice:

```text
M01-C2-SERVER-AUTHORITATIVE-PATIENT-WORKSPACE-v1
```

Proven at exact tested runtime head `27d5248e51daaa61bf29c70d3bebf5d73b93d6a2`:

- protected `patient_id + encounter_id + updated_at` authoritative encounter identity/version after server creation/load;
- immediate protected server draft creation for a new visit under an active protected patient;
- debounced serialized autosave;
- server hydration on encounter open/reload from another device;
- optimistic concurrency with stale write → HTTP 409 and no silent overwrite;
- explicit conflict state with clinician-triggered server reload;
- retry de-duplication by stable payload `internal_uuid` within the active patient if a create response is lost;
- clinician-facing `PILOT CASE` / local `Cases` workflow retired in favor of protected patients and `Επισκέψεις`;
- generic encounter completion semantics while preserving the single authoritative Finish owner;
- no G-2 evidence/rule change and no DB migration.

Cross-device continuity in C2 v1 means server autosave + later open/reload from another device. It is not realtime push/websocket mirroring between already-open tabs.

---

# 4. C2 test / review evidence

Workflow:

```text
C2 server-authoritative patient workspace
run: 33544223692
job: 99977647943
head: 27d5248e51daaa61bf29c70d3bebf5d73b93d6a2
result: SUCCESS
```

Passed:

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

The exact tested runtime delta from the required G-3 predecessor is ahead-only with merge base equal to `f12b32d...` and contains only the bounded C2 workspace/runtime/test/canonical files. No PR-1/PR-2, physiotherapy/RF, or osteoporosis evidence-rule leakage was found.

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

Broader authentication/authorization review, access audit trail, retention/data-minimization and GDPR/privacy work remain production-readiness concerns. C2 does not create a whole-service compliance claim.

---

# 6. Current authority / exact next action

The product owner authorized continuation after C2 implementation/test completion. This authorizes canonical closeout and fresh release-readiness review; it does **not** by itself authorize merge or deploy.

Current sequence:

```text
finish C2 canonical closeout
→ exact-head C2 release-readiness review
→ preserve G-3 → C2 release ordering
→ STOP before merge/deploy without separate explicit authority
```

If C2 release-readiness passes while G-3 remains unmerged, retain predecessor HOLD. Do not silently bypass G-3 by releasing C2 directly to `main`.

No runtime mutation is authorized in this closeout/review step unless a release blocker is discovered that requires a new bounded implementation decision.
