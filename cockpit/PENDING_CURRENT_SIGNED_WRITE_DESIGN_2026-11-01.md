# Pending current-signed writes — approved design with metadata replay correction

## Authority and frozen review target

Source base: `01f0f4bf07fdcd00cdbb08eafe51bbcb40c2e21d` (fresh remote main, 2026-11-01).

The Product Owner approved the separate P2 design and then explicitly approved the small metadata-replay correction identified by the independent pre-code R2 review. Current authority is design integration and ONE independent delta plus affected cumulative closure review only. No runtime implementation, new test execution, real-provider/PHI/database access, PR publication, merge or deploy is authorized by this document.

This is separate from released PR #275. Its navigation-only PASS and production release do not close the existing checkpoint-write P2. No runtime writer is active. This local design branch is not a release candidate.

## Problem and intended behavior

The existing `_signed_source` proves the source encounter is signed but does not prove the Pending still belongs to its current signed candidate. An amendment cancels removed/replaced tasks, but existing restore and subsequent checkpoint actions do not verify current membership. This is a source-proven reachable sequence, not a new executed write experiment.

For example, after an amendment removes an MRI obligation, trying to restore it or record further progress must return 409 without another event or status change. The original clinician-owned task, signed visit and history remain the sole owners; current members retain their legitimate checkpoint, metadata and lifecycle behavior.

## Existing-owner scope

Extend `clinical_pending_one_tap.py`; reuse `EncounterORM`, `ClinicalPendingORM` and `PendingCheckpointEventORM`. No new table, migration, grant, identity/authentication, API field, email association, provider action or clinical writer. Reuse existing checkpoint/amendment test owners and add only the targeted concurrency evidence needed for this seam in a future implementation.

Do not change signed payloads to authorize a checkpoint. Do not use `clinical_visit_amendments.source_item` for current eligibility: it intentionally searches previous signed versions for historical provenance.

## Transaction and lock contract

1. Retain existing protected-route authentication, actor checks, patient/source ownership and terminal-deletion policies.
2. In the shared write loader, discover source encounter ID without treating that preliminary read as authorization. Avoid retaining a preliminary ORM object as authoritative state.
3. On PostgreSQL, acquire the encounter row lock FIRST, then the Pending row lock; hold both through event/status commit or rollback. This matches the amendment owner's Encounter -> Pending ordering. Re-read/refresh and validate both objects after locking, including exact patient, Pending and source-encounter identity. Do not authorize from the ORM identity map's stale values. The shared loader covers metadata and lifecycle callers as well as the checkpoint endpoints.
4. For SQLite, acquire an explicit write-intent transaction, such as BEGIN IMMEDIATE, before any Session read. A concurrent writer must either finish before eligibility is read or cause a rollback/409. Handle relevant lock conflicts at acquisition, write and commit; no blind retry and no partial event/status write. Do not claim SQLite SELECT FOR UPDATE provides PostgreSQL row-lock evidence.
5. `cancel_pending`, when invoked inside an amendment, reuses that caller's transaction and locks. It must not start a nested transaction or invert lock order.

## Current-membership contract

A common helper in the existing owner validates the current signed payload, not its history: same patient and source encounter, completed/amended encounter, valid signed flag and signed_at, structurally valid `pending_ids` and candidate pending lists of equal length, and exactly one matching Pending ID with its corresponding candidate item. Malformed or ambiguous membership fails closed with 409.

Membership is based on identity. Do not require mutable current `row.item_type` or `row.review_date` to equal their original signed values: existing audited metadata corrections legitimately change them without rewriting the visit. Existing revision/status/plan/checkpoint validity checks remain in force.

After any valid replay return, enforce membership before creating a NEW event or changing status in:

- checkpoint-plan, including its optional first observation;
- checkpoint, including a retraction that can reopen a task;
- lifecycle restore, because it reactivates a task.

Current-member undo/restore remains available. Removed/replaced-member progress or restore returns 409 with no mutation. Historical cancel and permanent-delete remain available under their existing rules. Metadata's existing new-event eligibility is preserved; this bounded change does not invent an additional metadata membership policy.

## Replay ordering and the approved P2-01 correction

Do not move replay handling ahead of authentication, source ownership or existing terminal-deletion policy. In particular, preserve metadata/checkpoint refusal for permanently deleted tasks and the special exact permanent-delete replay supported by lifecycle.

For checkpoint-plan/checkpoint/restore, a valid existing replay returns before the NEW membership guard. It reports the existing event/current projection and never creates an event, changes metadata/status or reactivates a task. Existing replay identity checks remain authoritative.

For metadata specifically, the corrected order is:

1. Complete existing authentication, locked loading, signed-source/ownership and terminal-deletion checks.
2. Look up the existing event by idempotency key; derive the proposed `desired` metadata from the already validated request.
3. If an event exists, verify exact patient, Pending, actor principal/role, `metadata_corrected` kind, metadata checkpoint key, stored after-values, and `existing.revision == action.expected_revision + 1`. A mismatch remains 409. A match returns `replayed=true` with the CURRENT projection, without writing or restoring the old values.
4. Only if no event exists, apply category eligibility dependent on the CURRENT row, including the allowlist and legacy-category retention rule; then the existing current status, expected revision and no-op checks, followed by normal event creation/commit.

This explicitly moves the existing current-category eligibility check after existing-event replay handling. It is a design correction, not a claim that current source already behaves this way. Static request-shape/date validation stays in place.

The no-event no-op response remains unchanged; it is not invented into an immutable event or given a new replay guarantee.

### Required regression for P2-01

Given a current signed Pending with a valid legacy item_type outside the metadata chooser's allowlist:

1. Request A changes only the date while retaining that legacy category and commits a metadata event.
2. Request B legitimately changes the category to `result` and commits.
3. Retry the EXACT A: same key, original expected revision and body.
4. Expect `replayed=true`, A's event ID and the CURRENT B projection. Assert event count, B's category/date/status, lifecycle revision and signed payload are unchanged by the retry.

Also assert a same-key changed body/expected revision/actor or foreign patient/Pending cannot replay; a genuinely new unsupported category still fails 422. Preserve existing deleted/cancelled replay boundaries.

## Finite future evidence and release boundary

Future implementation evidence must establish:

- missing/foreign/unsigned/malformed/duplicate/removed/replaced current membership rejects NEW progress and restore with no state/event changes;
- unchanged Pending membership across an amendment remains eligible;
- metadata -> plan -> metadata -> checkpoint uses shared revisions while leaving the signed payload unchanged;
- the P2-01 A/B/A exact-replay sequence above, plus conflicting keys and preserved terminal-deletion policies;
- legitimate current-member undo/cancel/restore and historical cancel/permanent-delete remain supported;
- PostgreSQL two-session amendment/checkpoint in BOTH execution orders: action before amendment serializes safely; amendment before action denies removed membership or permits an unchanged member under existing revision rules. No deadlock, stale-source authorization or partial write;
- SQLite BOTH execution orders, including acquisition/write/commit conflict rollback; sequential SQLite tests alone do not prove concurrency safety.

No tests or database experiments are run for this pre-code review. Runtime race protection remains unverified until implemented and covered by the targeted evidence. Review classification is R2 because the future change affects clinical writes. A closure PASS closes only this design gate; implementation requires separate explicit authority and later exact-code fidelity review. Merge/deploy remain separate decisions.

## Independent review disposition

Original independent pre-code review: BLOCK, P0:P1:P2=0:0:1. P2-01 was current-category validation before an immutable metadata replay. The Product Owner approved the correction above. One independent delta plus affected cumulative closure review is pending on this frozen revised document. Do not begin implementation from this pending state.
