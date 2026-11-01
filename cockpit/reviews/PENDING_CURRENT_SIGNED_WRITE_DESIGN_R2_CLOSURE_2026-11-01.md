# Independent Pending-write R2 design closure — PASS

Date: 2026-11-01. User authorized integration of the metadata replay design correction and ONE independent delta plus affected cumulative closure review only.

- Frozen reviewed design commit: `c975960af6f40a2e143c40fa6aeb9f2dbe3dfd3d`.
- Source base: `01f0f4bf07fdcd00cdbb08eafe51bbcb40c2e21d`.
- Design: `cockpit/PENDING_CURRENT_SIGNED_WRITE_DESIGN_2026-11-01.md`.
- Reviewer: the independent delegated reviewer from the original R2, separate from the design author.
- Verdict: **PASS; P0:P1:P2=0:0:0**.
- **ALL_MATERIAL_FINDINGS_CLOSED=YES; COMPLETE_FOR_DECLARED_SCOPE=YES**.

## Original finding and closure

The original review BLOCK P2-01 identified a source-proven metadata replay failure: date-only request A retains a legacy category, later request B changes the category, and exact replay A is rejected by mutable current-category validation before its already committed event is examined.

Frozen design lines53–60 now explicitly place exact metadata event replay after existing auth, locked ownership, signed-source and terminal-deletion guards but before mutable current-category eligibility. Existing event/patient/Pending/actor/body identity and stored revision == requested expected_revision + 1 remain. Replay returns A's event identity and the CURRENT B projection, without restoring A metadata or creating an event. No-event/no-op behavior is not given a new replay guarantee.

Lines64–71 require the concrete A/B/A regression, unchanged event count/current metadata/status/revision/signed payload, conflicting request rejection, and refusal of a genuinely new unsupported category. **P2-01 is closed at design level.**

## Affected cumulative coverage

- Lines25–29: shared PostgreSQL Encounter -> Pending order, postlock refresh/revalidation for shared callers, locks through transaction completion; SQLite write-intent before Session reads and relevant acquisition/write/commit conflicts rollback/409. Amendment cancellation reuses its caller's transaction.
- Lines33–43: current identity-based membership after a valid replay and before NEW plan/first observation, checkpoint/reopening retraction or restore. No equality requirement against mutable metadata; current-member undo/restore and historical cancel/permanent-delete retained.
- Lines47–49: existing cancelled/deleted boundaries and special permanent-delete replay preserved; replay does not bypass auth/source/deletion guards.
- Lines75–85: finite future PostgreSQL and SQLite concurrency tests in both execution orders, plus metadata/replay/current-membership/history evidence. These are requirements, not claimed test results.

The reviewed diff is documentation only: one design and three canonical pointers/checkpoints. No new material inconsistency was found in the declared affected scope. No further design correction or design review is needed for this gate.

## Stop and authority boundary

This PASS closes the design gate only. Runtime P2 remains unresolved. Implementation, targeted runtime evidence and exact-code fidelity review require separate explicit authority; merge/deploy remain separate decisions.

The reviewer performed source/design inspection only: no file changes, test execution, provider/PHI/database access, publication, merge or deploy. The coordinator recorded this receipt and disposition locally after review; no runtime code changed and no remote publication occurred. **STOP.**
