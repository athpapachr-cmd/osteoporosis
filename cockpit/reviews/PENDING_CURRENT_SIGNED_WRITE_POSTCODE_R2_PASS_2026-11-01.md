# Pending current-signed writes — independent exact-code R2 PASS

User authorized the approved P2 implementation, targeted synthetic PostgreSQL/SQLite tests and ONE independent final-code review. Merge/deploy remain unauthorized.

- Exact reviewed implementation/tests commit: `02755139b1f2d307909baf471a58d528a1f06936`.
- Source base/main: `01f0f4bf07fdcd00cdbb08eafe51bbcb40c2e21d`.
- Approved design: `c975960af6f40a2e143c40fa6aeb9f2dbe3dfd3d`, with independent design closure PASS.
- Reviewer: the independent delegated R2 reviewer, not the implementation author.
- **PASS; P0:P1:P2=0:0:0.**
- **ALL_MATERIAL_FINDINGS_CLOSED=YES; COMPLETE_FOR_DECLARED_SCOPE=YES.**
- Smallest required correction: none in the declared examined scope. Review chain CLOSED.

## Exact-code findings and coverage

The reviewer verified current signed payload/unique membership validation at `clinical_pending_one_tap.py:136–157`, without comparing mutable metadata against original signed values.

`clinical_pending_one_tap.py:223–254` reserves SQLite write intent before reads and maps specific lock failures to rollback/409. PostgreSQL locks Encounter -> Pending. Discovery returns scalar identity; locked ORM rows use populate_existing and identity revalidation. The existing amendment POST reuses this same transaction entry at `clinical_visit_amendments.py:87–94`. This is necessary for the approved SQLite transaction contract; it adds no clinical owner or nested transaction to cancel_pending.

At `clinical_pending_one_tap.py:274–303`, metadata exact-event replay precedes mutable category eligibility, retains patient/Pending/actor/kind/key/after-values/expected_revision+1 identity, returns CURRENT values without restoring A and preserves no-event/no-op behavior. The original metadata replay design finding is implemented and closed for this candidate.

At `clinical_pending_one_tap.py:333–335,360–362,388–395`, valid replay precedes membership guard for NEW plan/initial observation, checkpoint/retraction and restore. Auth/ownership/deletion policies and historical cancel/permanent-delete remain.

## Evidence examined independently

The reviewer inspected test code, the implementation evidence receipt and actual local JUnit: targeted **111 PASS / 1 expected SKIP**, zero failures/errors; browser **7 PASS**. Existing API21 PASS is separately recorded by the coordinator.

- `test_pending_signed_write_integrity.py:162–187`: A/B/A exact metadata replay, current B projection and unchanged state.
- `:258–320`: real independent connections/SQL barriers in both amendment/checkpoint orderings, with removed and unchanged membership.
- `:323–348`: actual acquisition lock conflicts and actual SQLite reader-held commit conflict.
- `:351–368`: explicitly labelled deterministic write-phase driver error injection; not misrepresented as a naturally occurring race.

The suite covers malformed/foreign/unsigned/orphan membership, lawful metadata progression, undo/restore, cancelled/deleted replay and historical deletion without reactivation. No new material owner/schema/auth/API/provider change or technical-contract deviation was found.

Local PostgreSQL evidence is version18.6. It does not certify CI PostgreSQL17 or production behavior. The existing CI workflow supplies independent ephemeral PostgreSQL17+SQLite execution; exact current CI state is on DRAFT PR #277 and must be checked separately.

## Handback and STOP

PR: https://github.com/athpapachr-cmd/osteoporosis/pull/277 (OPEN/DRAFT, unmerged). Later checkpoint/receipt commits change documentation only; code/test bytes remain the exact reviewed candidate. No new review is required for these disposition-only updates.

The reviewer performed source/evidence inspection only, with no edits, new test execution, publication, live database/provider/PHI, merge or deploy. The coordinator keeps the candidate in release HOLD. Implementation is source-tested/reviewed; production still runs the prior release and the P2 fix is not deployed. Next step requires a separate explicit release decision after all applicable final-head CI. **STOP before merge/deploy.**
