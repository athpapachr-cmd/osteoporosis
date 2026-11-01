# Pending-write P2 — targeted implementation evidence

Status: implemented and locally tested candidate; independent exact-code R2 pending. No merge/deploy or real patient/provider/production DB access.

## Source and bounded delta

- Fresh source base/main: `01f0f4bf07fdcd00cdbb08eafe51bbcb40c2e21d`.
- Approved design: `c975960af6f40a2e143c40fa6aeb9f2dbe3dfd3d`; independent design closure PASS0:0:0 recorded separately.
- Runtime changes: existing `clinical_pending_one_tap.py` and the existing amendment owner's session entry in `clinical_visit_amendments.py` only. No frontend, API payload, auth, model, production schema or provider change.
- Shared write loader locks Encounter then Pending and refreshes authoritative rows. SQLite reserves its writer before Session reads. Acquisition/write/commit lock failures roll back to409; unrelated operational errors are not falsely classified as conflicts.
- Current membership guards NEW plan/first-observation, checkpoint/reopening retraction and restore, after valid historical replay. Audited mutable metadata need not equal signed originals. Existing historical cancel/delete/terminal policies remain.
- Metadata exact-event replay occurs before mutable category eligibility and returns CURRENT metadata, not old request values.

## Why the amendment session entry also reuses the transaction helper

A first candidate reserving SQLite writes only in checkpoint endpoints failed the deterministic two-connection tests. With checkpoint first, an amendment could read the previous token before waiting for its first write and then return200 with stale authorization; with amendment first, its initial reads held no SQLite write-intent reservation and the contender bypassed the expected serialization.

The smallest correction is reuse of the SAME existing-owner `pending_write_session` at the amendment POST transaction entry. This reserves SQLite writes before amendment reads as well; PostgreSQL keeps its existing source-first row locking. `cancel_pending` remains inside its caller's transaction, with no nested BEGIN. These are implementation details satisfying the approved transaction contract; no new clinical behavior or owner was added. The same two-order tests now pass on both databases.

## Actual local evidence

Environment: Python3.12.2, SQLAlchemy2.1.6, SQLite3.54.0, disposable loopback PostgreSQL18.6 (Homebrew). All data and schemas were newly created synthetic test fixtures. PostgreSQL uses a new isolated schema per case, with teardown restricted to that schema; the test URL guard requires loopback and a named disposable test database. No real DB service/config/grants were used.

- `pytest -q test_pending_signed_write_integrity.py`, with P2_TEST_POSTGRES_URL pointing to the disposable local database: **111 PASS, 1 SKIP**, 28.13s. SQLite56/56 PASS; PostgreSQL55 PASS and one SQLite-only commit-conflict case not applicable. No PostgreSQL case was skipped for absent access.
- `pytest -q test_cockpit_cp2.py test_cockpit_visit_amendments.py -k 'not browser and not ui'`: **21 PASS**, 8.15s; the seven complementary browser tests were selected separately.
- `pytest -q test_cockpit_cp2.py test_cockpit_visit_amendments.py -k 'browser or ui'`: **7 PASS**, 22.74s. Actual synthetic Chromium; no provider calls or production access.
- Python AST and workflow YAML syntax PASS; diff hygiene PASS.

The new suite covers malformed/missing/foreign/unsigned/ambiguous/removed membership for plan with first observation, checkpoint, retraction and restore; actual amendment/removal; source-preserving metadata progression; legacy-category A/B/A exact replay, conflicting request/actor and new unsupported category; no-op, cancelled/deleted replay boundaries; orphan historical cancel/delete; both amendment/checkpoint execution orders with removed AND unchanged members; no partial mutations on conflict.

Concurrency uses actual distinct database connections and real SQL barriers, not mocked DB responses. It observes the contender attempting the lock and not completing it while the holder owns the transaction. PostgreSQL and SQLite acquisition conflicts are actual held locks. SQLite commit conflict uses an actual held read transaction in rollback-journal mode. Write-phase rollback uses explicitly labelled deterministic driver lock-error injection after write processing begins; this is not presented as a real provider or production race.

Local JUnit evidence: `/tmp/cockpit-p2-evidence.CCXSd7/targeted.xml`, `/tmp/cockpit-p2-evidence.CCXSd7/browser.xml`. These paths are local evidence, not deployment artifacts.

## Repeatability and next gate

The existing Visit Capture workflow includes the new test file in its triggers and runs the targeted suite against an ephemeral PostgreSQL17 service plus SQLite. This is CI-only synthetic infrastructure, not a production table/database installation. CI PostgreSQL17 results must be reported separately from the local PostgreSQL18.6 result.

Next: freeze exact code/tests, ONE independent final-code R2 fidelity review of this approved scope and its evidence. A PASS does not authorize merge/deploy. Real clinical/provider/production validation remains NOT_VERIFIED.
