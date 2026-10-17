# Cockpit #164 — A5 one-transaction migration + runtime grants, independent affected R2

**Date:** 2026-10-17 Asia/Nicosia. **Mode:** fresh independent **READ-ONLY** one bounded post-code R2; STOP after handback. No modification, merge, deploy, Aiven login, secret viewing, patient reads/writes or actual migration execution.

Repository `athpapachr-cmd/osteoporosis`. Parent #161/#163 code has already been merged in `main=b20458fa37bee8ce1a517887f81d5637f680cf7a`, Render deployment `dep-db9qocnlk1mc73c7n4q0` LIVE. The new feature remains OFF pending activation gates.

Child **DRAFT/HOLD PR #164** adds **only** an additive PostgreSQL migration and synthetic qualification on top of the merged parent. Exact **CODE+TEST tested HEAD** `32c47c2dae40ed481cf11b2ec2f8c2926c63aadd`. A later review-request-only commit does not change schema/test source; confirm precise blob identity.

## Affected source
- `cockpit/migrations/20261017_visit_brief_staff_access.sql` — already reviewed empty-table DDL originating in #163.
- `cockpit/migrations/20261017_visit_brief_cockpit_app_grants.sql` — gated role/schema/PK check and least-privilege grants.
- `cockpit/migrations/20261017_visit_brief_atomic_clinic_install.sql` — **one BEGIN/COMMIT** combining previously reviewed DDL and the new grants.
- `test_visit_brief_a5_postgres.py` — disposable PostgreSQL 17 positive and negative DML, exact schema re-run, schema drift rollback test.
- `.github/workflows/cockpit-a5-postgres-grants.yml` — no Aiven/Render secrets; synthetic-only service.
- `cockpit/CURRENT.md` — scope and release hold.

## Only four material questions

1. **Atomic and metadata-only:** first create *three empty* Visit Brief tables and indexes, then verify precise columns/types/nullability/PK and grant. Failure / altered schema must roll back the **whole transaction**. No patient rows, no migration on boot, no automatic link deletion. Check a typo in migration cannot commit partial grants or partial DDL.
2. **Least-privilege compatibility:** `cockpit_app` has no public schema CREATE; linkage requires SELECT/INSERT/UPDATE but **no DELETE**, link and staff audit need SELECT/INSERT **without UPDATE/DELETE**. Staff audit is clinician-readable only via existing server authority. Confirm exact table definitions and `sqlalchemy` ORM requirements match.
3. **Preflight quality and rollback:** correct DB name `osteoporosis`, correct role (not `cockpit_app` admin), no elevated role, no schema CREATE, exact three tables; no silent IF NOT EXISTS drift. Idempotent rerun and malformed schema fail closed. Explicit 3-second lock wait and 20-second statement timeout.
4. **Synthetic evidence and regressions:** `Cockpit A5 PostgreSQL runtime grants` [39053026123](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39053026123) SUCCESS, `Canonical Impact` [39053026117](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39053026117) SUCCESS, `Cockpit surgery queue` [39053026109](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39053026109) SUCCESS. Previous Visit Brief independent R2 PASS and all unaffected #161/#163 tests inherited; **do not rerun them**.

## Release / activation separation

A **PASS** here establishes only migration+grant code readiness. It **does not authorize** running SQL on Aiven, switching protected flags on, enabling Secretary access, introducing machine-to-machine access, or using clinical PHI. Those require a separately authorised clinic operator action with current backup/rollback and attested staff authentication, access policy, retention and terms. Staff MFA/rate limits / private role boundary remain separate A5 gates. The intentionally disabled live feature remains so after any code merge.

## Handback

Return exactly PASS/BLOCK/UNKNOWN, COMPLETE_FOR_DECLARED_SCOPE/PARTIAL, P0:P1:P2, evidence with GitHub file lines, and only the smallest necessary next action. STOP READ-ONLY.
