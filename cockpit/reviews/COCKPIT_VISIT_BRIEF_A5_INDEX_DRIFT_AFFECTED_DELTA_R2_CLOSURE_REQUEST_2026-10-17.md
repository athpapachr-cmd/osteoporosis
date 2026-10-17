# Cockpit #164 — A5 PostgreSQL index-drift correction, fresh independent DELTA + AFFECTED R2 CLOSURE

**Date:** 2026-10-17 Asia/Nicosia. **Mode:** independent **READ-ONLY** affected R2 closure. STOP after verdict. Do not edit code, SQL, docs, review/request, canonical, PR, Aiven/Render, settings or real patient data; no merge/deploy/activation/migration.

Repository: `athpapachr-cmd/osteoporosis`
PR: [#164](https://github.com/athpapachr-cmd/osteoporosis/pull/164) — **DRAFT/HOLD** on `main=b20458fa37bee8ce1a517887f81d5637f680cf7a`.
Previous reviewed CODE+TEST HEAD: `32c47c2dae40ed481cf11b2ec2f8c2926c63aadd`.
Independent R2 handback: **BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:1:1**, findings P1-01 missing audit-index drift validation and P2-01 missing atomic-install path in CI filters. Previous unaffected migration, role, ORM, security and parent #161/#163 review PASS are inherited.
Corrected immutable **CODE+TEST HEAD**: `c54de9bca14e7aabbf74aee1f1e07b37e271a295`. Later review-request/canonical-only commits are not code changes; verify blobs.

## Exact corrected blobs

- `cockpit/migrations/20261017_visit_brief_atomic_clinic_install.sql` — `4887a86254aea5f8089c5c8b57174fd4000766f3`
- `cockpit/migrations/20261017_visit_brief_cockpit_app_grants.sql` — `c961e6d56221afb10e2e7f4705d90eeec7735aef`
- `test_visit_brief_a5_postgres.py` — `af393e899b4c8973e456f96ca551bf759f7a04bc`
- `.github/workflows/cockpit-a5-postgres-grants.yml` — `3fdce1a08b0a898dd61ed52580341ac30dd34611`
- Unchanged prior DDL `cockpit/migrations/20261017_visit_brief_staff_access.sql` — `d9115878ea939d737639e966166d5bea0f4239f8`.

## Exact-head author evidence (not independent acceptance)

- [PostgreSQL 17 PR run 39055597435](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39055597435): SUCCESS.
- [PostgreSQL 17 push run 39055592686](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39055592686): SUCCESS.
- [Canonical Impact run 39055597295](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39055597295): SUCCESS.
- [Cockpit surgery queue run 39055597262](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39055597262): SUCCESS.
All runs refer to exact code HEAD `c54de9bca14e7aabbf74aee1f1e07b37e271a295`; synthetic-only ephemeral PostgreSQL 17, no real patients, no Aiven/Render credentials.

## Exactly four bounded review questions

1. **P1-01 closure:** in *both* standalone grants and one-transaction installer, ensure both named audit indexes are verified for existence, correct table, btree method, one exact key, validity/ready/live, no partial/expression/INCLUDE/unique/unexpected options; same-name wrong-key or missing index **RAISE EXCEPTION and rollback** despite `CREATE INDEX IF NOT EXISTS`. Check structural query cannot silently accept a wrong definition.
2. **P2-01 closure:** workflow path filters on *both* push and pull_request include `20261017_visit_brief_atomic_clinic_install.sql`, not just DDL, grants, tests and workflow.
3. **Synthetic failure oracle:** check disposable PostgreSQL test first injects a malformed late GRANT into atomic installer and verifies no new tables remain after rollback; then confirms successful exact re-run plus positive/negative role DML; tests wrong named-index and missing-index fail closed, no partial grants, previous altered-column failure preserved.
4. **Cumulative affected safety:** three tables and 2 indexes remain additive, one BEGIN/COMMIT, no patient rows, no automatic startup or deletion, preflight locks/timeouts/database/role, ORM-compatible types and least privilege unchanged. Reuse previous independent #161/#163 and prior A5 unaffected PASS; do not rerun them.

## Verdict and STOP

Return **PASS/BLOCK/UNKNOWN**, **COMPLETE_FOR_DECLARED_SCOPE/PARTIAL**, **P0:P1:P2**, finding closure and line-specific GitHub evidence, only the minimal necessary next action. **PASS is code/test readiness only**, never permission to run SQL in Aiven, merge/deploy, turn on Visit Brief/Secretary, use PHI, or bypass outstanding staff MFA/rate-limit, data-access, clinic-specific retention and operator release gates. Remain DRAFT/HOLD until separate explicit clinic owner decision. Do not expand into another review loop.
