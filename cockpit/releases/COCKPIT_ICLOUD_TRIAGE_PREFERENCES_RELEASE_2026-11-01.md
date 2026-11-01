# iCloud sorting preferences — authorized release checkpoint

Status: PREFLIGHT BLOCKED BY OPERATOR NETWORK ACCESS. No provisioning, merge or deploy.

Product Owner approved one operational-preference table in the existing database, PR279 merge, native deploy and no-patient-record-write smoke. Independent design and final-code R2 are closed PASS; no new review chain.

- Fresh source main: `b8637bcc8af0deebee3716cf7cd98bc99bf2e912`.
- PR279 frozen HEAD: `9fc712273db157837b5b19344e27add9174b0a14`, OPEN/DRAFT/MERGEABLE; all21 raw check runs SUCCESS on release preflight (no pending/failure).
- Corrected substantive code/tests: `c39f00cece7905f27f4f17bd0a466d673e0721f0`, independent closure0:0:0, complete declared scope and all material findings closed.
- Existing Render workspace `tea-d5qfh5vpm1nc738pvbv0`, service `srv-d5qfk31r0fns73di596g`, osteoporosis, main/on-commit autoDeploy. Current live `dep-dbjeohjncjis73dgmvv0` / source `7956aceb2e0a436b4289bfeb161406748bef99f3`.
- Existing Render configuration was read through authenticated Render API solely to qualify DATABASE_URL destination (secret values not printed or persisted): existing Aiven PostgreSQL service `pg-3c0d5bd0`, database `osteoporosis`, application role `cockpit_app`. No configuration changed.
- Aiven Console after Product Owner login confirms service Running, PostgreSQL17.11, network allowlist Restricted. Latest backup displayed22 hours ago, oldest15days,742MB stored. This is backup inventory evidence, not a restore test.
- Local read-only metadata connection with the existing application credentials timed out; credentials were passed only in memory and not printed/persisted. Render native SSH closed before command output. No network restriction bypass or new service/ephemeral instance requested.
- Aiven SQL editor selected osteoporosis, but metadata-only SELECT was refused by the current operator IP allowlist. No query result, current schema, table absence or role privilege proof was obtained. No DDL/grant or clinical-table access executed.

Next: separately confirm a temporary single-address network exception, retaining all existing entries, if the Product Owner chooses this path. Qualify exact DB/schema/role/table absence; then execute the reviewed one-table DDL transaction and only SELECT/INSERT/UPDATE for the existing application role. Verify no schema CREATE/DELETE grant; remove temporary network exception. Reverify exact PR HEAD/all CI, merge once, checkpoint native exact-source deploy and bounded smoke. Production category/rule writes need an approved synthetic source fixture or the clinician's deliberate operation; no automatic experiment on real messages.

Release coordinator owns only the bounded release checkpoints on `docs/icloud-triage-release-20261101`, outside frozen PR279. Runtime/tests unchanged. No merge/deploy claim.

## Temporary network exception

Product Owner explicitly approved a single operator IPv6 /128 entry at action time. Saved in Aiven with a TEMP PR279/removal comment, preserving all three existing allowlist entries. Must remove immediately after provisioning verification or if provisioning cannot complete. This temporary configuration change is distinct from the as-yet unexecuted DDL/merge/deploy.

## Table creation checkpoint

Metadata confirmed osteoporosis/avnadmin/public, table absent, existing cockpit_app, schema USAGE=true/CREATE=false, owner default-table-ACL count0. SQL editor split a DO block at semicolons and rejected it with an unterminated dollar quote; post-error metadata still showed table absent. Executed only the exact reviewed standalone schema-qualified CREATE TABLE. Verification: table exists, owner avnadmin,0rows, app SELECT=false. No clinical table inspected or changed. Grant still pending.

## Minimum grant verification

Standalone GRANT SELECT, INSERT, UPDATE applied only to public.icloud_triage_preferences for existing cockpit_app. Metadata result: SELECT=true, INSERT=true, UPDATE=true, DELETE=false, schema CREATE=false, preference rows0, columns4. No wider grant or clinical change. Temporary operator network exception cleanup is next.

## Provisioning and network cleanup complete

Verified primary key(mailbox_key), bounded positive revision CHECK and JSON-object CHECK. Removed only TEMP PR279 IPv6 /128 entry and saved; Aiven visibly shows exactly the three original allowlist entries. No temporary access remains. Four-column table remains empty; no category/rule row or patient record written. Provisioning complete; PR merge/deploy still not executed.

## Frozen release gate

PR279 marked ready; release authority/provisioning reflected in its body. Fresh exact HEAD9fc712273db157837b5b19344e27add9174b0a14, READY/OPEN/MERGEABLE; all23 check runs SUCCESS after ready/body events. None pending. Next SHA-bound squash merge.

## Merge checkpoint

SHA-bound squash merge succeeded: `445c5ee2f7af8f085a4316c5ddfff5fe96767927`, frozen PR HEAD `9fc712273db157837b5b19344e27add9174b0a14`. All23 final check runs SUCCESS. Native deploy and smoke still pending; no duplicate manual deploy.

## Native deploy identified

Auto-deploy `dep-dbjhivtg1s2s73fvbcg0`, created2026-11-01T10:50:39Z, source `445c5ee2f7af8f085a4316c5ddfff5fe96767927`, trigger new_commit, build_in_progress. Full source tree equals frozen PR279 tree `9b8c6b4dfe400a610eadc0c2747aadd225e35ad3`. No manual deploy.
