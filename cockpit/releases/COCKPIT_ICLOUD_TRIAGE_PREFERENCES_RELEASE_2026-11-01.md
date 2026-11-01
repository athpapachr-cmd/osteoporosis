# iCloud sorting preferences — authorized release checkpoint

Status: TECHNICAL RELEASE COMPLETE. Exact source LIVE; bounded read-only API and connected-clinician browser smoke PASS. Earlier preflight/build blockers below are historical and resolved. Production real-message Save/undo acceptance remains untested.

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

## Exact-source LIVE

Native deploy terminal LIVE2026-11-01T10:52:49.547728Z (12:52:49 Asia/Nicosia), exact source445c5ee2f7af8f085a4316c5ddfff5fe96767927, deploy dep-dbjhivtg1s2s73fvbcg0. Startup complete; no duplicate deploy. Bounded smoke next.

## Bounded read-only API smoke

{
  "started_at": "2026-11-01T10:53:43.427827+00:00",
  "source": "445c5ee2f7af8f085a4316c5ddfff5fe96767927",
  "assets": [
    {
      "path": "/static/cockpit/index.html",
      "status": 200,
      "sha256": "f97122fa644173c401961c2a867cd093ea29e3d64d371aa9dfaf5a8f067c419a",
      "matches_source": true
    },
    {
      "path": "/static/cockpit/icloud-triage.js",
      "status": 200,
      "sha256": "5577c2bfa6f9164bdb7dedb7f70643b5bb677b1e7c226588216d274aa68d24c9",
      "matches_source": true
    },
    {
      "path": "/static/cockpit/icloud-inbox.js",
      "status": 200,
      "sha256": "8b0081e8953af814a6710727c705bbe8328c083e209e77ede8084b109b65e201",
      "matches_source": true
    },
    {
      "path": "/static/cockpit/doctor-shell.js",
      "status": 200,
      "sha256": "667e26cb4f8ba5d3779b859907b360f6c8ac2185d049cf76baeacd3d720bb229",
      "matches_source": true
    },
    {
      "path": "/static/cockpit/patient-chart.js",
      "status": 200,
      "sha256": "c3e57b96a6939da8321f21b8ab34380aa0c928116225f1c057012688079cda67",
      "matches_source": true
    },
    {
      "path": "/static/cockpit/clinical-inbox.css",
      "status": 200,
      "sha256": "207eff610373f76fb6df324e06b03274fb3c2be489c4e7e084000e0cf5a52b3c",
      "matches_source": true
    }
  ],
  "anonymous": [
    {
      "path": "/clinical/status",
      "status": 401
    },
    {
      "path": "/clinical/icloud-mail/preferences",
      "status": 401
    },
    {
      "path": "/clinical/icloud-mail?view=all&limit=1",
      "status": 401
    },
    {
      "path": "/clinical/icloud-mail/peek",
      "status": 401
    },
    {
      "path": "/clinical/patients?query=SYNTHETIC-NO-PHI&limit=1",
      "status": 401
    },
    {
      "path": "/clinical/patient/SYNTHETIC-NO-PHI/pending?signed_only=true",
      "status": 401
    }
  ],
  "authenticated_read_only": [
    {
      "path": "/clinical/icloud-mail/preferences",
      "status": 200,
      "preferences_available": true,
      "revision": 0,
      "no_store": true,
      "same_scope_epoch_revision": true,
      "rule_count": 0
    },
    {
      "path": "/clinical/icloud-mail?view=all&limit=1",
      "status": 200,
      "preferences_available": true,
      "revision": 0,
      "no_store": true,
      "same_scope_epoch_revision": true,
      "item_count": 1
    },
    {
      "path": "/clinical/icloud-mail/peek",
      "status": 200,
      "preferences_available": true,
      "revision": 0,
      "no_store": true,
      "same_scope_epoch_revision": true,
      "item_count": 0
    }
  ],
  "authenticated_mutation_count": 0,
  "health": 200,
  "anonymous_preference_post_denied": true,
  "old_action_router": 404,
  "finished_at": "2026-11-01T10:54:13.939505+00:00",
  "result": "PASS_BOUNDED_READ_ONLY_RELEASE_SMOKE"
}

Only operational booleans/counts/hashes retained; all message payloads discarded. Existing Clinical Key used only in memory against its existing application destination; no cookie, key or patient/mail payload persisted. No authenticated POST or clinical mutation. Production Save/undo on actual messages deliberately not tested; successful writes/concurrency were validated in disposable PostgreSQL/SQLite and synthetic browser.

## Browser and final disposition

Actual existing authenticated clinician browser session was inspected through the supported browser-control API, without extracting cookies/keys. This was not a fresh anonymous browser; anonymous protection was tested independently by HTTP requests above. Opened unread peek/full Inbox/empty personal-rule manager; confirmed explicit rule criteria, future-only explanation and four categories. Opened only a non-patient administrative-provider message to inspect the category editor: Clinical/Admin/Other/Unclassified/reset-to-automatic, optional future rule, explicit Save and refresh controls visible before body. No field value changed, no Save/undo/disable pressed, no patient folder or attachment opened.

Desktop1280 and mobile390: no horizontal overflow; browser error log empty. Screenshot artifacts contain only the empty rule form or category controls, with no message/patient payload: `/tmp/cockpit-pr279-release/rules-desktop.jpg`, `/tmp/cockpit-pr279-release/category-mobile.jpg`. Public release-window app ERROR and HTTP5xx filtered logs2026-11-01T10:50:39Z–10:59:30Z returned0/hasMore=false; bounded evidence only.

`PROVISIONED=YES`; `DEPLOYED_EXACT_SOURCE=YES`; `BOUNDED_READ_ONLY_SMOKE=PASS`; `PRODUCTION_REAL_MESSAGE_WRITE_ACCEPTANCE=NOT_TESTED`. No patient/provider/category/rule writes or duplicate deploy. Original network restrictions restored. Technical release complete; independent review chain stays closed. Next clinician deliberate use; actual production Save/undo cannot be claimed from read-only smoke and was validated only synthetically before release. Release coordinator released after final docs-only publication using `[skip render]`.
