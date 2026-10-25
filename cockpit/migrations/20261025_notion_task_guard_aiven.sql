-- Priority 2: install the empty Notion confirmation guard on the existing
-- osteoporosis PostgreSQL database. Run once as the existing Aiven migration
-- owner, never as cockpit_app and never on application startup.
-- Nine ordinary statements fit Aiven PG Studio's 10-query execution limit.
-- Select and run the entire batch. A failed assertion aborts the transaction.
-- No Notion task text, transcript, audio, or patient rows are read or written.
BEGIN;
SELECT
  set_config('lock_timeout', '3s', true),
  set_config('statement_timeout', '20s', true),
  1 / ((current_database() = 'osteoporosis'
        AND current_user <> 'cockpit_app'
        AND to_regclass('public.cockpit_notion_task_attempts') IS NULL
        AND EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cockpit_app'
                    AND rolcanlogin AND NOT rolsuper AND NOT rolcreatedb
                    AND NOT rolcreaterole AND NOT rolreplication)
        AND has_schema_privilege('cockpit_app', 'public', 'USAGE')
        AND NOT has_schema_privilege('cockpit_app', 'public', 'CREATE'))::integer)
    AS correct_database_role_and_absent_table;
CREATE TABLE public.cockpit_notion_task_attempts (
  attempt_id varchar(36) PRIMARY KEY,
  source_page_id varchar(36) NOT NULL,
  source_revision varchar(64) NOT NULL,
  state varchar(24) NOT NULL,
  CONSTRAINT uq_notion_task_page_revision UNIQUE (source_page_id, source_revision)
);
REVOKE ALL ON TABLE public.cockpit_notion_task_attempts FROM PUBLIC;
REVOKE ALL ON TABLE public.cockpit_notion_task_attempts FROM cockpit_app;
GRANT SELECT, INSERT ON TABLE public.cockpit_notion_task_attempts TO cockpit_app;
GRANT UPDATE (state) ON TABLE public.cockpit_notion_task_attempts TO cockpit_app;
SELECT
  'NOTION TASK GUARD READY' AS result,
  1 / ((has_table_privilege('cockpit_app', 'public.cockpit_notion_task_attempts', 'SELECT')
        AND has_table_privilege('cockpit_app', 'public.cockpit_notion_task_attempts', 'INSERT')
        AND has_column_privilege('cockpit_app', 'public.cockpit_notion_task_attempts', 'state', 'UPDATE')
        AND NOT has_column_privilege('cockpit_app', 'public.cockpit_notion_task_attempts', 'attempt_id', 'UPDATE')
        AND NOT has_column_privilege('cockpit_app', 'public.cockpit_notion_task_attempts', 'source_page_id', 'UPDATE')
        AND NOT has_column_privilege('cockpit_app', 'public.cockpit_notion_task_attempts', 'source_revision', 'UPDATE')
        AND NOT has_table_privilege('cockpit_app', 'public.cockpit_notion_task_attempts', 'DELETE')
        AND NOT has_table_privilege('cockpit_app', 'public.cockpit_notion_task_attempts', 'TRUNCATE')
        AND NOT has_schema_privilege('cockpit_app', 'public', 'CREATE'))::integer)
    AS least_privilege_verified;
COMMIT;

-- Before first use only, rollback is DROP TABLE public.cockpit_notion_task_attempts
-- as the migration owner. After any confirmation attempt, preserve the table
-- and its one-shot history even if the application is rolled back.
