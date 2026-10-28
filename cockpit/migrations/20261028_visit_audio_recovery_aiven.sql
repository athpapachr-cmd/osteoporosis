-- Install the empty, opaque Visit Audio recovery-capability table on the
-- existing osteoporosis PostgreSQL database. Run once as the existing Aiven
-- migration administrator, never as cockpit_app and never on app startup.
-- The table stores only a random capability ID, one fixed state and expiry;
-- it never stores audio, transcript, patient identity or clinical content.
BEGIN;
SET LOCAL lock_timeout = '3s';
SET LOCAL statement_timeout = '20s';

DO $$
BEGIN
  IF current_database() <> 'osteoporosis'
     OR current_user = 'cockpit_app'
     OR NOT EXISTS (
       SELECT 1 FROM pg_roles WHERE rolname = 'cockpit_app'
         AND rolcanlogin AND NOT rolsuper AND NOT rolcreatedb
         AND NOT rolcreaterole AND NOT rolreplication
     )
     OR NOT has_schema_privilege('cockpit_app', 'public', 'USAGE')
     OR has_schema_privilege('cockpit_app', 'public', 'CREATE')
     OR to_regclass('public.clinical_visit_audio_recovery') IS NOT NULL THEN
    RAISE EXCEPTION 'Visit Audio recovery migration precondition failed';
  END IF;
END $$;

CREATE TABLE public.clinical_visit_audio_recovery (
  id varchar PRIMARY KEY,
  state varchar NOT NULL,
  expires_at timestamp without time zone NOT NULL
);
CREATE INDEX ix_clinical_visit_audio_recovery_state
  ON public.clinical_visit_audio_recovery (state);
CREATE INDEX ix_clinical_visit_audio_recovery_expires_at
  ON public.clinical_visit_audio_recovery (expires_at);

REVOKE ALL ON TABLE public.clinical_visit_audio_recovery FROM PUBLIC;
REVOKE ALL ON TABLE public.clinical_visit_audio_recovery FROM cockpit_app;
GRANT SELECT, INSERT, DELETE ON TABLE public.clinical_visit_audio_recovery TO cockpit_app;
GRANT UPDATE (state) ON TABLE public.clinical_visit_audio_recovery TO cockpit_app;

DO $$
BEGIN
  IF NOT has_table_privilege(
       'cockpit_app', 'public.clinical_visit_audio_recovery', 'SELECT')
     OR NOT has_table_privilege(
       'cockpit_app', 'public.clinical_visit_audio_recovery', 'INSERT')
     OR NOT has_table_privilege(
       'cockpit_app', 'public.clinical_visit_audio_recovery', 'DELETE')
     OR NOT has_column_privilege(
       'cockpit_app', 'public.clinical_visit_audio_recovery', 'state', 'UPDATE')
     OR has_column_privilege(
       'cockpit_app', 'public.clinical_visit_audio_recovery', 'id', 'UPDATE')
     OR has_column_privilege(
       'cockpit_app', 'public.clinical_visit_audio_recovery', 'expires_at', 'UPDATE')
     OR has_table_privilege(
       'cockpit_app', 'public.clinical_visit_audio_recovery', 'TRUNCATE')
     OR has_schema_privilege('cockpit_app', 'public', 'CREATE')
     OR (SELECT count(*) FROM public.clinical_visit_audio_recovery) <> 0
     OR to_regclass('public.ix_clinical_visit_audio_recovery_state') IS NULL
     OR to_regclass('public.ix_clinical_visit_audio_recovery_expires_at') IS NULL THEN
    RAISE EXCEPTION 'Visit Audio recovery least-privilege postcondition failed';
  END IF;
END $$;
COMMIT;

-- Before first use only, rollback is DROP TABLE
-- public.clinical_visit_audio_recovery as the migration owner. After a recovery
-- capability is issued, preserve one-shot history and roll back runtime only.
