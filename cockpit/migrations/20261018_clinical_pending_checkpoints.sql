-- CP2: new append-only event table reused from #166, no existing rows modified.
-- Run once in osteoporosis as existing migration owner; never application startup.
-- Existing table causes transaction rollback; do not overwrite an unknown schema.
-- Rollback before use: DROP TABLE public.clinical_pending_checkpoint_events;
-- After use retain events; roll application back to a5dfa5b instead.
BEGIN;
SET LOCAL lock_timeout = '3s';
SET LOCAL statement_timeout = '20s';
DO $preflight$
BEGIN
 IF current_database() <> 'osteoporosis' OR current_user = 'cockpit_app'
 OR NOT EXISTS(SELECT 1 FROM pg_roles WHERE rolname='cockpit_app' AND NOT rolsuper AND NOT rolcreatedb AND NOT rolcreaterole)
 THEN RAISE EXCEPTION 'Incorrect database or migration role'; END IF;
END $preflight$;
CREATE TABLE public.clinical_pending_checkpoint_events (
    id varchar(36) PRIMARY KEY,
    idempotency_key varchar(100) NOT NULL UNIQUE,
    pending_id varchar(36) NOT NULL,
    patient_id varchar(120) NOT NULL,
    checkpoint_key varchar(64) NOT NULL,
    plan_version integer NOT NULL,
    kind varchar(32) NOT NULL,
    retracts_event_id varchar(36),
    revision integer NOT NULL,
    recorded_at timestamp without time zone NOT NULL,
    actor_principal_id varchar(64) NOT NULL,
    actor_role varchar(32) NOT NULL,
    plan_json json,
    CONSTRAINT uq_clinical_pending_checkpoint_event_revision UNIQUE(pending_id, revision)
);
CREATE INDEX ix_clinical_pending_checkpoint_events_pending_id
    ON public.clinical_pending_checkpoint_events (pending_id);
CREATE INDEX ix_clinical_pending_checkpoint_events_patient_id
    ON public.clinical_pending_checkpoint_events (patient_id);
REVOKE ALL ON TABLE public.clinical_pending_checkpoint_events FROM PUBLIC;
GRANT SELECT, INSERT ON TABLE public.clinical_pending_checkpoint_events TO cockpit_app;
DO $verify$
BEGIN
 IF NOT has_table_privilege('cockpit_app','public.clinical_pending_checkpoint_events','SELECT')
 OR NOT has_table_privilege('cockpit_app','public.clinical_pending_checkpoint_events','INSERT')
 OR has_table_privilege('cockpit_app','public.clinical_pending_checkpoint_events','UPDATE')
 OR has_table_privilege('cockpit_app','public.clinical_pending_checkpoint_events','DELETE')
 THEN RAISE EXCEPTION 'Append-only grants incorrect'; END IF;
 RAISE NOTICE 'CP2 CHECKPOINT STORAGE READY';
END $verify$;
COMMIT;
