-- Cockpit #162: FUTURE-ONLY PostgreSQL candidate, NOT applied or reviewed for production.
-- Synthetic code stores clinician-approved plans as revision-zero events
-- so deployed clinical_pending_items columns are NOT changed.
-- No GRANT, no startup DDL, no real-patient approval.
BEGIN;
SET LOCAL lock_timeout = '3s';
SET LOCAL statement_timeout = '20s';
CREATE TABLE IF NOT EXISTS public.clinical_pending_checkpoint_events (
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
CREATE INDEX IF NOT EXISTS ix_clinical_pending_checkpoint_events_pending_id
    ON public.clinical_pending_checkpoint_events (pending_id);
CREATE INDEX IF NOT EXISTS ix_clinical_pending_checkpoint_events_patient_id
    ON public.clinical_pending_checkpoint_events (patient_id);
COMMIT;
