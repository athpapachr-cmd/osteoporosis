-- #162 Clinical Pending One-Tap FUTURE SCHEMA, NO PRODUCTION EXECUTION AUTHORITY.
-- Deliberately NOT run automatically; #162 remains synthetic-only.
-- A separate clinic-approved PostgreSQL rollout and actor/retention gate is required.
-- Do not run this with cockpit_app or on live identifiable clinical rows.
BEGIN;
SET LOCAL lock_timeout = '3s';
SET LOCAL statement_timeout = '20s';

ALTER TABLE public.clinical_pending_items
    ADD COLUMN IF NOT EXISTS confirmation_revision integer NOT NULL DEFAULT 0;
ALTER TABLE public.clinical_pending_items
    ADD COLUMN IF NOT EXISTS checkpoint_plan_json json;

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
    CONSTRAINT uq_clinical_pending_checkpoint_event_revision
        UNIQUE (pending_id, revision)
);
CREATE INDEX IF NOT EXISTS ix_clinical_pending_checkpoint_events_pending_id
    ON public.clinical_pending_checkpoint_events (pending_id);
CREATE INDEX IF NOT EXISTS ix_clinical_pending_checkpoint_events_patient_id
    ON public.clinical_pending_checkpoint_events (patient_id);

-- No GRANT, feature activation, automatic rollout, or clinical attestation.
COMMIT;
