-- Clinical Inbox B1. NOT executed automatically or approved for production.
-- One additive, empty, synthetic-only schema; PostgreSQL 17 disposable CI.
-- Patient identity, clinical result, source body, referral, ADT and providers excluded.
-- B2 requires NEW reviewed migration, actor qualification, privacy/retention
-- adoption, installation authority and explicit real-data activation.
-- Operator must use a privileged migration principal, never cockpit_app.
BEGIN;
SET LOCAL lock_timeout = '3s';
SET LOCAL statement_timeout = '20s';

DO $b1_preflight$
BEGIN
  IF current_database() <> 'osteoporosis' OR current_user = 'cockpit_app' THEN
    RAISE EXCEPTION 'B1 STOP: incorrect database or migration actor';
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_roles WHERE rolname = 'cockpit_app'
      AND rolcanlogin AND NOT rolsuper AND NOT rolcreatedb
      AND NOT rolcreaterole AND NOT rolreplication
  ) OR has_schema_privilege('cockpit_app', 'public', 'CREATE') THEN
    RAISE EXCEPTION 'B1 STOP: unqualified cockpit_app role';
  END IF;
  IF to_regclass('public.clinical_attention_items') IS NOT NULL
     OR to_regclass('public.clinical_attention_events') IS NOT NULL THEN
    RAISE EXCEPTION 'B1 STOP: table already exists, investigate schema';
  END IF;
END $b1_preflight$;

CREATE TABLE public.clinical_attention_items (
  item_id varchar(64) PRIMARY KEY,
  logical_source_item_key varchar(100) NOT NULL UNIQUE,
  source_type varchar(32) NOT NULL,
  active_source_revision varchar(16) NOT NULL,
  source_subject_scope varchar(24) NOT NULL,
  origin_kind varchar(16) NOT NULL,
  patient_id varchar(120),
  identity_state varchar(24) NOT NULL,
  review_state varchar(24) NOT NULL,
  disposition varchar(24) NOT NULL,
  version integer NOT NULL,
  created_at timestamp without time zone NOT NULL,
  updated_at timestamp without time zone NOT NULL,
  CONSTRAINT b1_item_synthetic_only CHECK (
    item_id ~ '^SYN-B1-[A-Z0-9-]+$'
    AND logical_source_item_key ~ '^SYN-B1-[A-Z0-9-]+$'
    AND origin_kind = 'synthetic'
    AND patient_id IS NULL
    AND identity_state = 'unlinked'
  ),
  CONSTRAINT b1_item_state CHECK (
    source_type IN ('lab_notice','imaging_notice','gesy_availability')
    AND active_source_revision ~ '^r[1-9][0-9]{0,3}$'
    AND source_subject_scope IN ('single_subject','unknown','mixed_subjects')
    AND review_state IN ('unreviewed','source_inspected','needs_re_review')
    AND disposition = 'undecided'
    AND version >= 0
  )
);

CREATE TABLE public.clinical_attention_events (
  event_id varchar(36) PRIMARY KEY,
  item_id varchar(64) NOT NULL REFERENCES public.clinical_attention_items(item_id),
  actor_id varchar(80) NOT NULL,
  actor_role varchar(16) NOT NULL,
  event_type varchar(32) NOT NULL,
  outcome varchar(24) NOT NULL,
  item_version integer NOT NULL,
  source_revision varchar(16) NOT NULL,
  idempotency_key varchar(36),
  request_hash varchar(64),
  occurred_at timestamp without time zone NOT NULL,
  CONSTRAINT uq_b1_actor_item_idempotency UNIQUE (item_id,actor_id,idempotency_key),
  CONSTRAINT b1_event_synthetic_only CHECK (
    actor_id ~ '^b1-synthetic-person-[A-Za-z0-9_-]+$'
    AND actor_role IN ('clinician','secretary')
    AND event_type IN ('fixture_created','inspect_source','correct_source','view')
    AND outcome IN ('allowed','denied_403','denied_409')
    AND item_version >= 0 AND source_revision ~ '^r[1-9][0-9]{0,3}$'
    AND (
      (idempotency_key IS NULL AND request_hash IS NULL)
      OR
      (idempotency_key IS NOT NULL AND request_hash ~ '^[a-f0-9]{64}$'
       AND event_type IN ('inspect_source','correct_source')
       AND outcome = 'allowed')
    )
  )
);
CREATE INDEX ix_b1_events_item_time
  ON public.clinical_attention_events (item_id, occurred_at);

REVOKE ALL ON TABLE public.clinical_attention_items FROM PUBLIC;
REVOKE ALL ON TABLE public.clinical_attention_events FROM PUBLIC;
GRANT SELECT, INSERT, UPDATE ON TABLE public.clinical_attention_items TO cockpit_app;
GRANT SELECT, INSERT ON TABLE public.clinical_attention_events TO cockpit_app;

DO $b1_verify$
BEGIN
  IF NOT (
    has_table_privilege('cockpit_app','public.clinical_attention_items','SELECT')
    AND has_table_privilege('cockpit_app','public.clinical_attention_items','INSERT')
    AND has_table_privilege('cockpit_app','public.clinical_attention_items','UPDATE')
    AND NOT has_table_privilege('cockpit_app','public.clinical_attention_items','DELETE')
    AND has_table_privilege('cockpit_app','public.clinical_attention_events','SELECT')
    AND has_table_privilege('cockpit_app','public.clinical_attention_events','INSERT')
    AND NOT has_table_privilege('cockpit_app','public.clinical_attention_events','UPDATE')
    AND NOT has_table_privilege('cockpit_app','public.clinical_attention_events','DELETE')
    AND NOT has_schema_privilege('cockpit_app','public','CREATE')
  ) THEN
    RAISE EXCEPTION 'B1 STOP: least-grants verification failed';
  END IF;
  IF (SELECT count(*) FROM public.clinical_attention_items) <> 0
     OR (SELECT count(*) FROM public.clinical_attention_events) <> 0 THEN
    RAISE EXCEPTION 'B1 STOP: expected empty tables';
  END IF;
END $b1_verify$;
COMMIT;
