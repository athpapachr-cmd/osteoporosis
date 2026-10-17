-- Cockpit A5 ONE TRANSACTION migration — NOT run automatically.
-- Intended only for a separately authorized clinic operator using avnadmin,
-- against osteoporosis, after pre-migration backup and an explicit rollout go/no-go.
-- No patient rows read or written. This will create 3 new empty Visit Brief tables
-- then assign only the necessary privileges to cockpit_app; any failure rolls back all.
-- IMPORTANT: only PostgreSQL psql with ON_ERROR_STOP=1; do not paste in Aiven SQL editor.
BEGIN;
SET LOCAL lock_timeout = '3s';
SET LOCAL statement_timeout = '20s';

-- Phase 1: additive metadata-only schema
-- REVIEW ONLY. Do not run against live PHI until backup, private DB,
-- lease/owner, TLS, role credentials and retention policies are verified.
-- Additive PostgreSQL migration for PR #161 (never startup create_all).

CREATE TABLE IF NOT EXISTS clinical_booking_patient_links (
    namespace varchar(48) NOT NULL,
    booking_uid varchar(160) NOT NULL,
    patient_id varchar(120) NOT NULL,
    state varchar(32) NOT NULL,
    revision integer NOT NULL,
    evidence_type varchar(64) NOT NULL,
    confirmed_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    PRIMARY KEY (namespace, booking_uid)
);

CREATE TABLE IF NOT EXISTS clinical_booking_patient_link_audit (
    id varchar(36) PRIMARY KEY,
    namespace varchar(48) NOT NULL,
    booking_uid varchar(160) NOT NULL,
    patient_id varchar(120),
    event varchar(24) NOT NULL,
    revision integer NOT NULL,
    evidence_type varchar(64),
    operator_type varchar(32) NOT NULL,
    occurred_at timestamp without time zone NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_clinical_booking_patient_link_audit_namespace
    ON clinical_booking_patient_link_audit (namespace);
CREATE INDEX IF NOT EXISTS ix_clinical_booking_patient_link_audit_booking_uid
    ON clinical_booking_patient_link_audit (booking_uid);

CREATE TABLE IF NOT EXISTS clinical_visit_brief_staff_access_audit (
    id varchar(36) PRIMARY KEY,
    principal_id varchar(64) NOT NULL,
    role varchar(32) NOT NULL,
    booking_uid varchar(160) NOT NULL,
    patient_id varchar(120),
    event varchar(40) NOT NULL,
    outcome varchar(32) NOT NULL,
    revision integer,
    occurred_at timestamp without time zone NOT NULL
);


-- Before production activation, explicitly inspect schema compatibility
-- without reading patient rows. IF NOT EXISTS does NOT upgrade existing tables.
-- No link or audit deletion is automatically scheduled by this migration.

-- Phase 2: fail-closed schema validation and least-privilege grants
-- A5 Cockpit Visit Brief runtime role grants — DRAFT, not executed in production.
-- Apply ONLY after the reviewed 20261017_visit_brief_staff_access.sql migration,
-- after a current backup/rollback, and with independent release authorisation.
-- No patient rows accessed; never run from the application's cockpit_app role.
SET LOCAL lock_timeout = '3s';
SET LOCAL statement_timeout = '20s';

DO $a5_preflight$
DECLARE
  table_names text[];
BEGIN
  IF current_database() <> 'osteoporosis' OR current_user = 'cockpit_app' THEN
    RAISE EXCEPTION 'A5 STOP: incorrect database or migration principal';
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_roles WHERE rolname='cockpit_app'
    AND rolcanlogin AND NOT rolsuper AND NOT rolcreatedb
    AND NOT rolcreaterole AND NOT rolreplication
  ) OR has_schema_privilege('cockpit_app','public','CREATE') THEN
    RAISE EXCEPTION 'A5 STOP: cockpit_app role or privileges unqualified';
  END IF;
  SELECT array_agg(c.relname::text ORDER BY c.relname::text) INTO table_names
    FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
   WHERE n.nspname='public' AND c.relkind='r'
     AND c.relname IN ('clinical_booking_patient_links','clinical_booking_patient_link_audit',
                       'clinical_visit_brief_staff_access_audit');
  IF table_names IS DISTINCT FROM ARRAY['clinical_booking_patient_link_audit',
    'clinical_booking_patient_links','clinical_visit_brief_staff_access_audit']::text[] THEN
    RAISE EXCEPTION 'A5 STOP: Visit Brief tables missing/unqualified';
  END IF;
  -- A CREATE TABLE IF NOT EXISTS cannot upgrade a stale table; compare every
  -- column, type and nullability before granting any permission.
  IF EXISTS (
    WITH expected(table_name,column_name,expected_type,expected_nullable) AS (VALUES
      ('clinical_booking_patient_links','namespace','character varying(48)',false),
      ('clinical_booking_patient_links','booking_uid','character varying(160)',false),
      ('clinical_booking_patient_links','patient_id','character varying(120)',false),
      ('clinical_booking_patient_links','state','character varying(32)',false),
      ('clinical_booking_patient_links','revision','integer',false),
      ('clinical_booking_patient_links','evidence_type','character varying(64)',false),
      ('clinical_booking_patient_links','confirmed_at','timestamp without time zone',false),
      ('clinical_booking_patient_links','updated_at','timestamp without time zone',false),
      ('clinical_booking_patient_link_audit','id','character varying(36)',false),
      ('clinical_booking_patient_link_audit','namespace','character varying(48)',false),
      ('clinical_booking_patient_link_audit','booking_uid','character varying(160)',false),
      ('clinical_booking_patient_link_audit','patient_id','character varying(120)',true),
      ('clinical_booking_patient_link_audit','event','character varying(24)',false),
      ('clinical_booking_patient_link_audit','revision','integer',false),
      ('clinical_booking_patient_link_audit','evidence_type','character varying(64)',true),
      ('clinical_booking_patient_link_audit','operator_type','character varying(32)',false),
      ('clinical_booking_patient_link_audit','occurred_at','timestamp without time zone',false),
      ('clinical_visit_brief_staff_access_audit','id','character varying(36)',false),
      ('clinical_visit_brief_staff_access_audit','principal_id','character varying(64)',false),
      ('clinical_visit_brief_staff_access_audit','role','character varying(32)',false),
      ('clinical_visit_brief_staff_access_audit','booking_uid','character varying(160)',false),
      ('clinical_visit_brief_staff_access_audit','patient_id','character varying(120)',true),
      ('clinical_visit_brief_staff_access_audit','event','character varying(40)',false),
      ('clinical_visit_brief_staff_access_audit','outcome','character varying(32)',false),
      ('clinical_visit_brief_staff_access_audit','revision','integer',true),
      ('clinical_visit_brief_staff_access_audit','occurred_at','timestamp without time zone',false)
    ), actual AS (
      SELECT c.relname::text, a.attname::text,
             format_type(a.atttypid,a.atttypmod)::text, NOT a.attnotnull
        FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        JOIN pg_attribute a ON a.attrelid=c.oid
       WHERE n.nspname='public' AND c.relkind='r'
         AND c.relname IN ('clinical_booking_patient_links','clinical_booking_patient_link_audit',
                           'clinical_visit_brief_staff_access_audit')
         AND a.attnum>0 AND NOT a.attisdropped
    ), drift AS (
      (SELECT * FROM expected EXCEPT SELECT * FROM actual)
      UNION ALL
      (SELECT * FROM actual EXCEPT SELECT * FROM expected)
    )
    SELECT 1 FROM drift
  ) THEN
    RAISE EXCEPTION 'A5 STOP: Visit Brief table schema drift';
  END IF;
  IF EXISTS (
    WITH primary_columns AS (
      SELECT c.relname::text AS tab, array_agg(a.attname::text ORDER BY k.ordinality) AS cols
        FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        JOIN pg_index ix ON ix.indrelid=c.oid AND ix.indisprimary
        JOIN unnest(ix.indkey) WITH ORDINALITY k(attnum,ordinality) ON true
        JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum=k.attnum
       WHERE n.nspname='public' AND c.relkind='r'
         AND c.relname IN ('clinical_booking_patient_links','clinical_booking_patient_link_audit',
                           'clinical_visit_brief_staff_access_audit')
       GROUP BY c.relname
    )
    SELECT 1 FROM (VALUES
      ('clinical_booking_patient_links',ARRAY['namespace','booking_uid']::text[]),
      ('clinical_booking_patient_link_audit',ARRAY['id']::text[]),
      ('clinical_visit_brief_staff_access_audit',ARRAY['id']::text[])
    ) e(tab,cols) LEFT JOIN primary_columns p ON p.tab=e.tab
    WHERE p.cols IS DISTINCT FROM e.cols
  ) THEN
    RAISE EXCEPTION 'A5 STOP: Visit Brief primary key drift';
  END IF;
END;
$a5_preflight$;

-- Least privilege: versioned links need reads/insert/correction/revoke;
-- append-only audit tables need only SELECT/INSERT. No DELETE or CREATE.
GRANT SELECT, INSERT, UPDATE ON TABLE public.clinical_booking_patient_links TO cockpit_app;
GRANT SELECT, INSERT ON TABLE public.clinical_booking_patient_link_audit TO cockpit_app;
GRANT SELECT, INSERT ON TABLE public.clinical_visit_brief_staff_access_audit TO cockpit_app;

DO $a5_postcheck$
BEGIN
 IF NOT (
  has_table_privilege('cockpit_app','public.clinical_booking_patient_links','SELECT')
  AND has_table_privilege('cockpit_app','public.clinical_booking_patient_links','INSERT')
  AND has_table_privilege('cockpit_app','public.clinical_booking_patient_links','UPDATE')
  AND NOT has_table_privilege('cockpit_app','public.clinical_booking_patient_links','DELETE')
  AND has_table_privilege('cockpit_app','public.clinical_booking_patient_link_audit','SELECT')
  AND has_table_privilege('cockpit_app','public.clinical_booking_patient_link_audit','INSERT')
  AND NOT has_table_privilege('cockpit_app','public.clinical_booking_patient_link_audit','UPDATE')
  AND NOT has_table_privilege('cockpit_app','public.clinical_booking_patient_link_audit','DELETE')
  AND has_table_privilege('cockpit_app','public.clinical_visit_brief_staff_access_audit','SELECT')
  AND has_table_privilege('cockpit_app','public.clinical_visit_brief_staff_access_audit','INSERT')
  AND NOT has_table_privilege('cockpit_app','public.clinical_visit_brief_staff_access_audit','UPDATE')
  AND NOT has_table_privilege('cockpit_app','public.clinical_visit_brief_staff_access_audit','DELETE')
  AND NOT has_schema_privilege('cockpit_app','public','CREATE')
 ) THEN
  RAISE EXCEPTION 'A5 STOP: post-grant least-privilege verification failed';
 END IF;
 RAISE NOTICE 'A5 ROLE GRANTS VERIFIED: links R/I/U; audit R/I; no CREATE/DELETE';
END;
$a5_postcheck$;

COMMIT;
