-- REVIEW ONLY. Do not run against live PHI until backup, private DB,
-- lease/owner, TLS, role credentials and retention policies are verified.
-- Additive PostgreSQL migration for PR #161 (never startup create_all).
BEGIN;

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

COMMIT;

-- Before production activation, explicitly inspect schema compatibility
-- without reading patient rows. IF NOT EXISTS does NOT upgrade existing tables.
-- No link or audit deletion is automatically scheduled by this migration.
