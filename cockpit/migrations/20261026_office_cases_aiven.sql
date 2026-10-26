-- Priority 3 private Office Cases schema. Run ONCE on a single verified Aiven
-- osteoporosis PostgreSQL connection as the existing migration administrator,
-- before deploying code that reads these tables. No patient rows are touched.
BEGIN;
SET LOCAL lock_timeout = '3s';
SET LOCAL statement_timeout = '30s';

DO $$
BEGIN
  IF current_database() <> 'osteoporosis' OR current_user = 'cockpit_app'
     OR NOT has_schema_privilege('cockpit_app', 'public', 'USAGE')
     OR has_schema_privilege('cockpit_app', 'public', 'CREATE')
     OR to_regclass('public.clinical_surgery_queue') IS NULL
     OR to_regclass('public.clinic_rf_applications') IS NULL
     OR to_regclass('public.clinical_surgery_details') IS NOT NULL
     OR to_regclass('public.clinical_accident_cases') IS NOT NULL
     OR to_regclass('public.clinical_accident_case_audit') IS NOT NULL
     OR to_regclass('public.clinical_accident_payments') IS NOT NULL THEN
    RAISE EXCEPTION 'Office Cases migration precondition failed';
  END IF;
END $$;

CREATE TABLE public.clinical_surgery_details (
  surgery_id varchar PRIMARY KEY REFERENCES public.clinical_surgery_queue(id) ON DELETE RESTRICT,
  rf_application_id varchar UNIQUE REFERENCES public.clinic_rf_applications(id) ON DELETE RESTRICT,
  room_entry_time varchar(5),
  created_at timestamp NOT NULL,
  updated_at timestamp NOT NULL,
  CONSTRAINT ck_surgery_room_time CHECK (room_entry_time IS NULL OR room_entry_time ~ '^([01][0-9]|2[0-3]):[0-5][0-9]$')
);

CREATE TABLE public.clinical_accident_cases (
  id varchar(36) PRIMARY KEY,
  patient_name varchar(200) NOT NULL,
  phone varchar(40) NOT NULL,
  accident_date varchar(10) NOT NULL,
  visit_date varchar(10),
  writing_started_date varchar(10),
  report_sent_date varchar(10),
  amount_due_cents integer,
  revision integer NOT NULL,
  created_at timestamp NOT NULL,
  updated_at timestamp NOT NULL,
  CONSTRAINT ck_accident_due CHECK (amount_due_cents IS NULL OR amount_due_cents >= 0),
  CONSTRAINT ck_accident_revision CHECK (revision >= 1)
);
CREATE INDEX ix_accident_cases_updated ON public.clinical_accident_cases(updated_at DESC, id DESC);

CREATE TABLE public.clinical_accident_case_audit (
  id varchar(36) PRIMARY KEY,
  case_id varchar(36) NOT NULL REFERENCES public.clinical_accident_cases(id) ON DELETE RESTRICT,
  revision integer NOT NULL,
  action varchar(30) NOT NULL,
  before_json json NOT NULL,
  after_json json NOT NULL,
  created_at timestamp NOT NULL,
  CONSTRAINT uq_accident_case_revision UNIQUE (case_id, revision)
);
CREATE INDEX ix_accident_case_audit_case ON public.clinical_accident_case_audit(case_id);

CREATE TABLE public.clinical_accident_payments (
  id varchar(36) PRIMARY KEY,
  case_id varchar(36) NOT NULL REFERENCES public.clinical_accident_cases(id) ON DELETE RESTRICT,
  payment_request_id varchar(36) NOT NULL,
  amount_cents integer NOT NULL,
  paid_on varchar(10) NOT NULL,
  payer varchar(200) NOT NULL,
  created_at timestamp NOT NULL,
  voided_at timestamp,
  void_reason varchar(500),
  CONSTRAINT uq_accident_payment_request UNIQUE (case_id, payment_request_id),
  CONSTRAINT ck_accident_payment_positive CHECK (amount_cents > 0),
  CONSTRAINT ck_accident_payment_void CHECK ((voided_at IS NULL) = (void_reason IS NULL))
);
CREATE INDEX ix_accident_payments_case ON public.clinical_accident_payments(case_id, created_at, id);

REVOKE ALL ON TABLE public.clinical_surgery_details, public.clinical_accident_cases,
  public.clinical_accident_case_audit, public.clinical_accident_payments FROM PUBLIC;
REVOKE ALL ON TABLE public.clinical_surgery_details, public.clinical_accident_cases,
  public.clinical_accident_case_audit, public.clinical_accident_payments FROM cockpit_app;
GRANT SELECT, INSERT ON TABLE public.clinical_surgery_details TO cockpit_app;
GRANT UPDATE (room_entry_time, updated_at) ON TABLE public.clinical_surgery_details TO cockpit_app;
GRANT SELECT, INSERT ON TABLE public.clinical_accident_cases TO cockpit_app;
GRANT UPDATE (patient_name, phone, accident_date, visit_date, writing_started_date,
  report_sent_date, amount_due_cents, revision, updated_at)
  ON TABLE public.clinical_accident_cases TO cockpit_app;
GRANT INSERT ON TABLE public.clinical_accident_case_audit TO cockpit_app;
GRANT SELECT, INSERT ON TABLE public.clinical_accident_payments TO cockpit_app;
GRANT UPDATE (voided_at, void_reason) ON TABLE public.clinical_accident_payments TO cockpit_app;

DO $$
BEGIN
  IF NOT has_table_privilege('cockpit_app', 'public.clinical_surgery_details', 'SELECT')
     OR NOT has_table_privilege('cockpit_app', 'public.clinical_accident_cases', 'INSERT')
     OR NOT has_table_privilege('cockpit_app', 'public.clinical_accident_case_audit', 'INSERT')
     OR NOT has_table_privilege('cockpit_app', 'public.clinical_accident_payments', 'INSERT')
     OR has_table_privilege('cockpit_app', 'public.clinical_accident_payments', 'DELETE')
     OR has_schema_privilege('cockpit_app', 'public', 'CREATE') THEN
    RAISE EXCEPTION 'Office Cases least-privilege postcondition failed';
  END IF;
END $$;
COMMIT;

-- Before first use, rollback is DROP of these four empty tables in reverse FK
-- order by the migration owner. After use, preserve their clinical/financial
-- history and roll back only runtime code if required.
