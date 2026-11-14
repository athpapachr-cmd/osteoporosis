-- Offline upgrade for a qualified legacy SQLite calendar database only.
-- Check a backup and exact old column set before execution; this is not run at startup.
BEGIN;
ALTER TABLE clinical_appointment_classifications ADD COLUMN visit_kind VARCHAR;
ALTER TABLE clinical_appointment_classifications ADD COLUMN clinic_override VARCHAR;
ALTER TABLE clinical_appointment_classifications ADD COLUMN revision INTEGER NOT NULL DEFAULT 0;
COMMIT;
-- The old runtime's delete-on-reset cannot safely run while label rows exist.
