-- Candidate additive migration. Review and qualify the target database and
-- application role before execution. This file is not run by the application.
-- Existing category values and appointment identifiers are preserved.
BEGIN;
ALTER TABLE clinical_appointment_classifications
  ADD COLUMN IF NOT EXISTS visit_kind VARCHAR,
  ADD COLUMN IF NOT EXISTS clinic_override VARCHAR,
  ADD COLUMN IF NOT EXISTS revision INTEGER NOT NULL DEFAULT 0;
COMMIT;

-- Do not roll back to a runtime whose category-reset operation deletes the
-- entire row while the new columns contain data. Use a compatible rollback.
