-- Native RF Tracking + Weekly Calendar v1: OPTIONAL PRE-RELEASE DATABASE MIGRATION.
-- NOT executed or applied by the app. Requires separate release approval.
-- PostgreSQL only; run each statement with AUTOCOMMIT, never inside BEGIN/COMMIT.
-- Inspect pg_indexes and pg_index.indisvalid before and after. A stale or
-- invalid same-named index must be reviewed, not treated as successfully migrated.
-- Does not add patient tables, outcome writers, or change any clinical value.
CREATE INDEX CONCURRENTLY IF NOT EXISTS
    ix_clinic_rf_application_progress_scheduled_date_id
    ON clinic_rf_application_progress (scheduled_date, application_id)
    WHERE scheduled_date IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS
    ix_clinic_rf_application_progress_performed_date_id
    ON clinic_rf_application_progress (performed_date, application_id)
    WHERE performed_date IS NOT NULL;

-- Verify individually after creation:
-- SELECT indexname, indexdef FROM pg_indexes
-- WHERE tablename='clinic_rf_application_progress';
-- SELECT c.relname AS indexname, i.indisvalid, i.indisready
-- FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
-- WHERE c.relname IN
-- ('ix_clinic_rf_application_progress_scheduled_date_id',
--  'ix_clinic_rf_application_progress_performed_date_id');
