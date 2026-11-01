-- OFFLINE CANDIDATE ONLY. Do not run before separate provisioning authority.
-- Operator must first verify the exact database, schema, backup and existing
-- application role. Execute with the intended schema as the current schema.
-- Deliberately no IF NOT EXISTS: an existing object needs inspection, not reuse.
-- Application startup and GET/POST handlers NEVER execute this statement.
CREATE TABLE icloud_triage_preferences (
    mailbox_key VARCHAR(64) PRIMARY KEY,
    revision BIGINT NOT NULL CHECK (revision > 0 AND revision <= 9007199254740991),
    state_json JSONB NOT NULL CHECK (jsonb_typeof(state_json) = 'object'),
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
);
-- Separately grant only SELECT, INSERT, UPDATE on this exact table to the
-- independently confirmed existing application role. Never schema CREATE,
-- table DELETE, a new role or privileges on clinical tables. No sample role
-- name is supplied here because production role identity was not inspected.
