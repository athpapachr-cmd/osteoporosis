-- Reviewed separately; run only with verified clinical-database administrator.
-- Runtime role receives only table-scoped SELECT, INSERT, UPDATE, DELETE.
CREATE TABLE clinical_visit_preparation_drafts (
    booking_key VARCHAR(64) PRIMARY KEY,
    booking_date VARCHAR(10) NOT NULL,
    expires_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    generation VARCHAR(64),
    nonce BYTEA,
    ciphertext BYTEA,
    pending_token VARCHAR(64),
    pending_until TIMESTAMP WITHOUT TIME ZONE
);
CREATE INDEX ix_clinical_visit_preparation_drafts_expires_at
    ON clinical_visit_preparation_drafts (expires_at);
-- Marker initialization is an exclusive operator step; it is not done at startup.
