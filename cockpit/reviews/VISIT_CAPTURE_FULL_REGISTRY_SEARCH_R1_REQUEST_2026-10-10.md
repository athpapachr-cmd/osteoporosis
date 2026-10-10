# Visit Capture — independent A4 search review request

Date: 2026-10-10
PR: #140
Exact implementation head: `b230197297307463e82eba28ee77adbde9c91ebb`

Review only the change from a latest-100 local picker to protected server-side search across the full registered-patient database. Check Greek name and identifier matching, result paging, stale-response handling, explicit patient selection and unchanged clinical Save authority. Reuse original R1 A1–A3 PASS and earlier R2; do not repeat unrelated checks. Evidence: Visit Capture CI `38025138254` SUCCESS, Home `38025138290` SUCCESS, canonical `38025138283` SUCCESS. Provide one PASS/BLOCK/UNKNOWN outcome with source evidence and STOP. No patient-data access or deployment.
