# Cockpit Visit Brief staff security — independent affected POST-CODE R2

Date: 2026-10-17 Asia/Nicosia. Mode: FRESH INDEPENDENT READ-ONLY, one bounded post-code implementation-fidelity R2. NO code/merge/deploy/PHI/provider/secret/configuration mutations.

Repository: athpapachr-cmd/osteoporosis. Child DRAFT PR #163 (https://github.com/athpapachr-cmd/osteoporosis/pull/163), stacked on DRAFT #161.
Frozen tested CODE+TEST HEAD: a918d82507f749fb138ac39151a79972ccc8d2b2. Branch may have later docs-only commits; DO NOT substitute HEAD.
Frozen parent #161 HEAD: 3999ea46402090c3d9912ce99995a02dc63c81d0. Reuse previous Visit Brief original independent design and P1/P2 post-code PASS. Four-visit purge design P1-01 delta independently PASS / 0:0:0, S5 live activation UNKNOWN.

## Pinned source/test blobs
clinical_staff_auth.py faa44447c6df35b6281b8b46adecbb598cd7ee80
clinical_auth.py 0b548a2dac04e565974f92420f214d2b7dd10794
clinical_calendar.py d4f4eb9da0d8c2153177dd493cf50a2ad2ac2058
clinical_visit_brief.py a704305b877297b0349aaeff64909a1a969b1720
static/cockpit/clinical-workspace.js 97c672cb6f1b764824b8a907a74b42e7a5ded834
static/cockpit/visit-brief.js 01a7181e9e2712367ad659b2b06d1616cb652de5
static/cockpit/shared-nav.js 67b9458f0cceb85dfb7015b3e03d99da59f1f9d9
test_clinical_visit_brief.py 4d75cf11d5979a5a2138e0fae6efd25f01f8e2da
test_cockpit_doctor_browser.py 2016917a7b4740ae2f32800c59644d432b8cec2a
test_cockpit_shared_navigation.js 86894c262bb825a2776b24cf530fb7cb5d31a31c
cockpit/migrations/20261017_visit_brief_staff_access.sql d9115878ea939d737639e966166d5bea0f4239f8

## Exact-code-head AUTHOR CI (not independent PASS)
Visit Brief Python/Node/actual Chromium: https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39008205419 SUCCESS
Cockpit Home Python/Node/actual Chromium: https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39008205415 SUCCESS
Clinical Calendar: https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39008205427 SUCCESS
Canonical Impact: https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39008205499 SUCCESS

## Independent gates
A1 Staff trust: independent doctor/secretary key and actor sessions, key rotation, expiry/forgery/logout, legacy Clinical Key never staff, secretary no broad Clinical Data or clinical Save or calendar writes. Inspect MFA/rate-limit readiness without inventing external protections.
A2 Medical brief+audit: secretary may read previous signed visit/plan/pending ONLY after booking-specific independent per-opening verification. Read/reconfirm/correction/revocation and denial auditable by true principal; no PHI URL/log/browser persistent state. Correct/revoke clinician-governed. Audit history restricted clinician.
A3 Registry: booking-scoped minimal full-registry matching, name/ID/Greek search/pagination, fresh Reception source and observation, stale CAS; no automatic same booking UID attendee inference or full-chart access.
A4 Retention/migration: withdraw four-visit and elapsed-time cleanup entirely. Revocation does not erase patient_id. Future policy-qualified true erasure must be separate, auditable and guard held/active work. Distinct audit retention. SQL additive/idempotent/reviewed ONLY; no VisitBriefBase DDL on runtime startup, check existing schemas and rollback, no patient chart or pending deletion.
A5 Production gate: independent role/private DB/backup/TLS/network/least-privilege/lawful link+audit retention/migration/MFA/rate limiting qualification; all new gates OFF by default, Render auto-deploy main means NO merge/deploy/activate. Aiven Europe+reported restore is not private DB or compliance proof.
A6 Tests: check Python/JS/real Chromium 1280/680/390, close/401 protected DOM purge, disabled feature, booking race and source drift, role authority, Home #160 and Visit Capture unchanged. Reuse prior unaffected PASS.

## Required output
Return PASS/BLOCK/UNKNOWN; COMPLETE_FOR_DECLARED_SCOPE/PARTIAL; P0:P1:P2; A1–A6 PASS/BLOCK/UNKNOWN with narrow GitHub line evidence and minimal affected corrections. No broad re-review. Independent R2 PASS does NOT authorize real data, production DB migration, merge or deployment. PR #163 and #161 stay DRAFT/HOLD.

STOP: READ-ONLY.