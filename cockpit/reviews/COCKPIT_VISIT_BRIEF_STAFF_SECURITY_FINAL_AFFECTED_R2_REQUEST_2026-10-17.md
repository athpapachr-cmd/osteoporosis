# Cockpit Visit Brief staff security — FINAL focused affected R2 request

Date: 2026-10-17 Asia/Nicosia.
MODE: fresh independent READ-ONLY post-code R2; ONE bounded correction review.
No code, merge, deploy, DB migration, credentials, production configuration, provider or real-PHI operations.

Repository: `athpapachr-cmd/osteoporosis`.
Child DRAFT/HOLD PR #163, stacked on parent DRAFT/HOLD #161.
**Exact corrected CODE+TEST HEAD: `4baa276dd3b5355ebe18b2da3e993ec57ba05a15`.**
The later request-only documentation commit is NOT the tested code head.
Previous correction baseline: `148e96adbf05aead719ad6d6b9c8cefe5f51317e`.
Reuse prior unaffected design and implementation PASS; do NOT re-review unrelated Cockpit components.

## Exact corrected blobs

- `clinical_visit_brief.py`: `e93311bbb8138367dac4d16514a00f3cacc7b596`
- `static/cockpit/clinical-workspace.js`: `86ce8e00fdc650597e94b7c307c67fe030929bb9`
- `static/cockpit/shared-nav.js`: `25231c58e83c505e23559e50c11d3847a0bfdc7b`
- `test_clinical_visit_brief.py`: `59ab72a88041107434a94d52cfeafbe95b930818`
- `test_cockpit_shared_navigation.js`: `309dea53c477276376230474013a295dc830f6d4`
- `test_cockpit_doctor_browser.py`: `838f3fae48c17944efb5d5afe15d1c75acfcb284`

## Inspect only the three corrections and direct regressions

**P1-02 — attributable audit or no booking disclosure.** Review `clinical_visit_brief.py` lines 257–264 and 417–440, plus Python tests. Staff-only protected reads now fail closed with 503 before Reception access when the reviewed audit/storage/link policy gates are OFF; no unaudited booking-only projection and no boot-time DB DDL. With gates ON, fresh-source denials are audited unconditionally by the guarded booking helper, and booking-only allowed reads are committed to staff audit. Reuse existing allowed/denied search/confirm/reconfirm/correct/revoke audit checks. Confirm the disabled fixture works even without migrated Visit Brief tables; distinguish policy-off requests with no data from permitted booking reads that MUST be attributable. Ensure no link/pending deletion and audit is clinician-readable only.

**P1-03 — late protected response cannot repopulate the DOM.** Review `static/cockpit/clinical-workspace.js` lines 73–123 and corresponding real Chromium synthetic test. The recent-encounter request captures the auth/workspace revision before await and discards late success/error after logout or principal switch. Existing protected registry search already guards request generations and is purged with the Dia preview, hidden modal and selected identity. Examine actual browser 1280/680/390 delayed success AFTER `clinical:unauthenticated`; stale patient names must not reappear.

**P2-01 — role bootstrap/probe race.** Review `static/cockpit/shared-nav.js` lines 62–205 and `test_cockpit_shared_navigation.js`: older `staff-session` and access-panel probes cannot overwrite a newer clinician/secretary login or logout. Server authority remains role-protected independently of `window.CockpitStaffRole`. Confirm clinician chart/revoke controls are visible after verified read, secretary controls (including linked booking-only unlink) are hidden; secretary chart/save/correct/revoke/broad Clinical Data remain denied server-side.

**P1-01 — regression-only.** The previously corrected one clinician staff key authorizes all endpoints formerly backed by the legacy Clinical Key cookie plus staff Visit Brief; no second key is required for Home, Visit Capture, Clinical Data, RF/Reports. Key rotation, expiry and logout return 401, and secretary cannot acquire broad clinical authority. Old Clinical Key keeps legacy screens but never creates a staff principal or grants staff Visit Brief. Reuse previous evidence; do not broaden implementation.

## Exact-code-head author CI (not independent verdict)

- Visit Brief Python, JS and real Chromium: https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39011488790 — SUCCESS
- Cockpit Home Python, JS and Chromium: https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39011488747 — SUCCESS
- Clinical Calendar snapshot: https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39011488659 — SUCCESS
- Canonical Impact: https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39011488660 — SUCCESS

## Inherited scope and hard release hold

Reuse A1 staff trust, A3 registry/source drift, A4 additive reviewed-only migration and withdrawal of four-visit/elapsed-time automatic purge; check only the directly affected regressions in A2/A6 plus clinician role UI in A1. No unrelated Home #160, Visit Capture Save, RF, Medical Reports or Reception changes.
A5 LIVE readiness remains UNKNOWN until separate private DB, TLS, backup/restore, least privilege, MFA, rate limits, credential issuance, lawful link/audit retention and reviewed migration qualifications are documented and passed.
Feature switches remain OFF by default; no automatic retention cleanup. `#163` and `#161` stay DRAFT/HOLD. Author CI or even independent R2 PASS must NOT be treated as merge/deploy/production-PHI authorization.

## Required reviewer handback

Return PASS/BLOCK/UNKNOWN; COMPLETE_FOR_DECLARED_SCOPE or PARTIAL; P0:P1:P2; P1-01/P1-02/P1-03/P2-01; affected A1–A6 with **exact GitHub code/test line evidence**, and only the smallest necessary follow-up. STOP READ-ONLY.
