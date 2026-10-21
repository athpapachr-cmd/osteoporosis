# Cockpit — Standalone Clinical Obligations, one affected post-code R1

**Date:** 2026-10-22 Asia/Nicosia. **Mode:** fresh independent **READ-ONLY**, ONE bounded post-code R1 implementation-fidelity review. Return PASS/BLOCK/UNKNOWN and STOP. No code, merge, deploy, backend, clinic DB or PHI actions.

Repository: `athpapachr-cmd/osteoporosis`. PR: [#205](https://github.com/athpapachr-cmd/osteoporosis/pull/205), DRAFT/HOLD. Verified base `main=7ca04734d697eaf573c59437e987b5575e5998c7`. **Frozen runtime+test author head**: `ecfa22f6dc639a3aacdf2a5febcb799ab239563e`. Follow-up review-request/checkpoint commits must not change runtime/test blobs.

## Product Owner requirement

The clinician requests the patient obligations formerly shown only in a narrow Tasks popup to have an independent **full Cockpit screen**, searchable/sortable by exact patient, recorded category and Cyprus review date, without another task registry. New emails must FIRST enter an **independent Clinical Inbox** for review, then only deliberately link to existing obligations; no email intake or automatic task completion is authorized in this UI step.

## Scope — only affected UI and read boundaries

- `static/cockpit/index.html`: new hidden sibling `doctorObligationsView`, search, status/category filters, grouped tabs (patient/category/date), numbers and explicit 100-row source cap; direct Tasks icon.
- `static/cockpit/obligations-view.js`: uses existing `GET /clinical/clinical-updates?scope=real&active_only=true` protected source with credentials same-origin, cache no-store and abort/generation; scopes synthetic-161 to explicitly marked existing synthetic source; groups patient by ID, never by namesake display string. Local filtering and counters **only on returned at-most-100 rows**, with clear user disclosure. No new backend/DB/clinical write or email connector.
- `static/cockpit/doctor-shell.js`: routes Tasks and Home attention to this screen, on item returns to existing protected `CockpitStoredChart.openPending(patient_id,pending_id)`; only synthetic-161 fixture can fall back to its already-authoritative synthetic `open`. Original surgery/Reports/RF remain accessible through Actions/library. Logout hides both views and purges patient data.
- `static/cockpit/doctor-shell.css`: responsive at 1280, 680, 390 without horizontal overflow, proper keyboard labels/focus.
- Focused tests `test_cockpit_doctor_browser.py`, `test_cockpit_doctor_shell.js`, `test_cockpit_unified_home.js`, `test_cockpit_synthetic_161.py`; one small Home workflow syntax-check addition. Root/workstream CURRENT updates record bounded writer and release HOLD.

## Finite fidelity questions

1. **Identity/data:** protected session needed; status 401/403 and synthetic fixture can't spill into real data; no names/Pending text fetched when hidden; late network resolution after close/logout cannot repopulate PHI; no patient ID/name in URL/local storage as a consequence of this view.
2. **Presentation:** true full-screen workspace, correctly returns Home; grouped exact patient IDs, recorded category and Cyprus due-date buckets; search/status/type filters and up-to-100 limitation true; counts not misrepresented as global; handles empty, unauthorized, unavailable.
3. **Existing task owner:** click passes unchanged IDs to original `CockpitStoredChart.openPending` patient chart/popup, no second task writer, no automatic acceptance/completion, no duplicate clinical task lifecycle.
4. **Inbox preserved:** email icon remains distinct and honestly pending; no Gmail/prototype connector, no automatic email-to-task association; unrelated Home/Calendar/VisitBrief/VisitCapture/surgery preserved. Legacy synthetic-161 session works with its own fixture owner.
5. **Evidence:** compare exact source/test blobs and focused author GitHub Actions: Home [39680844094](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39680844094), Visit Brief+real Chromium/synthetic-161 [39680844268](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39680844268), surgery [39680844173](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39680844173), Canonical Impact [39680844031](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39680844031). Check final Visit Capture result separately. Reuse previously accepted clinical pending/CP1/CP2 and Home R2; don't reopen unrelated medical authority.

## Scope exclusions and handback

Full-clinic pagination and global server-side filter/count are future scope: the existing protected endpoint caps at 100, so report **limited coverage** instead of claiming completeness. Clinical Inbox live Gmail intake/identity verification/review is another later slice. No real patient writes, email reads, secrets, new endpoints, tables, flags, merge or Render release.

Report **PASS/BLOCK/UNKNOWN; COMPLETE_FOR_DECLARED_SCOPE/PARTIAL; P0:P1:P2; exact tested HEAD; evidence for each question; minimum next action.** STOP READ-ONLY.
