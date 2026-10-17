# Cockpit Verified Appointment ↔ Patient & Visit Brief v1 — P1/P2 affected POST-CODE delta R2

**Date:** 2026-10-17 Asia/Nicosia. **MODE:** fresh independent READ-ONLY, one bounded affected-delta correction review, then STOP. No edits, CI reruns as substitute for analysis, provider/clinical data, merge or deploy.

**Repository:** `athpapachr-cmd/osteoporosis`
**DRAFT implementation PR:** [#161](https://github.com/athpapachr-cmd/osteoporosis/pull/161), branch `feat/cockpit-verified-appointment-visit-brief-v1-20261016`, base **design-only** branch `design/cockpit-verified-appointment-visit-brief-v1-20261016` (NOT `main`).
**Exact frozen corrected code+synthetic test HEAD:** `3a98f7c046a83f05e0590661118c92c972483026`.
**Original post-code reviewed HEAD (BLOCK):** `f3b7fb654155907df663b10a00435a71f9d031b2`; handoff docs checkpoint before corrections `0d9740811d0af2cadddc03d25c17a4409c2cf89d`.
**Frozen pre-code corrected design:** commit `199e37c28024391387546fb662350dbabc2b5c1b`, blob `2015dd1db0e2fd7c12a26664be456da10c77f6d7`.
**Independent original post-code disposition:** `BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:1:1`. Original R2 identified P1 lingering hidden protected DOM after close and P2 inability to explicitly re-link a different patient after a revoked association. **Do not self-certify these closed on author CI.** Reuse prior other-gate PASS; do not restart Home, RF, Reports, Clinical Inbox or Visit Capture design audits.

## Frozen changed-file blob oracle (verify at corrected code HEAD)

| Affected owner | Expected blob SHA |
| --- | --- |
| `static/cockpit/visit-brief.js` | `d08469b6a994209c280daedc2dae762314150b97` |
| `test_cockpit_visit_brief_ui.js` | `c7f803b172d307f18ff69786fd24b96009ffb573` |
| `test_clinical_visit_brief.py` | `258e10ddc43824323a7443818151927607d254d1` |
| `test_cockpit_doctor_browser.py` | `d721dd0b5119e6088dca652c75c3738f9cbe1fb9` |

Exactly these four implementation/test files differ between the handoff docs checkpoint and corrected code HEAD. All other code owners, particularly `clinical_visit_brief.py` (unchanged `b1489752e02866de2bf35bf7bcf5104fc7310456`), `clinical_calendar.py`, `clinical_data.py`, previous/now/next Home positions, Visit Capture Save, RF, Inbox and Reports, remain untouched. Documentation-only commits after code HEAD may update `cockpit/CURRENT.md` and this request, but do NOT replace the frozen code HEAD.

## Inspect only two affected corrections

### P1 — purge identifiable DOM and patient selection (original P1)

Recheck `static/cockpit/visit-brief.js` `invalidate()`, `closed()`, `clearProtected()`, `renderConfirmed()`, `revoke()` and `clinical:authenticated/unauthenticated`. After a successfully confirmed synthetic P1 brief, close, auth loss, context switch or re-open must remove the protected ID, patient identity string, booking confirmation text, reason/time/name/freshness, all prior visit/plan/pending/check rows, checkbox marks and chart action authorization from the **hidden DOM and controller state**. Clearing just `.hidden` is insufficient. Opening the same UID after close remains booking-only, never a reusable authorization. During an intentionally open confirmed brief, selected protected chart identity MUST still be visible; do not inadvertently blank it on render. Check late network responses/ticket invalidation, 401/403/503 and no local/session storage or URL PHI. Do not reset or erase deliberately opened patient workspace outside this overlay.

Synthetic regressions: Node actual-controller DOM assertions in `test_cockpit_visit_brief_ui.js`; actual Chromium widths 1280/680/390 in `test_cockpit_doctor_browser.py`. Check actual assertions rather than just workflow green.

### P2 — explicit correction after revoked booking link (original P2)

Existing backend already enforces `replace_existing=True` when patient ID differs, including revoked rows, with expected_revision CAS/409 and audit event. Do **not** relax backend to auto-reassign or accept UID/name matching. A revoked server row must still return booking-only `reason=unlinked, linked=false, revision>0`; previously `visit-brief.js` hid the correction checkbox whenever linked=false, preventing a clinician from sending an explicitly corrected P2 selection.

Now `selectPatient()` exposes the existing correction checkbox for `linked || revision>0`; `confirmSelection()` requires independent verification and, for a revoked row, checked explicit correction before any network write. Verify server still rejects unflagged P1→P2 and stale revision, then accepts flagged independent correction at revision 2, yielding revision 3, audit `confirmed, revoked, corrected`, confirmed_once P2 brief **only in that direct response**, followed by booking-only read with `needs_reconfirmation`. No different booking UID or automatic same-person continuity permission. Inspect Python synthetic CAS/audit test and Node UI payload test.

## Exact corrected-code-head Actions (verify run HEAD/status and individual jobs/steps)

- [Focused Visit Brief #38967892160](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38967892160) — author observed SUCCESS; 42 Python tests, Node controller, actual Chromium 1280/680/390.
- [Cockpit Home #38967892268](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38967892268) — SUCCESS.
- [Clinical Calendar #38967892194](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38967892194) — SUCCESS.
- [Visit Capture #38967892227](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38967892227) — SUCCESS.
- [Canonical Impact #38967892163](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38967892163) — SUCCESS.

All five pull_request runs have `head_sha=3a98f7c046a83f05e0590661118c92c972483026`, verified by author. Independently confirm GitHub Actions status and tested SHA.

## Reviewer disposition and hard release boundary

Return **AFFECTED DELTA POST-CODE R2 PASS / BLOCK / UNKNOWN** and **COMPLETE_FOR_DECLARED_SCOPE / PARTIAL**; explicit **P1 PASS/BLOCK/UNKNOWN** and **P2 PASS/BLOCK/UNKNOWN**, P0:P1:P2, narrow file+line citations and exact CI evidence. If BLOCK, prescribe only one minimal affected correction. No new unrelated audit loops.

Even if both issues PASS, this closes only the correction's **implementation fidelity**. Reception `ActualBooking` remains unqualified for immutable attendee continuity: later UID reads MUST remain booking-only until independent per-opening confirmation. Production merge, deploy, enabling `CLINICAL_VISIT_BRIEF_LINKS_ENABLED`, real identifiable clinical processing, retention/privacy/access policy and automatic one-click release remain separately held. Do not claim automatic one-click PASS.

**STOP — fresh independent READ-ONLY.**
