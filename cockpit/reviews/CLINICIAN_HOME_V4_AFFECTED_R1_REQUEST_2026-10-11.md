# Cockpit Home V4 — independent clinician-first UX implementation fidelity R1

Date: 2026-10-11. **Fresh independent READ-ONLY post-code R1; one finite affected scope then STOP.**

Repository `athpapachr-cmd/osteoporosis`. [Draft PR #145](https://github.com/athpapachr-cmd/osteoporosis/pull/145); branch `feat/cockpit-apple-inspired-home-2026-10-11`.
Production base/main at review preparation `7028c1459ff51c0d6edd0df74298f742f77ccd3f`.
**Exact implementation+test+CI head `2d2e97f3b6dff0ea56d7a8b53605266df2b42ef0`.**
Design Product Owner contract `cockpit/CLINICIAN_HOME_V4_UI_CONTRACT_2026-10-11.md`.

## Frozen affected blobs

| File | Git blob |
|---|---|
| `static/cockpit/index.html` | `e52791f271dddef3d696904ad9b55c5203e96a74` |
| `static/cockpit/clinical-workspace.js` | `1e9be939500896e5b37cb70b8fc630de26824dd3` |
| `static/cockpit/app.js` | `4debceff28433512747b648f501b435a742b1515` |
| `static/cockpit/doctor-shell.js` | `6aec0fdc8d378d45a64040b48e450311791de9c7` |
| `static/cockpit/doctor-shell.css` | `9ff4f2c33667be205c555b5012abee5677917650` |
| `.github/workflows/cockpit-home-tests.yml` | `2fda9f4c586fa5ef365ca660a8b9a8f8c9aae617` |
| `test_cockpit_home.py` | `261d69baaa56668ff3c5f8a6a72868fbcc1a1ad8` |
| `test_cockpit_unified_home.js` | `e7c22d838c073dda330927c0fd586661ba79fc00` |
| `test_cockpit_doctor_shell.js` | `3797aae1196f4b23c892d054135b68f80e121cae` |
| `test_cockpit_doctor_browser.py` | `fe27518b408eb530757239c5d39aead779e69794` |

## Exactly four finite questions

**A1 — Surface/one-click work.** Is the default Home uncluttered: search, up to three actual recent encounters and source-backed attention, with the previous sprawling `Πρόγραμμα και πηγές`, large Visit Capture panel and heavy surgery/modules kept outside default Home? Can clinician navigate to every existing tool/queue and back without loss of existing surgery functionality? On half-width 680 px, does top date/action bar remain to the right, sidebar collapse/reopen without horizontal overflow, and old rail appointment Peek retain the unverified-identity boundary?

**A2 — Popover and actions.** Does top-right date open on-demand relevant-category appointments only from the existing protected `/clinical/calendar/appointments?start&end` for Asia/Nicosia local midnight boundaries (including independently computed DST boundaries), labeled as PARTIAL, with full Reception route accessible? Clicking row opens *unlinked* appointment context, never silently a confirmed patient ID. Escape/close/outside click and late response switching keep one floating panel, proper focus/aria, no stale clinical text. Raycast-style action search reaches existing physio/sick leave/RF/report/visit capture/Module 01/calendar/Reception/learning routes.

**A3 — Attention truth & privacy.** The aggregate surgery count is emitted from the already loaded protected queue, no second source fetch or PHI/name disclosure, clears if source becomes unavailable, and displays no invented count. Email Inbox popover is clearly NOT connected: hidden unread badge, no user emails, credentials, provider requests or inferred summaries. Recent list admits the current API supplies only encounter metadata; absent signed brief excerpt displays a truthful message, not fabricated medical content. Browser-only prose Dia and protected Save remain unchanged/no newly reachable write path.

**A4 — Focused evidence / untouched owners.** Existing Home/Clinical Calendar/Visit Capture/surgery source/test functionality is unchanged except designated Home UI adaptations. Real Chromium browser synthetic desktop 1280 px, Dia halfwidth 680 px and date-popover screenshots are authentic produced by focused workflow; no real patient/provider reads. All code in PR #145 stays a fresh R1 read-only UI, not an implicit protected R2 change. Reuse existing closed Stage A R1, protected 3+3 read R2, Visit Capture R1 and #140/#143 production passes; don't rerun their design reviews.

## Existing exact-head evidence (reuse)

- [Cockpit Home GitHub Actions #38088464470](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38088464470) SUCCESS, including real Chromium synthetic desktop + halfwidth smoke, command palette, day popup, identity boundary and no writes.
- [Visit Capture #38088464499](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38088464499) SUCCESS.
- [Canonical impact #38088464488](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38088464488) SUCCESS.
- [Visual smoke artifact `cockpit-v4-visual-smoke`](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38088464470#artifacts).
- Inherited Clinical Learning L0 design-only changed-file scope assertion may be red for this non-L0 diff; do not assert all workflows green and do not turn an inapplicable scope check into a Cockpit defect.

## Review/output discipline

Return one verdict **PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 0:0:0** if source/evidence matches; otherwise bounded **BLOCK** or **UNKNOWN** on real material issues, smallest correction and STOP. Distinguish product-accepted partial scope (not a live Clinical Inbox or all-specialty day/signed encounter-summary source) from an accidental user-facing overclaim. Any protected source extension is a separate R2 product/technical contract; do not silently change schema or require another broad design review here.

No implementation, merge/deploy, identifiable data, Gmail/Zadarma, patient writes, or repeat of prior R1/R2.
