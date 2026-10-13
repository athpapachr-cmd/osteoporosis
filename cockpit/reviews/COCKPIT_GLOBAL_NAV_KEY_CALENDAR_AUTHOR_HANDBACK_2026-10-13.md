# Cockpit global shortcuts / Clinical Key / calendar — implementation author handback

**Date:** 2026-10-13. **Mode:** bounded implementation complete, release HOLD, next independent affected R1.

- Repo: `athpapachr-cmd/osteoporosis`.
- Product Owner explicitly approved three persistent actions on all core screens, without redesigning Home V4, and correction of Cockpit calendar availability/account UX.
- Base main: `151312fc95b2f128bb9fb13190ac8041991df192` (verified before this docs checkpoint).
- [DRAFT PR #151](https://github.com/athpapachr-cmd/osteoporosis/pull/151), branch `feat/cockpit-global-shortcuts-auth-calendar-20261013`. No merge/deploy/real records.
- Frozen implementation + tests: `51b874bfa1a6d8d754747a4d26f43f89cd786ea7`, tree `c6fc00837f23894f4aff96565870b136d6faeb2c`.
- Subsequent author commits, if any, are documentation-only checkpoints with all affected source/test Git blobs identical.

## Implemented

1. **Shared bar**, `static/cockpit/shared-nav.js` and `.css` used in **nine** core entrypoints: Cockpit Home, Visit Capture, Osteoporosis, weekly calendar, Learning, RF, Physio, Sick Leave and Medical Report. Always visible/sticky; direct Cockpit Home; same-origin list of eight already-existent destinations; small Clinical Key panel reusing protected `POST /clinical/login` and `GET /clinical/status`. Key is cleared immediately from DOM on submit and never stored in JS/localStorage/sessionStorage. HttpOnly secure cookie is set solely by the existing server. Failed key does not claim login success. `clinical:authenticated` refreshes Home/surgery/doctor-day, weekly calendar, clinical recent and Visit Capture auth state, without auto-reloading dirty forms.
2. **Cockpit day context**, `clinical_calendar.py`: the existing Reception-owned schedule-context request passes `target_date` using today's **Asia/Nicosia** date (same request convention already used by the week calendar). Existing source freshness ≤5min, explicit full-day coverage, read-only and no artificial fallback remain unchanged; cross-day Next continues if provider supplies future horizon. `static/cockpit/app.js` distinguishes 401/403 Clinical Key from 503 absent/stale source, shows last trustworthy fetch only where present, provides explicit retry on Home. An unavailable Reception feed is not relabeled as a bad Key.
3. **Finite evidence:** `test_cockpit_shared_navigation.js` executes actual common JS in deterministic synthetic DOM, tests all nine entrypoints, password clearing/nonstorage, wrong vs successful key, same-origin cookie request, shortcuts, Escape and change event. `test_clinical_calendar_snapshot.py` checks exact local-date query, cross-day dates and truthy coverage; relevant Cockpit existing Home Chromium / Visit Capture / Surgery validations reused. `.github/workflows/cockpit-home-tests.yml` now executes shared-nav test.

## Verified focused CI at frozen tested SHA

- [Cockpit Home CI 38468051478](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38468051478) **SUCCESS**, including actual Chromium desktop + narrow viewport smoke and `Cockpit shared sticky shortcuts / secure login UI: PASS`.
- [Calendar snapshot CI 38468051401](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38468051401) **SUCCESS**.
- [Visit Capture CI 38468051427](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38468051427) **SUCCESS**.
- [Surgery CI 38468051203](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38468051203) **SUCCESS**.
- [Canonical Impact 38468059492](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38468059492) **SUCCESS**.
- Other unrelated PHYSIO/Learning scope-only guard failures should not be presented as Cockpit runtime regressions; separate medical-report workflow's inline Home HTML-link test was already contradicted by frozen production main (`doctor-shell.js` renders the navigation dynamically). Do not modify unrelated module to silence legacy assertion; reviewer may assess as needed.

## Frozen Git blobs (selected)

```
static/cockpit/shared-nav.js                d5d2d0609816c7a135756f5f9486ed9d72e4dc22
static/cockpit/shared-nav.css               aab1c415f4c106577d67a25b74a05577d9abcb62
test_cockpit_shared_navigation.js           4274c42fc43cafb8bcacbc15ff346d393f66669b
clinical_calendar.py                        024a95d71bf1fbd26ee24d242fb8c7081458f65e
test_clinical_calendar_snapshot.py          30680553adf41b0b47fc84575a5db11f125cf999
static/cockpit/index.html                   b7c5ea8c50cac4b328f36262ccde6475c7e2b7ee
static/cockpit/app.js                       a4d9c5c4317f264acf36d81cc3bd4853632781dc
static/cockpit/doctor-shell.js              e3a4f0bc9b93342ca842178078fff76aec58f238
static/cockpit/clinical-workspace.js        9aaafdb76da54a32d53e65a0787ecf2fa321f502
static/cockpit/visit-capture/app.js         ba0c22c5f0a3b965518a51262274243023c90b27
static/clinical-calendar/calendar.js        d0031e8b490e942f3cd7fc1acdbe3de5cf3ddfed
.github/workflows/cockpit-home-tests.yml    c8e0d7bd42a8f967337d135567a9e170d1c36477
```

**Next:** one fresh independent exact-head affected post-code **R1** review. No new pre-code design review needed because shared authority unchanged. Return PASS/BLOCK/UNKNOWN, P0:P1:P2, STOP. Author writer released; explicit separate Product Owner merge/deploy authorization needed later.
