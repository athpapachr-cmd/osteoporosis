# Cockpit PR #207 — Final Independent R2 PASS and Product Owner Release Gate

**Date:** 2026-10-23 (Asia/Nicosia).
**Repo:** `athpapachr-cmd/osteoporosis` · [PR #207](https://github.com/athpapachr-cmd/osteoporosis/pull/207).
**Final READ-ONLY R2 closure verdict:** **PASS | COMPLETE_FOR_DECLARED_SCOPE | P0:P1:P2 = 0:0:0**.
**Exact frozen substantive code+tests:** `13bd91426d1f7f09ce5c0a9d17fc153df6eefce4`.
**Docs-only review handback at approval:** `28ff3315cedb756114ab265fbd0496e11b5af9e0`.
**Production main verified before release:** `8708a37c0d4083c3f811ced6b186e9ef7e817355`.
**Live Render baseline before release:** `osteoporosis`, service `srv-d5qfk31r0fns73di596g`, deploy `dep-dbco4a3bc2fs738lqs8g` LIVE exact `8708a37c0d4083c3f811ced6b186e9ef7e817355`.

## Independent R2 chain and source-verified closure

1. Original independent R2 on `4088f7a...`: BLOCK 0:1:1 for patient-name search surviving key/principal switch and late Undo reply repaint/reload.
2. First bounded code+test correction `ee6e1f9...`: P1 CLOSED; Undo success/error protected by generation and scope. Independent follow-up found P2 delayed HTTP 401 could emit `clinical:unauthenticated` in `postTask()` before caller staleness guard (BLOCK 0:0:1).
3. Final correction `13bd914...`: `static/cockpit/obligations-view.js` captures originating generation and scope **before** fetch, dispatches auth-invalidating events only for a current active request, retains genuine-current-session 401 behavior. `test_cockpit_cp2.py` extends real Chromium on isolated FastAPI/SQLite: 680px genuine `/clinical/login` success then deferred stale Undo HTTP 401; asserts no stale logout or login request, no stale feedback and list still visible. Original 1280px stale HTTP 200, 390px stale network failure and search-field privacy regression retained. Previous unaffected R2 provenance/lifecycle, atomic clinical writer/audit and UX findings reused.
4. Final independent READ-ONLY closure concluded PASS, COMPLETE_FOR_DECLARED_SCOPE, 0:0:0, no material defect within final affected scope. This is a bounded source/review conclusion, not a claim of live clinician acceptance or whole-service GDPR compliance.

## Frozen CI evidence

On exact code+test `13bd914...`: [Visit Capture CP1/CP2 + Chromium](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766333973) (**37 passed**), [Cockpit Home](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766334219), [Visit Brief](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766334234), [Surgery](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766334096), [Pending Demo](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766333983), [Canonical Impact](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766334474): all SUCCESS.

On docs-only `28ff331...`: [Capture](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766880069) (**37 passed**), [Home](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766880082), [Visit Brief](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766880025), [Surgery](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766880156), [Pending Demo](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766880029), [Canonical Impact](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766947163): all SUCCESS. GitHub compare confirmed docs-only successor after `13bd914...`.

## Product Owner release authority and strict scope

On **2026-10-23**, after the independent R2 PASS handback, the clinician explicitly requested **“production release”** of PR #207. This grants a **single bounded release** of the R2-reviewed source to the existing Cockpit production environment: guarded **squash merge** of exact PR HEAD, existing Render `osteoporosis` main auto-deploy (no manual duplicate deploy), deployment status verification and **no-PHI startup/log smoke** only.

**Authorized:** installation of the new obligation category/due-date and same-screen metadata/reversible Delete/Undo UI for usual clinician-authorized clinic use after a successful production deploy.

**Not authorized:** changing patient data on the user's behalf, permanent task deletion, uncontrolled acceptance of title-derived due dates, new role/grant/schema/secret changes, Gmail/email ingestion, Aiven operations, unrelated changes or public PHI screenshots/logging. The independent Clinical Inbox and 100-active-row limit remain unchanged.

**Pre-merge state:** Release writer on PR #207 scoped to documentation checkpoint and exact guarded merge; no merge/deploy yet. Following merge, checkpoint the exact main SHA before any deliberate next stage, observe native Render deployment, verify LIVE same SHA and no-PHI boot, then release the writer with accurate canonical closeout. No further architecture review.
