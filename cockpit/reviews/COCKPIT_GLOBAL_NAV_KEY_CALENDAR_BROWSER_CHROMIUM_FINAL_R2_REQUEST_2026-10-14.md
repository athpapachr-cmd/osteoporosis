# Cockpit Global Navigation / Clinical Key / Calendar — final browser-evidence independent R2

**MODE:** fresh independent **READ-ONLY**, one bounded Chromium-evidence delta + cumulative affected R2. No code edit, merge, deploy, live patient/provider access or new design. **STOP** after verdict.

Repository: `athpapachr-cmd/osteoporosis`
Implementation: DRAFT [PR #151](https://github.com/athpapachr-cmd/osteoporosis/pull/151), OPEN / UNMERGED / UNDEPLOYED.
Activation `main`: `151312fc95b2f128bb9fb13190ac8041991df192`.
Previously reviewed corrected implementation/test HEAD: `de26841d8575b3d19a479b6f1bec58b87f76a5c7`; prior governance HEAD `ab90aece1039c78b29387e8ad6d9eefd2f47a2f5`.
**New exact tested code/test HEAD:** `b905c5fc213f112e408381555f50ca3734671b4d`. Resolve exact PR HEAD afresh; later docs-only commits do not change browser test.

## Prior independent disposition (reuse; do not reopen)

P1-01 source-proven unsaved Visit Capture Dia draft deletion on shared/inline Clinical Key re-authentication **CLOSED**. Focused implementation-fidelity **PASS**, P0:P1:P2 = 0:0:0. Cumulative **PARTIAL** solely because existing Chromium Home smoke loaded old Home assets without `static/cockpit/shared-nav.js` and `.css`; so it could not prove real-browser shared toolbar display, interaction and narrow-width behavior. Reuse prior R1 gates for auth/Clinical Calendar/Save/Reception and frozen relevant CI, with unaffected blobs; this is **not** a new generalized review.

Review source: `cockpit/reviews/COCKPIT_GLOBAL_NAV_KEY_CALENDAR_P1_01_AFFECTED_R1_CLOSURE_REQUEST_2026-10-14.md`.

## One test-only delta for the evidence gap

Only `test_cockpit_doctor_browser.py` changed in `b905c5f`, frozen blob **`10c6aa73904e864777f9e27804283dbaf9c5301c`**. No changes to runtime shared bar, Home, Calendar, Visit Capture, patient Save, backend, auth, or other tests.

The actual Playwright/Chromium harness now injects the **real** `static/cockpit/shared-nav.js` and `.css` alongside existing Home assets into synthetic-origin DOM. All outbound requests are intercepted; Clinical Key login/status are mocked with synthetic data. It checks:

- Desktop **1280px**, half-width **680px**, mobile **390px** (mobile exercises the shared toolbar without expanding unrelated Home coverage).
- Visible sticky toolbar, Home destination, both top-right controls, no toolbar overflow.
- Eight real menu entries, popover within viewport, accessible Escape close.
- Clinical Key panel, 401 unauthenticated status, popover viewport bounds, wrong-key error and proper value clearing, synthetic successful auth and value clearing.
- Desktop/half-width existing Home/Visit Capture identity boundaries, controls and no real Save still execute.
- Screenshot `cockpit-shared-nav-mobile.png` and retained Home screenshot artifacts.

## Exact CI for new tested HEAD

- [Cockpit Home 38507619765](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38507619765) — **SUCCESS**, with actual Chromium `PASS Cockpit V4 Chromium + actual shared toolbar: desktop, half-width, mobile, key and popovers`.
- [Visit Capture 38507619711](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38507619711) — **SUCCESS**.
- [Clinical Calendar 38507619740](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38507619740) — **SUCCESS**.
- [Surgery 38507619809](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38507619809) — **SUCCESS**.
- [Canonical Impact 38507619746](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38507619746) — **SUCCESS**.
- [Chromium visual artifact](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38507619765/artifacts/11868507090) `cockpit-v4-visual-smoke` (artifact ID `11868507090`).

## Exact independent reviewer questions

1. Verify latest DRAFT PR HEAD and delta from `ab90aece` to `b905c5f`: **only** `test_cockpit_doctor_browser.py`, with later docs-only commits (if any), and unchanged main/production.
2. Inspect the actual loaded assets, route mocks and Chromium assertions: did the old browser evidence gap really close, without silently loading a synthetic substitute for the bar or performing external clinical/provider operations?
3. Validate 1280/680/390 width checks, keyboard Escape, panel bounds, wrong/correct Key behavior, overlay non-interference, and screenshot CI/run provenance. Distinguish genuine omissions from claims not made by the test.
4. Reuse the prior P1-01 closure plus R1 PASS gates. Return **IMPLEMENTATION-FIDELITY PASS / BLOCK / UNKNOWN**, **COMPLETE_FOR_DECLARED_SCOPE / PARTIAL**, **P0:P1:P2**, exact HEAD, smallest remaining gap if any and **STOP**.

**Authority:** test author does not independently certify cumulative closure. No PR merge, deploy, production enablement or real patient-use approval is implied.
