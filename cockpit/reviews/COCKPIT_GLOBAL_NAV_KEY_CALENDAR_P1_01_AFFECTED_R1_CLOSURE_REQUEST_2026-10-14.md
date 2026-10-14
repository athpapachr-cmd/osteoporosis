# Cockpit Global Navigation / Clinical Key / Calendar — P1-01 independent affected R1 closure

**MODE: fresh independent READ-ONLY, ONE bounded delta + affected cumulative R1.** No implementation, merge, deploy, live patient/provider operations or new design.

Repository: `athpapachr-cmd/osteoporosis`; DRAFT [PR #151](https://github.com/athpapachr-cmd/osteoporosis/pull/151), unmerged/undeployed.
Activation main: `151312fc95b2f128bb9fb13190ac8041991df192`.
Prior frozen implementation/test HEAD: `51b874bfa1a6d8d754747a4d26f43f89cd786ea7`.
Prior R1 request: `cockpit/reviews/COCKPIT_GLOBAL_NAV_KEY_CALENDAR_AFFECTED_POSTCODE_R1_REQUEST_2026-10-13.md`.
Prior independently reported R1: **BLOCK / PARTIAL / P0:P1:P2 = 0:1:0**. One source-proven P1-01; browser evidence gap described separately.
Original reviewed PR HEAD: `5c1dc4884a925b70a1f5e3023908c060c9ef44e7`.

## Author's bounded correction (NOT independent closure)

Corrected tested code/test HEAD: `de26841d8575b3d19a479b6f1bec58b87f76a5c7` (exact PR HEAD: **fresh resolve**, later governance-only commits may follow).
Delta from original reviewed HEAD consists of exactly:
- `static/cockpit/visit-capture/app.js`, blob `25790c54e6020447f2cc26f0232f608a57a6bfa7`;
- `test_visit_capture_smoke_ui.js`, blob `08ba3eabd0f6fbfa7b19f9b821e00cac7f6129bb`.

P1-01 original reachable behavior: on a `?patient_id=...` deep link, the existing `clinical:authenticated` handler called `checkAuth()`, which silently invoked `selectMode("record")` from a demo editor and `clearCandidateState()` deleted an unsaved Dia draft and preview.

Small correction: automatic record-mode selection is allowed only on a **clean initial page load**, never on shared-key or inline-key re-authentication; initial async status response also checks whether the user has started a draft. The patient ID remains suggested for later **explicit** record choice. No schema, protected Save, patient binding, credentials, Reception, Calendar, other utilities or backend changes.

Focused `test_visit_capture_smoke_ui.js` executes the actual UI handler in synthetic DOM: old failure path with URL patient ID and drafted Dia text, successful shared auth event, inline login, preview retention, no auto-Save, explicit record choice/deep-link selection, clean initial entry, and user typing while the initial status request is pending. Synthetic data only.

## Exact corrected CI

- [Visit Capture 38506205714](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38506205714) — SUCCESS
- [Cockpit Home 38506205680](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38506205680) — SUCCESS
- [Canonical Impact 38506205682](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38506205682) — SUCCESS

Reuse frozen Calendar, Surgery, original Home and shared-nav synthetic success from the first request where blobs are unchanged. No unrelated generalized reruns.

## Finite reviewer questions

1. Verify current DRAFT PR exact HEAD and the above corrected code/test blob oracle, docs-only later commits if present, and unchanged main/release state.
2. Re-execute the original P1-01 path by reading the actual handler/test: shared-key and inline-key success must preserve unsaved candidate and preview without switching to record, changing Save authority or writing patient data. Test counterexamples: clean initial deep link still routes correctly; draft entered during initial pending status is preserved.
3. Verify unaffected protected patient selection/context Save behavior and unchanged data authority; only follow direct affected call sites.
4. Reuse prior R1 PASS gates. **Important evidence limitation:** the original R1 also found the pre-existing Chromium Home suite does **not** inject the new shared-nav JS/CSS, so it does not directly prove the shared toolbar's visual/narrow-width behavior. P1-01 correction did NOT modify that browser harness. Do not label the full R1 `COMPLETE_FOR_DECLARED_SCOPE` unless that declared browser evidence gap is independently resolved; do not silently count old Home Chromium as toolbar coverage.

Return `IMPLEMENTATION-FIDELITY PASS | BLOCK | UNKNOWN` for P1-01 and an explicit cumulative `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, `P0:P1:P2`, smallest remaining gap if any, and **STOP**. Author does not self-certify independent closure. PR remains DRAFT, no merge/deploy or permission for real patient use.
