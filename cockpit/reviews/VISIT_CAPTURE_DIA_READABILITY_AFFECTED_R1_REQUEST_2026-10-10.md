# Visit Capture — independent affected R1, Dia clinical readability

Date: 2026-10-10 · **Mode: ONE fresh separate independent READ-ONLY exact-head UI-fidelity review, then STOP.**

Repository: `athpapachr-cmd/osteoporosis`  
Draft implementation PR: [#143](https://github.com/athpapachr-cmd/osteoporosis/pull/143)  
Branch: `fix/cockpit-visit-capture-clinical-readability-2026-10-10`  
Fresh production base: `987d5aef0e4191fbae436b3aa9df310db2a54a97`  
**Exact substantive implementation+test head: `a2c9fc09b0372b058e8f622b02a1cc20b6b87b1b`**  
Unchanged existing protected backend and clinical Save path.

## Source evidence: five exact reviewed blobs

| File | Immutable Git blob |
|---|---|
| `static/cockpit/visit-capture/index.html` | `b169b1d393987365cacace114525917ab6546fdc` |
| `static/cockpit/visit-capture/styles.css` | `2d9b3222e632376382e7a7042fbeb448a35149ad` |
| `static/cockpit/visit-capture/app.js` | `718cefebdb937dfcc91de3d3d64e37cf7df7916b` |
| `test_visit_capture_ui.py` | `a72efef1e3644b580d2ed58098d168cc6c2021d9` |
| `test_visit_capture_smoke_ui.js` | `7e4e25359c3b7a755f63b75e3f93e5762d59330e` |

All clinical examples/fixtures are invented. No screenshot, trace, personal identifier or real clinical numerical value from the clinician's case entered source/tests.

## Why (Product Owner clinical UX clarification)

The clinician explicitly clarified that the investigations were ordered, not performed, with no results yet, and the cited FRAX MOF denotes ten-year risk of **major osteoporotic fracture**, separate from hip. PO approved clearer Snapshot/Visit Brief, expandable structured Encounter Detail and salient separately visible source-flagged 'points for confirmation'. The implementation modifies only the **generic Dia copy prompt and browser-local demo presentation**, NOT clinical records or the authoritative interpretation of a patient case.

## Four finite checks

**A1 Prompt/source integrity.** The copyable three-view Greek Dia instruction distinguishes ordered/planned/performed/results, describes FRAX MOF versus hip correctly, asks the same source-backed clinical decisions in all levels, never asserts findings from unavailable input and retains identifiers exclusion. Does not claim to have corrected already-pasted content; re-run Dia or clinician changes are separate.

**A2 Source-only formatting.** Snapshot and Visit Brief use only explicitly labeled source headings; Encounter Detail groups existing labeled source text into accessible expandable sections without dropping data. A freeform/unlabeled source remains verbatim, not assigned invented headings or pseudo-clinical semantics. Rendering is `textContent`/DOM methods, never untrusted `innerHTML` or executable markup. Test that long detailed output remains available.

**A3 Explicit check-items only.** The badge/count is based solely on numbered/bulleted notes under a supplied `Σημεία προς έλεγχο/επιβεβαίωση` heading, not an app-generated diagnosis or inferred contradiction. The UI labels notes unverified, permits expansion, preserves panel state across tabs and hides it after source clear/mode change. No check-item promotion into Save or CareTasks.

**A4 Protected state unchanged.** The affected client does not alter registry, protected patient selection/context, existing server-side preview/versioning/signed Save, no-patient no-write mode, clinical endpoints, or provider effects. Verify synthetically sourced data only. Reuse parent R2 and PR #140 A4 independent PASS; do NOT rerun those settled reviews.

## Reused focused evidence

- [Visit Capture #38058818300](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38058818300) SUCCESS.
- [Cockpit Home #38058818251](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38058818251) SUCCESS.
- [Canonical impact #38058818134](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38058818134) SUCCESS.
- Earlier intermediate source CI failures were limited to test tab-selection expectation, a changed Greek privacy/test string, prompt accent match and docs trailing whitespace; repaired before exact tested head. Do not report historical failures as still open.
- The inherited unrelated Learning L0 design-only diff-scope assertion may be red on a non-L0 PR; check actual applicability rather than claim all CI green.

## Authority and stop

Classify **R1 UI/presentational**: no new clinical inference, patient identity authority, clinical/longitudinal state, authoritative write, provider integration or patient data access. If a source-reachable new semantic authority or unsafe write path is demonstrated, return bounded BLOCK/reclassification rather than silently expanding scope.

Return A1–A4 PASS/BLOCK/UNKNOWN and one finite verdict: **PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0** if no material findings, otherwise source-evidenced BLOCK or UNKNOWN with smallest correction. STOP.

No implementation/merge/deploy, no real Dia/Heidi/GESY access, no patient record write, no new general research or repeated closed R2/A4 review.
