# Cockpit #205 — Focused R1 delta + affected closure after three corrections

**2026-10-22 Asia/Nicosia.** Reviewer disposition from frozen source-level correction and synthetic evidence. No production data, connected email, database or provider action used.

**R1 = PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0.**
Original R1 findings P1-01/P1-02/P1-03 from [BLOCK receipt](COCKPIT_STANDALONE_OBLIGATIONS_AFFECTED_R1_BLOCK_RECEIPT_2026-10-22.md) **CLOSED within affected scope.** No new material regression found in the bounded delta and inherited original UI.

## Source and tests

- Reviewed original code/test head: `36aa39369d9f6ea6b05fa4e09e65fd203074fa18`.
- Corrected exact full source+test head: **`f63eb7d84f60a104dcd8761d2a1b0d4cf0fe11b9`**.
- Subsequent commits are documentation and review-only; source/test blob identity must be checked again at release.
- Home+real Chromium 1280/680/390 [39715477276](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39715477276) **SUCCESS**; Visit Capture/CP1-CP2/pending ID continuity [39715477212](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39715477212) **SUCCESS**; Visit Brief including synthetic #161 [39715477224](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39715477224) **SUCCESS**; Surgery [39715477270](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39715477270) **SUCCESS**; Canonical Impact [39715477155](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39715477155) **SUCCESS**.

### P1-01 CLOSED — badge authority
`static/cockpit/doctor-shell.js` now hides and clears `doctorTasksBadge` in valid and invalid surgery-summary paths. Existing surgical counters still feed the Home attention/surgery library; an icon labelled patient obligations does not claim a fabricated count. `test_cockpit_doctor_shell.js` and actual `test_cockpit_doctor_browser.py` assert hidden status.

### P1-02 CLOSED — patient-name input lifecycle
`static/cockpit/obligations-view.js` `close()` clears typed patient-name search and selection values, in addition to zeroing patient rows, counters, protected status and aborting pending requests. Actual Chromium types a patient name, navigates to the existing chart, then asserts the hidden search value is empty; late successful fetch cannot reinstate rows after close.

### P1-03 CLOSED — uncommitted chart/draft
`static/cockpit/doctor-shell.js` calls the existing `CockpitStoredChart.canLeave()` before hiding Home or showing the obligations screen; refusal leaves the clinical draft visible with the existing chart prompt. The Node shell test explicitly exercises both refusal and accepted open.

## Affected cumulative disposition

- Existing `/clinical/clinical-updates?scope=real&active_only=true` remains the protected **read-only** source; per-item click delegates unchanged patient/pending IDs to existing protected chart/popup action owner, no clinical writer duplicated.
- Search, grouping by exact `patient_id`, recorded category, and Cyprus review-date buckets work on the returned projection and do not claim full-clinic coverage. **Existing 100-row backend limit remains explicitly disclosed**; server pagination is out of scope.
- Clinical Inbox still does not ingest email in this step; no email→task auto-create/auto-closure, no new Gmail reads or secrets.
- Existing Home, Calendar/Brief, surgery, synthetic patient and pending lifecycle show no material regression in applicable synthetic CI.
- **No automatic deployment authorisation from a review verdict alone.** Separate Product Owner approval and guarded branch/head check required for squash and exact Render LIVE.

**STOP review. Next release action:** as separately authorised by the Product Owner, verify unchanged production `main`, exact PR head and source/test freeze, mark ready, squash-merge, observe native Render auto-deploy and bounded no-PHI startup logs. Clinical user acceptance remains distinct from tested implementation fidelity.
