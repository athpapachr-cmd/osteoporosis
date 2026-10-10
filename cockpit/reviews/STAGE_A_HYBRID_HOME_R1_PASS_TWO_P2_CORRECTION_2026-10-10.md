# Stage A Hybrid Home — independent R1 PASS + two nonblocking P2 corrections
Date: 2026-10-10. Mode: Product Owner-authorized bounded correction checkpoint, NOT a new independent review.

## Earlier independent affected R1 (already received)
- Target/UI+test commit: `2de828e656f85cd4139d407edd34da893031611b`.
- Verdict: **IMPLEMENTATION-FIDELITY PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:2**. R1–R6 PASS for bounded UI authority, no merge/deploy authorization.
- P2-01: static `3` badges could imply actual loaded records even when fewer/unavailable.
- P2-02: `visitPatientSearch` `aria-expanded` did not reflect results visibility.
- Review also explicitly preserved the separate protected read pre-code R2 and parent PR #140 affected A4 holds.

## User-approved minimum correction + author verification
- Exact correction code/test commit: `cbb0afd355903cb85482bca1a15a89171608c5da` (draft PR #142, stacked on draft PR #140).
- Changed files only: `static/cockpit/index.html` blob `b273b641563abd6581d796c749d2ec956552909a`; `static/cockpit/clinical-workspace.js` blob `9ec13c4e267928bc6b1ac8619afcf276ba23f926`; `test_cockpit_home.py` blob `93c8e9acc610455093a3df0252c70bfda2d3f9ff`; `test_cockpit_unified_home.js` blob `22f270814f687e164cbc5318fd08a839adc989e5`.
- P2-01 corrected: `έως 3` in each quick-view heading, independent of currently loaded rows; markup tests updated.
- P2-02 corrected: search input ARIA expanded is set to true with visible query results and false on clear, selected patient and return; synthetic DOM regression assertions added.
- Focused CI on exact corrected commit: [Cockpit Home 38048298303](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38048298303) SUCCESS; [Visit Capture 38048298281](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38048298281) SUCCESS; [canonical impact 38048298279](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38048298279) SUCCESS; Learning L1/L1B/L1C SUCCESS.
- L0 contract validation step succeeded but inherited design-only changed-file scope assertion failed; do not report all CI green.
- This checkpoint does NOT pretend that the earlier independent R1 reviewer re-reviewed the new correction commit. No material BLOCK existed in R1; the corrections are nonblocking and covered by narrowly changed synthetic tests.

## Release / next independent gate
PR #142 stays DRAFT, UNMERGED, UNDEPLOYED. No protected backend 3+3 reads, identity writer, Dia/Heidi/GESY/provider access, Clinical Inbox adapter or production activation was implemented. Writer is RELEASED.

Next independent review is the **already prepared, unchanged** `cockpit/reviews/UNIFIED_VISIT_V3_READ_PRECODE_R2_REQUEST_2026-10-10.md` of the two proposed protected read extensions, under frozen design blob `9e107112020e2d98d542867c64fbb08d47e278dd`. Do not implement until its independent R2 pre-code PASS. Parent PR #140 affected full-registry A4 remains its own pending R1.
