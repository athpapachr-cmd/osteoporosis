# Visit Capture PR #143 — A3 P2-01 independent delta / affected-cumulative R1 closure

Date: 2026-10-10. **Fresh separate independent READ-ONLY review, ONE bounded P2 closure; STOP after finite verdict.**
Repository: `athpapachr-cmd/osteoporosis`.
Draft PR: [#143](https://github.com/athpapachr-cmd/osteoporosis/pull/143) (not merged/deployed).
Branch: `fix/cockpit-visit-capture-clinical-readability-2026-10-10`.
Production base: `987d5aef0e4191fbae436b3aa9df310db2a54a97`.
Original independently reviewed exact code/test head: `a2c9fc09b0372b058e8f622b02a1cc20b6b87b1b`.
**Exact P2 parser correction + synthetic test head:** `de0e25e49f0b92fa3e4a34e3992a4af773ececfa`.

## Received original independent R1 finding (REUSE)
Original handback receipt: `cockpit/reviews/VISIT_CAPTURE_DIA_READABILITY_A3_P2_BLOCK_RECEIPT_2026-10-10.md`.
**Original result: A1 PASS, A2 PASS, A3 BLOCK (one P2), A4 PASS / P0:P1:P2=0:0:1.** DO NOT rerun A1/A2/A4 or parent Visit Capture R2 and PR #140 A4; no new design review.

A3 source-proven defect: `readExplicitReviewNotes()` previously continued to collect numbered/bulleted content after next `Ιστορικό:`, `Ευρήματα:`, `Φαρμακευτική αγωγή:`, despite these being recognized clinical section headings, erroneously inflating review notes.

## Exactly changed executable/test blobs for A3 correction
- `static/cockpit/visit-capture/app.js` current blob **`844846be18479d3c64fe5eda553f906165bc8562`**, previously `718cefebdb937dfcc91de3d3d64e37cf7df7916b`.
- `test_visit_capture_smoke_ui.js` current blob **`2de4e47bfdc684f0dd49576edf3c1e29603adc48`**, previously `7e4e25359c3b7a755f63b75e3f93e5762d59330e`.

The UI HTML/CSS, Greek prompt, `test_visit_capture_ui.py`, protected backend, full-registry search and real Save endpoints have **unchanged blobs** from original A1/A2/A4 review. Changes after corrected code/test head are only review and CURRENT documentation. The correction replaces the old partial-heading regex with `Object.keys(sectionNames).some(level => sourceSectionHeading(line, level))`, preserving the existing explicit rendering grammar; no new parser or clinical inference engine.

## Finite questions, nothing more
**C1 — Original P2 fixed:** Starting from `Σημεία προς επιβεβαίωση:`, a numbered note is emitted, but the subsequent `Ιστορικό:` ends collection and bullets beneath `Ιστορικό:` and `Ευρήματα:` do not become review notes. The synthetic test also includes `Φαρμακευτική αγωγή:`. Expect EXACTLY one item `Έλεγχος χρονολογίας.`.

**C2 — Affected cumulative UI integrity:** Original source remains intact and readable, no clinical sections are dropped or converted into warnings; existing explicitly supplied notes still work when at the end of Encounter Detail; tab changes, clearing the input and switching to record/demo correctly reset visible source-only cues; no client-side generated medical facts or HTML injection.

**C3 — Unaffected authority and evidence:** The correction does not add/fetch clinical records, change patient selection, protected Save/versioning, expose PII, introduce provider calls or change the original Dia prompt/HTML/CSS. Reuse the original R1 A1/A2/A4 PASS and settled R2/A4. Review the latest scoped CI only where relevant.

## Reuse focused correction evidence (author, not independent PASS)
- [Visit Capture 38061805111](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38061805111): SUCCESS including Node synthetic browser simulation and existing Python tests.
- [Cockpit Home 38061805107](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38061805107): SUCCESS.
- [Canonical impact 38061805098](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38061805098): SUCCESS.

The Clinical Learning L0 design-only scope assertion, when red on an unrelated runtime PR, is not an A3 regression; do not claim all workflows green.

## Finite independent closure
Return C1/C2/C3 **PASS/BLOCK/UNKNOWN** with exact lines/blobs/evidence and one global disposition:
`PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0`
if no material risk remains, otherwise a source-grounded bounded BLOCK or UNKNOWN with the smallest corrective action. **STOP.**

No implementation, PR merge, deployment, new broad review, repeated prior A1/A2/A4 tests, clinical patient data, live Dia/Heidi/GESY, provider activity or signed encounter write.
