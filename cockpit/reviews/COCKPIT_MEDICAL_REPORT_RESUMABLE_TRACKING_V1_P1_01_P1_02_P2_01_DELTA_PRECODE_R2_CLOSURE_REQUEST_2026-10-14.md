# Cockpit Medical Reports Tracking v1 — P1-01/P1-02/P2-01 delta + affected PRE-CODE R2 closure

**2026-10-14 · ONE fresh independent READ-ONLY bounded delta + affected PRE-CODE R2 · STOP.** No implementation, runtime/schema/test modification, provider/DB operations, real clinical records, merge or deploy.

Repository: `athpapachr-cmd/osteoporosis`.
PR: [#153](https://github.com/athpapachr-cmd/osteoporosis/pull/153), `design/cockpit-medical-report-resumable-tracking-20261014`; fresh-resolve exact current DRAFT/unmerged HEAD, base `244a3264f119252b1bbfc3562a00648cfdf113bd`, current diff, and current design blob at `cockpit/COCKPIT_MEDICAL_REPORT_RESUMABLE_TRACKING_V1_DESIGN_2026-10-14.md`.
Original independent PRE-CODE R2 identity: HEAD `aa37748ac02081f3c9c4715dddf841be1f70ae16`, design blob `e158510d0c482a1363c389deb836b44bbea4571e`. Original verdict **BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:2:1**. That blob is the **original** baseline, not the new corrected expected blob. Reviewer must pin the corrected blob independently.

## Exact limited delta under review

* **P1-01 / G4 — completion against unsaved edits.** Inspect C3/C4: collect exact DOM-edited current snapshot (including accepted refine and research); wait for latest successful save acknowledgement, block during dirty/pending/failed/409/auth; require fresh clinician attestation tied to the still-current server revision, validate again atomically; edits after acknowledgement revoke completion eligibility. Verify no PDF/AI/preview-driven completion. Synthetic oracle H2/H4.
* **P1-02 / G3 — cross-workspace async isolation.** Inspect C2: shared `app.js` + `v1-1-refine.js` active workspace/generation; all open/create/save/analyze/refine/research and intercepted delayed callbacks, successes **and failures**, guarded; A→B/B→A switching safely clears transient source/file/signature/research/proposal/confirmation, respects dirty-input warning, hydrates the selected snapshot only; no provider reanalysis or persistent browser fallback. Synthetic oracle H2.
* **P2-01 / G6 — report-date filters.** Inspect D2/F/H3: name + status + inclusive optional strict ISO report-date range and 422 reversed range, all-record indexed paginated search with stable ordering and filtered totals, bounded date controls; no new endpoint, no PHI query strings/logging.

## Affected invariants (reuse accepted prior gates, do not reopen unrelated work)

* **G1:** source identity and docs-only diff; no runtime, schema, migrations, tests or provider operations. Original report API session-only behavior remains unchanged.
* **G2/G7:** single protected Medical Reports database owner, strict bounded snapshots, no raw source page texts/signature/PDF, shared Clinical Key limitation explicitly truthful, PHI feature OFF by default, actual encryption/backup/retention/access proofs deferred to independent release gate. No assumption of individual doctor identity.
* **G3/G4:** accepted refinements/DOM edits/valid research/clinician resolutions and historical source manifest resume exactly; 409/non-destructive error; clinician confirmation and signature not restored.
* **G5:** one count-only authoritative Home reader, `draft + awaiting_information` pending, `completed` manual only, `void` excluded, no PHI Home state, invalid/error/stale read unavailable not zero, #152 surgery/RF untouched.
* **G6:** synthetic test plan only, 390/680/1280, full search >100, Greek text exact restore, report AI/PDF and Visit Capture Save unaffected.
* **G7:** distinct future authorities: independent delta PRE-CODE R2 → separate Product Owner approval to implement → synthetic tests + independent affected POST-CODE R2 → separate merge/deploy and real-PHI/privacy decision. Do not treat design correction as authorization.

## Required reviewer answer — STOP

Return `PRE-CODE R2 PASS | BLOCK | UNKNOWN`, `COVERAGE = COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, `P0:P1:P2 = x:y:z`, current PR #153 HEAD and corrected design blob, 4-file original docs-only scope + any newly added docs-only diff, each P1-01/P1-02/P2-01 CLOSED or still OPEN with source evidence, affected G1–G7, minimum **design-only** correction if still blocked, and `STOP`.

The implementing author **must not self-PASS**, and no code/database/provider/PHI/release action is permitted in this review.
