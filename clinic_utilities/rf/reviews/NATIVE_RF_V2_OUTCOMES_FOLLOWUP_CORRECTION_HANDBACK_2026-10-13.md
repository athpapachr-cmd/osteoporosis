# Native RF v2 outcomes/follow-up — bounded R2 correction author handback (2026-10-13)

**CORRECTION IMPLEMENTED / TESTED — independent R2 closure PENDING; writer RELEASED; STOP.**

- Repository: `athpapachr-cmd/osteoporosis`
- PR: [#150](https://github.com/athpapachr-cmd/osteoporosis/pull/150), `feat/rf-v2-outcomes-followup-20261012`, **DRAFT / UNMERGED / UNDEPLOYED**.
- Fresh main at correction activation and code verification: `6ee184f9c76bf45a6c43e1f9fae90d43c1dfde00`.
- Independently reviewed original PR HEAD: `61087bbc262a5d9fed2e01fb2f3dbe689ea28077`; frozen original code+tests `606478b1df29390792f72a79eced3d7fbd840449`.
- Previous independent post-code R2: **BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:2:1**.
- Approved pre-code closure: PASS/COMPLETE/P0:P1:P2=0:0:0. Previous RF CI `38291533010` SUCCESS at original tested head, reused.
- Exact corrected **code+test HEAD**: `d72290b0c83be76c9f59448f05bc20b7ed0e5d87`.
- Exact corrected **code+test tree**: `ff9051d556163f846c6fc607bc798aad4021870f`.
- Tests: [RF hotfix CI 38348261081](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38348261081) **SUCCESS** for exact corrected head. All **45 native tests** PASS; real synthetic browser handler A→B, wrong-selection write guard, rendered existing audit/unknown and original reuse handlers PASS. RF template bytes/PDF generation, prior RF tests, adjacent suites, Python/JS syntax and `git diff --check` PASS. [Canonical impact 38348264773](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38348264773) **SUCCESS**. Initial [38348144886](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38348144886) FAILURE was solely a new test expecting an error for chronologically valid submission correction; test oracle fixed, and new exact-head CI passed.

## P1-01 — Submitted date, downstream correction and chronology

- Explicit `correct_submission` requires an existing manual Submitted date. It changes only its date under optimistic revision, in the same DB transaction as append-only audit; **no stage or mine-to-follow / explicit claim/release mutation**.
- Every actual event date is checked against the issued RF date; existing submitted→decision→scheduled→performed ordering is enforced including corrections.
- `correct_decision` is now available in `scheduled` and `performed` stages, but cannot silently reverse an approved downstream act or move its decision date after its schedule/performance. Valid date/reason edits keep the downstream stage.
- A previously rejected original with an already issued linked correction cannot be changed to approved; a corrected rejection date after that issued correction is rejected. Contradictory requests return 422 and do not write a revision/audit.
- The issued application/PDF, linkage semantics, procedure history and server-attested own/colleague issuer identity remain untouched.

## P1-02 — Stale browser selection

- Independent `outcomeSelectionToken`, application ID and search/scope context guard the real detail handler, event handler and correction reuse handler. Selecting B immediately disables A while B loads.
- Search input/scope change invalidate selection; stale list cards cannot be used in a different search context; delayed A detail or event response cannot replace B.
- The synthetic browser regression executes **select A → select B → resolve B → resolve A**, confirms B remains selected, and sends no wrong-ID event.

## P2-01 — Existing audit provenance

- The protected selected-application detail and event response project `last_decision_report` from the last relevant existing RF progress audit: `reported_source`, actual decision event date from audited after-state, and audit entry time in UTC.
- If historical decision audit is missing, source/entry-time remain **unknown**, never inferred or backfilled; no new tables, duplicate events or external provider source.
- The UI displays Greek source labels, event date and Cockpit entry time in the selected record using **textContent**, not raw HTML. Synthetic malicious rejection reason is displayed literally.

## Changed source/test Git blobs (code+test freeze)

| Path | Corrected blob SHA |
|---|---|
| `clinic_utilities/rf/persistence.py` | `0385e2cb56ca20b197241ca7a0f70e01acf71ed7` |
| `clinic_utilities/rf/api.py` | `391878ff4ff2ab55c6d03872f4a420b42c768b1c` |
| `static/clinic-utilities/rf/app.js` | `389421107c6cf4b0f3adf3c300968fd42394f5ae` |
| `test_rf_v2_native.py` | `8294131d402e5142e58216fa5834fc5d47f4b897` |
| `test_rf_v2_ui.js` | `cf8ec799c6730b96769f0a3fe44b24cc3ccb5cda` |

The unchanged checked oracles: `clinic_utilities/rf/pdf.py` `b3573bd74501631eb5c2b767a79a7ead719a8aa7`; `prior_pdf.py` `07ca2ebd8874f8b1aebd803f2f7a95a0539bdc2a`; `static/cockpit/doctor-shell.js` `6aec0fdc8d378d45a64040b48e450311791de9c7`. No unapproved source files changed.

**Next:** Exactly one separate read-only independent delta+affected-cumulative R2 closure review on fresh-resolved DRAFT PR HEAD with the immutable tested code+test head above; see `NATIVE_RF_V2_OUTCOMES_FOLLOWUP_POSTCODE_R2_CLOSURE_REQUEST_2026-10-13.md`. Review PASS is not merge/deploy permission. **STOP**.
