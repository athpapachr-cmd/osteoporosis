# Native RF v2 — managed doctors issuer P1-01: ONE independent delta + affected R2 closure

**MODE: fresh, separate, independent READ-ONLY, one bounded post-correction fidelity review.** No implementation, merge, deploy, production configuration, provider/patient data or full RF re-review.

Repository: `athpapachr-cmd/osteoporosis`. PR [#146](https://github.com/athpapachr-cmd/osteoporosis/pull/146) DRAFT / OPEN / UNMERGED at author handback.
Baseline remote main at correction: `8469e12e8803ab1f9a882437527ed69a59a5e9b6` (fresh verify).
Original independently BLOCKED PR head: `9a27bae55e2a5f57bffd54e94bc01583e5aff15e` (tree `4e560c22b5a764022ce2370d4935dfcd10ddecc6`).
**Frozen corrected source+test HEAD:** `d7f1a790317f279bf0e346acc2a630835f1f3f5f`, tree `32dc19115966a8cfe75b239bd5085c646a3d12e1`.
**Exact RF CI:** [38124963792](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38124963792) — SUCCESS, head exactly `d7f1a790317f279bf0e346acc2a630835f1f3f5f`; logs include `RF v2 managed-doctor issuer-selection/retained-draft regression: PASS`, RF focused regressions, official A1/A2 and JPEG integrity.
**Actual PR HEAD:** fresh resolve at review. After corrected frozen HEAD, author may add only operational/review Markdown; verify unchanged code/test Git blobs at actual HEAD.

## Original independent finding and decision

Independent affected post-code R2 at `9a27bae55e2a5f57bffd54e94bc01583e5aff15e` returned **BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:1:0**. R1–R3 and R5 PASS reused. Only **R4 P1-01**: editing doctor B while current RF issuer was doctor A called `loadContract(saved.key)`; full `populateContract` selected B silently, cleared laterality, appended duplicate indication/product options, and left visible issuer summary stale.

Decide whether that one P1-01 is **CLOSED**, without a newly introduced affected material defect. Do not repeat pre-code R2, prior JPEG P1-01 closure, RF profile/backend/transaction reviews or new general architecture review.

### C1 — Corrected behavior and actual browser regression
- Inspect only `static/clinic-utilities/rf/app.js` and `test_rf_v2_ui.js` correction delta.
- Saving edited doctor B (or adding doctor C) must preserve chosen issuer A and all current RF inputs; correcting doctor A while B is selected must preserve B.
- Directory refresh may replace **only doctor options**; must not reinitialize products, indications, selected side, free-text or medical draft.
- Deactivating the current issuer must yield **no selected doctor** with explicit prompt, never silently choose another (even the default); `createPdf` must reject until deliberate selection.
- Visible `reviewSummary` and readiness state must immediately reflect current issuer, and cannot claim a different issuer on success.
- Require a true browser-handler synthetic regression rather than checking mere element existence. Reuse successful RF CI and review test source.

### C2 — Affected cumulative preservation
- Compare exact old/fixed/actual PR source blobs, scope limited to the two JS files plus checkpoint/review Markdown:
  - `static/clinic-utilities/rf/app.js`: corrected `dc86985ca1e24e1e04376219aeaab048533a1fc0` (prior `7b78c47d6359c78777227ad0a4e2763cc3412027`)
  - `test_rf_v2_ui.js`: corrected `a130ba0634cf67aab5c69ff0c7c8c1c2ba800532` (prior `9bdf8808dfc782792659cb260764bdac8ba359d9`).
  - `clinic_utilities/rf/pdf.py` JPEG unchanged `b3573bd74501631eb5c2b767a79a7ead719a8aa7`.
  - `clinic_utilities/rf/api.py` unchanged `2359f859d8d57a1c3711fbd158408c857eaf9921`, `catalog.py` `bf0745a865f983d87961907345fea33413fdf714`, `persistence.py` `2d8f2c804427d14954e8bbf4c8cde398ff7f4bb8`.
  - Original HTML/CSS and Python RF tests frozen; no Home/other service/schema/provider changes.
- Reuse original R1–R3/R5 independent PASS and successful RF code/test CI; unrelated Learning-only scope-check failures are not RF regressions.

### C3 — Identity/decision and STOP
- Fresh verify main, actual draft PR HEAD/tree, frozen corrected tree/blobs and docs-only post-test commits.
- Return **IMPLEMENTATION-FIDELITY PASS | BLOCK | UNKNOWN** + COMPLETE_FOR_DECLARED_SCOPE | PARTIAL + P0:P1:P2 + C1–C3 explicit disposition, original P1-01 CLOSED or smallest residual finding, exact reviewed SHA. **STOP**. No release authority.
