# Native RF v2 — Independent P1-01/P1-02/P2-01 delta + affected-cumulative post-code R2 closure

**MODE:** Fresh separate independent READ-ONLY bounded source-grounded R2 closure. No implementation, design, new general review, tests without a declared evidence need, merge, deploy, or real clinical data.

**Repository:** `athpapachr-cmd/osteoporosis`; implementation [PR #149](https://github.com/athpapachr-cmd/osteoporosis/pull/149) DRAFT / OPEN / UNMERGED.
**Fresh-verify** remote main against at-issuance `7186a26105e6edd37ed02b089781ce3909c2eaf3`; original design-only PR #148 stays separate and unmerged.
**Original independent BLOCK:** original frozen source+tests `ac1dd54ce2f17575e160a025367557fcf8afe598`; actual originally reviewed PR HEAD `dbff8a23832e6001d2ac3a11de806818583a382d`. Independent R2 BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:2:1. User supplied its three source-grounded findings. R2 old official PDF and R3 saved drafts PASS remain settled.
**Corrected frozen implementation+test commit:** `390296d421f3d1f0c47e6475ec327c91f02b8367`.
**Corrected frozen tree:** `13c4ad1fe7754c789d39710febf7ad3bd0183c55`.
**Corrected blob manifest:**
- `clinic_utilities/rf/persistence.py`: `8b6a59fbd15be65361a69ce51fb41c3a5322ba98`
- `clinic_utilities/rf/api.py`: `720777a0f3ef6e7b524addd54164fb04837cdbf0`
- `static/clinic-utilities/rf/app.js`: `6541769134f831c220d464a7af0ef8459a9a1418`
- `static/clinic-utilities/rf/index.html`: `4b0c984037bcadf19d1a071b6e4aa02f60b58610`
- `static/clinic-utilities/rf/styles.css`: `fd62bac121acced3ce7df0b5de5214c4f39bcb46`
- `test_rf_v2_native.py`: `b606e33a86aa3ad339660114cb8707ad0d16eb08`
- `test_rf_v2_ui.js`: `bcc33d3de951dd14d9e978d63a6a95e71ab5bd5c`
Unchanged prior official PDF helper `clinic_utilities/rf/prior_pdf.py`: `07ca2ebd8874f8b1aebd803f2f7a95a0539bdc2a`.
**Focused corrected RF CI:** [38172386948](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38172386948) SUCCESS on exact corrected commit (35 native tests, actual-browser-handler VM shim tests, previously accepted RF/JPEG/doctor/A1/A2/PDF and adjacent regressions); canonical impact PASS. Source-sha and CI must not be inferred from the docs-only HEAD.

## R1 — Original P1-01: no unrelated historical product

Prior defect: `get_rf_source(kind="procedure")` selected a prior application on patient identity/site/laterality without `indication_code`, silently transferring a product from a different RF shoulder indication. In the correction, historical identity/name may come from an identity-linked record, but **product_key** must come only from latest identity + site + laterality + exact indication compatible application, else remain **empty**. Review actual persistence code and test `test_previous_procedure_never_prefills_product_from_other_shoulder_indication` for both incompatible-blank and compatible-positive paths. Verify A2 issuance already enforces exact indication and no new product substitution risk.

## R2 — Original P1-02: never issue with unresolved patient name/identity conflict

Prior defect: typed name B + blank identity, select historical source A, UI retained name B and filled identity A, showed warning but could issue without explicit resolution; backend compared only identity. Review actual source-selection UI, identity/name provenance and protected `/api/create` path. Expected:
- selected prior source A with name A + typed B / blank identity → current B not silently overwritten, identity A shown, conflict clearly visible, **create blocked** before PDF submission and **422** server-side without explicit name resolution;
- explicit clinician choice to use previous name A or keep confirmed current name B; use-current confirmation is transmitted to API only as explicit affirmative and does not weaken source identity/site/side/indication checks;
- editing patient name/identity or reopening a stored draft invalidates stale confirmation and rechecks protected prior source; benign case/accent difference not spuriously blocked.
Tests: `test_previous_rf_name_conflict_blocks_issue_until_explicit_name_resolution` and actual-handler `testOldRFNameMismatchResolution`. Follow only affected consumers, not full UX/clinical privacy programme.

## R3 — Original P2-01: A2 source-first visible layout

Prior defect: HTML placed old RF lookup/upload below patient/product, diagnosis and A1-only cards, despite not having an actual three-field pre-search gate. Verify DOM order at *corrected head*: after pathway selection, A2 source-search/old-PDF upload appears before drafts, patient and diagnosis, hidden for A1; later actual-procedure and current VAS cards remain below. The native CSS `.builder>section{display:grid}` must not defeat `[hidden]`, so check the narrowly added visibility rule and pathway handler. Test asserts real HTML DOM order, JS display selection and unchanged A1. No unrelated UI redesign.

## R4 — Affected cumulative, unchanged authority

- No new RF doctor selection, no historical issuing-doctor substitution or loss of typed unrelated RF fields.
- Source and prior actual procedure require identity, site, unilateral side and exact indication; previously submitted RF never implies procedure performed.
- A1/A2 template, Horos JPEG validation/ordered append, imaging attachment/zero-write barriers unchanged.
- No PDF/JPEG binary persistence in saved drafts; prior saved-draft CRUD/revision and atomic finalization unchanged.
- Source/test delta restricted to P1-01/P1-02/P2-01 plus directly necessary CSS visibility and focused regressions; operational/checkpoint files change only as R0.

## R5 — Report & STOP

Use prior independent original post-code R2 **R2-PASS old-PDF** and **R3-PASS saved-draft** and prior design R2 PASS; avoid a second broad architecture or multi-round test programme. Check actual PR HEAD/tree, compare original source commit with corrected frozen commit, and verify docs-only HEAD preserved the corrected 7 code/test blobs. Evaluate P1-01, P1-02, P2-01 with source evidence, any genuinely new material affected risks, and no speculation.

Return exactly:

`IMPLEMENTATION-FIDELITY PASS | BLOCK | UNKNOWN`
`COVERAGE = COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`
`P0:P1:P2 = x:x:x`

plus exact source identities, 3 original finding dispositions, bounded affected-cumulative evidence, and STOP. Even PASS does not authorize merge, production deployment, or identifiable patient data.
