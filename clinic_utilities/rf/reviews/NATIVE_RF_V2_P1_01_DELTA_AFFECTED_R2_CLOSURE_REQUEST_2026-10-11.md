# Native RF v2 P1-01 — ONE independent delta + affected-cumulative R2 closure

**MODE: separate fresh independent READ-ONLY / ONE bounded post-correction implementation-fidelity R2 review.** No code, merge, deploy, production configuration, or real patient images/data.

Repository: `athpapachr-cmd/osteoporosis`.
PR: [#146](https://github.com/athpapachr-cmd/osteoporosis/pull/146) (DRAFT/OPEN/UNMERGED when prepared).
Verified baseline main at correction: `8469e12e8803ab1f9a882437527ed69a59a5e9b6`.
**Previously BLOCKED original exact PR HEAD:** `37180026325a0e165881a315c166d5cf36b8d560`.
**Corrected immutable code+tests HEAD:** `e9b2845bf5843e5bd70314a785e68f578f2c5618`.
**Exact corrected RF CI (SUCCESS):** [38120599122](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38120599122).
**Exact PR HEAD to review:** fresh-resolve `feat/rf-v2-doctor-horos-jpeg-2026-10-11` PR #146 HEAD, and pin the full Git SHA in the verdict. The PR body is updated with final HEAD after author writer release. Verify `clinic_utilities/rf/pdf.py` and `test_rf_v2_native.py` Git blobs at that HEAD exactly match corrected tested commit `e9b2845bf5843e5bd70314a785e68f578f2c5618`; unrelated six RF code/UI and one JS test blobs should match the original frozen tested head `6ca642635c7415be3c0d3fec492143e75d146ffb`. After `e9b2845bf5843e5bd70314a785e68f578f2c5618`, further author commits may change ONLY documentation/checkpoints and this request.

## Decision to support

Is the **sole original P1-01** materially closed by the minimum JPEG integrity correction, without creating a new material issue in affected R2/R3 or regressing reused R1/R4/R5?

Reuse the independent pre-code R2 PASS and original post-code R2 R1/R4/R5 PASS (details preserved in [BLOCK receipt](clinic_utilities/rf/reviews/NATIVE_RF_V2_P1_01_INDEPENDENT_R2_BLOCK_RECEIPT_2026-10-11.md)). Apply `AGENTS.md`, six canonicals, `PROCEDURES.md` P5 and RF CURRENT. No repeated comprehensive architecture/release review.

### C1 — Original failure, exact reproduction, write barrier

- Inspect only `clinic_utilities/rf/pdf.py` and `test_rf_v2_native.py` correction delta. Verify direct early-EOI / truncated entropy with spoofed `FF D9` is rejected **despite tolerant PyMuPDF**; check progressive/sequential EXIF-safe normal JPEG remains usable.
- The synthetic regression `test_repaired_eoi_cannot_hide_truncated_jpeg_or_write_database` attempts A.1 with damaged JPEG and A.2 with good image followed by damaged JPEG, verifies HTTP 422 and **zero** `clinic_rf_applications` and `clinic_rf_procedure_history` rows. Verify no residual write path prior to PDF full completion.
- Inspect narrow input resource limits (bytes/dimensions/pixels) before scan processing. New decoder warnings, "verify only" or magic suffix checks alone are not closure.
- Evaluate whether the correction fails closed on unsupported JPEG entropy coding without breaking the existing Horos/EXIF cases. Do not require Pillow unless source-proven unavoidable.

### C2 — Affected cumulative preservation

- Reuse existing validations of doctor-key authority, A.1/A.2, imaging PDF mandatory/semantic validation, two doctors, EXIF 1–8/mirrored, zero/one/multiple ordered JPEG PDF pages, browser thumbnails/remove, medication/unilateral/procedure identity.
- Confirm `clinic_utilities/rf/catalog.py`, `clinic_utilities/rf/api.py`, three static RF files and RF JS regression are unchanged from original frozen code/test blob identities, plus official template, `main.py`, gateway, schema, dependencies, Home and other systems untouched.
- Use [CI 38120599122](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38120599122) as supporting evidence; do not claim unrelated design-only scope CI is RF-regression green.

### C3 — Exact head, review and STOP

- Fresh verify current main, PR #146 actual DRAFT/merged state and exact Git HEAD/tree, inspect original BLOCK receipt and corrected tests.
- Distinguish: author-tested source != reviewer PASS; documentation-only after tested source must preserve its blobs.
- Report **IMPLEMENTATION-FIDELITY PASS | BLOCK | UNKNOWN**, **COMPLETE_FOR_DECLARED_SCOPE | PARTIAL**, **P0:P1:P2**, explicit C1–C3 status, closure of original P1-01 or smallest correction required, exact reviewed SHA and **STOP**. No further implementation, general research, release decision, merge, deploy or real-data run.
