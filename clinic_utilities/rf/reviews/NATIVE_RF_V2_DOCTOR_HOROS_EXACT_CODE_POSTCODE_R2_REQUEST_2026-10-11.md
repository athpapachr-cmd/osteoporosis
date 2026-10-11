# Native RF v2 — doctor choice + Horos JPEG independent post-code R2

**MODE:** fresh, separate, independent READ-ONLY implementation-fidelity R2; declared scope only; no implementation/merge/deploy or clinical data.

**Repository:** `athpapachr-cmd/osteoporosis`
**Base verified at author bootstrap:** `8469e12e8803ab1f9a882437527ed69a59a5e9b6`.
**Frozen reviewed code+tests commit (tested):** `6ca642635c7415be3c0d3fec492143e75d146ffb`.
**CI (exact above commit):** [RF hotfix workflow 38118950747](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38118950747) — **SUCCESS**, including both native RF suites and the added synthetic tests.
**Implementation branch:** `feat/rf-v2-doctor-horos-jpeg-2026-10-11`.
**PR head:** resolve exact SHA fresh from its DRAFT PR (the PR body also pins this SHA after author closeout); subsequent documentation-only checkpoints must preserve the exact six production-code and two test Git blobs from the frozen tested commit. Never infer that docs commit equals tested code identity.

## Independent question / gate

Does the bounded implementation at the exact PR head faithfully implement only the two independently pre-code approved additions, while preserving native RF v2 authority, imaging validation, history identity and no-write-on-failure?

Fresh-verify remote main and current PR/ref; apply `AGENTS.md`, six canonicals, `PROCEDURES.md`, `clinic_utilities/rf/CURRENT.md` and inspect actual code/test Git blobs at PR head. Reuse the independent **pre-code R2 PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1=0:0**; do not restart design. Check `rf_gateway.py` remains rollback-only and prior PR #116 MERGED, with old RF CURRENT HOLD historical.

## Finite evidence questions

**R1 — Doctor authority:** Only a missing `doctor_key` uses the original `RF_DOCTOR_PROFILE_JSON` default; explicit null/empty/unknown/unconfigured is rejected; select `maxoutlou` uses *private server* `RF_DOCTOR_MAXOUTLOU_PROFILE_JSON`. Browser sends only `doctor_key`, contract exposes labels/configured booleans only. Six populated official PDF doctor fields correct for A.1/A.2 and both synthetic profiles. New application `payload_json.doctor_key` persisted without schema change, historical records without key remain unattributed; old actual A.2 procedure has no reassignment.

**R2 — Image validation, EXIF, resource limits:** Original imaging PDF remains mandatory with its existing semantic check/clinician-confirmation path. Optional file selection accepts 0/1/N JPEG, actual JPEG/integrity, count/byte/dimension/pixel ceilings before uncontrolled decoding; no persistent images or thumbnails. EXIF rotations and *mirrored* orientations 1–8 produce correct readable images with PyMuPDF, no gratuitous Pillow.

**R3 — Assembly and persistence:** For both A.1/A.2, PDF pages ordered exactly official selected A pages → every original imaging PDF page → one page per JPEG in attachment order. Any invalid/truncated/oversized/corrupt image blocks the entire operation **before** `record_legacy_procedure` or `record_application`. No new write side-effect or history identity mutation.

**R4 — Browser UX:** Normal dropdown defaults to Papachristou, shows unavailable secondary as not configured, previews thumbnails, allows removing specific attachments, revokes object URLs, keeps upload order, displays useful errors. Browser submits only `doctor_key` and ordered multipart `horos_jpegs`; verify synthetic JavaScript handler test and original UI integrity.

**R5 — Bounded preservation:** Inspect source diff (only six RF code/UI files, existing RF test changes, two status checkpoints and this request). `main.py`, Home, official PDF template, database schema/persistence, native route, unilateral/anatomical/indication/medication/physio behavior, imaging semantic guard, legacy RF gateway unchanged. Reuse actual CI run `38118950747` and existing RF native/hotfix gate, not unrelated redesign or redundant broad tests.

## Required reviewer handback

Return exactly `IMPLEMENTATION-FIDELITY PASS | BLOCK | UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, `P0:P1:P2`, R1–R5 with source/blob/CI evidence, exact reviewed PR HEAD, any material issue + minimum bounded correction, and **STOP**. Do not write, merge, deploy, activate private config or access real patients. An author cannot independently close this R2 review.
