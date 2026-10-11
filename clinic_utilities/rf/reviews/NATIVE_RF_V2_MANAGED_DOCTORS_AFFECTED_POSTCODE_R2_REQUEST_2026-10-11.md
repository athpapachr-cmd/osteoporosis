# Native RF v2 managed doctors — independent affected post-code R2 request (2026-10-11)

**MODE: fresh separate independent READ-ONLY / ONE affected exact-head implementation-fidelity R2. No code changes, merge, deploy or real data.**

Repository: `athpapachr-cmd/osteoporosis`; PR [#146](https://github.com/athpapachr-cmd/osteoporosis/pull/146) DRAFT, unmerged at request writing.
Expected remote main `8469e12e8803ab1f9a882437527ed69a59a5e9b6` (fresh verify).
Frozen **managed-doctor implementation+test SHA**: `19db2c6b6c4111259df7ef8cfca2888ca785ce06`. Actual PR HEAD is later docs-only; fresh-resolve and pin it, verify all affected source/test blobs match frozen commit. Exact successful [RF v2 hotfix regression CI #38123786948](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38123786948) at frozen SHA.

Approved pre-code design exact HEAD `6f70d63657536662ce330fb000fd0269994c7401`: `clinic_utilities/rf/RF_V2_MANAGED_DOCTORS_BOUND_DESIGN_2026-10-11.md`. Independent pre-code PASS/COMPLETE/0:0:0 supplied by Product Owner. Prior independent JPEG P1-01 closure PASS/0:0:0 is reused; **do not repeat JPEG core review**. Check source `clinic_utilities/rf/pdf.py` blob remains `b3573bd74501631eb5c2b767a79a7ead719a8aa7`.

**Decision:** Is the bounded dynamic managed-doctors replacement complete, faithful to accepted design and compatible with RF/JPEG reuse, with no new material risk in the affected contracts?

## Bounded evidence questions
- **R1 — Source/identity:** Actual main/PR head/tree; exact cumulative main-to-head RF change list, source/test hashes, inherited JPEG unchanged; docs-only commits after frozen code/test SHA. No new PDF engine/AI/external service/Home or unrelated program changes.
- **R2 — Protected profile authority:** RFBase `clinic_rf_doctor_profiles` existing database, persistent across restart; GET/list, POST add, PUT edit, status deactivate/reactivate, field validation and synthetic multi-doctor tests. RF legacy `RF_DOCTOR_PROFILE_JSON` imported only for default and never overwrites managed edits; no new per-colleague Render env. Client only sends stable key, server resolves protected active profile. Unauthorized/invalid/stale revision rejected.
- **R3 — Snapshot, races and write authority:** Within a transaction/row lock for protected RF create, consistent exact six-field issuer snapshot generated from active managed doctor and recorded with key in new application's `payload_json`; concurrent edit/deactivate cannot cause mixed issuer identity; historic records lacking key remain unattributed; prior A.2 procedure ownership unaffected; errors/invalid profile or broken JPEG cannot write new application or legacy-history row. Include post-change edit/deactivate/restart synthetic oracle.
- **R4 — UX:** Existing compact dropdown + on-page expandable managed directory; Nth doctor can be added/edited/deactivated/reactivated with no code or Render access; no browser storage, free-text doctor payload in application, XSS or separate app; preserve existing JPEG thumbnails/remove and official PDF.
- **R5 — Responsibilities/preservation:** Issuer != obligation owner. Creating an RF for a colleague does **not** create approvals/scheduling/procedure tasks, Home attention items or automatic clinician follow-up. Rejected-RF reason/reapplication tracking intentionally deferred; no new tracking in this PR. Reuse pre-existing A.1/A.2, pharmacology, mandatory imaging PDF/validation, JPEG EXIF/integrity and cumulative RF CI without an unrelated architecture review.

## Frozen affected source/test Git blobs
- `clinic_utilities/rf/catalog.py`: `bf0745a865f983d87961907345fea33413fdf714`
- `clinic_utilities/rf/persistence.py`: `2d8f2c804427d14954e8bbf4c8cde398ff7f4bb8`
- `clinic_utilities/rf/api.py`: `2359f859d8d57a1c3711fbd158408c857eaf9921`
- `static/clinic-utilities/rf/index.html`: `16c01d6314c86b0d08e000c4021f806c7f27526f`
- `static/clinic-utilities/rf/app.js`: `7b78c47d6359c78777227ad0a4e2763cc3412027`
- `static/clinic-utilities/rf/styles.css`: `bf9db5359294c45b30c73cc4c19d0143c34dd117`
- `test_rf_v2_native.py`: `0398feea53a2feec7e0aa0f6e0a3e20bcff088af`
- `test_rf_v2_ui.js`: `9bdf8808dfc782792659cb260764bdac8ba359d9`
- `clinic_utilities/rf/pdf.py` **unchanged**: `b3573bd74501631eb5c2b767a79a7ead719a8aa7`.

Apply `AGENTS.md` six canonical bootstrap, `PROCEDURES.md` P4–P5 and RF CURRENT. Do not invent new product scope, extra reviewer rounds or insist on modifying unrelated Learning-only CI scope checks.

**Return:** IMPLEMENTATION-FIDELITY **PASS | BLOCK | UNKNOWN**, COMPLETE_FOR_DECLARED_SCOPE | PARTIAL, P0:P1:P2, explicit R1–R5 evidence and smallest findings if any; actual reviewed PR HEAD + verified frozen blob identities, and STOP. Writer is released; no merge/release authority.
