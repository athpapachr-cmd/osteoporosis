# SLICE_PLAN_CURRENT.md — CLINICAL DOCUMENTS PHASE 1 / SICK LEAVE V1

> **STATUS:** MERGED / DEPLOYED — PRODUCTION FUNCTIONAL SMOKE PENDING.
> **Activated:** 2026-09-12 Asia/Nicosia.
> **Release:** 2026-09-12 Asia/Nicosia.
> **Slice:** `CU-CLINICAL-DOCUMENTS-P1-SICK-LEAVE-2026-09-12`.
> **Bootstrap main:** `bbfa26f3820f520d7f3ea312e6793fa55f399f01`.
> **Implementation branch:** `feat/clinic-documents-p1-sick-leave-2026-09-12`.
> **PR:** `#97` — squash merged.
> **Exact reviewed/tested runtime head:** `fd88fa3b7ebce8aa9d97dfea12e2f2667495ed39`.
> **Clinical Documents PR gate:** `34717351567` — SUCCESS.
> **Release commit:** `a3ae5dd792a715a8306f5c279cc4bdbec4ca5b5e`.
> **Render deploy:** `dep-dairee0jo6nc73bt03b0` — LIVE.
> **Writer:** none.
> **Persistence:** NONE for patient/document/signature state.

## 1. Objective — achieved

The first bounded Clinical Documents runtime slice is released:

```text
ClinicianProfile
+ PatientIdentity
+ ephemeral SignatureAsset
+ DocumentMetadata
+ PDF rendering/re-import seam
→ Sick Leave Certificate V1
→ protected Clinic Utilities page
```

This release does not include Accident/Medico-Legal AI report functionality.

## 2. Runtime owners released

```text
clinic_utilities/clinical_documents/__init__.py
clinic_utilities/clinical_documents/models.py
clinic_utilities/clinical_documents/sick_leave.py
clinic_utilities/clinical_documents/api.py
static/clinic-utilities/sick-leave/index.html
static/clinic-utilities/sick-leave/app.js
static/clinic-utilities/sick-leave/styles.css
main.py
static/baseline-audit/g4-workspace-ergonomics.js
test_clinical_documents_sick_leave.py
.github/workflows/clinical-documents-p1-tests.yml
```

No existing RF/physio business-rule owner was rewritten.

## 3. Released Sick Leave V1 contract

`SickLeaveDraftV1` contains:

```text
patient_name
id_type: ADT | ARC
id_number
diagnosis
leave_from
leave_to
issued_on
derived_from_document_id? 
relation? = extension | new_leave_same_patient
```

Rules released:

- user text trimmed and bounded;
- `leave_to >= leave_from`;
- inclusive duration derived;
- no arbitrary numeric-only validation for ID;
- relation requires derived document id;
- whole `draft_json` bounded to 16 KiB before parsing;
- no patient state written to database/browser storage.

## 4. Browser workflow released

New document:

`Form → Preview → Download PDF → Clear`

Signature:

- explicit PNG/JPEG chooser;
- bounded size and parsed image validation;
- held only in JavaScript memory for current page lifetime;
- not persisted server-side;
- PDF remains usable without uploaded signature.

Previous PDF:

- explicit clinician-selected V1 PDF upload only;
- metadata-only application round-trip;
- no OCR/text guessing;
- `Επέκταση ίδιας άδειας`: retains identity + diagnosis and next start = prior leave-through + 1 day;
- `Νέα άδεια στον ίδιο ασθενή`: retains identity only and clears diagnosis/dates.

## 5. PDF / metadata contract released

Generated certificate is A4 portrait with Greek Unicode text, clinician header, `ΒΕΒΑΙΩΣΗ ΑΣΘΕΝΕΙΑΣ`, identity, diagnosis, leave range, issue date and signature/closing block.

Filename includes Greek patient name and issue date but excludes diagnosis and identity number.

Embedded V1 metadata includes:

```text
schema = sick_leave_certificate_v1
version = 1.0
document_id
patient_name
id_type
id_number
diagnosis
leave_from
leave_to
issued_on
derived_from_document_id?
relation?
```

Imported metadata is schema/type/value validated, including date-order consistency and relation provenance. It is solely a clinician-selected PDF round-trip mechanism, not a server patient database.

## 6. Released protected API

```text
GET  /clinical/clinic-utilities/sick-leave
GET  /clinical/clinic-utilities/sick-leave/api/contract
POST /clinical/clinic-utilities/sick-leave/api/preview
POST /clinical/clinic-utilities/sick-leave/api/pdf
POST /clinical/clinic-utilities/sick-leave/api/import-previous
```

Every page/API route uses the existing protected clinical access dependency.

## 7. Security / privacy boundary preserved

- no Clinical Documents database mutation;
- no localStorage/sessionStorage/indexedDB patient/signature state;
- no autosave;
- no patient identifiers in query strings;
- request/file sizes bounded;
- signature only validated PNG/JPEG;
- previous PDF bounded and parsed as PDF;
- unknown/malformed metadata fails closed;
- repository tests use synthetic/non-identifiable data only.

## 8. Acceptance evidence

Exact reviewed/tested runtime head:

`fd88fa3b7ebce8aa9d97dfea12e2f2667495ed39`

Clinical Documents PR workflow `34717351567` completed SUCCESS and covers the Phase-1 deterministic acceptance contract, including ADT/ARC handling, Greek PDF generation, date validation, metadata round-trip/reuse, unknown-PDF failure, signature handling, filename safety, authentication boundary, no persistence APIs, malformed metadata and oversized draft rejection.

Adjacent same-head evidence:

- CU-1 focused tests `34717351536`: SUCCESS;
- G3 guidance/longitudinal summary `34717351537`: SUCCESS;
- Clinical Learning L1 substantive runtime/contracts/schema steps passed before its expected scope/adjacent-owner guard failed for this non-Learning slice.

## 9. Release evidence

Product Owner explicitly authorized `Merge και deploy`.

PR #97 was squash merged as:

`a3ae5dd792a715a8306f5c279cc4bdbec4ca5b5e`

Render service `osteoporosis` has `autoDeploy=yes` on `main`, so no manual duplicate deploy was triggered. Automatic deploy:

`dep-dairee0jo6nc73bt03b0`

reached `LIVE` for the exact release commit.

No production environment variable or secret was changed.

## 10. Explicit exclusions retained

Not in this slice:

- Accident Report intake;
- medico-legal Evidence Ledger;
- AI drafting/provider call;
- targeted literature retrieval;
- causation matrix;
- prognosis engine;
- billing/fee-note/receipt runtime;
- persistent case store;
- user-uploaded reusable templates;
- Siri/Gemini;
- RF/physio changes beyond navigation exposure;
- production environment/secret mutation.

## 11. Closure boundary

Implementation, test, review, merge and deployment gates are satisfied.

`DEPLOYED != PRODUCTION-SMOKE-VERIFIED`.

An authenticated synthetic/product-owner functional smoke is still required before adding the production-smoke-verified label. Writer lock is released. Any Phase 2 implementation requires a fresh bounded slice and fresh Product Owner authority.
