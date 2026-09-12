# CURRENT_OPERATIONAL.md — Clinical Documents Engine Phase 1

> **STATUS:** IMPLEMENTATION ACTIVE — COMMON CORE + SICK LEAVE V1.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh bootstrap main:** `bbfa26f3820f520d7f3ea312e6793fa55f399f01`.
> **Active branch:** `feat/clinic-documents-p1-sick-leave-2026-09-12`.
> **Active slice:** `CU-CLINICAL-DOCUMENTS-P1-SICK-LEAVE-2026-09-12`.
> **Writer:** this implementation conversation — bounded to Clinical Documents Phase 1 owners and required Clinic Utilities navigation/composition seams.
> **Production config/secrets authority:** NONE.
> **Patient persistence authority:** NONE.
> **Real-patient data in code/tests:** FORBIDDEN.

## 1. Product Owner authority

On 2026-09-12 the Product Owner explicitly authorized implementation after the Clinical Documents Engine design was frozen in the design conversation.

Implementation is intentionally phased. This active writer lock covers **Phase 1 only**:

```text
Common Clinical Documents Core
+
Sick Leave Certificate V1
+
protected Clinic Utilities navigation/integration
```

Accident/medico-legal multi-document ingestion, Evidence Ledger, live AI provider integration, literature retrieval, billing persistence and commercial user-template upload remain later separately reviewable implementation phases.

## 2. Frozen Phase-1 product contract

Sick Leave V1 fields:

- patient full name;
- ID type: `ADT` or `ARC`;
- ID number;
- diagnosis;
- leave from;
- leave through inclusive;
- issue date, defaulted in the browser to today but editable.

Actions:

- Preview;
- Download PDF;
- Clear;
- Load signature for current browser session;
- Import a previously generated Sick Leave V1 PDF.

Previous-PDF reuse modes:

```text
Extension
→ keep identity + diagnosis
→ next start = prior leave-through + 1 day
→ clear new end date

New leave, same patient
→ keep identity only
→ clear diagnosis and leave dates
```

## 3. Privacy / persistence boundary

Hard Phase-1 rules:

- no patient PostgreSQL write;
- no patient history registry;
- no localStorage/sessionStorage patient data;
- no autosave;
- no automatic reopening of prior patient/document;
- previous-document reuse only after explicit clinician file selection;
- signature is browser-session memory only and is never committed or persisted server-side;
- PDF generation/parsing is request-scoped/in-memory;
- unknown/non-V1 PDFs fail safely; no OCR guessing;
- no identifiable patient data or signature asset in repository fixtures/tests.

## 4. PDF/document contract

Generated PDF must be A4 portrait and contain:

- clinician identity/header;
- `ΒΕΒΑΙΩΣΗ ΑΣΘΕΝΕΙΑΣ`;
- patient name;
- the selected identity label (`ΑΔΤ` or `ARC`) and number;
- diagnosis;
- leave range;
- issue date;
- signature block.

Filename keeps Greek patient name and issue date but excludes diagnosis and identity number.

Generated PDFs carry machine-readable application metadata sufficient for reliable V1 re-import. The metadata must not become a separate server-side patient store.

## 5. Validation

Server remains authoritative for:

- required fields;
- allowed identity type;
- valid ISO dates;
- leave-through >= leave-from;
- bounded field lengths;
- accepted signature file type/size;
- recognized previous-PDF metadata/version.

No arbitrary numeric-format restriction is imposed on ADT/ARC values.

## 6. Common-core boundary

Phase 1 should establish reusable owners for:

```text
ClinicianProfile
PatientIdentity
SignatureAsset (ephemeral)
DocumentTemplate identity
DocumentMetadata
PDF renderer
```

The implementation must not prematurely introduce Evidence Ledger, causation, prognosis, billing database or generic AI authority into Sick Leave V1.

## 7. Adjacent-owner isolation

Do not modify:

- Clinical Learning contracts/runtime;
- osteoporosis guidance rules/evidence;
- RF request/PDF semantics;
- physiotherapy referral clinical taxonomy/evidence;
- current CY_GESY jurisdiction overlay semantics;
- patient/encounter/lab persistence schemas.

Navigation/composition-only edits to expose the new protected Clinic Utility are allowed.

## 8. Next legitimate action

Implement the bounded Phase-1 runtime and deterministic tests on the active branch, then run exact-head review/gates before any PR. No merge/deploy/config change is authorized merely by implementation completion.
