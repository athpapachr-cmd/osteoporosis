# CURRENT_OPERATIONAL.md — Clinical Documents Engine Phase 2 / Medical Report V1

> **STATUS:** IMPLEMENTED / TESTED / EXACT-HEAD REVIEWED — RELEASE HOLD.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh bootstrap main / merge base:** `8993a1c4b585c590a543c907ea6b2eba32bbccdc`.
> **Implementation branch:** `feat/clinical-documents-p2-medical-report-v1-2026-09-13`.
> **Active slice:** `CU-CLINICAL-DOCUMENTS-P2-MEDICAL-REPORT-V1-2026-09-13`.
> **Exact reviewed/tested runtime head:** `3a19a09280ff682badfd4866e19d6a4f3cb9c208`.
> **Clinical Documents P2 gate:** run `34733760918` — SUCCESS.
> **Writer:** none — implementation checkpoint closed in RELEASE HOLD.
> **Production config/secrets authority:** NONE; no production AI configuration was changed.
> **Patient/case persistence authority:** NONE.
> **Real-patient data in repository/tests:** FORBIDDEN / NOT USED.

## 1. Product Owner authority

On 2026-09-13 the Product Owner explicitly authorized development of the medical-report workflow and clarified the intended operating model:

```text
clinician supplies the history/instructions
+ uploads relevant GeSY/clinical/specialist/imaging/procedure/discharge documents
→ system extracts source-bounded evidence and chronology
→ AI-assisted report draft
→ prognosis/future-needs research layer
→ clinician review/edit/confirmation
→ final signed PDF
```

That bounded implementation is now complete and tested on the active branch. This checkpoint does **not** authorize merge/deploy, production OpenAI key/config mutation, persistent case storage, billing runtime or commercial template upload.

## 2. Implemented product scope

Protected **Medical Report V1** workspace inside Clinic Utilities supports:

- `accident_medical_report`;
- `medico_legal_expert_report`.

The clinician can provide:

- a pasted history / own synthesis / special instructions;
- GeSY visit history;
- own notes;
- other doctors' notes;
- specialist reports;
- hospital/emergency records;
- imaging reports;
- procedure/operation notes;
- admission/discharge summaries;
- laboratory or physiotherapy reports;
- prior reports / sick-leave certificates.

The browser exposes clinician-editable source classification before analysis. Uploaded files remain explicit current-session inputs only.

## 3. Authority model preserved

Hard invariant:

```text
SOURCE FACT
!= PATIENT-REPORTED FACT
!= CLINICIAN FINDING
!= SPECIALIST OPINION
!= LITERATURE EVIDENCE
!= AI INFERENCE
!= FINAL CLINICIAN OPINION
```

The AI extracts, summarizes, organizes and drafts. It does not silently promote inference to fact, diagnosis, causation, permanence or prognosis. Diagnosis/causation/pre-existing-condition/prognosis/future-needs outputs remain review-required. Final PDF generation requires explicit clinician confirmation.

Source documents are also treated as untrusted quoted data for prompt-injection purposes: instructions/links/commands embedded in an uploaded record are not model instructions.

## 4. Implemented workflow

```text
Case details
→ clinician pasted history/instructions
→ upload + clinician classification of source documents
→ local request-scoped text extraction
→ AI Evidence Ledger + Timeline + structured report draft
→ deterministic provenance/reference checks
→ work-incapacity interval overlap/gap checks
→ clinician review/edit
→ targeted prognosis/literature research using generalized identity-free questions
→ clinician review/edit of prognosis/future needs/research text
→ explicit final confirmation
→ optional session-only signature
→ professional multi-page Greek PDF
```

## 5. Source/file boundary

Accepted V1 file types:

- PDF with extractable text;
- TXT;
- Markdown;
- DOCX via local OOXML/XML extraction.

Implemented safety/resource bounds include:

- max 20 uploaded files;
- max 12 MiB per file;
- max 40 MiB aggregate uploaded bytes;
- max 350,000 extracted characters;
- max 5,000 PDF pages;
- bounded DOCX decompressed `word/document.xml` size;
- bounded typed request/final payloads.

Image-only/scanned PDFs are surfaced as `no_extractable_text`; V1 does not OCR or guess their contents.

## 6. Evidence / timeline integrity

AI output is typed and post-validated against deterministic source truth.

Validation rejects or flags:

- nonexistent source IDs;
- nonexistent page references;
- duplicate evidence IDs;
- timeline references to nonexistent sources/evidence;
- diagnosis/future-needs references to nonexistent evidence;
- work-incapacity interval references to nonexistent sources/evidence;
- work-incapacity intervals whose end precedes start;
- overlapping work-incapacity periods;
- gaps between extracted work-incapacity periods;
- repeated conflict keys across source evidence.

The AI instructions require exact work/sick-leave interval boundaries to be extracted only when both dates are source-supported; missing boundaries must not be inferred.

## 7. AI provider boundary

The runtime contains an OpenAI Responses API provider behind explicit server-side gates:

```text
OPENAI_API_KEY
+ CLINICAL_DOCUMENTS_AI_ENABLED=true
```

Identifiable-record analysis additionally requires:

```text
CLINICAL_DOCUMENTS_PHI_PROVIDER_APPROVED=true
```

Provider calls use `store=false`; no OpenAI Files/Vector Store persistence is introduced. The implementation records token/web-search usage metadata for later cost accounting.

No provider secret, AI enable flag or PHI-approval flag was changed in production during this implementation.

## 8. Literature / prognosis

The case analysis generates targeted prognosis questions rather than invented citations.

The explicit research action:

- builds generalized clinical questions from diagnoses/problems/prognosis/future-needs topics;
- defensively excludes direct patient name, identity number and instructing reference from the external research prompt;
- uses hosted web search;
- returns research prose separately from clickable URL citations;
- keeps research editable and under final clinician confirmation.

## 9. Privacy / persistence boundary

V1 remains request/session scoped:

- no Clinical Documents patient/case database;
- no automatic patient history;
- no localStorage/sessionStorage/indexedDB case or PHI state;
- no autosave;
- uploaded source bytes parsed in memory and not persisted by this package;
- no source files committed to the public repository;
- no feature-owned database writes;
- no patient identifiers in query strings;
- signature held only in current browser memory and request payload;
- browser refresh/close loses the working case.

All test data are synthetic/non-identifiable.

## 10. Final PDF

The implemented renderer produces professional multi-page A4 Greek reports with:

- clinician header;
- case/patient information;
- editable report sections;
- automatic page breaks;
- page numbering;
- optional literature synthesis/citations;
- optional clinician-authored declaration;
- optional session-only signature.

Long unbroken references/URLs are wrapped. Filename contains report type + Greek patient name + report date and excludes diagnosis/identity number.

## 11. Exact-head evidence

Exact reviewed/tested runtime head:

`3a19a09280ff682badfd4866e19d6a4f3cb9c208`

Clinical Documents P2 workflow run:

`34733760918` — **SUCCESS**.

That exact run passed:

- Python syntax;
- browser JavaScript syntax;
- OpenAI Responses SDK contract guard without a live provider call;
- Medical Report V1 deterministic tests;
- source classification tests;
- work-incapacity consistency tests;
- existing Sick Leave V1 regression;
- existing Clinic Utilities navigation regression.

Exact branch comparison against fresh `main`/merge-base `8993a1c4b585c590a543c907ea6b2eba32bbccdc` is `ahead 31 / behind 0`. Changed runtime scope is confined to Clinical Documents owners plus `main.py` composition and a four-line Clinic Utilities navigation extension. No RF/physio/Clinical Learning/osteoporosis-guidance business-rule or persistence-schema owner was modified.

## 12. Explicit exclusions retained

Not implemented/authorized in this checkpoint:

- persistent patient/case database;
- automatic GeSY API integration;
- OCR/vision for image-only documents;
- autonomous diagnosis/causation/prognosis;
- permanent-impairment scoring;
- compensation/damages estimation;
- billing / Fee Note / Receipt runtime;
- tax/VAT logic;
- user-uploaded reusable template platform;
- Siri/Gemini/native app;
- automatic jurisdiction-specific declaration wording;
- RF/physio/osteoporosis-guidance/Clinical-Learning business-rule mutation;
- production environment/secret mutation.

## 13. Next legitimate action

Implementation is in **RELEASE HOLD**.

A future release requires separate explicit Product Owner authority for PR/merge/deploy and a separate, deliberate production-provider/privacy/config decision before identifiable patient records may be sent to the AI provider. No merge/deploy or production AI configuration is implied by this implementation completion.