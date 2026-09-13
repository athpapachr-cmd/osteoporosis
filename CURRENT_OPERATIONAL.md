# CURRENT_OPERATIONAL.md — Clinical Documents Engine Phase 2 / Medical Report V1

> **STATUS:** IMPLEMENTATION ACTIVE — EPHEMERAL MEDICAL REPORT WORKSPACE V1.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh bootstrap main:** `8993a1c4b585c590a543c907ea6b2eba32bbccdc`.
> **Active branch:** `feat/clinical-documents-p2-medical-report-v1-2026-09-13`.
> **Active slice:** `CU-CLINICAL-DOCUMENTS-P2-MEDICAL-REPORT-V1-2026-09-13`.
> **Writer:** current Clinical Documents implementation conversation, bounded to the owners listed in `SLICE_PLAN_CURRENT.md`.
> **Production config/secrets authority:** NONE in this implementation slice.
> **Patient/case persistence authority:** NONE.
> **Real-patient data in repository/tests:** FORBIDDEN.

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

This activates Phase 2 implementation only. Merge/deploy, production OpenAI key/config mutation, persistent case storage, billing runtime and user-uploaded template marketplace remain separate gates.

## 2. Active product scope

Implement a protected **Medical Report V1** workspace inside Clinic Utilities for:

- `accident_medical_report`;
- `medico_legal_expert_report`.

The clinician may provide:

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

V1 source files are explicitly selected by the clinician for the active browser session only.

## 3. Authority model

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

The AI may extract, summarize, organize and draft. It may not silently promote inference to fact, causation, diagnosis, permanence or prognosis. Consequential opinion text remains marked for clinician review until explicitly confirmed.

## 4. V1 workflow

```text
Case details
→ clinician pasted history/instructions
→ upload source documents
→ local server text extraction
→ AI source classification + Evidence Ledger + Timeline + structured report draft
→ clinician review/edit
→ targeted prognosis/literature research (generalized, identity-free query)
→ clinician review/edit of prognosis/future needs
→ final confirmation
→ signed multi-page PDF
```

## 5. Privacy / persistence boundary

V1 is request/session scoped:

- no Clinical Documents patient/case database;
- no automatic patient history;
- no localStorage/sessionStorage/indexedDB case or PHI state;
- no autosave;
- uploaded source bytes are parsed in memory and are not persisted by this package;
- no source files are committed to the public repository;
- no request body or extracted source text is deliberately logged by the feature;
- browser refresh/close loses the working case unless a future explicit export feature is separately added.

## 6. AI provider boundary

Runtime must support an OpenAI provider behind explicit server-side gates:

- `OPENAI_API_KEY` present;
- `CLINICAL_DOCUMENTS_AI_ENABLED=true`;
- identifiable-record processing additionally requires `CLINICAL_DOCUMENTS_PHI_PROVIDER_APPROVED=true`.

No provider secret or production flag is authorized to be added/changed in this slice.

Provider calls use `store=false` and do not use OpenAI Files/Vector Store persistence. Identifiable source text is not sent unless the explicit PHI-provider approval gate is active.

Literature/web research is built from generalized clinical questions and must exclude direct patient identifiers.

## 7. V1 source/file boundary

Initial accepted file types:

- PDF with extractable text;
- TXT;
- Markdown;
- DOCX via local XML extraction.

No OCR guessing in this slice. Image-only/scanned PDFs must be surfaced as `no_extractable_text` and require clinician-provided text or a later separately approved vision/OCR path.

Files/count/bytes/extracted-character totals must be bounded server-side.

## 8. V1 report structure

Default editable sections:

1. Patient / Case Details
2. Purpose and Instructions
3. Sources and Documents Reviewed
4. History of the Incident
5. Initial Medical Management
6. Chronological Clinical Course
7. Current / Reported Symptoms
8. Objective Clinical Findings
9. Investigations
10. Treatment / Procedures
11. Diagnoses
12. Pre-existing Conditions
13. Causation
14. Functional Consequences
15. Work Incapacity
16. Prognosis
17. Future Needs
18. Summary of Opinion
19. Bibliography
20. Signature / optional declaration

Irrelevant sections may remain empty/hidden. Formal jurisdiction-specific declaration text is not auto-inserted in V1.

## 9. Literature / prognosis rule

The AI first proposes targeted prognosis questions from the case. A separate research action may use hosted web search to gather current sources.

The UI must expose source links/citations and keep literature separate from patient facts. Research text is AI draft and must remain editable/reviewable before final PDF inclusion.

## 10. Finalization gate

Final PDF generation requires explicit clinician confirmation that:

- the report text has been reviewed;
- diagnosis/causation/prognosis/future-needs wording is clinician-approved;
- bibliography/research content has been reviewed where included.

No unreviewed AI draft should enter the final PDF by default.

## 11. Explicit exclusions

Not authorized in this slice:

- persistent patient/case database;
- automatic GeSY integration;
- OCR/vision for image-only documents;
- autonomous diagnosis/causation/prognosis;
- permanent-impairment scoring;
- compensation/damages estimation;
- billing / Fee Note / Receipt runtime;
- tax/VAT logic;
- user-uploaded reusable template platform;
- Siri/Gemini/native app;
- RF/physio/osteoporosis-guidance/Clinical-Learning business-rule mutation;
- production environment/secret mutation.

## 12. Next legitimate action

Implement the bounded Medical Report V1 runtime, browser workspace, provider abstraction, deterministic source parsing, structured AI contract, literature-research seam, PDF renderer and focused synthetic regression suite on the active branch. Then run exact-head review/gates. No PR/merge/deploy is implied by implementation completion.