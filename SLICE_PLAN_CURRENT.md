# SLICE_PLAN_CURRENT.md — CLINICAL DOCUMENTS PHASE 2 / MEDICAL REPORT V1

> **STATUS:** IMPLEMENTATION ACTIVE.
> **Activated:** 2026-09-13 Asia/Nicosia.
> **Slice:** `CU-CLINICAL-DOCUMENTS-P2-MEDICAL-REPORT-V1-2026-09-13`.
> **Bootstrap main:** `8993a1c4b585c590a543c907ea6b2eba32bbccdc`.
> **Branch:** `feat/clinical-documents-p2-medical-report-v1-2026-09-13`.
> **Writer:** current Clinical Documents implementation conversation.
> **Scope:** ephemeral multi-source medical-report workspace + Evidence Ledger/Timeline + AI-assisted draft + prognosis/literature seam + reviewed final PDF.
> **Persistence:** NONE for patient/case/source/report state.

## 1. Objective

Deliver the first clinically useful accident/medico-legal report workflow on top of the already released Clinical Documents core.

The Product Owner workflow is:

```text
pasted clinician history/instructions
+ explicitly uploaded relevant records
→ evidence/provenance extraction
→ chronology
→ structured editable report draft
→ targeted prognosis/future-needs research
→ clinician review
→ final signed PDF
```

The output is an **AI-assisted draft under clinician authority**, not autonomous medico-legal opinion.

## 2. Runtime owners

New preferred owners:

```text
clinic_utilities/clinical_documents/report_models.py
clinic_utilities/clinical_documents/report_sources.py
clinic_utilities/clinical_documents/report_ai.py
clinic_utilities/clinical_documents/report_pdf.py
clinic_utilities/clinical_documents/report_api.py
static/clinic-utilities/medical-report/index.html
static/clinic-utilities/medical-report/app.js
static/clinic-utilities/medical-report/styles.css
```

Composition/navigation seams:

```text
clinic_utilities/clinical_documents/__init__.py
main.py
static/baseline-audit/g4-workspace-ergonomics.js
```

Focused tests/workflow:

```text
test_clinical_documents_medical_report.py
.github/workflows/clinical-documents-p2-tests.yml
```

Do not rewrite Sick Leave V1 semantics except shared helpers where backward-compatible and explicitly regression-tested.

## 3. Case contract

`MedicalReportCaseV1`:

```text
report_type: accident_medical_report | medico_legal_expert_report
patient_name: required
id_type: ADT | ARC | OTHER
id_number: optional/required by clinician use
birth_date: optional
occupation: optional
incident_date: optional
report_date: required
instructing_party: optional
instructing_reference: optional
purpose_and_questions: optional
clinician_context: optional free text
```

The browser defaults report date to today but keeps it editable.

Patient identifiers are excluded from literature-search prompts.

## 4. Source ingestion contract

Accepted V1 files:

```text
.pdf
.txt
.md
.docx
```

Limits:

- max 20 uploaded files;
- max 12 MiB per file;
- max 40 MiB aggregate uploaded bytes;
- max 350,000 extracted characters across all sources;
- filename length and extracted page count bounded;
- unsupported files fail clearly.

PDF extraction is local/request-scoped with PyMuPDF. Text is retained page-by-page for provenance.

DOCX is extracted locally from OOXML/XML without uploading the original to a third-party file store.

Image-only/scanned PDF with insufficient extractable text returns `no_extractable_text`; V1 performs no OCR guessing.

The clinician-context textarea is represented as a dedicated `clinician_note_self` source so it is distinguishable from external records.

## 5. Source object

`ReportSourceV1`:

```text
source_id
filename
source_type
status: extracted | no_extractable_text
page_count
character_count
pages[]:
  page_number
  text
```

AI may propose source metadata/classification, but source identity and extracted text remain deterministic server output.

## 6. Evidence Ledger contract

`EvidenceItemV1` contains:

```text
evidence_id
source_id
page_numbers[]
date_optional
date_text_optional
evidence_type
statement
certainty
conflict_key_optional
requires_clinician_review
```

Evidence types include:

```text
patient_reported
clinician_observed
specialist_opinion
imaging_finding
lab_finding
procedure_performed
treatment_given
medication
diagnosis_recorded
functional_limitation
work_absence
pre_existing_condition
causation_opinion
prognostic_opinion
recommended_future_care
other
```

Post-AI validation rejects references to nonexistent source IDs/pages and duplicate evidence IDs.

## 7. Timeline contract

`TimelineEventV1` contains:

```text
event_id
date_optional
date_text_optional
title
summary
source_ids[]
evidence_ids[]
conflict_flags[]
```

Chronology must not silently resolve source disagreements. Unknown/approximate dates remain explicit.

Deterministic post-processing checks at minimum:

- same-date duplicate/overlap warning candidates;
- sick-leave overlap/gap calculations when intervals are represented;
- source/evidence referential integrity;
- impossible source page references.

## 8. AI report contract

`MedicalReportAnalysisV1` returns:

```text
source_summaries[]
evidence_items[]
timeline[]
diagnosis_analyses[]
report_sections[]
prognosis_questions[]
future_needs[]
warnings[]
usage
```

Every generated diagnosis/causation/prognosis/future-needs item is explicitly marked `requires_clinician_review=true`.

The model instructions must:

- write the report draft in professional Greek;
- preserve attribution and uncertainty;
- never invent absent dates/findings/tests/diagnoses;
- distinguish patient history, own findings, specialist opinion and investigation results;
- flag conflicts instead of reconciling them silently;
- avoid compensation-value/damages calculations;
- avoid jurisdiction-specific legal declarations unless explicitly configured later.

## 9. OpenAI provider seam

The implementation may use the installed OpenAI Python SDK and Responses API.

Default server-side model may be configured by:

```text
CLINICAL_DOCUMENTS_AI_MODEL
CLINICAL_DOCUMENTS_RESEARCH_MODEL
```

with a high-quality current default permitted in code.

Provider availability requires:

```text
OPENAI_API_KEY
+ CLINICAL_DOCUMENTS_AI_ENABLED=true
```

Identifiable patient source processing additionally requires:

```text
CLINICAL_DOCUMENTS_PHI_PROVIDER_APPROVED=true
```

Provider calls use `store=false`.

Tests never call the live provider; they use deterministic synthetic/fake provider output.

## 10. Research/literature seam

The analysis generates targeted prognosis/future-needs questions.

A separate endpoint may run hosted web search using a prompt containing only generalized clinical facts/questions required for the research task, not patient name/ID/contact details.

Research response contains:

```text
research_text
citations[]:
  title
  url
queries[] when available
usage
```

The UI must display citations as clickable sources. Research text is not silently promoted to final report text.

## 11. Browser workflow

### Step 1 — Case
Case fields + report type + purpose/instructions.

### Step 2 — Sources
- large clinician-history/instructions textarea;
- multi-file chooser;
- file list and extraction warnings.

### Step 3 — Generate draft
Server performs request-scoped extraction + AI analysis.

### Step 4 — Review
Visible tabs/panels for:
- source summaries;
- Evidence Ledger;
- Timeline;
- diagnoses/causation;
- editable report sections.

All AI draft content visibly carries a review-required state until final confirmation.

### Step 5 — Prognosis & literature
Explicit button runs research only when requested. Sources/citations shown separately.

### Step 6 — Final report
Clinician edits final section text, optionally loads session-only signature, checks confirmation, previews/downloads PDF.

## 12. PDF contract

Professional multi-page A4 Greek report:

- clinician header on first page;
- patient/case information block;
- section hierarchy;
- readable long-form typography;
- automatic page breaks;
- page number/footer;
- bibliography when included;
- signature/clinician closing block;
- optional declaration section only if clinician-authored/confirmed text is supplied.

Filename contains report type + Greek patient name + report date, but excludes diagnosis and ID number.

## 13. Finalization contract

Final PDF request requires:

```text
clinician_confirmed = true
```

The final payload is the clinician-edited report text, not an opaque provider response.

No AI endpoint may directly emit a final authoritative signed report without the explicit clinician-confirmation step.

## 14. Privacy/security

- existing protected clinical auth dependency on all report routes;
- no PHI in query strings;
- no browser persistence APIs for case/source/report state;
- no feature-owned database writes;
- no request/source content deliberately logged;
- bounded upload and extracted-text size;
- source filenames sanitized for display;
- provider error messages sanitized;
- no API secret exposed to browser;
- literature research prompt excludes direct identifiers.

## 15. Acceptance gates

At minimum synthetic tests prove:

1. valid accident case;
2. valid medico-legal case;
3. Greek/unicode case fields;
4. PDF text extraction with page provenance;
5. TXT/MD extraction;
6. DOCX extraction;
7. unsupported file rejected;
8. oversize file/aggregate/extracted text rejected;
9. image-only PDF surfaced as `no_extractable_text` without OCR;
10. clinician textarea becomes distinct own-note source;
11. fake AI result validates source/page/evidence references;
12. forged nonexistent source/page reference fails closed;
13. duplicate evidence IDs rejected;
14. draft sections are editable payload objects;
15. diagnosis/causation/prognosis output is review-required;
16. AI endpoint fails closed when provider not enabled/configured;
17. identifiable AI endpoint additionally fails closed without PHI provider approval;
18. research request builder excludes patient name/identifier;
19. research citations returned distinctly from research prose;
20. final PDF requires clinician confirmation;
21. final PDF parseable and multi-page for long content;
22. filename includes Greek patient/date and excludes diagnosis/ID;
23. optional session signature works without persistence;
24. all routes require existing clinical auth;
25. package/browser contain no new patient/case persistence owner;
26. existing Sick Leave V1 regression still passes;
27. Clinic Utilities navigation exposes Medical Reports.

## 16. Explicit exclusions

Not in this slice:

- persistent case database/history;
- direct GeSY API integration;
- OCR/vision ingestion;
- automatic permanent-impairment/disability scoring;
- damages/claim valuation;
- billing/Fee Note/Receipt runtime;
- tax/VAT logic;
- commercial template upload/marketplace;
- Siri/Gemini/native app;
- automated jurisdiction declaration wording;
- production secret/env mutation;
- unrelated RF/physio/Clinical Learning/osteoporosis-guidance changes.

## 17. Exit boundary

Implementation completion means:

```text
runtime + UI implemented
+ synthetic deterministic provider tests pass
+ existing Sick Leave regression passes
+ exact-head review passes
```

It does not mean merged, deployed, provider-configured for PHI, production-smoke-verified or clinically validated. Those remain separate gates.