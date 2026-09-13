# SLICE_PLAN_CURRENT.md — CLINICAL DOCUMENTS PHASE 2 / MEDICAL REPORT V1

> **STATUS:** IMPLEMENTED / TESTED / EXACT-HEAD REVIEWED — RELEASE HOLD.
> **Activated:** 2026-09-13 Asia/Nicosia.
> **Implementation checkpoint:** 2026-09-13 Asia/Nicosia.
> **Slice:** `CU-CLINICAL-DOCUMENTS-P2-MEDICAL-REPORT-V1-2026-09-13`.
> **Bootstrap main / merge base:** `8993a1c4b585c590a543c907ea6b2eba32bbccdc`.
> **Branch:** `feat/clinical-documents-p2-medical-report-v1-2026-09-13`.
> **Exact reviewed/tested runtime head:** `3a19a09280ff682badfd4866e19d6a4f3cb9c208`.
> **Clinical Documents P2 gate:** `34733760918` — SUCCESS.
> **Writer:** none — checkpoint frozen in release hold.
> **Scope:** ephemeral multi-source medical-report workspace + Evidence Ledger/Timeline + AI-assisted draft + prognosis/literature seam + reviewed final PDF.
> **Persistence:** NONE for patient/case/source/report state.

## 1. Objective — achieved for V1 implementation

The first clinically useful accident/medico-legal report workflow is implemented on top of the released Clinical Documents core.

Product Owner workflow:

```text
pasted clinician history/instructions
+ explicitly uploaded relevant records
+ clinician-editable source classification
→ evidence/provenance extraction
→ chronology
→ structured editable report draft
→ targeted prognosis/future-needs research
→ clinician review
→ explicit final confirmation
→ signed PDF
```

The output remains an **AI-assisted draft under clinician authority**, not autonomous medico-legal opinion.

## 2. Runtime owners

Implemented owners:

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
test_clinical_documents_medical_report_consistency.py
test_clinical_documents_medical_report_source_types.py
test_clinical_documents_openai_sdk_contract.py
.github/workflows/clinical-documents-p2-tests.yml
```

Sick Leave V1 semantics were not rewritten; its full existing regression is part of the P2 gate.

## 3. Case contract

`MedicalReportCaseV1`:

```text
report_type: accident_medical_report | medico_legal_expert_report
patient_name: required
id_type: ADT | ARC | OTHER
id_number: optional
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

Direct patient identifiers are excluded from external literature-search prompts.

## 4. Source ingestion contract

Accepted V1 files:

```text
.pdf
.txt
.md
.docx
```

Implemented limits:

- max 20 uploaded files;
- max 12 MiB per file;
- max 40 MiB aggregate uploaded bytes;
- max 350,000 extracted characters;
- max 5,000 PDF pages;
- max 4 MiB decompressed DOCX `word/document.xml`;
- bounded filename/display and typed request payloads.

PDF extraction is local/request-scoped with PyMuPDF and retains page-level provenance.

DOCX extraction is local OOXML/XML parsing. The original is not placed in an external file store.

Image-only/scanned PDF with insufficient extractable text returns `no_extractable_text`; V1 performs no OCR guessing.

The clinician-context textarea is represented as a dedicated `clinician_note_self` source so it remains distinguishable from external records.

Uploaded sources expose an editable type selector before AI generation, including GeSY, own note, other clinician, specialist, hospital/emergency, admission/discharge, procedure, imaging, lab, physiotherapy, prior report, sick leave and other/auto.

## 5. Evidence Ledger contract

`EvidenceItemV1` contains:

```text
evidence_id
source_id
page_numbers[]
event_date_optional
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

Post-AI validation rejects nonexistent source/page references and duplicate evidence IDs.

## 6. Timeline and work-incapacity contract

`TimelineEventV1` contains:

```text
event_id
event_date_optional
date_text_optional
title
summary
source_ids[]
evidence_ids[]
conflict_flags[]
```

`WorkAbsenceIntervalV1` contains:

```text
interval_id
leave_from
leave_to
source_ids[]
evidence_ids[]
note
requires_clinician_review
```

Rules:

- chronology never silently resolves source disagreement;
- unknown/approximate dates remain explicit;
- work-absence end cannot precede start;
- exact work/sick-leave intervals are requested from AI only when both boundaries are source-supported;
- deterministic post-processing flags overlaps and gaps;
- every work-absence source/evidence reference must exist.

## 7. AI report contract

`MedicalReportAnalysisV1` returns:

```text
source_summaries[]
evidence_items[]
timeline[]
work_absence_intervals[]
diagnosis_analyses[]
report_sections[]
prognosis_questions[]
future_needs[]
warnings[]
```

Every generated diagnosis/causation/prognosis/future-needs object is explicitly review-required.

The model instructions require:

- professional Greek draft prose;
- strict source-bounded extraction;
- attribution and uncertainty preservation;
- no invented dates/findings/tests/diagnoses/treatments/outcomes;
- distinction among patient history, own findings, specialist opinion, investigation findings and AI inference;
- conflict surfacing instead of silent reconciliation;
- no compensation/damages/permanent-impairment percentage calculation;
- no jurisdiction-specific legal declaration insertion;
- source documents treated as untrusted data, not as instructions to the model.

## 8. OpenAI provider seam

The implementation uses the installed OpenAI Python SDK / Responses API behind explicit server-side gates.

Configurable model variables:

```text
CLINICAL_DOCUMENTS_AI_MODEL
CLINICAL_DOCUMENTS_RESEARCH_MODEL
```

Current code default is `gpt-5.6`, configurable server-side.

Provider availability requires:

```text
OPENAI_API_KEY
+ CLINICAL_DOCUMENTS_AI_ENABLED=true
```

Identifiable source processing additionally requires:

```text
CLINICAL_DOCUMENTS_PHI_PROVIDER_APPROVED=true
```

Provider calls use `store=false`, bounded output tokens and no Files/Vector Store persistence. Token/web-search usage metadata are returned for later cost accounting.

Tests never call the live provider. A dedicated SDK-contract guard proves the expected `responses.parse` / `responses.create` parameters exist in the installed SDK.

## 9. Research/literature seam

Analysis generates targeted prognosis/future-needs questions.

The explicit research endpoint runs hosted web search using generalized clinical problems/questions. Direct patient name, patient identity number and instructing reference are defensively excluded/redacted from the external research prompt.

Research output keeps separate:

```text
research_text
citations[]:
  title
  url
queries[] when available
usage
```

The UI shows citations as clickable sources. Research prose remains editable and requires final clinician confirmation before inclusion.

## 10. Browser workflow

### Step 1 — Case
Case fields + report type + purpose/instructions.

### Step 2 — Sources
- large clinician-history/instructions textarea;
- multi-file chooser;
- clinician-editable source classification;
- source list + extraction state/warnings.

### Step 3 — Generate draft
Request-scoped local extraction + guarded AI analysis.

### Step 4 — Review
Visible panels for:
- source summaries;
- Evidence Ledger;
- Timeline;
- diagnoses/causation;
- editable report sections.

AI draft content is visibly review-required.

### Step 5 — Prognosis & literature
Explicit button only. Research sources/citations remain separate from case facts.

### Step 6 — Final report
Clinician edits final section text, optionally loads a session-only signature, confirms review and previews/downloads the PDF.

Changing source classification after a draft invalidates that draft and requires regeneration.

## 11. PDF contract

Professional multi-page A4 Greek report includes:

- clinician header;
- patient/case information;
- section hierarchy;
- readable long-form typography;
- automatic page breaks;
- page number/footer;
- optional bibliography/research synthesis;
- signature/clinician closing block;
- optional declaration only if explicitly clinician-supplied/confirmed.

Long unbroken references/URLs are safely wrapped.

Filename contains report type + Greek patient name + report date, but excludes diagnosis and identity number.

## 12. Finalization contract

Final PDF request requires:

```text
clinician_confirmed = true
```

The final payload is the clinician-edited report text, not an opaque provider response.

No AI endpoint directly emits an authoritative signed report without explicit clinician confirmation.

## 13. Privacy/security

- existing protected clinical auth dependency on every report route;
- no PHI in query strings;
- no browser persistence APIs for case/source/report state;
- no feature-owned database writes;
- no request/source content deliberately logged by this feature;
- bounded upload/extracted-text/page/XML/request sizes;
- source filenames sanitized for display;
- provider errors sanitized;
- no API secret exposed to browser;
- literature research prompt excludes direct identifiers;
- all repository tests use synthetic/non-identifiable data.

## 14. Acceptance evidence — satisfied at exact runtime head

Synthetic gates prove at minimum:

1. valid accident case;
2. valid medico-legal case;
3. Greek/unicode case fields;
4. PDF text extraction with page provenance;
5. TXT/MD extraction;
6. DOCX extraction;
7. unsupported/oversize source failure;
8. image-only PDF surfaced as `no_extractable_text` without OCR;
9. clinician textarea is distinct own-note source;
10. clinician-editable uploaded-source classification is preserved and validated;
11. forged/misaligned source classification fails closed;
12. fake AI result validates source/page/evidence references;
13. forged nonexistent source/page reference fails closed;
14. duplicate evidence IDs rejected;
15. diagnosis/causation/future-needs output is review-required;
16. work-absence inverted dates rejected;
17. work-absence overlap and gap warnings are deterministic;
18. work-absence forged source/evidence references fail closed;
19. AI endpoint fails closed without provider enable/config;
20. identifiable AI analysis additionally fails closed without PHI-provider approval;
21. research prompt excludes direct identifiers;
22. research citations returned separately from research prose;
23. final PDF requires clinician confirmation;
24. long final report produces parseable multi-page PDF;
25. filename includes Greek patient/date and excludes diagnosis/ID;
26. optional session signature works without persistence;
27. all routes use existing clinical auth;
28. browser contains no case persistence APIs;
29. existing Sick Leave V1 regression passes;
30. Clinic Utilities navigation exposes Medical Reports;
31. installed OpenAI SDK supports required Responses API contract.

Exact reviewed/tested runtime head:

`3a19a09280ff682badfd4866e19d6a4f3cb9c208`

GitHub Actions:

```text
workflow: Clinical Documents P2 tests
run:      34733760918
result:   SUCCESS
```

Fresh exact comparison against `main`/merge-base `8993a1c4b585c590a543c907ea6b2eba32bbccdc`:

```text
ahead / behind: 31 / 0
runtime owners:  Clinical Documents only
composition:     main.py router mount
navigation:      four-line Clinic Utilities extension
adjacent RF/physio/Clinical Learning/osteoporosis-rule mutations: NONE
```

## 15. Explicit exclusions

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

## 16. Exit / release boundary

Implementation exit is satisfied:

```text
runtime + UI implemented
+ synthetic provider/contract tests pass
+ source-provenance/consistency tests pass
+ existing Sick Leave regression passes
+ exact-head review passes
```

Current state is **RELEASE HOLD**.

It does not mean merged, deployed, production AI/PHI configured, production-smoke-verified or clinically validated. Those remain separate gates requiring explicit Product Owner authority and, for identifiable external-AI processing, a deliberate provider/privacy/configuration decision.