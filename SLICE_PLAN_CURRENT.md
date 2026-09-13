# SLICE_PLAN_CURRENT.md — CLINICAL DOCUMENTS PHASE 2 / MEDICAL REPORT V1

> **STATUS:** RELEASED / DEPLOYED — AUTHENTICATED PRODUCTION SMOKE PENDING.
> **Activated:** 2026-09-13 Asia/Nicosia.
> **Released:** 2026-09-13 Asia/Nicosia.
> **Slice:** `CU-CLINICAL-DOCUMENTS-P2-MEDICAL-REPORT-V1-2026-09-13`.
> **Implementation branch:** `feat/clinical-documents-p2-medical-report-v1-2026-09-13`.
> **Release PR:** `#99` — squash merged.
> **Implementation review head:** `3a19a09280ff682badfd4866e19d6a4f3cb9c208`.
> **Release commit:** `261015be5a2921c6d67ad6b48d14196c17b1c34f`.
> **Implementation gate:** `34733760918` — SUCCESS.
> **PR Clinical Documents gate:** `34734187010` — SUCCESS.
> **Release deploy:** `dep-daj11b95efls739bog40` — LIVE.
> **Configuration deploy:** `dep-daj128u7bikc73acabt0` — LIVE.
> **Writer:** none.
> **Persistence:** NONE for Clinical Documents patient/case/source/report state.

## 1. Objective

Medical Report V1 delivers the first released multi-source accident / medico-legal workflow:

```text
clinician case context
+ pasted own history/instructions
+ explicitly selected source documents
+ clinician-editable source classification
→ request-scoped text extraction
→ source/page-bounded Evidence Ledger
→ chronology / Timeline
→ structured work-incapacity intervals
→ AI-assisted report draft
→ clinician review/edit
→ optional targeted prognosis/literature research
→ clinician review/edit
→ explicit final confirmation
→ multi-page Greek PDF
```

The output is an AI-assisted draft under clinician authority, not an autonomous medico-legal opinion.

## 2. Released owners

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

Composition/navigation seams are limited to:

```text
clinic_utilities/clinical_documents/__init__.py
main.py
static/baseline-audit/g4-workspace-ergonomics.js
```

Sick Leave V1 is preserved and regression-tested.

## 3. Source and evidence contract

Accepted files:

```text
PDF with extractable text
TXT
Markdown
DOCX
```

V1 deliberately performs no OCR guessing for image-only PDFs.

Source processing is bounded by file count, per-file bytes, aggregate bytes, extracted characters, PDF pages and DOCX XML expansion. The clinician's own free-text context is a distinct own-note source.

Evidence retains source ID and page provenance. Forged source/page/evidence references and duplicate evidence IDs fail closed.

## 4. Timeline / work-incapacity contract

Timeline items preserve source/evidence references and do not silently reconcile conflicts.

Work-incapacity intervals require explicit start/end dates; end before start is rejected. Deterministic processing surfaces overlapping periods and gaps. AI instructions prohibit inferring a missing interval boundary.

## 5. AI authority contract

The provider may extract, summarize, organize and draft, while preserving attribution and uncertainty.

It must not silently invent or promote absent dates, findings, investigations, diagnoses, treatment, causation, permanence or prognosis to fact.

Consequential diagnosis/causation/prognosis/future-needs text remains review-required. Uploaded documents are treated as quoted/untrusted data rather than model instructions.

No compensation value, damages calculation, permanent-impairment percentage or jurisdiction-specific declaration is generated automatically.

## 6. Research contract

Literature/prognosis research is an explicit separate action. The research prompt uses generalized clinical problems/questions and excludes direct patient name, identity number and instructing reference.

Research prose and clickable citations remain separate from patient facts and remain editable before final inclusion.

## 7. Finalization contract

Final PDF requires explicit clinician confirmation.

The PDF supports:

- Greek multi-page A4 output;
- clinician/case information;
- editable report sections;
- page breaks and numbering;
- optional reviewed bibliography/research text;
- optional clinician-authored declaration;
- optional session-only signature.

Filename includes report type, Greek patient name and report date, while excluding diagnosis and patient identity number.

## 8. Privacy / persistence

The release introduces no Clinical Documents case database, browser PHI persistence, autosave or automatic patient history. Uploaded sources are request scoped. Signature remains session/request scoped. No real-patient source files or identifiable fixtures are committed to the public repository.

## 9. Release/config evidence

Product Owner release authority was provided after the implementation hold with `Προχωρά μέχρι τέλους`.

PR #99 was squash merged to `main` as `261015be5a2921c6d67ad6b48d14196c17b1c34f`.

Render auto-deploy `dep-daj11b95efls739bog40` reached LIVE. The already-designed Medical Report AI runtime gates were then enabled and configuration deploy `dep-daj128u7bikc73acabt0` reached LIVE at `2026-09-13T02:59:14Z`.

No provider credential value was exposed or changed by this release session. The runtime remains fail-closed if a required provider credential is unavailable.

## 10. Acceptance evidence

The focused Clinical Documents implementation and PR gates passed, covering:

- Python and JavaScript syntax;
- OpenAI Responses SDK contract without a live network call;
- accident + medico-legal typed cases;
- PDF/TXT/MD/DOCX extraction;
- page provenance;
- unsupported/oversize/image-only handling;
- clinician source classification;
- referential integrity;
- work-incapacity validation/overlap/gap checks;
- review-required consequential opinions;
- identifier-free research prompt construction;
- final confirmation requirement;
- long multi-page Greek PDF generation;
- no browser case persistence;
- existing Sick Leave regression;
- Clinic Utilities navigation.

## 11. Explicit exclusions retained

Not in V1:

- persistent report/case history;
- direct GeSY API integration;
- OCR/vision ingestion;
- automatic permanent-impairment scoring;
- damages/claim valuation;
- billing / Fee Note / Receipt runtime;
- commercial user-uploaded template platform;
- automatic jurisdiction declaration wording;
- unrelated RF/physio/Clinical Learning/osteoporosis-guidance mutations.

## 12. Remaining release-validation boundary

The deployed routes intentionally remain protected by existing clinical authentication. The deployment connector does not inherit the Product Owner's authenticated browser session, so a real end-to-end browser smoke cannot honestly be claimed from this release session alone.

Remaining smoke:

```text
synthetic authenticated case
→ AI draft
→ Evidence Ledger / Timeline review
→ literature research
→ final clinician confirmation
→ PDF preview/download
```

Once that succeeds, this slice may be marked `PRODUCTION-SMOKE-VERIFIED / CLOSED`.