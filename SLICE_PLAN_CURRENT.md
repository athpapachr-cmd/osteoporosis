# SLICE_PLAN_CURRENT.md — Clinical Documents Medical Report V1.1

> **STATUS:** IMPLEMENTATION ACTIVE / FOCUSED GATE PASSING.
> **Activated:** 2026-09-13 Asia/Nicosia.
> **Slice:** `CU-CLINICAL-DOCUMENTS-V1-1-SOURCE-RESOLUTION-2026-09-13`.
> **Bootstrap main:** `53ea38318d99e81218e5402e3c206ba30c51a8a9`.
> **Branch:** `fix/clinical-documents-v1-1-source-resolution-r2-2026-09-13`.
> **Focused gate:** `34740415927` — SUCCESS at `f4c547f6e2e9b0552a06d58c65dbfeda301191ab`.
> **Writer:** current Clinical Documents conversation.

## Objective

Improve Medical Report V1 from real-use feedback while preserving the existing clinician-authority and no-persistent-case boundaries.

## Scope

- Add Remove for uploaded files and invalidate stale drafts after source changes.
- Add source types: `prescription`, `imaging_referral`, `specialist_referral`, `lab_or_service_referral`, `heidi_transcript`.
- Preserve `referral != completed care`, `prescription != medication taken`, and `pending != result available`.
- Add bounded visual reading for image-only PDFs, marked as AI-derived and review-required.
- Parse explicit sick-leave start/end dates into structured intervals before general report synthesis when possible.
- Add clinician resolutions as separate objects; never rewrite original evidence.
- Add session-only AI discussion/refinement after the first analysis.
- Do not warn about missing occupation unless work-capacity reasoning needs it.
- Show elapsed progress during long AI calls.

## Implemented V1.1 contract

### Source handling

The original PDF/TXT/MD/DOCX intake remains. Uploaded files now have a removable browser-row control. New clinician-declared source classes distinguish prescriptions, imaging referrals, specialist referrals, laboratory/service referrals and Heidi transcripts.

### Image-only PDF

A PDF without a usable text layer remains deterministically identifiable as such. V1.1 may then render a bounded number of pages locally and ask the already-approved AI provider to read the visible document. This output is marked `visual_ai`, `visual_extracted`, `review_required=true` and remains page-provenanced. It is not represented as deterministic text extraction.

### Sick leave

For a source classified as `sick_leave_certificate`, V1.1 attempts structured leave extraction before broad report synthesis, preferring compatible embedded Clinical Documents metadata and then explicit text dates. Structured leave evidence/intervals remain clinician-review-required.

### Clinician resolution and AI refinement

The original Evidence Ledger is immutable during refinement. A clinician can explain a wrong date in another doctor's note, state that an investigation is pending, or ask the AI to reconsider the report. Any accepted clarification is represented separately as a clinician resolution and may update downstream chronology/report prose without deleting or rewriting original evidence.

The refinement conversation is browser-session/request scoped and is not stored as a patient/case thread.

### Semantics

```text
referral != completed examination or consultation
prescription != medication actually taken/administered
requested/pending != completed/result available
source fact != clinician resolution
visual AI extraction != deterministic text extraction
AI draft != final clinician opinion
```

## Safety and privacy

No new patient/case database, browser persistence or autosave. Existing protected auth and provider gates remain. Real production files are not repository fixtures; automated tests use synthetic files only. No user-uploaded real patient file is committed to the public repository.

## Acceptance evidence so far

Focused run `34740415927` passed:

- Python syntax;
- JavaScript syntax including the V1.1 extension scripts;
- OpenAI Responses SDK contract;
- Medical Report V1 + V1.1 deterministic tests;
- visual-fallback fake-provider path;
- deterministic sick-leave extraction;
- immutable-evidence refinement checks;
- clinician-resolution behavior;
- source-type semantics;
- contextual occupation warning;
- existing Sick Leave V1 regression;
- existing Clinic Utilities navigation regression.

## Owners

Only Clinical Documents runtime/UI/tests/workflow owners are mutable. Physio, RF, Clinical Learning and Osteoporosis clinical rules are out of scope.

## Exit

Complete exact-head review/hardening and rerun the focused gate on the final runtime head. Then set the slice to implementation-complete/release-HOLD and update `CURRENT_OPERATIONAL.md` plus the append-only changelog. Merge/deploy remains a separate Product Owner gate.