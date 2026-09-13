# SLICE_PLAN_CURRENT.md — Clinical Documents Medical Report V1.1

> **STATUS:** IMPLEMENTED / EXACT-HEAD TESTED — RELEASE HOLD.
> **Activated:** 2026-09-13 Asia/Nicosia.
> **Slice:** `CU-CLINICAL-DOCUMENTS-V1-1-SOURCE-RESOLUTION-2026-09-13`.
> **Bootstrap main:** `53ea38318d99e81218e5402e3c206ba30c51a8a9`.
> **Branch:** `fix/clinical-documents-v1-1-source-resolution-r2-2026-09-13`.
> **Exact tested runtime head:** `4c30867e8aea00842772b6df7d624a79bbf3f19e`.
> **Focused gate:** `34740684612` — SUCCESS.
> **Writer:** none — implementation candidate frozen in release HOLD.

## Objective — achieved at implementation level

Refine Medical Report V1 from real-use feedback while preserving clinician authority, source provenance and the no-persistent-case boundary.

## Implemented V1.1 scope

- removable uploaded source files;
- source types `prescription`, `imaging_referral`, `specialist_referral`, `lab_or_service_referral`, `heidi_transcript`;
- explicit semantics that referral is not completed care, prescription is not medication taken and pending is not a result;
- bounded controlled visual AI reading for image-only PDFs, explicitly review-required;
- structured sick-leave start/end extraction from compatible metadata or explicit text, including after visual extraction;
- provenance-bearing work-absence interval creation;
- separate `ClinicianResolutionV1` objects without source mutation;
- protected session-only AI refinement/discussion after the initial draft;
- explicit Apply / Discard of proposed refinement;
- refinement integrity guard preventing Evidence Ledger/source-summary rewriting;
- invalidation of stale literature after an applied refinement;
- context-sensitive occupation warnings;
- elapsed-time progress for long AI operations;
- synthetic-only V1.1 regression coverage.

## Core invariants

```text
ORIGINAL SOURCE != CLINICIAN RESOLUTION
REFERRAL != COMPLETED EXAMINATION / CONSULTATION
PRESCRIPTION != MEDICATION TAKEN / ADMINISTERED
REQUESTED / PENDING != RESULT AVAILABLE
VISUAL AI EXTRACTION != DETERMINISTIC TEXT EXTRACTION
AI DRAFT != FINAL CLINICIAN OPINION
```

An erroneous external note remains unchanged. The clinician's resolution is a separate current-workspace object that may guide downstream chronology and report wording.

## Image-only PDF contract

A PDF without usable text may be rendered locally to page images and passed through the existing gated AI provider, up to 12 pages. Successful output is marked:

```text
status = visual_extracted
extraction_method = visual_ai
review_required = true
```

Page provenance is retained. Failure falls back to the original unreadable-source state rather than fabricated text.

## Sick-leave contract

For `sick_leave_certificate`:

1. compatible embedded application metadata is preferred when available;
2. otherwise explicit text `from/to` dates are parsed deterministically;
3. image-only certificates may gain explicit dates after the controlled visual-reading step;
4. the resulting interval remains review-required and source-linked.

The general AI receives these structured source notes, so it is no longer solely responsible for rediscovering leave dates.

## Refinement contract

The clinician may send a session-only clarification/question after the first analysis. The AI may propose resolutions and revise downstream timeline/report prose, but source summaries and original evidence items must remain byte-for-byte structurally unchanged under server validation.

The browser applies a proposed refinement only after an explicit clinician action. Any prior research is marked stale and excluded until research is rerun.

## Privacy / persistence

Unchanged:

- no Medical Report patient/case database;
- no localStorage/sessionStorage/indexedDB PHI state;
- no autosave;
- request-scoped source processing;
- session-only refinement discussion;
- existing protected clinical auth and AI/PHI gates;
- no real patient files in repository tests/fixtures.

## Acceptance evidence

Exact runtime head `4c30867e8aea00842772b6df7d624a79bbf3f19e`, workflow `34740684612` SUCCESS, covering:

- Python + all Medical Report JavaScript syntax;
- OpenAI SDK contract guard;
- inherited Medical Report V1 tests;
- new source-class semantics;
- deterministic sick-leave parsing;
- image-only PDF visual-source semantics;
- actual synthetic PDF page rendering to JPEG data URLs;
- fake-provider visual API flow;
- clinician-resolution and immutable-evidence refinement checks;
- contextual occupation warnings;
- V1.1 browser extension order/static privacy checks;
- inherited Sick Leave V1 and Clinic Utilities navigation regressions.

## Owners / exclusions

Only Clinical Documents runtime/UI/tests/workflow owners were changed. No Physio, RF, Clinical Learning or Osteoporosis clinical-rule owner was modified. No patient persistence, billing, tax/VAT, compensation calculation or autonomous final opinion was added.

## Release boundary

Implementation exit is satisfied. The candidate is **not merged or deployed**. A fresh-main release review/PR and Render auto-deploy require separate explicit Product Owner release authority.