# SLICE_PLAN_CURRENT.md — Clinical Documents Medical Report V1.1

> **STATUS:** IMPLEMENTATION ACTIVE.
> **Activated:** 2026-09-13 Asia/Nicosia.
> **Slice:** `CU-CLINICAL-DOCUMENTS-V1-1-SOURCE-RESOLUTION-2026-09-13`.
> **Bootstrap main:** `53ea38318d99e81218e5402e3c206ba30c51a8a9`.
> **Branch:** `fix/clinical-documents-v1-1-source-resolution-2026-09-13`.
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

## Safety and privacy

No new patient/case database, browser persistence or autosave. Existing protected auth and provider gates remain. Real production files are not repository fixtures; automated tests use synthetic files only.

## Acceptance

Tests cover file removal, new source types, sick-leave extraction, visual fallback via fake provider, immutable original evidence during refinement, clinician resolutions, pending-vs-missing semantics, optional occupation, elapsed-progress UI, and inherited Medical Report/Sick Leave/navigation regressions.

## Owners

Only Clinical Documents runtime/UI/tests/workflow owners are mutable. Physio, RF, Clinical Learning and Osteoporosis clinical rules are out of scope.

## Exit

Implement and exact-head test on the active branch, then stop at release HOLD unless merge/deploy is separately authorized.