# CURRENT_OPERATIONAL.md — Clinical Documents Medical Report V1.1

> **STATUS:** IMPLEMENTATION ACTIVE.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Fresh bootstrap main:** `53ea38318d99e81218e5402e3c206ba30c51a8a9`.
> **Branch:** `fix/clinical-documents-v1-1-source-resolution-2026-09-13`.
> **Slice:** `CU-CLINICAL-DOCUMENTS-V1-1-SOURCE-RESOLUTION-2026-09-13`.
> **Writer:** current Clinical Documents conversation.

## Product Owner authority

Implement the Medical Report fixes found during real production use and keep canonicals current for continuation in another conversation.

## Current scope

- remove selected files before analysis;
- add prescription and referral source categories plus Heidi transcript;
- preserve referral/requested/completed distinctions;
- add controlled visual reading for image-only PDFs;
- extract structured sick-leave dates reliably;
- add clinician conflict-resolution and AI refinement discussion;
- make occupation warning context-sensitive;
- show elapsed AI progress;
- use synthetic tests only.

## Invariants

Original source content is never silently rewritten. Clinician resolution remains separate from source evidence. Referral is not proof of completed care, prescription is not proof of medication taken, and requested/pending is not a result.

No new patient/case persistence is authorized.

## Adjacent state

Knee-OA V5 is separately closed and production-smoke-verified. Do not mutate Physio/RF/Clinical Learning/Osteoporosis clinical rules in this slice.

## Next action

Implement and test V1.1 on this branch, then perform exact-head review and stop at release HOLD unless merge/deploy is separately authorized.