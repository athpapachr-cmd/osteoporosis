# CURRENT_OPERATIONAL.md — Clinical Documents Medical Report V1.1

> **STATUS:** IMPLEMENTATION ACTIVE / FOCUSED GATE PASSING.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Fresh bootstrap main:** `53ea38318d99e81218e5402e3c206ba30c51a8a9`.
> **Branch:** `fix/clinical-documents-v1-1-source-resolution-r2-2026-09-13`.
> **Slice:** `CU-CLINICAL-DOCUMENTS-V1-1-SOURCE-RESOLUTION-2026-09-13`.
> **Latest focused gate:** `34740415927` — SUCCESS at runtime head `f4c547f6e2e9b0552a06d58c65dbfeda301191ab`.
> **Writer:** current Clinical Documents conversation.

## Product Owner authority

Implement the Medical Report fixes found during real production use and keep canonicals current for continuation in another conversation.

## Implemented in the active candidate

- removable uploaded files before analysis;
- source categories for prescription, imaging referral, specialist referral, laboratory/other-service referral and Heidi transcript;
- explicit referral/prescription/requested-vs-completed semantics in AI instructions;
- controlled AI visual reading for image-only PDFs, marked review-required;
- deterministic structured sick-leave interval extraction from compatible metadata or explicit text;
- clinician resolutions kept separate from immutable source evidence;
- session-only AI refinement endpoint and browser discussion layer;
- context-sensitive occupation warning;
- elapsed-time indicator during long AI work;
- synthetic-only V1.1 regression coverage.

## Invariants

Original source content is never silently rewritten. Clinician resolution remains separate from source evidence. Referral is not proof of completed care, prescription is not proof of medication taken, and requested/pending is not a result.

No new patient/case persistence is authorized. Real patient files used during product-owner testing are not repository fixtures and must never be committed.

## Adjacent state

Knee-OA V5 is separately closed and production-smoke-verified. Do not mutate Physio/RF/Clinical Learning/Osteoporosis clinical rules in this slice.

## Next action

Complete exact-head code review/hardening of the V1.1 candidate, rerun the focused Clinical Documents gate on the final runtime head, update canonicals/changelog, then stop at release HOLD unless merge/deploy is separately authorized.