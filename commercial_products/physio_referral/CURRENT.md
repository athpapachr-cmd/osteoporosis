# CURRENT.md — Physio Referral commercial product track

> **STATUS:** KNEE-OA V1 COCKPIT RELEASE CANDIDATE — IMPLEMENTED / EXACT-HEAD TESTED / RELEASE REVIEW PASS / PR NEXT.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Diagnosis vertical:** Knee Osteoarthritis only.
> **Release-candidate head:** `57d879b072fcaf2bef350d5dbb0547ce465764e1`.
> **Integration branch:** `feat/physio-knee-oa-cockpit-integration-v1-2026-09-12`.
> **Root operational authority:** `CURRENT_OPERATIONAL.md`.

## Product state

The reviewed Knee-OA experience has moved from a synthetic-only prototype into the existing authenticated Clinical Excellence Cockpit physiotherapy utility. The product philosophy and interaction contract remain intact: live deterministic referral, few meaningful routine actions, evidence complexity under progressive disclosure, direct editing with fail-closed reconciliation, and scan-first advanced access rather than a comprehensive visible form.

Commercial/product authority now lives under `commercial_products/physio_referral/`; technical contracts/runtime/tests remain under `clinic_utilities/physio_referral_product/`.

## Review state

The four independent specialist reviews remain:

1. Clinical / Evidence
2. Physiotherapy / receiving-professional utility
3. UX / Product
4. Commercial / Product-Market

Their material findings were synthesized and amended through the reviewed Knee-OA lineage. A separate final release/red-team review at the Cockpit integration stage is recorded at:

`releases/KNEE_OA_V1_RELEASE_REVIEW.md`

Verdict: **PASS / no open release blocker**.

## Exact release-candidate evidence

At `57d879b072fcaf2bef350d5dbb0547ce465764e1`:

- Cockpit integration gate `34679427725` — SUCCESS
- inherited Knee-OA product gate `34679427741` — SUCCESS
- CU-1 focused gate `34679427822` — SUCCESS

Underlying v3 product candidate remains anchored by substantive run `34677022119` and exact-head closeout run `34677243751`, both SUCCESS.

## Product truth that remains unproven

A technically releasable authenticated Cockpit feature is not the same thing as a validated business:

- no receiving-physiotherapist field validation yet;
- no actual iPhone Safari/VoiceOver acceptance yet;
- no paid conversion/retention evidence yet;
- no real clinical pilot validation yet;
- Cyprus/GeSY recommendation-by-recommendation overlay is not yet activated.

## Exact next product boundary

Complete PR → squash merge → exact Render deploy verification → bounded non-identifiable production smoke → canonical closeout. After that, do not automatically add another diagnosis. The next commercial/product learning step should be chosen from real receiver/device/market evidence rather than feature-count pressure.
