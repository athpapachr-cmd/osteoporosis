# CURRENT_OPERATIONAL.md — Knee-OA Cockpit production integration

> **STATUS:** PHYSIO REFERRAL KNEE-OA — PRODUCTION INTEGRATION ACTIVE / RELEASE AUTHORIZED THROUGH PR + MERGE + DEPLOY VERIFICATION.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Parent closed v3 head:** `b7f2db6fd86f90a32875d9b9cfd356cfa6129c69`.
> **Implementation branch:** `feat/physio-knee-oa-cockpit-integration-v1-2026-09-12`.
> **Active slice:** `CU1-PRODUCT-KNEE-OA-COCKPIT-INTEGRATION-V1-20260912`.
> **ACTIVE RUNTIME / DESIGN / CANONICAL WRITER:** this bounded integration/release session.
> **Product-owner authority:** commercial-canonical reorganization, Cockpit integration, final review, PR, squash merge, Render auto-deploy verification and bounded production smoke.
> **Real-patient data:** NOT AUTHORIZED for smoke; use synthetic/non-identifiable test state only.

## 1. Release objective

Finish the Knee-OA product track cleanly before the next conversation:

```text
commercial product canonicals reorganized
→ tested Knee-OA product UI integrated into existing authenticated Cockpit physiotherapy utility
→ full inherited + production-integration regression
→ final release/red-team review
→ PR
→ squash merge
→ Render auto-deploy verification
→ bounded synthetic production smoke
→ canonical closeout / writer NONE
```

The existing production route is:

`/clinical/clinic-utilities/physio-referral`

It currently serves the older large CU-1 form. The new Knee-OA product must integrate into that protected utility rather than create a parallel public service.

## 2. Hard integration boundaries

- existing CU-1 validation/safety remains authoritative;
- existing Clinical Excellence authentication remains authoritative;
- no new patient persistence;
- no analytics/billing/account system in this slice;
- no second diagnosis;
- no unaudited Cyprus/GeSY recommendation changes;
- no LLM-generated routine referral;
- manual text reconciliation remains explicit and fail-closed;
- evidence/selection/safety/provenance states remain distinct;
- commercial product canonicals move under `commercial_products/physio_referral/`;
- technical contracts/runtime/tests remain under `clinic_utilities/physio_referral_product/`.

## 3. Current validated product lineage

Latest closed synthetic v3 substantive head:

`9deafa2db43d3498c5becf20f77a804b03849d53`

Final v3 closeout head:

`b7f2db6fd86f90a32875d9b9cfd356cfa6129c69`

Substantive run `34677022119` and closeout run `34677243751` both succeeded.

## 4. Release rule

Do not merge merely because the synthetic v3 was green. Production integration must prove the protected Cockpit route, static assets/API behavior, CU-1 safety, privacy/no-storage boundary, mobile behavior and adjacent-owner regressions on the exact release head.

## 5. Exact next action

Reorganize commercial canonicals, implement the protected Cockpit integration on this branch, run the full gate, perform a fresh final release review, then proceed through the explicitly authorized PR/merge/deploy path only if clean.
