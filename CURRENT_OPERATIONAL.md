# CURRENT_OPERATIONAL.md — Knee-OA Cockpit release candidate

> **STATUS:** PHYSIO REFERRAL KNEE-OA — IMPLEMENTED / EXACT-HEAD TESTED / FINAL RELEASE REVIEW PASS / PR NEXT.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified main / merge base:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Parent closed v3 head:** `b7f2db6fd86f90a32875d9b9cfd356cfa6129c69`.
> **Integration branch:** `feat/physio-knee-oa-cockpit-integration-v1-2026-09-12`.
> **Active slice:** `CU1-PRODUCT-KNEE-OA-COCKPIT-INTEGRATION-V1-20260912`.
> **Exact reviewed/tested release-candidate head:** `57d879b072fcaf2bef350d5dbb0547ce465764e1`.
> **Release review:** `commercial_products/physio_referral/releases/KNEE_OA_V1_RELEASE_REVIEW.md` — PASS / no open blocker.
> **ACTIVE RUNTIME / DESIGN / CANONICAL WRITER:** this bounded release session until merge/deploy/smoke closeout.
> **Product-owner authority:** PR, squash merge, Render auto-deploy verification and bounded production smoke already granted for this slice.
> **Real-patient data:** NOT AUTHORIZED for release smoke; synthetic/non-identifiable state only.

## 1. Release candidate delivered

The reviewed Knee-OA product is integrated into the existing authenticated Clinical Excellence physiotherapy utility:

`/clinical/clinic-utilities/physio-referral`

The prior large Generate-form surface is replaced by the reviewed live-referral product experience while preserving existing Clinical Excellence authentication and real CU-1 validation/safety authority.

Commercial/product authorities are separated under:

`commercial_products/physio_referral/`

Technical contracts/runtime/tests remain under:

`clinic_utilities/physio_referral_product/`

## 2. Production architecture

```text
ClinicalCookieMiddleware / X-Clinical-Key authority
→ protected physiotherapy Cockpit route
→ production product transport
→ shared deterministic knee_oa_projection.py
→ real CU-1 validation/safety + reviewed evidence/template interaction
→ live Greek referral UI
```

The local loopback prototype is not mounted in production. It is only an alternate test transport over the same shared projection.

No new database, patient persistence, analytics, billing system or unauthenticated public product endpoint is introduced.

## 3. Exact release-candidate gates

At head `57d879b072fcaf2bef350d5dbb0547ce465764e1`:

```text
Physio Knee OA Cockpit integration gate   34679427725   SUCCESS
Physio Knee OA prototype gate             34679427741   SUCCESS
CU-1 focused tests                        34679427822   SUCCESS
```

The integration gate protects the authenticated page/API, real CU-1 path, full inherited product browser behavior, production Chromium flow, no-storage boundary, mobile reflow, adjacent Learning Hub/RF isolation and package closure.

## 4. Release review findings closed before PASS

Two material technical release findings were corrected before PASS:

1. production transport initially imported `prototype.server`; projection ownership was extracted to shared production-owned `knee_oa_projection.py`, with the prototype reduced to loopback transport only;
2. production requests initially inherited `synthetic_only:true`; the Cockpit boundary now reports/sends `synthetic_only:false`, rejects synthetic-marked production requests and only then adapts internally to the frozen reviewed compatibility envelope.

No clinical rule or evidence state changed in those corrections.

## 5. Hard boundaries still active

```text
NO second diagnosis
NO new GeSY item-level recommendation activation
NO patient draft persistence
NO analytics/billing/account system
NO LLM-generated referral text
NO real-patient smoke data
NO claim of iPhone VoiceOver acceptance
NO claim of receiving-physiotherapist or commercial/pilot validation
```

## 6. Exact next action

Open one bounded PR from `feat/physio-knee-oa-cockpit-integration-v1-2026-09-12` to current `main`. Require exact PR-head checks/review to settle cleanly. If clean, squash-merge using the exact expected head, verify Render auto-deploy at the resulting merge SHA, perform bounded production smoke without patient data, update final canonicals/changelogs and release the writer to `NONE`.
