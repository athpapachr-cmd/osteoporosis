# SLICE_PLAN_CURRENT.md — Knee-OA Cockpit integration + commercial-product canonical split

> **STATUS:** IMPLEMENTATION ACTIVE.
> **Slice:** `CU1-PRODUCT-KNEE-OA-COCKPIT-INTEGRATION-V1-20260912`.
> **Branch:** `feat/physio-knee-oa-cockpit-integration-v1-2026-09-12`.
> **Parent closed v3:** `b7f2db6fd86f90a32875d9b9cfd356cfa6129c69`.
> **Fresh main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Writer:** ACTIVE for bounded commercial-canonical reorganization + physiotherapy Cockpit integration + release verification.

## 1. Problem

The Knee-OA product is technically mature as a synthetic loopback prototype but is not yet the actual Cockpit physiotherapy experience. Product/commercial canonicals are also mixed into the implementation directory.

## 2. Target architecture

```text
six root canonicals
→ repo-wide operational authority

commercial_products/physio_referral/
→ product/commercial strategy, current state, product history, review/market/release navigation

clinic_utilities/physio_referral_product/
→ clinical/UX technical contracts, validators, implementation tests and product runtime

static/clinic-utilities/physio-referral/
+ clinic_utilities/physio_referral_api.py
→ authenticated Cockpit production surface
```

## 3. Production UI contract

The existing protected Cockpit route `/clinical/clinic-utilities/physio-referral` becomes the Knee-OA product surface.

Preserve from the tested candidate:

- explicit OA diagnosis assertion and laterality;
- live deterministic referral, no Generate button;
- direct `✎ Επεξεργασία` + safe manual reconciliation;
- evidence-aware defaults/suggestions;
- `Άλλες n προτάσεις` discoverability;
- `★ Συχνά / Σχετικά τώρα / Όλα` scan-first advanced architecture;
- qualifier semantics and FFD/weakness/pes-anserine safeguards;
- evidence conflict transparency;
- export/readiness safety precedence;
- no patient draft persistence.

## 4. Server/runtime ownership

Production must reuse authenticated Clinical Excellence and real CU-1 authority. Any Knee-OA product projection helper extracted from the synthetic prototype must be server-owned, deterministic and testable; do not register the loopback HTTP server itself in production.

No new database or patient write path.

## 5. Canonical split

Product canonicals move to:

`commercial_products/physio_referral/`

At minimum:

- `CURRENT.md`
- `PRODUCT_CONTEXT_CURRENT.md`
- `PRODUCT_PLAN.md`
- `PRODUCT_CHANGELOG.md`
- `releases/KNEE_OA_V1_CURRENT.md`
- navigation for reviews/market records.

Old product-canonical paths become redirects/pointers rather than competing authorities.

## 6. Acceptance

Before PR:

1. exact production route requires existing authentication;
2. Knee-OA bootstrap/projection is server authoritative;
3. all prior supported selections/evidence/safety behavior remains reachable;
4. no storage/analytics/patient persistence introduced;
5. protected route and static assets work with production-style tests;
6. inherited CU-1 and Knee-OA clinical/browser gates pass;
7. adjacent Learning Hub/RF/Osteoporosis owners unchanged except root operational docs;
8. fresh final release/red-team review reports no blocker;
9. exact PR head is tested.

After merge:

- verify Render auto-deploy corresponds to exact merge SHA;
- bounded production smoke with synthetic/non-identifiable state only;
- update canonicals to MERGED/DEPLOYED/SMOKE state and release writer.

## 7. Out of scope

```text
second diagnosis
billing/subscription engine
analytics
patient persistence
favorites account persistence
new GeSY clinical recommendations
new evidence surveillance pipeline
public unauthenticated physio product
```

## 8. Exact next action

Move the commercial product authorities, then adapt the tested Knee-OA runtime/UI to the existing protected Cockpit physiotherapy route and add production-integration regressions.
