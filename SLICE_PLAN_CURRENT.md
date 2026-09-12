# SLICE_PLAN_CURRENT.md — Knee-OA Cockpit integration + release

> **STATUS:** IMPLEMENTED / EXACT-HEAD TESTED / RELEASE REVIEW PASS / PR NEXT.
> **Slice:** `CU1-PRODUCT-KNEE-OA-COCKPIT-INTEGRATION-V1-20260912`.
> **Branch:** `feat/physio-knee-oa-cockpit-integration-v1-2026-09-12`.
> **Parent closed v3:** `b7f2db6fd86f90a32875d9b9cfd356cfa6129c69`.
> **Fresh main / merge base:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Reviewed/tested RC head:** `57d879b072fcaf2bef350d5dbb0547ce465764e1`.
> **Writer:** ACTIVE only through PR / merge / deploy / smoke closeout.
> **Release / real clinical pilot validation:** release path authorized; pilot validation remains separate.

## 1. Delivered architecture

```text
six root canonicals
→ repo-wide operational authority

commercial_products/physio_referral/
→ product/commercial strategy, current state, product history, reviews/market/releases

clinic_utilities/physio_referral_product/
→ clinical/UX technical contracts, shared Knee-OA projection, validators/tests

static/clinic-utilities/physio-referral/
+ clinic_utilities/physio_referral_api.py
+ clinic_utilities/physio_referral_product/production_api.py
→ authenticated Cockpit production surface
```

Old product-canonical paths under the implementation tree are redirect-only compatibility files.

## 2. Production UI contract delivered

The existing protected Cockpit route `/clinical/clinic-utilities/physio-referral` now carries the reviewed Knee-OA product surface:

- explicit OA diagnosis assertion and laterality;
- live deterministic referral, no routine Generate button;
- direct `✎ Επεξεργασία` + safe manual reconciliation;
- evidence-aware defaults/suggestions;
- `Άλλες n προτάσεις` discoverability;
- `★ Συχνά / Σχετικά τώρα / Όλα` scan-first advanced architecture;
- qualifier semantics and FFD/weakness/pes-anserine safeguards;
- evidence conflict transparency;
- export/readiness safety precedence;
- no patient draft persistence.

## 3. Server/runtime ownership delivered

`knee_oa_projection.py` is the shared server-authoritative deterministic projection. Production and local prototype both use it.

The local `prototype/server.py` is only loopback transport and is never mounted in production.

The protected production boundary truthfully identifies Cockpit use as non-synthetic before adapting internally to the frozen compatibility envelope.

No database migration or patient write path was added.

## 4. Commercial canonical split delivered

Product authorities live under:

`commercial_products/physio_referral/`

Current key authorities:

- `CURRENT.md`
- `PRODUCT_CONTEXT_CURRENT.md`
- `PRODUCT_PLAN.md`
- `PRODUCT_CHANGELOG.md`
- `releases/KNEE_OA_V1_CURRENT.md`
- `releases/KNEE_OA_V1_RELEASE_REVIEW.md`

This subtree is not a seventh root canonical authority.

## 5. Exact acceptance evidence

At release-candidate head `57d879b072fcaf2bef350d5dbb0547ce465764e1`:

```text
Cockpit integration gate   34679427725   SUCCESS
Inherited product gate     34679427741   SUCCESS
CU-1 focused gate          34679427822   SUCCESS
```

The full release/red-team review is PASS with no open blocker. Its two material technical findings were corrected before the reviewed head.

Acceptance proven includes:

1. production route/API requires existing clinical authentication;
2. projection/safety remains server authoritative and real-CU1-backed;
3. all reviewed advanced/evidence/safety behavior remains reachable;
4. no browser storage/analytics/patient persistence was introduced;
5. production FastAPI + Chromium flow works at the exact RC head;
6. inherited CU-1 and Knee-OA regressions pass;
7. adjacent Learning Hub and RF regression smoke passes;
8. production clipboard export contains the clean referral rather than prototype markings;
9. mobile 390px reflow remains within viewport;
10. synthetic-marked requests are rejected at the production boundary.

## 6. Explicit non-claims

```text
actual iPhone Safari / VoiceOver acceptance    NOT YET PROVEN
receiving-physiotherapist field validation     NOT YET PROVEN
paid conversion / retention                    NOT YET PROVEN
GeSY item-level overlay audit                   NOT ACTIVATED
real clinical pilot                            NOT YET PROVEN
```

None of these are silently converted into release evidence.

## 7. PR / merge / deploy acceptance

Before squash merge:

- PR head must equal the reviewed/re-gated intended head;
- relevant PR-triggered workflows must settle cleanly;
- fresh compare to `main` must remain behind by 0 and bounded;
- no new material review finding may remain open.

After merge:

- verify exact new `main` SHA;
- verify normal Render auto-deploy source commit equals that merge SHA;
- production smoke must use no identifiable patient data;
- close canonicals with exact deploy/smoke evidence and writer `NONE`.

## 8. Exact next action

Open the bounded PR to current `main`, verify exact PR-head checks, then follow the already-authorized squash-merge/deploy/smoke path if clean.
