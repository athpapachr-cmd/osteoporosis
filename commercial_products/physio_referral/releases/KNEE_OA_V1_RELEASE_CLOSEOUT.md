# Knee OA v1 — release/deployment closeout

> **Runtime release PR:** `#87`
> **Squash-merge SHA / runtime release SHA:** `eeec7f82b4f5a9054e7df51354803dd75e7dc9eb`
> **Render service:** `osteoporosis` / `srv-d5qfk31r0fns73di596g`
> **Render deploy:** `dep-daifjt0jo6nc73biqhpg`
> **Deploy source commit:** `eeec7f82b4f5a9054e7df51354803dd75e7dc9eb`
> **Deploy status:** `live`
> **Deploy finished:** 2026-09-12 07:08:08 UTC
> **Production URL:** `https://ortho-reception-backend.onrender.com`
> **External live boundary-smoke run:** `34680029691` — SUCCESS

## 1. Merge and PR evidence

PR `#87` was squash-merged only after the exact PR head remained mergeable and current with `main`.

Relevant release/product checks passed, including:

- Knee-OA Cockpit integration gate
- Knee-OA inherited prototype gate
- CU-1 focused tests
- Knee-OA evidence design validator
- Knee-OA template design validator
- Knee-OA interaction design validator
- G3 adjacent-owner regression

Four Clinical Learning workflows showed red PR conclusions because their own scope guards rejected a non-Learning slice. In each of those workflows, the actual Learning syntax/contracts/runtime/inherited regression steps completed successfully before the scope guard failed. They were not treated as Learning regressions and their workflows were not weakened merely to make the PR dashboard green.

## 2. Deployment verification

Render auto-deploy was enabled on `main`; no manual deploy was triggered.

The deployment record `dep-daifjt0jo6nc73biqhpg` reported `live` and its source commit exactly matched the squash-merge runtime SHA:

`eeec7f82b4f5a9054e7df51354803dd75e7dc9eb`

This establishes the `DEPLOYED` lifecycle state for the Knee-OA Cockpit runtime.

## 3. External production boundary smoke

A temporary, no-secret GitHub Actions smoke workflow was run from the closeout branch and removed immediately after completion.

Run `34680029691` completed `SUCCESS`.

It verified against the live Render URL:

- deployed Knee-OA static `index.html` returned `200` and contained the reviewed product surface;
- deployed `production-env.js` returned `200` and exposed the non-synthetic production boundary;
- deployed `product-more-v3.js` returned `200` and contained the scan-first advanced UI;
- `/clinical/clinic-utilities/physio-referral` returned `401` without authentication;
- `/clinical/clinic-utilities/physio-referral/api/product/bootstrap` returned `401` without authentication.

Render logs independently recorded the same `200` asset requests and `401` protected-route/API requests.

No clinical key, login password, session cookie, patient identifier or patient data was supplied.

## 4. Lifecycle wording

The following claims are supported:

```text
MERGED                                             YES
DEPLOYED                                           YES
PUBLIC-ASSET LIVE SMOKE                            PASS
UNAUTHENTICATED AUTH-BOUNDARY LIVE SMOKE           PASS
FULL AUTHENTICATED LIVE END-TO-END SMOKE           NOT YET PERFORMED
PILOT-VALIDATED                                    NO
COMMERCIAL/PAID VALIDATED                          NO
```

Because no authorized production credential/session was used, this closeout does **not** claim a full authenticated live end-to-end production smoke. Authenticated route/API behavior is extensively covered at the exact release code by protected FastAPI and Chromium integration tests, but that is distinct from a live authenticated production session.

## 5. Review-history archive

The four specialist reviews plus one supplementary combined review are archived byte-for-byte under:

`commercial_products/physio_referral/reviews/archive/2026-09-11-knee-oa-candidate-6539351c/`

See `MANIFEST.md` for exact source filenames, roles, sizes and hashes.

## 6. Remaining validation boundaries

Still not proven by this release:

- actual iPhone Safari / VoiceOver acceptance;
- authenticated live end-to-end smoke using an authorized production session;
- receiving-physiotherapist field validation;
- real clinical pilot validation;
- paid conversion/retention and willingness-to-pay;
- item-level Cyprus/GeSY overlay activation;
- readiness for a second diagnosis.

The technical runtime release is closed. Future product expansion should be driven by receiver/device/market evidence rather than by feature-count pressure.
