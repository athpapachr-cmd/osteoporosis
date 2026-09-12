# CURRENT.md — Physio Referral commercial product track

> **STATUS:** KNEE-OA V1 RUNTIME RELEASED / DEPLOYED / LIVE PUBLIC-ASSET + AUTH-BOUNDARY SMOKE PASS.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Diagnosis vertical:** Knee Osteoarthritis only.
> **Runtime release PR:** `#87`.
> **Runtime release SHA:** `eeec7f82b4f5a9054e7df51354803dd75e7dc9eb`.
> **Render deploy:** `dep-daifjt0jo6nc73biqhpg` — LIVE.
> **Release closeout:** `releases/KNEE_OA_V1_RELEASE_CLOSEOUT.md`.
> **Review archive:** `reviews/archive/2026-09-11-knee-oa-candidate-6539351c/`.
> **Root operational authority:** `CURRENT_OPERATIONAL.md`.

## Product state

The reviewed Knee-OA experience is now the deployed Clinical Excellence Cockpit physiotherapy surface. The product keeps the accepted interaction philosophy: live deterministic referral, few meaningful routine actions, evidence complexity under progressive disclosure, direct editing with fail-closed reconciliation, and scan-first advanced access rather than a comprehensive visible form.

Commercial/product authority lives under `commercial_products/physio_referral/`; technical contracts/runtime/tests remain under `clinic_utilities/physio_referral_product/`.

## Review history

The four specialist review axes remain:

1. Clinical / Evidence
2. Physiotherapy / receiving-professional utility
3. UX / Product
4. Commercial / Product-Market

A fifth source file is archived as a supplementary combined / multi-axis review, not a second specialist physiotherapy review.

All five original uploads are now preserved byte-for-byte in the review archive with exact hashes.

## Release state

Supported lifecycle claims:

- merged: yes;
- deployed: yes;
- live public product assets: verified;
- unauthenticated auth boundary: verified;
- full authenticated live end-to-end smoke: not yet performed.

The last distinction is deliberate. Release-code authenticated behavior passed protected FastAPI and Chromium tests, but no production credential/session was used in the external live smoke.

## Product truth still unproven

A released technical feature is not yet a validated business or clinical pilot:

- no receiving-physiotherapist field validation yet;
- no actual iPhone Safari/VoiceOver acceptance yet;
- no paid conversion/retention evidence yet;
- no real clinical pilot validation yet;
- Cyprus/GeSY recommendation-by-recommendation overlay is not activated;
- no second diagnosis has been selected.

## Next product boundary

Do not expand diagnosis count automatically. The next product-learning step should come from real receiver/device/market evidence, plus an authorized authenticated live verification when operationally safe.
