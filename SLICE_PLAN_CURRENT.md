# SLICE_PLAN_CURRENT.md — Knee-OA v1 release closed

> **STATUS:** CLOSED — runtime merged and deployed; live public-asset/auth-boundary smoke passed; archive/closeout recorded.
> **Closed runtime slice:** `CU1-PRODUCT-KNEE-OA-COCKPIT-INTEGRATION-V1-20260912`.
> **Runtime release PR:** `#87`.
> **Runtime release SHA:** `eeec7f82b4f5a9054e7df51354803dd75e7dc9eb`.
> **Render deploy:** `dep-daifjt0jo6nc73biqhpg` — LIVE at exact runtime release SHA.
> **External live boundary-smoke:** `34680029691` — SUCCESS.
> **Writer:** NONE after this bounded archival closeout merges.
> **Next implementation slice:** NONE SELECTED.

## 1. Completed

The Knee-OA product has completed:

```text
design/evidence contract
→ deterministic referral contract
→ interaction/traceability contract
→ functional prototype
→ qualifier refinement
→ four specialist reviews + supplementary combined review
→ cross-review amendment
→ usability refinement
→ scan-first advanced redesign
→ protected Cockpit production integration
→ final release review
→ PR #87
→ squash merge
→ exact-SHA Render deploy
→ external live public-asset + auth-boundary smoke
→ immutable 4+1 review archive
```

## 2. Production state

The deployed production route is:

`/clinical/clinic-utilities/physio-referral`

Existing authentication and real CU-1 validation/safety remain authoritative. The local loopback prototype is not mounted in production.

No database migration, patient persistence, analytics, billing, new GeSY clinical rule or second diagnosis was added by the release.

## 3. Smoke precision

The live smoke proved deployment visibility and the unauthenticated protection boundary. It did not use or retrieve production credentials.

Therefore full authenticated live end-to-end smoke remains a separate future verification step and must not be silently inferred from local/authenticated FastAPI + Chromium release tests.

## 4. Historical review archive

Immutable review originals and checksums:

`commercial_products/physio_referral/reviews/archive/2026-09-11-knee-oa-candidate-6539351c/`

The archive contains four specialist sources plus one supplementary combined source. The supplementary file is not a fifth specialist axis.

## 5. Deferred validation, not release blockers retroactively

```text
actual iPhone Safari / VoiceOver
authorized authenticated live E2E smoke
receiving-physiotherapist field validation
real clinical pilot
paid conversion / retention
Cyprus/GeSY item-level activation
second-diagnosis selection
```

## 6. Exact next governance state

No active engineering writer. Future work requires a fresh bounded slice and explicit writer claim. A second diagnosis should be chosen only after product/receiver/market evidence justifies it.
