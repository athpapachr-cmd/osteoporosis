# CURRENT_OPERATIONAL.md — Step 5 Knee-OA prototype

> **STATUS:** STEP 5 IMPLEMENTATION ACTIVE; NON-PRODUCTION / SYNTHETIC ONLY.
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Canonical home:** athpapachr-cmd/osteoporosis.
> **Verified main:** d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37.
> **Frozen Step-4 parent:** 4e0e3206dd1e12a2e55a6abd18d2a6f7dbc3f7c6.
> **Branch:** feat/physio-referral-knee-oa-prototype-v1-2026-09-11.
> **Slice:** CU1-PRODUCT-KNEE-OA-PROTOTYPE-V1-20260911.
> **ACTIVE WRITER:** this conversation, bounded prototype and its supporting canonicals/tests only.
> **Authority:** product owner requested progression to Step 5 after Step-4 closeout, latest instruction `Προχωρά`.
> **Production runtime/API/UI/database/config/secrets/patient-data mutation:** NONE.
> **PR / merge / deployment / production smoke authority:** NONE.

## Bootstrap and preserved authority

Fresh GitHub reads verified main and the closed Step-4 branch at the exact identities above. The six root canonicals at this unchanged main and the stacked ancestry have been read in this conversation; main records no active writer and Clinical Learning L-1D closed. Steps 1–4 are inherited at their frozen identities, not re-certified as independent clinical review. Step 4's own writer review is not independent review.

## Bounded implementation

Create one isolated, loopback-only functional prototype under `clinic_utilities/physio_referral_product/prototype/`. It must not be registered with the production application, placed under production static routes, or deployed. It uses synthetic test interactions, no patient history/persistence, no account/billing code, and no external services except deliberate activation of static evidence links.

Reuse the actual existing CU-1 validation engine through a narrow adapter; do not replace clinical validation with a green UI flag. Reuse the frozen Step-3 composition and Step-4 evidence/suggestion functions as explicitly prototype-only dependencies. Extraction into production modules is a later boundary. The existing CU-1 runtime, source contracts and design validators remain unchanged.

The interface must implement live referral, neutral selection versus evidence cues, compact power-user disclosure, explicit source-backed suggestions, one evidence sheet, no timed evidence messages, all-source mixed-guidance display, revision-bound export, manual-buffer protection and reset/BFCache cleanup. Exported samples are prominently marked non-clinical demonstration material.

## Verification boundary

Focused Python integration tests must call the real CU-1 engine. Browser tests must exercise the actual prototype server and DOM, not only a synthetic state model. Any local mocked transport used for visual development is not integration evidence. Actual iPhone/Safari/VoiceOver and product-owner usability remain separate acceptance. Clinical source positions are inherited, not newly reviewed by UI work.

## Exact next action

Record the narrow implementation contract, implement prototype, run focused integration/browser acceptance, inspect failures and fix only bounded defects, then record exact results and release writer. If incomplete, preserve exact remaining work without declaring Step 5 closed.

## HOLD

No second diagnosis; no production registration/rewrite; no clinical evidence rule changes; no real patient data; no persistence/analytics; no billing/auth/entitlements; no autonomous source updates; no PR/merge/deploy/production smoke. Step-5 implementation authority does not waive independent review or commercial validation.
