# SLICE_PLAN_CURRENT.md — Step 5 functional Knee-OA prototype

> **STATUS:** IMPLEMENTED / FOCUSED TECHNICAL ACCEPTANCE PASS / CLOSED FOR PRODUCT-OWNER TESTING.
> **Slice:** `CU1-PRODUCT-KNEE-OA-PROTOTYPE-V1-20260911`.
> **Branch:** `feat/physio-referral-knee-oa-prototype-v1-2026-09-11`.
> **Frozen parent:** `4e0e3206dd1e12a2e55a6abd18d2a6f7dbc3f7c6`.
> **Verified main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Tested substantive head:** `6595bf4cc41388dbd796f9ba5b53ae7c49bafdee`.
> **Writer:** NONE.
> **Release / real clinical use:** NOT AUTHORIZED.

## 1. Implemented boundary

Only new `clinic_utilities/physio_referral_product/prototype/` code, `.github/workflows/physio-knee-oa-prototype.yml` and supporting canonical/progress documents. No production router/static registration or mutation of existing CU-1, Learning Hub, RF, database, clinical contracts or evidence positions.

Entrypoint: `python clinic_utilities/physio_referral_product/prototype/server.py`. Bind: `127.0.0.1:8765`, with optional local port override. No public or LAN service is authorized.

The actual CU-1 engine owns validation/safety. Existing Greek labels and frozen Step-3 composition / Step-4 evidence and suggestion functions are reused read-only. Reuse of design-checker functions is a deliberate synthetic-prototype dependency, not the approved production architecture. Later extraction/production integration requires a new bounded review.

## 2. Data, transport and authority

The ephemeral request contains UUID draft identity, nonnegative revision, fixed package version, `synthetic_only=true`, bounded selected state and draft-local dismissal keys. Unknown keys/types/IDs and stale suggestions are rejected. Neither the browser nor caller may forge validation, safety acknowledgement or clearance.

Allowed state preserves two phenotype booleans, positive selected findings/functions, selected rehabilitation/adjuncts/goals, explicit restrictions and normalized literal clinician note. Stiffness is not measured ROM; generic weakness is not objective weakness. No patient identity or history store was added.

Transport is allowlisted local GET and bounded JSON POST. Exact loopback Host/peer, same-origin policy, custom request header, no CORS, no directory listing, no-store/CSP/referrer headers and silent request logging were implemented. Source links are deliberate static HTTPS navigation only. No analytics or AI requests.

## 3. Clinical flow and export

Explicit clinician diagnosis assertion and side precede a copy-ready result. Selected state updates the real CU-1 gate and deterministic Greek text. Every change immediately invalidates export until a matching draft/revision/package response arrives. Stale or failing responses cannot preserve an old exportable referral.

The UI separates neutral selection from six evidence cues. Advanced options stay compact and retain unique selected IDs; selected optional interventions move to the visible plan rather than creating duplicate visible controls. A suggestion cannot add treatment without an explicit action and current-candidate validation.

One evidence message has no timer; one modal host presents summary, source direction, scope, original recorded wording/strength and source links. Mixed guidance preserves all positions at first disclosure. Greek display translations do not renew the clinical-review date. No exact recommendation/page locator is invented.

Manual text is a separate ephemeral buffer. Structured changes retain it but require reconciliation before export. Copy and browser print/PDF share the guard and carry the synthetic/non-clinical stamp. Browser print-to-PDF is not a separate server PDF generator or proof of real print-dialog interoperability.

## 4. Executed focused acceptance

Run `34569247051`, job `103167619691`, succeeded at the exact substantive head:

- 15 real CU-1 adapter/HTTP test methods, including 15 exact inherited Greek output fixtures;
- 54 source positions with Greek display summaries and preserved source scope;
- 12 actual Chromium tests against the actual local HTTP server and CU-1 engine;
- packaged real-CU1 dependency-closure smoke;
- syntax and changed-path scope checks.

The preceding run exposed modal keyboard focus escape. The correction changed the UI, retained the failing test, and passed the full focused gate. Local mocked transport was used only for preparatory visual inspection; it is not clinical or integration evidence. Actual CI screenshots are supplied separately.

## 5. Deferred acceptance and stop rule

Real iPhone Safari/VoiceOver, actual browser BFCache navigation, complete contrast/accessibility audit, clinical acceptance of source summaries, external usability, independent clinical/physio/UX/commercial review and willingness-to-pay remain unproven. Simulated lifecycle events and narrow Chromium viewports must not be called real-device tests.

No broad taxonomy change: hidden walking-aid, weight-management, dry-needling and true-locking seams remain as previously scoped. No second diagnosis, real-patient use, persistence, billing, public hosting or production release.

The bounded implementation objective is complete. Next is **Step 6 product-owner trial**, then the independent review. Do not continue adding features under the closed Step-5 writer.
