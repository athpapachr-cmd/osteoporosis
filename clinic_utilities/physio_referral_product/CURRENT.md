# CURRENT.md — Physio Referral product track

> **STATUS:** STEPS 1–4 FROZEN; STEP 5 FUNCTIONAL PROTOTYPE IMPLEMENTED / TECHNICAL GATE PASS.
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Branch:** `feat/physio-referral-knee-oa-prototype-v1-2026-09-11`.
> **Main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Step-4 parent:** `4e0e3206dd1e12a2e55a6abd18d2a6f7dbc3f7c6`.
> **Tested substantive head:** `6595bf4cc41388dbd796f9ba5b53ae7c49bafdee`.
> **Writer:** NONE. Root `CURRENT_OPERATIONAL.md` remains sole operational authority.
> **Production registration / PR / merge / deploy:** NONE.

## What exists now

A real, runnable synthetic Knee-OA prototype, not just design contracts. `prototype/README.md` contains Greek setup and test instructions. `prototype/server.py` runs locally on `127.0.0.1` and uses the real existing CU-1 validation engine. The website cannot be opened as standalone HTML or accessed from another device without a separately designed hosting change.

Implemented flow: diagnosis/side, selected phenotype/function, three defaults, compact advanced groups, dynamic Greek referral, evidence cues, explicit suggestions, all-source disagreement, one evidence sheet, preserved manual edits, guarded synthetic copy/print and no draft persistence.

## Proven technical results

GitHub Actions `34569247051` at the substantive head passed: 15 backend/HTTP tests, 15 exact-output fixtures within that suite, 54 source-summary display positions, 12 actual Chromium tests and packaged dependency closure. Actual browser tests use the real server and CU-1 engine. The first run's keyboard containment failure was corrected in the UI, not suppressed in tests.

The six pinned parent files remain unchanged. Existing production code and clinical evidence positions were not rewritten. The author review is `KNEE_OA_PROTOTYPE_REVIEW_V1.md`, explicitly not independent review.

## Not yet proven

```text
product-owner usability / clinical-copy acceptance      PENDING
actual iPhone Safari / VoiceOver                         NOT TESTED
real BFCache lifecycle, full accessibility audit         NOT PROVEN
independent clinical / physio / UX / commercial review    NOT PERFORMED
independent source-to-claim audit                         NOT COMPLETED
exact per-recommendation/page locators                    NOT PROVIDED
willingness to pay / commercial pilot                     NOT VALIDATED
production release / public preview                      NOT AUTHORIZED
```

The clinical corpus is inherited. Greek display translations are not a fresh literature review. Manual prose is not automatically evidence-validated. Source-level links are not precise verified recommendation locators.

## Exact next action

**STEP 6 — product-owner trial of the local prototype using synthetic cases.**

Assess actual ease, text usefulness, clarity of evidence, suggestion burden, extra-option discoverability and what should be removed. Then request the planned separate independent review against a pinned candidate. No second diagnosis or release before those gates.

No real patient data, production rewrite, public/LAN hosting, persistence, billing/auth/entitlements, autonomous evidence updating, PR/merge/deploy/production smoke is authorized by this closeout.
