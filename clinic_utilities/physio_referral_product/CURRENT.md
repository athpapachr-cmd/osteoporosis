# CURRENT.md — Physio Referral product track

> **STATUS:** STEPS 1–4 DESIGN FROZEN; STEP 4 CLOSED.
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Branch:** `design/physio-referral-knee-oa-interaction-v1-2026-09-11`.
> **Main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Step-3 parent:** `8489e2ee32f7aeae6f678c7db2838930c0759eb4`.
> **Step-4 substantive head:** `e1039809818ddf4061e6d0350905578ff2ca16aa`.
> **Writer:** NONE. Root `CURRENT_OPERATIONAL.md` remains sole operational authority.
> **Runtime / PR / merge / deploy:** NOT AUTHORIZED.

## Completed design

Steps 1–3 preserve the minimal/mobile-first UX, six-state source-specific evidence architecture and deterministic Greek referral contract. The explicit clinician diagnosis assertion, laterality, supported-input scope, restriction preservation and safety/export boundaries remain unchanged.

Step 4 adds the interaction/traceability contract:

- separate selection, evidence, availability and safety axes;
- six non-colour evidence cues, one small evidence control and one sheet host;
- one expanded contextual bubble without a timer;
- all-source first disclosure for mixed guidance;
- explicit source scope/version/review-date and source-level locator precision;
- positive-only, source-backed suggestions with explicit add and stale-candidate rejection;
- preserved advanced choices and notes on collapse;
- safety/revision precedence for every export action;
- ephemeral traceability without patient storage or analytics.

Human specification: `KNEE_OA_EVIDENCE_INTERACTION_DESIGN_V1.md`.
Machine contract: `contracts/knee_oa_evidence_interaction_v1.yaml`.
Exact freeze/review: `KNEE_OA_EVIDENCE_INTERACTION_REVIEW_V1.md`.

## Obtained evidence

GitHub Actions run `34565131646`, job `103155584052`, passed on the exact Step-4 substantive head: 46 synthetic scenario/mutation checks plus three parent-blob identity checks. Active-writer design review passed; independent review was not performed.

The reviewed substantive artifacts retain their creation-state candidate headers. Their subsequent frozen lifecycle is owned by the exact review record and root canonicals, not by a silent rewrite of their content.

## Not yet proven

```text
functional Knee-OA prototype              NOT BUILT
new UI / runtime                          NOT IMPLEMENTED
actual contrast / Safari / VoiceOver      NOT TESTED
product-owner usability                   NOT TESTED
independent multi-axis review             NOT PERFORMED
independent source-to-claim clinical audit NOT PERFORMED in Step 4
exact per-recommendation locators          NOT PROVIDED by inherited registry
willingness to pay / commercial pilot      NOT VALIDATED
```

Walking-aid UI exposure, weight-management machine representation and true-locking safety mapping remain separate inherited seams. This stage did not reopen them.

## Exact next step

**STEP 5 — bounded functional Knee-OA prototype implementation gate.**

Fresh bootstrap → define narrow prototype entrypoint and adapter scope → separate implementation authority/writer → implement one synthetic Knee-OA flow → real browser/product-owner checks → independent review after the functional slice.

No second diagnosis, patient persistence, billing/auth/entitlement work, autonomous evidence update or production release is authorized by this closeout.
