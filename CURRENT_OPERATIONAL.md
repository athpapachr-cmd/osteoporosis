# CURRENT_OPERATIONAL.md — Step 5 Knee-OA prototype closeout

> **STATUS:** STEP 5 FUNCTIONAL PROTOTYPE IMPLEMENTED / FOCUSED TECHNICAL GATE PASS / WRITER CLOSED.
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Frozen Step-4 parent:** `4e0e3206dd1e12a2e55a6abd18d2a6f7dbc3f7c6`.
> **Branch:** `feat/physio-referral-knee-oa-prototype-v1-2026-09-11`.
> **Slice:** `CU1-PRODUCT-KNEE-OA-PROTOTYPE-V1-20260911`.
> **Tested substantive head:** `6595bf4cc41388dbd796f9ba5b53ae7c49bafdee`.
> **ACTIVE RUNTIME / DESIGN / CANONICAL WRITER:** NONE; bounded Step-5 writer released.
> **PR / merge / deploy / production smoke authority:** NONE.
> **Production API/UI/database/config/secrets/patient-data mutation:** NONE.

## 1. What was implemented

A functional, synthetic-only Knee-OA prototype exists at `clinic_utilities/physio_referral_product/prototype/`. It runs through `server.py` on `127.0.0.1` only. It is not registered with the production app, deployed or available through a public preview URL. Learning Hub, RF, production CU-1 and frozen Steps 1–4 are unchanged.

Implemented: Greek responsive UI, explicit diagnosis/laterality, three reviewed defaults, phenotype/function input, compact advanced selections, live deterministic referral, independent evidence/selection cues, source-backed explicit suggestions, all-source disagreement display, one evidence sheet, non-timed bubble, manual text reconciliation, common export guard, synthetic export stamp, reset and page-lifecycle cleanup.

The actual existing CU-1 engine validates the clinical draft. The adapter intentionally reuses frozen Step-3 composition and Step-4 evidence/suggestion functions as prototype-only dependencies. This is not approval of that coupling for a production release. No new clinical recommendation or autonomous literature update was introduced.

## 2. Exact technical evidence

```text
workflow: Physio Knee OA prototype gate
run: 34569247051
job: 103167619691
head: 6595bf4cc41388dbd796f9ba5b53ae7c49bafdee
result: SUCCESS
real CU-1 adapter / local HTTP tests: 15 / 15 PASS
exact frozen Greek output fixtures through adapter: 15 PASS (within that suite)
Greek source-summary display coverage: 54 source positions
actual Chromium / HTTP / CU-1 browser tests: 12 / 12 PASS
packaged dependency-closure smoke: PASS
scope and syntax gates: PASS
```

The first run `34568902516` at `2e9798b668fa65870e7831a4670c04acf67946b6` correctly failed one browser focus-containment test. The app was corrected; the test was not weakened. The successful run above includes that same test.

Browser checks cover normal and failing paths, explicit suggestion add/dismiss, all five acupuncture source positions, retained advanced selections, manual reconciliation, actual inherited safety block, deliberate network failure, 320/390/800/1280-pixel viewports, 200% text sizing, visible 44-pixel button hit boxes in tested views, forced-colour cue identity, modal keyboard containment, no browser storage and simulated BFCache events.

These are bounded automated technical results, not clinical validation, complete accessibility compliance, real Safari/VoiceOver acceptance, product-owner usability or independent review. No broad unrelated production regression was requested merely for reassurance.

## 3. Runnable package and traceability

CI artifact `10187149429` from the successful substantive run contains `physio-knee-oa-prototype.zip` plus actual browser screenshots. The inner package passed its own real-CU1 dependency-closure smoke. Its `BUILD_PROVENANCE.json` records the build source commit and SHA-256 hashes of packaged source files.

The downloaded substantive package was inspected: 77 manifest file hashes matched; no font binary was included. Inner ZIP SHA-256: `9e92586d3f597dbec1c75052943fbb508caa9f45938e7d54401c4ade3132724f`.

A subsequent documentation-only closeout build has a different package hash/source commit but must retain the same substantive prototype code. Its workflow result is obtained from GitHub by exact head, not predicted in this record. The reviewed/tested substantive identity above remains unchanged.

Run instructions: `prototype/README.md`; author review and limitations: `KNEE_OA_PROTOTYPE_REVIEW_V1.md` in the product directory. The archive is a local Python application, not a double-click standalone HTML file. Python 3.12 and the isolated requirements are needed. No production credentials are needed or included.

## 4. Remaining gates

```text
Step 5 functional implementation / focused technical checks   COMPLETE
Step 6 product-owner usability / clinical copy acceptance      PENDING
actual iPhone Safari / VoiceOver / live BFCache lifecycle       NOT TESTED
complete measured contrast/accessibility audit                 NOT PERFORMED
independent clinical / physio / UX / commercial review          NOT PERFORMED
independent source-to-claim audit / precise claim locators      NOT COMPLETED
production integration / public preview / commercial pilot     NOT AUTHORIZED
```

Greek source summaries are faithful display translations of inherited positions, not a renewed clinical literature review. Current source links remain source-level. Manual prose is clinician-owned and is not automatically evidence-validated. Synthetic-only markers are usage restrictions, not an automatic PHI detector.

## 5. Exact next action and HOLD

**Step 6:** product-owner testing of this exact local prototype using invented cases, evaluating speed, clarity, useful evidence, retained choices and what should be removed. Record feedback before adding functionality. Then prepare the agreed separate independent multi-axis review against a pinned candidate; do not relabel this author's review as independent.

No second diagnosis, production CU-1 rewrite, real-patient use, patient persistence, account/billing/entitlements, autonomous source updates, PR, merge, deployment, production smoke or public/LAN hosting. A future preview-hosting or production-integration request needs a separately bounded authority and access/security review.
