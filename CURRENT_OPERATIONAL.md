# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PHYSIO REFERRAL STEP 4 — EVIDENCE INTERACTION / TRACEABILITY DESIGN ACTIVE.
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Frozen Step-3 parent:** `8489e2ee32f7aeae6f678c7db2838930c0759eb4`.
> **Active branch:** `design/physio-referral-knee-oa-interaction-v1-2026-09-11`.
> **Active slice:** `CU1-PRODUCT-KNEE-OA-INTERACTION-V1-2026-09-11`.
> **ACTIVE DESIGN/CANONICAL WRITER:** this bounded Step-4 session only.
> **ACTIVE RUNTIME WRITER:** NONE.
> **PR/merge/deploy/production smoke authority:** NONE.
> **Patient-data / production config / secrets authority:** NONE.

## Bootstrap and ancestry

Remote main and the Step-3 branch were freshly verified. Main has no active writer; Clinical Learning L-1D is closed. The six canonicals were read on the exact Step-3 ancestry, with main operational state checked separately. The Step-3 parent is 48 commits ahead of main, behind 0, and contains design/contracts/validators/workflows only. Earlier design branches remain unmerged and unchanged.

Prior milestones remain recorded, not re-certified independently here:
- Step 1 UX frozen, including the explicit Step-2 six-state replan.
- Step 2 evidence design frozen; clinical source positions are read-only in Step 4.
- Step 3 template design frozen; substantive head `cd4a42b4582921df7eb64d6ff3fb7c718a141c3a`, review artifact `11f04d9ea316ba7fc30a03f4aaa13b60ea5c1e65`.
- Existing CU-1 and Learning Hub production runtime are unchanged.

## Authorized scope

The product owner's continuation authorizes Step-4 DESIGN ONLY: evidence cues, compact contextual messages, information sheet, honest source/version/review-date display, suggestion provenance, disclosure behavior, accessibility acceptance criteria and executable synthetic design-contract checks.

Writable scope: this operational file, `SLICE_PLAN_CURRENT.md`, supporting `clinic_utilities/physio_referral_product/` Step-4 documents/contracts/fixtures/validator, bounded design-only workflow, and necessary product navigation/history updates. A machine-check PASS is not clinical revalidation, visual usability validation or independent review.

## Hard boundaries

- Six evidence states, Step-2 source positions and Step-3 clinical/text semantics remain unchanged.
- Selection, evidence, source availability and safety are separate axes.
- Suggestions never select treatment automatically.
- No patient data in repository, tests, logs, URLs or browser storage.
- No runtime/API/formatter/production UI/database changes.
- No new diagnosis, billing, authentication, entitlements, autonomous literature updates, PR, merge or deployment.
- Do not rerun inherited tests for reassurance; test new design behavior with a bounded purpose.

## Exact next action

Specify the Step-4 contract and interaction cases, verify the applicable Apple/W3C accessibility guidance, validate the new contract against exact frozen Step-2/3 identities, perform an explicitly active-writer review, then record the result and release this design writer. Step 5 requires a separate bounded prototype implementation gate.
