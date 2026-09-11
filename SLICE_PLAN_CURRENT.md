# SLICE_PLAN_CURRENT.md — Physio Referral Step 4

> **STATUS:** DESIGN ACTIVE; not runtime implementation.
> **Slice:** `CU1-PRODUCT-KNEE-OA-INTERACTION-V1-2026-09-11`.
> **Branch:** `design/physio-referral-knee-oa-interaction-v1-2026-09-11`.
> **Verified main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Frozen parent:** `8489e2ee32f7aeae6f678c7db2838930c0759eb4`.
> **Runtime, PR, merge, deploy, production smoke:** NOT AUTHORIZED.

## Objective

Define one implementable evidence interaction layer for Knee Osteoarthritis: quiet evidence cues, concise contextual bubbles, one information sheet, faithful source attribution and evidence-backed suggestions. Preserve the frozen Step-2 clinical positions and Step-3 deterministic referral behavior.

## Owners

- `CURRENT_OPERATIONAL.md`: sole writer/operational authority.
- `clinic_utilities/physio_referral_product/UX_CONTRACT_CURRENT.md`: inherited product direction.
- `contracts/knee_oa_evidence_contract_v1.yaml`: clinical source positions, six evidence states and suggestion eligibility.
- `contracts/knee_oa_template_contract_v1.yaml`: supported input scope, referral composition and Copy authority.
- `KNEE_OA_EVIDENCE_INTERACTION_DESIGN_V1.md` and `contracts/knee_oa_evidence_interaction_v1.yaml` in the product directory: Step-4 interaction/provenance contract.

## Deliverables and acceptance

Human design; machine contract; executable synthetic design validator; focused workflow; active-writer review with exact head and scope proof. Machine checks must verify parent blob identities, six distinct non-colour cues, selection/evidence separation, complete conflict attribution, source-year/review-date separation, honest claim scope, suggestion eligibility and dismissal, advanced-selection visibility, no clinical-state mutation and no safety/Copy bypass.

Browser rendering, contrast measurement on actual pixels, VoiceOver/Safari behavior, measured tap count, usability and independent review remain NOT TESTED in this design slice.

## Scope and stop rules

No production runtime/API/formatter/UI/database mutation, second diagnosis, patient persistence, billing/auth, new clinical recommendation, autonomous source updating, PR or release. No active Step-2/3 contract is silently rewritten. The Step-3 source corpus is retained read-only; Step 4 is not an independent clinical re-review.

If an exact source locator is unavailable, label its precision honestly instead of inventing a recommendation number. If an evidence dependency is unavailable or unverified, suppress new endorsement/promotion, retain clinician selections and show the last reviewed state with its limitation. Inherited safety blocks always outrank quiet evidence messages.

The roadmap's parked-utility language does not authorize general CU-1 work. The product owner's explicit continuation enables this narrow design track only. Existing unrelated roadmap/history drift is not corrected opportunistically.

## Next boundary

After the focused design gate and active-writer review: close Step 4, release its writer and identify a separately authorized Step-5 prototype gate. No prototype or commercial-readiness claim follows from a design PASS.
