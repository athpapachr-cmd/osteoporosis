# SLICE_PLAN_CURRENT.md — Physio Referral Step 4

> **STATUS:** DESIGN FROZEN / CLOSED; runtime NOT IMPLEMENTED.
> **Slice:** `CU1-PRODUCT-KNEE-OA-INTERACTION-V1-2026-09-11`.
> **Branch:** `design/physio-referral-knee-oa-interaction-v1-2026-09-11`.
> **Verified main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Frozen Step-3 parent:** `8489e2ee32f7aeae6f678c7db2838930c0759eb4`.
> **Reviewed substantive head:** `e1039809818ddf4061e6d0350905578ff2ca16aa`.
> **Writer:** NONE.
> **Runtime, PR, merge, deploy, production smoke:** NOT AUTHORIZED.

## Frozen objective

One evidence interaction layer for Knee Osteoarthritis: quiet evidence cues, concise anchored messages, one information sheet, honest source attribution and evidence-backed suggestions. The frozen Step-2 clinical positions and Step-3 deterministic referral behavior remain unchanged.

## Normative owners

`CURRENT_OPERATIONAL.md` owns the sole writer/operational state. In the product directory:

- `UX_CONTRACT_CURRENT.md` owns the inherited product direction.
- `contracts/knee_oa_evidence_contract_v1.yaml` owns source positions, six evidence states and suggestion eligibility.
- `contracts/knee_oa_template_contract_v1.yaml` owns supported inputs, referral composition and Copy authority.
- `KNEE_OA_EVIDENCE_INTERACTION_DESIGN_V1.md` and `contracts/knee_oa_evidence_interaction_v1.yaml` own Step-4 interaction/provenance behavior.
- `KNEE_OA_EVIDENCE_INTERACTION_REVIEW_V1.md` records the exact active-writer review and limitations.

Candidate headers in the reviewed substantive artifacts describe their creation state. This subsequent canonical freeze applies to their exact content at the reviewed head; it does not imply clinical re-review or implemented UI.

## Acceptance obtained

Focused workflow `34565131646`, job `103155584052`, passed on the reviewed substantive head:

```text
46 synthetic scenario/mutation checks
3 unchanged parent blob identities
6 distinct evidence states and non-colour cues
```

The checked model covers evidence/selection separation, full mixed-source preservation, availability overlay, claim scope, publication/review dates, positive-only triggers, omission/context suggestion order, aliasing/deduplication, explicit add, stale-candidate rejection, draft-local dismissal, quiet disclosure state and export-readiness precedence.

Static contract assertions and synthetic model transitions are not DOM, VoiceOver, clinical-effectiveness or independent-review tests.

## Preserved invariants

No indication is inferred merely by opening a flow. Evidence is not clinical safety clearance. Suggestion is not selection. A source's native recommendation strength is not an app evidence verdict or a narrower item's strength. Unknown/absent findings do not become positive trigger facts. A source outage does not turn disagreement into consensus. A hidden selected advanced item is not discarded. Evidence UI never writes back clinical facts or manual text.

## Explicit remaining acceptance

Step 5/6 must implement and verify the actual adapter, neutral selection check, six visible cues, hit targets, focus/inert behavior, one-sheet replacement, review summary, preserved advanced counts, export guards and manual-buffer reconciliation. Verify 320-CSS-pixel reflow, 200% text enlargement, actual contrast, reduced motion and Safari/VoiceOver.

Clinical source URLs currently have source-level precision. Do not claim exact recommendation/page verification or generate invented locators. Final Greek source-summary copy must preserve the inherited claims. Independent clinical/physio/UX/commercial review follows the functional vertical slice.

## Scope and next step

No production runtime/API/formatter/UI/database mutation, second diagnosis, patient persistence, billing/auth, new clinical recommendation, autonomous source updating, PR or release occurred. The root roadmap's parked-utility language is not broad CU-1 authorization; this product-owner-authorized track remains a separate bounded design effort.

Next: a separately authorized and freshly bootstrapped **Step-5 functional Knee-OA prototype implementation gate**. Do not extend this closed Step-4 writer into runtime work.
