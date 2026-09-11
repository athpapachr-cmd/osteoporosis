# Physio Referral Productization — supporting control plane

> **Status:** Steps 1–4 design frozen; functional prototype next, not implemented.
> **Repository:** `athpapachr-cmd/osteoporosis`.
> **Parent:** existing Clinic Utilities / CU-1 Physiotherapy Referral.
> **Scope:** Knee Osteoarthritis only.
> **Authority:** root six canonicals remain authoritative; this directory is not a seventh root authority.

## Navigation

| Document | Purpose |
|---|---|
| `PRODUCT_PLAN.md` | Product direction, €9.99 pricing hypothesis, staged commercial validation |
| `CURRENT.md` | Current product-track status and exact next gate |
| `CHANGELOG.md` | Append-only product history |
| `UX_CONTRACT_CURRENT.md` | Step-1 minimal/mobile-first UX with Step-2 evidence-state replan |
| `KNEE_OA_EVIDENCE_DESIGN_V1.md` | Step-2 source-specific clinical evidence architecture |
| `contracts/knee_oa_evidence_contract_v1.yaml` | Evidence states, source positions, defaults and eligibility |
| `KNEE_OA_EVIDENCE_DESIGN_REVIEW_V1.md` | Step-2 active-writer review |
| `KNEE_OA_TEMPLATE_DESIGN_V1.md` | Step-3 deterministic Greek composition and ownership |
| `contracts/knee_oa_template_contract_v1.yaml` | Template and supported-input contract |
| `KNEE_OA_TEMPLATE_DESIGN_REVIEW_V1.md` | Step-3 active-writer review |
| `KNEE_OA_EVIDENCE_INTERACTION_DESIGN_V1.md` | Step-4 evidence cues, bubbles, sheet, suggestions, accessibility |
| `contracts/knee_oa_evidence_interaction_v1.yaml` | Step-4 interaction and provenance contract |
| `contracts/knee_oa_evidence_interaction_fixtures_v1.yaml` | Synthetic interaction cases |
| `validate_knee_oa_evidence_interaction_v1.py` | Step-4 design checker, not production code |
| `KNEE_OA_EVIDENCE_INTERACTION_REVIEW_V1.md` | Exact Step-4 review/freeze and explicit limitations |

## Current boundary

Step-4 content at `e1039809818ddf4061e6d0350905578ff2ca16aa` passed the focused design gate and is frozen by its review/closeout record. Existing production CU-1 runtime and clinical taxonomy are unchanged. No design branch has been merged by this work.

A source-level evidence link is not an independently verified recommendation-level locator. A synthetic design PASS is not clinical revalidation, a browser/accessibility PASS, independent product review or commercial validation.

Next is the separately governed Step-5 functional prototype for one diagnosis. Walking-aid exposure, weight-management representation and true-locking safety mapping remain explicit separate seams.

```text
DESIGNED != IMPLEMENTED != BROWSER-TESTED != INDEPENDENTLY REVIEWED
!= MERGED != DEPLOYED != COMMERCIAL-PILOT-VALIDATED
```
