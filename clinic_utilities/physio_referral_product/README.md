# Physio Referral Productization — supporting control plane

> **Status:** Steps 1–4 frozen; Step 5 functional synthetic prototype implemented and technically tested; Step 6 owner trial next.
> **Repository:** `athpapachr-cmd/osteoporosis`.
> **Scope:** Knee Osteoarthritis only.
> **Authority:** the root six canonicals remain authoritative; this directory is not a seventh root authority.

## Start with the prototype

`prototype/README.md` contains Greek instructions. Run `prototype/server.py` from a repository checkout or the CI-generated standalone source ZIP using Python 3.12 and `prototype/requirements.txt`. The app binds to `127.0.0.1` only and must be used with invented cases, not real patient data.

`KNEE_OA_PROTOTYPE_REVIEW_V1.md` records exact implemented/tested scope and limitations. It is an author review, not an independent review. Actual CI produces a runnable ZIP with a source/hash manifest and desktop/mobile/evidence screenshots. The prototype is not a static HTML-only download or a publicly hosted website.

## Navigation

| Document | Purpose |
|---|---|
| `PRODUCT_PLAN.md` | Product direction, €9.99 pricing hypothesis and staged commercial validation |
| `CURRENT.md` | Product-track status and next acceptance gate |
| `CHANGELOG.md` | Append-only product history |
| `UX_CONTRACT_CURRENT.md` | Frozen minimal/mobile-first UX |
| `KNEE_OA_EVIDENCE_DESIGN_V1.md` | Step-2 source-specific evidence architecture |
| `contracts/knee_oa_evidence_contract_v1.yaml` | Clinical source positions, six states, defaults and eligibility |
| `KNEE_OA_EVIDENCE_DESIGN_REVIEW_V1.md` | Step-2 author review |
| `KNEE_OA_TEMPLATE_DESIGN_V1.md` | Step-3 deterministic Greek composition |
| `contracts/knee_oa_template_contract_v1.yaml` | Supported input and template contract |
| `KNEE_OA_TEMPLATE_DESIGN_REVIEW_V1.md` | Step-3 author review |
| `KNEE_OA_EVIDENCE_INTERACTION_DESIGN_V1.md` | Step-4 evidence disclosure, provenance and interaction |
| `contracts/knee_oa_evidence_interaction_v1.yaml` | Frozen Step-4 machine contract |
| `KNEE_OA_EVIDENCE_INTERACTION_REVIEW_V1.md` | Step-4 author review and limitations |
| `prototype/` | Isolated functional UI/server, tests, Greek summaries and package builder |
| `KNEE_OA_PROTOTYPE_REVIEW_V1.md` | Step-5 exact technical evidence and remaining acceptance |

## Current boundary

The tested substantive prototype head is `6595bf4cc41388dbd796f9ba5b53ae7c49bafdee`; focused run `34569247051` passed 15 backend/HTTP and 12 actual Chromium tests plus packaged dependency closure. Production CU-1, clinical taxonomy and frozen evidence are unchanged. No branch was merged or deployed by this work.

Source-level attribution is not independently verified recommendation-level evidence. Technical tests are not clinical certification, complete accessibility compliance, independent product review or commercial validation. Real Safari/VoiceOver and owner acceptance remain pending.

Next is **Step 6 product-owner trial**, then the separately agreed independent review. Walking-aid exposure, weight-management representation and true-locking safety mapping remain explicit separate seams. Do not add a second diagnosis or expose the loopback prototype publicly under this closeout.
