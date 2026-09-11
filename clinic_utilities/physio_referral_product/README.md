# Physio Referral Productization — supporting control plane

> **STATUS:** ACTIVE supporting product-design control plane; Steps 1–2 designed, Step 3 next.
> **Repository:** `athpapachr-cmd/osteoporosis`.
> **Parent area:** `Clinic Utilities / CU-1 Physiotherapy Referral`.
> **Vertical slice:** Knee Osteoarthritis only.
> **Important:** this directory does **not** create a seventh root canonical authority. The repository-wide six canonicals in `AGENTS.md` remain authoritative for repo-wide operational state and writer locks.

This directory preserves the commercial/product-design evolution of the existing CU-1 Physiotherapy Referral runtime so durable decisions are not left only in chat history.

## Local supporting documents

```text
PRODUCT_PLAN.md
→ product/commercial direction and staged roadmap

UX_CONTRACT_CURRENT.md
→ approved minimal/mobile-first interaction contract
→ includes the Step-2 six-state evidence REPLAN

KNEE_OA_EVIDENCE_DESIGN_V1.md
→ human-readable Step-2 Knee-OA evidence architecture

contracts/knee_oa_evidence_contract_v1.yaml
→ machine-readable source positions, evidence states, defaults and suggestion rules

validate_knee_oa_evidence_contract_v1.py
→ machine integrity gate against the existing CU-1 registry/catalog

KNEE_OA_EVIDENCE_DESIGN_REVIEW_V1.md
→ exact-head active-writer design review / Step-2 PASS

CURRENT.md
→ local product-track NOW / exact next action

CHANGELOG.md
→ append-only supporting product-design history
```

## Current proven state

```text
STEP 1 UX CONTRACT              COMPLETE / six-state evidence replan incorporated
STEP 2 EVIDENCE DESIGN          FROZEN / COMPLETE
STEP 2 MACHINE GATE             PASS
STEP 2 DESIGN REVIEW            PASS / material open finding none
STEP 3 DYNAMIC REFERRAL DESIGN  NEXT
RUNTIME PRODUCTIZATION          NOT IMPLEMENTED
```

## Boundary with existing CU-1

The existing CU-1 clinical taxonomy, machine contract and deployed runtime remain separate unless a later reviewed maintenance finding proves a concrete contradiction.

```text
existing CU-1 clinical/runtime foundation
!=
productization evidence/UX layer
```

Step 2 found no need for a broad CU-1 taxonomy rewrite. Two explicit future seams remain: walking-aid presentation scope and weight-management machine representation.

No second diagnosis is added until the complete Knee-OA experience has been prototyped, reviewed and accepted.

## Lifecycle vocabulary

```text
DESIGNED
!= IMPLEMENTED
!= TESTED
!= MERGED
!= DEPLOYED
!= PRODUCTION-SMOKE-VERIFIED
!= COMMERCIAL-PILOT-VALIDATED
```
