# Physio Referral Productization — supporting control plane

> **STATUS:** ACTIVE supporting product-design control plane.
> **Repository:** `athpapachr-cmd/osteoporosis`.
> **Parent area:** `Clinic Utilities / CU-1 Physiotherapy Referral`.
> **Important:** these files do **not** create a seventh root canonical authority. The repository-wide six canonicals in `AGENTS.md` remain authoritative for repo-wide state and writer locks.

This directory preserves the commercial/product-design evolution of the existing CU-1 Physiotherapy Referral runtime so durable decisions are not left only in chat history.

## Local supporting documents

```text
PRODUCT_PLAN.md
→ long-range product/commercial direction for Physio Referral

UX_CONTRACT_CURRENT.md
→ current approved UX/product interaction contract

CURRENT.md
→ local product-track NOW / exact next action

CHANGELOG.md
→ append-only history of accepted product-design decisions
```

## Boundary with existing CU-1

The existing CU-1 clinical taxonomy, machine contract and production runtime remain separate and frozen unless a later evidence/maintenance review proves a concrete contradiction.

```text
existing CU-1 clinical/runtime foundation
!=
new productization / evidence UX layer
```

The first productization vertical slice is deliberately **Knee Osteoarthritis only**. No second diagnosis is added until the complete Knee-OA experience has been prototyped, reviewed and accepted.

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
