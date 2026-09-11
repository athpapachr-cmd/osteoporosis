# SLICE_PLAN_CURRENT.md — Knee-OA independent review slice

> **STATUS:** REVIEW-ONLY / CANDIDATE PINNED / NO IMPLEMENTATION AUTHORITY.
> **Slice:** `CU1-PRODUCT-KNEE-OA-INDEPENDENT-REVIEW-V1-20260911`.
> **Review branch:** `review/physio-referral-knee-oa-independent-v1-2026-09-11`.
> **Pinned accepted candidate:** `6539351c592c1dc3e49931057b63925dea3cb94d`.
> **Tested substantive implementation:** `243095ca9545bd2f96be8986520aeae8c3551c27`.
> **Verified main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Writer:** NONE.
> **Release / real clinical use:** NOT AUTHORIZED.

## 1. Purpose

Obtain an independent, reviewer-neutral assessment of the accepted single-diagnosis Knee-OA Physio Referral candidate before any second diagnosis, commercial pilot or production-integration planning.

The product owner has accepted the design/functional direction. That acceptance does not replace independent clinical, physiotherapy, UX/accessibility, source-to-claim or commercial review.

## 2. Exact reviewer inputs

```text
clinic_utilities/physio_referral_product/KNEE_OA_INDEPENDENT_REVIEW_PACKET_V1.md
clinic_utilities/physio_referral_product/KNEE_OA_INDEPENDENT_REVIEW_PROMPT_V1.md
```

The reviewer must inspect the pinned functional candidate and current evidence/contracts rather than relying on prior author PASS statements.

## 3. Required review axes

```text
clinical / evidence integrity
physiotherapy usefulness / professional autonomy
UX / accessibility / cognitive load
commercial differentiation / recurring value / willingness to pay
```

Material evidence claims also require separate source-to-claim integrity checking.

## 4. Required finding severity

Every finding must be classified:

```text
BLOCKER
MATERIAL BEFORE SECOND DIAGNOSIS
MATERIAL BEFORE COMMERCIAL PILOT
IMPROVEMENT
LATER / OPTIONAL
```

The review must explicitly answer what should be removed, not only what could be added.

## 5. Review invariants

```text
technical PASS != clinical validation
product-owner acceptance != independent validation
insufficient evidence != ineffective
symptom/location != diagnosis
suggestion != treatment selection
guideline disagreement must remain visible
```

No reviewer score may conceal a material safety/evidence finding.

## 6. Review output

Return one verdict only:

```text
REVIEW PASS
CONDITIONAL PASS
REVIEW HOLD
```

and include:

```text
MATERIAL OPEN FINDINGS: <number>
BLOCKERS: <number>
TOP 3 CORRECTIONS
TOP 3 THINGS NOT TO CHANGE
```

## 7. Stop rule

Review is analysis only. Do not implement fixes during review. Findings first return to the product owner for disposition and prioritization.

No second diagnosis, feature expansion, production CU-1 rewrite, public hosting, auth/billing, merge, deployment or commercial pilot before review findings are dispositioned.
