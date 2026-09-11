# SLICE_PLAN_CURRENT.md — Knee-OA four independent reviews

> **STATUS:** REVIEW-ONLY / CANDIDATE PINNED / FOUR REVIEW PROMPTS READY / NO IMPLEMENTATION AUTHORITY.
> **Slice:** `CU1-PRODUCT-KNEE-OA-FOUR-INDEPENDENT-REVIEWS-V1-20260911`.
> **Review branch:** `review/physio-referral-knee-oa-independent-v1-2026-09-11`.
> **Pinned accepted candidate:** `6539351c592c1dc3e49931057b63925dea3cb94d`.
> **Tested substantive implementation:** `243095ca9545bd2f96be8986520aeae8c3551c27`.
> **Verified main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Writer:** NONE.
> **Release / real clinical use:** NOT AUTHORIZED.

## 1. Purpose

Obtain four separate independent assessments of the accepted single-diagnosis Knee-OA Physio Referral candidate before any second diagnosis, commercial pilot or production-integration planning.

Product-owner design/functional acceptance does not replace any of these independent reviews.

## 2. Shared candidate packet

All four reviewers inspect the same pinned candidate and may use this common candidate/background packet:

```text
clinic_utilities/physio_referral_product/KNEE_OA_INDEPENDENT_REVIEW_PACKET_V1.md
```

The former combined multi-axis prompt is superseded and is not an active review instruction.

## 3. Four independent review prompts

### A — Clinical / Evidence

```text
clinic_utilities/physio_referral_product/KNEE_OA_CLINICAL_EVIDENCE_REVIEW_PROMPT_V1.md
```

Owns clinical correctness, evidence fidelity, safety/non-misleading behavior, guideline disagreement and source-to-claim integrity.

### B — Physiotherapy

```text
clinic_utilities/physio_referral_product/KNEE_OA_PHYSIOTHERAPY_REVIEW_PROMPT_V1.md
```

Owns usefulness to the receiving physiotherapist, actionability, professional autonomy and referral signal-to-noise.

### C — UX / Product

```text
clinic_utilities/physio_referral_product/KNEE_OA_UX_PRODUCT_REVIEW_PROMPT_V1.md
```

Owns routine speed, cognitive load, progressive disclosure, discoverability, state clarity, evidence interaction and accessibility direction.

### D — Commercial / Product-Market

```text
clinic_utilities/physio_referral_product/KNEE_OA_COMMERCIAL_REVIEW_PROMPT_V1.md
```

Owns differentiation, recurring value, conversion/retention logic, pricing fit and willingness-to-pay experiment design.

## 4. Independence rule

```text
review A must not inherit B/C/D
review B must not inherit A/C/D
review C must not inherit A/B/D
review D must not inherit A/B/C
```

Do not show one completed review to another reviewer before that reviewer finishes.

All four reviewers also must not inherit prior author/Product Owner PASS statements as review conclusions.

## 5. Common finding severity

Every finding is classified:

```text
BLOCKER
MATERIAL BEFORE SECOND DIAGNOSIS
MATERIAL BEFORE COMMERCIAL PILOT
IMPROVEMENT
LATER / OPTIONAL
```

Every review must explicitly answer what should be **removed** rather than only what should be added.

## 6. Synthesis boundary

There is no fifth product review. After all four reviews are complete:

```text
A + B + C + D
→ Product Owner cross-review synthesis
→ agreements / contradictions / unique findings
→ material-finding disposition
→ bounded implementation authority if granted
```

The source-to-claim audit is an evidence-integrity responsibility within/alongside the Clinical/Evidence review, not a separate fifth product-review axis.

## 7. Stop rule

Review is analysis only. Do not implement fixes during any of the four reviews.

No second diagnosis, feature expansion, production CU-1 rewrite, public hosting, auth/billing, merge, deployment or commercial pilot before all four reviews are completed and material findings are dispositioned by the Product Owner.
