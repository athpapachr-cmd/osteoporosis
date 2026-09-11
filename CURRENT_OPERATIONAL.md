# CURRENT_OPERATIONAL.md — Knee-OA four-review gate ready

> **STATUS:** PRODUCT-OWNER FUNCTIONAL/DESIGN ACCEPTANCE RECORDED / FOUR INDEPENDENT REVIEW PROMPTS READY.
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Pinned accepted candidate branch/head:** `feat/physio-referral-knee-oa-prototype-v1-2026-09-11` @ `6539351c592c1dc3e49931057b63925dea3cb94d`.
> **Tested substantive implementation head:** `243095ca9545bd2f96be8986520aeae8c3551c27`.
> **Review branch:** `review/physio-referral-knee-oa-independent-v1-2026-09-11`.
> **ACTIVE RUNTIME / DESIGN / CANONICAL WRITER:** NONE.
> **PR / merge / deploy / production smoke authority:** NONE.
> **Real-patient use / production integration:** NOT AUTHORIZED.

## 1. Product-owner acceptance

The product owner reviewed the refined Knee-OA prototype direction, liked the design and authorized progression to review. This is product-owner design/functional-direction acceptance only; it is not independent clinical, physiotherapy, UX/accessibility, commercial or source-to-claim validation.

## 2. Review architecture corrected to four independent reviews

The former single multi-axis prompt is superseded. The accepted review architecture is:

```text
A — Clinical / Evidence
B — Physiotherapy
C — UX / Product
D — Commercial / Product-Market
```

Each review must be performed separately and must not inherit the other reviewers' findings or verdicts before completion.

Active reviewer prompts:

```text
clinic_utilities/physio_referral_product/KNEE_OA_CLINICAL_EVIDENCE_REVIEW_PROMPT_V1.md
clinic_utilities/physio_referral_product/KNEE_OA_PHYSIOTHERAPY_REVIEW_PROMPT_V1.md
clinic_utilities/physio_referral_product/KNEE_OA_UX_PRODUCT_REVIEW_PROMPT_V1.md
clinic_utilities/physio_referral_product/KNEE_OA_COMMERCIAL_REVIEW_PROMPT_V1.md
```

Shared candidate/background packet:

```text
clinic_utilities/physio_referral_product/KNEE_OA_INDEPENDENT_REVIEW_PACKET_V1.md
```

The former combined prompt remains only as a superseded redirect:

```text
clinic_utilities/physio_referral_product/KNEE_OA_INDEPENDENT_REVIEW_PROMPT_V1.md
```

## 3. Cross-review rule

Do not synthesize findings until all four independent reviews are complete.

After all four are returned:

```text
four independent outputs
→ Product Owner cross-review synthesis
→ identify agreements / contradictions / unique findings
→ disposition every material finding
→ authorize bounded corrections only after disposition
```

A separate source-to-claim integrity audit remains part of the Clinical/Evidence review responsibility and may produce a dedicated appendix; it is not a fifth product review.

## 4. Finding severity

All reviewers use the same severity vocabulary:

```text
BLOCKER
MATERIAL BEFORE SECOND DIAGNOSIS
MATERIAL BEFORE COMMERCIAL PILOT
IMPROVEMENT
LATER / OPTIONAL
```

Every reviewer must explicitly state what should be removed rather than added.

## 5. Exact next action

Run the **four reviews in four separate fresh conversations** against the same pinned candidate. No reviewer implements fixes.

Only after all four reviews return should this project perform cross-review synthesis and Product Owner disposition.

## 6. HOLD

No new diagnosis, feature expansion, production CU-1 rewrite, public hosting, auth/billing, merge or deployment before the four independent reviews are completed and their material findings are dispositioned.
