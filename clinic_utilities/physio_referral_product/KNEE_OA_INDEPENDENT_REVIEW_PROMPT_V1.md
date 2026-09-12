# Independent Reviewer Prompt — SUPERSEDED

> **STATUS:** SUPERSEDED on 2026-09-11.
>
> The former single multi-axis reviewer prompt is no longer the active review instruction. Product-owner clarification requires **four separate independent reviews**, performed independently before cross-review synthesis.

Use these four prompts in four separate review conversations:

```text
clinic_utilities/physio_referral_product/KNEE_OA_CLINICAL_EVIDENCE_REVIEW_PROMPT_V1.md
clinic_utilities/physio_referral_product/KNEE_OA_PHYSIOTHERAPY_REVIEW_PROMPT_V1.md
clinic_utilities/physio_referral_product/KNEE_OA_UX_PRODUCT_REVIEW_PROMPT_V1.md
clinic_utilities/physio_referral_product/KNEE_OA_COMMERCIAL_REVIEW_PROMPT_V1.md
```

All four review the same pinned accepted candidate:

```text
candidate head: 6539351c592c1dc3e49931057b63925dea3cb94d
tested substantive implementation: 243095ca9545bd2f96be8986520aeae8c3551c27
shared candidate packet: clinic_utilities/physio_referral_product/KNEE_OA_INDEPENDENT_REVIEW_PACKET_V1.md
```

Rules:

```text
review A does not inherit review B/C/D
no reviewer inherits prior PASS conclusions
no reviewer implements fixes
all four outputs return to Product Owner
only then perform cross-review synthesis and finding disposition
```

The shared packet remains common candidate/background material. Axis-specific scope and verdict requirements are owned by the four prompt files above.
