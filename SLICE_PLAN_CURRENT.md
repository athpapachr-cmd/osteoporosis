# SLICE_PLAN_CURRENT.md — Physio Referral Knee-OA Evidence Knowledge Module v1

> **STATUS:** DESIGN FROZEN / STEP 2 COMPLETE / ACTIVE-WRITER REVIEW PASS
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CU1-PRODUCT-KNEE-OA-EVIDENCE-V1-2026-09-11`.
> **Base `main`:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Branch:** `design/physio-referral-knee-oa-evidence-v1-2026-09-11`.
> **Vertical slice:** Knee Osteoarthritis only.
> **Parent runtime:** existing CU-1 Physiotherapy Referral v2 — unchanged.
> **Runtime implementation authority:** NONE.
> **Merge/deploy authority:** NONE.

---

# 1. Frozen deliverables

```text
clinic_utilities/physio_referral_product/UX_CONTRACT_CURRENT.md
clinic_utilities/physio_referral_product/KNEE_OA_EVIDENCE_DESIGN_V1.md
clinic_utilities/physio_referral_product/contracts/knee_oa_evidence_contract_v1.yaml
clinic_utilities/physio_referral_product/validate_knee_oa_evidence_contract_v1.py
clinic_utilities/physio_referral_product/KNEE_OA_EVIDENCE_DESIGN_REVIEW_V1.md
.github/workflows/physio-knee-oa-evidence-design.yml
```

Reviewed substantive head:

```text
6b82691c8431b699d20752c83b443793989f6402
```

Post-review artifact head:

```text
f20f01f34abf1796f075640a397d085443834c59
```

Machine evidence:

```text
run 34559461372 — SUCCESS on 6b82691...
run 34559680326 — SUCCESS on f20f01f...
```

Design review:

```text
DESIGN PASS
MATERIAL OPEN FINDING NONE
```

The review is an active-writer exact design review, not the later independent multi-axis product review.

---

# 2. Frozen evidence model

App-facing states:

```text
recommended_or_supported
conditional_or_context_dependent
limited_or_insufficient_evidence
guideline_conflict_or_mixed
recommendation_against_routine_use
not_yet_assessed
```

The sixth state `guideline_conflict_or_mixed` was required after real guideline review showed that acupuncture/manual-therapy positions cannot be represented honestly by a single support/against axis.

Greek surface semantic:

```text
Οι οδηγίες διαφέρουν
```

Source claim scope:

```text
direct_item_recommendation
named_component_of_broader_recommendation
broader_recommendation_only
contextual_clinical_mapping
```

This prevents broad evidence strength from being silently transferred to narrower product items.

---

# 3. Frozen Knee-OA default

Implicit:

```text
individualized physiotherapy assessment / active rehabilitation
```

Visible selected core:

```text
therapeutic exercise
progressive strengthening
education & self-management
```

Context-driven rather than universal default:

```text
graded activity
endurance/capacity work
mobility if restricted
neuromuscular/balance work
gait practice
functional task retraining
home programme
```

---

# 4. Adjunct and integration decisions

```text
manual therapy       → guideline_conflict_or_mixed
soft-tissue work     → guideline_conflict_or_mixed
acupuncture          → guideline_conflict_or_mixed
dry needling         → excluded from Knee-OA surface; recommendation-against-routine state with source detail preserved
walking aid          → canonical ID exists; current Knee UI scope does not expose it
weight management    → strongly supported when applicable; no dedicated current CU-1 selectable ID; advisory-only for now
```

No broad CU-1 taxonomy rewrite is required by Step 2.

---

# 5. Hard invariants

```text
INSUFFICIENT EVIDENCE != EVIDENCE AGAINST
GUIDELINE CONFLICT != CONSENSUS
SOURCE YEAR != PRODUCT REVIEW DATE
BROAD RECOMMENDATION != ITEM-SPECIFIC STRONG RECOMMENDATION
MISSING CONTEXT != NEGATIVE CONTEXT
SUGGESTION != CLINICIAN SELECTION
NO exact exercise prescription invented
NO adjunct replaces active rehabilitation
NO autonomous literature-to-live-rule mutation
```

---

# 6. Scope proof

Exact diff review against base `main` showed only:

- root operational/slice canonicals;
- supporting Physio product-design files;
- evidence YAML;
- contract validator;
- design workflow.

No CU-1 production runtime/API/formatter/static UI/database file changed.

---

# 7. Lifecycle

```text
STEP 1 UX CONTRACT                  COMPLETE / EVIDENCE REPLAN INCORPORATED
STEP 2 EVIDENCE DESIGN              FROZEN / COMPLETE
STEP 2 MACHINE CONTRACT             PASS
STEP 2 MACHINE GATES                SUCCESS
STEP 2 ACTIVE-WRITER REVIEW         PASS
MATERIAL OPEN FINDING               NONE
RUNTIME IMPLEMENTED                 NO
PR/MERGE                            NO
DEPLOYED                            NO
```

---

# 8. Exact next boundary

```text
STEP 3 — dynamic Knee-OA referral/template contract
```

Step 3 must define deterministic live text composition from diagnosis/laterality + phenotype/findings + function + evidence-aware plan + power-user choices while preserving physiotherapist autonomy.

No production runtime implementation is authorized by Step-2 closure.
