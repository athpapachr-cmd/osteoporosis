# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PHYSIO REFERRAL PRODUCTIZATION — STEP 2 KNEE-OA EVIDENCE DESIGN FROZEN / CLOSED
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Base `main`:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Design branch:** `design/physio-referral-knee-oa-evidence-v1-2026-09-11`.
> **Closed slice:** `CU1-PRODUCT-KNEE-OA-EVIDENCE-V1-2026-09-11`.
> **Reviewed substantive head:** `6b82691c8431b699d20752c83b443793989f6402`.
> **Review artifact head:** `f20f01f34abf1796f075640a397d085443834c59`.
> **ACTIVE DESIGN/CANONICAL WRITER:** NONE — Step 2 closed.
> **ACTIVE RUNTIME WRITER:** NONE.
> **Production config/secret authority:** NONE.
> **Patient-data mutation authority:** NONE.
> **Merge/deploy authority from this closeout:** NONE.

---

# 1. Prior program state preserved

Clinical Learning Hub L-1D remains `MERGED / DEPLOYED / PRODUCTION-SMOKE-VERIFIED / CLOSED` and was not modified by this Physio Referral design slice.

Existing CU-1 Physiotherapy Referral v2 remains the deployed production runtime. No CU-1 runtime/API/formatter/static production UI/database file was changed in Step 2.

---

# 2. Step-2 closure evidence

Human evidence design:

```text
clinic_utilities/physio_referral_product/KNEE_OA_EVIDENCE_DESIGN_V1.md
```

Machine evidence contract:

```text
clinic_utilities/physio_referral_product/contracts/knee_oa_evidence_contract_v1.yaml
```

UX contract after evidence-state REPLAN:

```text
clinic_utilities/physio_referral_product/UX_CONTRACT_CURRENT.md
```

Exact design review:

```text
clinic_utilities/physio_referral_product/KNEE_OA_EVIDENCE_DESIGN_REVIEW_V1.md
DESIGN PASS / MATERIAL OPEN FINDING NONE
```

Machine gates:

```text
run 34559461372
head 6b82691c8431b699d20752c83b443793989f6402
SUCCESS

run 34559680326
head f20f01f34abf1796f075640a397d085443834c59
SUCCESS
```

Exact ancestry review showed merge base exactly fresh `main` `d9f312f6...`, `behind_by = 0` at the substantive review point, and no runtime leakage.

---

# 3. Frozen Step-2 decisions

## Evidence-state model

```text
recommended_or_supported
conditional_or_context_dependent
limited_or_insufficient_evidence
guideline_conflict_or_mixed
recommendation_against_routine_use
not_yet_assessed
```

`guideline_conflict_or_mixed` was added after real source review showed that five states could not honestly represent acupuncture/manual-therapy disagreement.

Greek surface semantic:

```text
Οι οδηγίες διαφέρουν
```

## Evidence claim scope

Source positions retain:

```text
direct_item_recommendation
named_component_of_broader_recommendation
broader_recommendation_only
contextual_clinical_mapping
```

A broad strong recommendation cannot silently become an item-specific strong recommendation.

## Reviewed Knee-OA smart default

```text
implicit individualized PT / active rehabilitation
+ therapeutic exercise
+ progressive strengthening
+ education & self-management
```

`graded_activity_exposure` is context-dependent rather than a universal default.

## Adjuncts

```text
manual therapy       → guidelines differ
soft-tissue work     → guidelines differ
acupuncture          → guidelines differ
dry needling         → excluded from Knee-OA surface; no positive Step-2 guideline position
```

## Integration seams

```text
walking aid
→ existing canonical CU-1 ID
→ not currently exposed in Knee UI relevance

weight management
→ strongly supported when overweight/obesity applies
→ no dedicated current selectable CU-1 ID
→ advisory-only in Step 2; no inference
```

---

# 4. Hard evidence/product invariants

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

# 5. Lifecycle

```text
STEP 1 UX CONTRACT                         COMPLETE / EVIDENCE REPLAN INCORPORATED
STEP 2 HUMAN EVIDENCE DESIGN              PASS
STEP 2 MACHINE CONTRACT                   PASS
STEP 2 MACHINE GATES                      SUCCESS
STEP 2 EXACT DESIGN REVIEW                PASS
MATERIAL OPEN FINDING                     NONE
STEP 2 DESIGN FROZEN                      YES
RUNTIME IMPLEMENTED                       NO
PR/MERGE                                  NO
DEPLOYED                                  NO
```

---

# 6. Exact next action

```text
STEP 3 — dynamic Knee-OA referral/template contract
```

Define deterministically how:

```text
diagnosis + laterality
+ phenotype/findings
+ functional limitations
+ selected evidence-aware plan
+ optional power-user selections
→ live concise referral text
```

Step 3 remains a design task unless separately authorized for runtime implementation.

Current hold:

```text
NO production UI rewrite
NO CU-1 runtime mutation
NO second diagnosis
NO billing/auth/entitlements
NO patient persistence
NO autonomous evidence updates
NO merge/deploy/production smoke from Step-2 authority
```
