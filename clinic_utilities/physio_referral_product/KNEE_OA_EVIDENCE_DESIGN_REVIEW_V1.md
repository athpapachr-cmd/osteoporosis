# KNEE_OA_EVIDENCE_DESIGN_REVIEW_V1.md

> **REVIEW TYPE:** exact-head active-writer design review — NOT independent external review.
> **Slice:** `CU1-PRODUCT-KNEE-OA-EVIDENCE-V1-2026-09-11`.
> **Reviewed substantive head:** `6b82691c8431b699d20752c83b443793989f6402`.
> **Merge base / fresh main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Machine gate:** `Physio Knee OA evidence design gate`, run `34559461372` — **SUCCESS**.
> **Disposition:** **DESIGN PASS / MATERIAL OPEN FINDING NONE**.
> **Runtime implementation authority:** NONE.
> **Merge/deploy authority:** NONE.

---

# 1. Review question

Does the Step-2 Knee-OA evidence design provide an honest, implementable and source-traceable evidence layer for the agreed minimal referral UX without:

- manufacturing guideline consensus;
- overstating evidence strength;
- silently changing the frozen broad CU-1 clinical taxonomy;
- adding runtime/UI/database behavior;
- turning suggestions into autonomous treatment selection?

Disposition: **YES** at the reviewed substantive head.

---

# 2. Exact ancestry / scope

Comparison:

```text
d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37
...
6b82691c8431b699d20752c83b443793989f6402
```

GitHub reports:

```text
status      ahead
behind_by   0
merge_base  exact fresh main d9f312f6...
```

The diff is limited to:

```text
CURRENT_OPERATIONAL.md
SLICE_PLAN_CURRENT.md
.github/workflows/physio-knee-oa-evidence-design.yml
clinic_utilities/physio_referral_product/README.md
clinic_utilities/physio_referral_product/PRODUCT_PLAN.md
clinic_utilities/physio_referral_product/UX_CONTRACT_CURRENT.md
clinic_utilities/physio_referral_product/CURRENT.md
clinic_utilities/physio_referral_product/CHANGELOG.md
clinic_utilities/physio_referral_product/KNEE_OA_EVIDENCE_DESIGN_V1.md
clinic_utilities/physio_referral_product/contracts/knee_oa_evidence_contract_v1.yaml
clinic_utilities/physio_referral_product/validate_knee_oa_evidence_contract_v1.py
```

No production CU-1 runtime, API, formatter, static production UI, database schema, patient persistence, billing/authentication, RF or Learning Hub runtime file is changed.

---

# 3. Machine evidence

Workflow:

```text
Physio Knee OA evidence design gate
run 34559461372
head 6b82691c8431b699d20752c83b443793989f6402
result SUCCESS
```

The validator proves at minimum:

- exact Knee-OA route exists in the frozen CU-1 registry;
- exactly six app evidence states exist;
- source/direction/support-scope enums are unique and resolved;
- every declared CU-1 target ID exists in the current option catalog;
- every source position resolves to a reviewed source;
- source-native strength cannot be laundered into a narrower item when the source claim is only broader/contextual;
- the visible default plan is exactly therapeutic exercise + progressive strengthening + education/self-management;
- omission suggestions match that reviewed default set;
- adjunct conflict states are explicit;
- dry needling remains non-selectable and is not mislabeled as positive evidence;
- weight management cannot silently become a selectable/runtime item without a future contract decision;
- the walking-aid UI seam is represented honestly;
- key safety/evidence invariants are present.

---

# 4. Material findings found and corrected before PASS

## F1 — five UX evidence states could not represent real guideline conflict

Original issue:

```text
recommended
conditional
limited
against
not assessed
```

could not honestly represent acupuncture, where reviewed major guidelines differ materially in direction.

Correction:

```text
guideline_conflict_or_mixed
Greek surface: Οι οδηγίες διαφέρουν
```

Result: disagreement is visible rather than silently averaged.

Status: **CLOSED**.

## F2 — broad exercise recommendations could leak false item-specific strength

Original risk:

A source that strongly recommends `exercise` could be misrepresented as having issued a separate strong recommendation for every narrower product option, such as an exact strengthening or mobility programme.

Correction:

Every source position now carries:

```text
direct_item_recommendation
named_component_of_broader_recommendation
broader_recommendation_only
contextual_clinical_mapping
```

The validator rejects `strong_for` when the support scope is not a direct item recommendation.

Status: **CLOSED**.

## F3 — `graded_activity_exposure` was initially over-promoted as a universal core default

Review showed that physical activity and progression are broadly supported, but the CU-1 `graded_activity_exposure` concept is narrower than the general guideline claim. EULAR also did not identify specific evidence for pacing/activity-maintenance as a stand-alone component.

Correction:

```text
graded_activity_exposure
→ conditional_or_context_dependent
→ not default selected
→ suggested only from explicit walking/exercise/patient-priority context
```

Status: **CLOSED**.

## F4 — gait and task-specific retraining risked being presented as stand-alone guideline mandates

Correction:

`gait_walking_practice` and `functional_task_retraining` remain clinically useful contextual mappings, but source positions explicitly distinguish direct recommendation from broader individualized-PT support.

Status: **CLOSED**.

## F5 — dry needling was initially placed in the mixed/conflict bucket

Reviewed Step-2 sources show:

```text
NICE       against
AAOS       unclear / more evidence required
VA/DoD     insufficient for or against
```

There is no positive Step-2 guideline recommendation.

Correction:

```text
app_state = recommendation_against_routine_use
product_selectable = false
```

The detailed evidence still preserves that AAOS/VA-DoD are insufficient rather than falsely calling all sources negative.

Status: **CLOSED**.

## F6 — two existing CU-1 integration seams needed explicit treatment

### walking aid

```text
canonical CU-1 ID exists
current Knee UI relevance does not expose it
```

Decision: document only in Step 2; future bounded presentation-scope amendment may expose the existing ID.

### weight management

```text
strongly supported when overweight/obesity applies
present in frozen Knee clinical profile prose
no dedicated selectable CU-1 machine ID
```

Decision: product advisory only for now; no inference, no hidden free-text automation, no new ID in Step 2.

Status: **CLOSED / DEFERRED BY DESIGN**.

---

# 5. Evidence-state integrity

Frozen Step-2 semantic model:

```text
recommended_or_supported
conditional_or_context_dependent
limited_or_insufficient_evidence
guideline_conflict_or_mixed
recommendation_against_routine_use
not_yet_assessed
```

Hard distinctions preserved:

```text
INSUFFICIENT EVIDENCE != EVIDENCE AGAINST
GUIDELINE CONFLICT != CONSENSUS
SOURCE YEAR != PRODUCT REVIEW DATE
BROAD RECOMMENDATION != ITEM-SPECIFIC STRONG RECOMMENDATION
MISSING CONTEXT != NEGATIVE CONTEXT
SUGGESTION != CLINICIAN SELECTION
```

No arithmetic evidence score is used.

---

# 6. Reviewed Knee-OA default

Visible smart default:

```text
therapeutic_exercise
progressive_strengthening
education_and_self_management
```

Implicit referral purpose:

```text
physiotherapy_assessment_and_individualized_active_rehabilitation
```

Everything else remains context-driven or power-user selected unless a later reviewed rule changes that contract.

This keeps the routine UI small while preserving clinical depth underneath.

---

# 7. Safety / autonomy / scope review

PASS:

- clinician can override non-safety recommendations;
- no suggestion writes treatment truth automatically;
- physiotherapist autonomy is preserved;
- no exact exercise sets/reps/protocol are invented;
- active rehabilitation remains the core rather than adjunct substitution;
- absent findings are not converted to normal/negative findings;
- no patient identifiers or referral persistence are introduced;
- no autonomous evidence-to-live-rule updating is authorized.

---

# 8. Review disposition

```text
STEP-2 HUMAN DESIGN                  PASS
STEP-2 MACHINE CONTRACT              PASS
MACHINE GATE                         PASS — run 34559461372
MATERIAL OPEN FINDING                NONE
BROAD CU-1 TAXONOMY REWRITE          NOT REQUIRED
RUNTIME IMPLEMENTED                  NO
PRODUCTION UI CHANGED                NO
PATIENT PERSISTENCE CHANGED          NO
MERGED                               NO
DEPLOYED                             NO
```

Step 2 is **eligible to freeze as a design/evidence contract**.

This review is deliberately not presented as the future independent multi-axis product review. That independent review belongs after a functional Knee-OA vertical slice exists and can be judged for clinical usefulness, physiotherapy autonomy, UX friction and willingness to pay.

---

# 9. Exact next design boundary

After canonical Step-2 freeze/closeout, the next product-design step is:

```text
STEP 3 — dynamic Knee-OA referral/template contract
```

That step should define how:

```text
diagnosis + laterality
+ phenotype/findings
+ functional limitations
+ selected evidence-aware plan
+ optional power-user choices
→ live concise referral text
```

without yet authorizing production runtime implementation.
