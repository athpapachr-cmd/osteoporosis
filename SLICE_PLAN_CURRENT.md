# SLICE_PLAN_CURRENT.md — Physio Referral Knee-OA Evidence Knowledge Module v1

> **STATUS:** DESIGN ACTIVE / STEP 2
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CU1-PRODUCT-KNEE-OA-EVIDENCE-V1-2026-09-11`.
> **Base `main`:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Branch:** `design/physio-referral-knee-oa-evidence-v1-2026-09-11`.
> **Vertical slice:** Knee Osteoarthritis only.
> **Parent runtime:** existing CU-1 Physiotherapy Referral v2 — read-only in this slice.
> **Frozen Step-1 UX:** `clinic_utilities/physio_referral_product/UX_CONTRACT_CURRENT.md`.
> **Runtime implementation authority:** NONE.
> **Merge/deploy authority:** NONE.

---

# 1. Problem

The existing CU-1 Knee-OA pathway has good structured clinical content but does not yet expose a product-grade, source-traceable evidence layer capable of driving:

```text
smart defaults
positive evidence-backed suggestions
limited-evidence cues
recommendation-against cues
cross-guideline disagreement
concise `i` evidence explanations
review/freshness metadata
```

A subscription product cannot honestly reduce every clinical intervention to a generic checkbox or a single untraceable recommendation state.

---

# 2. Scope

Step 2 designs only the evidence knowledge module for the existing route:

```text
profile_id = knee
route_id   = knee_osteoarthritis
```

The design maps only clinically relevant existing CU-1 findings/functions/rehab directions/adjuncts for this route. It does not expand the global physiotherapy taxonomy.

Primary candidate evidence authorities include current/relevant international guidelines and consensus recommendations, with each source retained separately rather than blended invisibly.

---

# 3. Required contract

Each material intervention/evidence item must be able to retain:

```text
item_id
display label / role
CU-1 target id(s)
default-selection behavior
app-facing evidence state
source-specific positions[]
source framework / organization
source publication/version year
recommendation direction
recommendation strength/certainty when supplied
applicability / phenotype trigger
expected purpose/outcomes summary
suggestion trigger(s)
caution trigger(s)
reviewed_on
freshness/status
```

Source-specific positions are authoritative. Any app-facing summary is a transparent projection over those positions, not a replacement for them.

---

# 4. Evidence-state integrity

The frozen Step-1 UX currently defines:

```text
recommended_or_supported
conditional_or_context_dependent
limited_or_insufficient_evidence
recommendation_against_routine_use
not_yet_assessed
```

Step 2 must explicitly test whether these five states can represent genuine cross-guideline disagreement without deception.

If not, this is an approved **REPLAN trigger** from the Step-1 contract. The design must add an explicit conflict/mixed-guidance state rather than silently classifying disagreement as either support or opposition.

Hard rule:

```text
INSUFFICIENT EVIDENCE != EVIDENCE AGAINST
FRAMEWORK DISAGREEMENT != CONSENSUS
SOURCE YEAR != PRODUCT REVIEW DATE
```

---

# 5. Candidate Knee-OA evidence targets

Initial evidence mapping should cover the current CU-1 concepts that may materially affect a physiotherapy referral, including where applicable:

```text
therapeutic exercise
progressive strengthening
physical activity / aerobic / graded walking
education and self-management
home exercise / adherence support
mobility/ROM when restricted
neuromuscular/balance training when relevant
functional task retraining / gait
structured/supervised physical therapy
weight-management support/referral when overweight/obesity context is known
walking aid assessment/training when relevant
brace/support/taping context
manual therapy as adjunct
acupuncture
dry needling
TENS/electrotherapy only if exposed by the future product surface
other existing CU-1 adjuncts only when route-relevant
```

Do not add modalities merely to make the evidence table look comprehensive.

---

# 6. Product behavior to support later

The Step-2 contract must be sufficient for a later UI to derive behavior such as:

```text
strong/core option omitted
→ evidence-backed one-tap suggestion

phenotype/context makes option relevant
→ contextual suggestion

selected option has uncertain evidence
→ compact non-intrusive uncertainty bubble

selected option is advised against by a reviewed framework
→ caution state with source trace

reviewed frameworks materially disagree
→ explicit “guidelines differ” state and `i` explanation
```

No suggestion becomes clinician-selected truth automatically.

---

# 7. Acceptance criteria

Step 2 is design-complete only if:

- each live app-facing evidence claim is traceable to reviewed source position(s);
- guideline disagreement is preserved explicitly;
- evidence strength is not invented when the source does not provide it;
- source publication/version dates are distinct from `reviewed_on`;
- no unsupported claim is presented as guideline consensus;
- existing CU-1 Knee-OA semantics are either compatible or a concrete contradiction is logged as REPLAN;
- machine-readable and human-readable contracts agree;
- no runtime/UI/database mutation is included.

---

# 8. Out of scope

```text
runtime implementation
visual prototype code
dynamic referral formatter rewrite
billing/subscription/account logic
patient persistence
autonomous evidence updating
second diagnosis
commercial release
```

---

# 9. Exact next gate

```text
fresh source review
→ source-specific evidence registry
→ intervention mapping
→ explicit conflict-state decision
→ machine contract
→ human design review
→ exact-head review/freeze
```

Only after a clean Step-2 freeze may Step 3 dynamic referral/template design begin.
