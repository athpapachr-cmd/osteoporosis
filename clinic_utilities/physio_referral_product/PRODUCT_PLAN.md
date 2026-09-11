# PRODUCT_PLAN.md — Physio Referral productization

> **STATUS:** PRODUCT-OWNER APPROVED DIRECTION — STEPS 1–3 DESIGNED / FROZEN.
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Parent:** existing CU-1 Physiotherapy Referral v2.
> **First vertical slice:** Knee Osteoarthritis only.
> **Runtime implementation authority:** NONE at this stage.

---

# 1. Product objective

Evolve the existing CU-1 referral utility from a capable deterministic referral form into a small subscription clinical product that a clinician can reasonably value at approximately:

```text
€9.99 / month
```

Initial commercial target is deliberately modest:

```text
~100 subscribers
→ ~€999 MRR
```

The first goal is product-market usefulness, not scale.

---

# 2. Product positioning

The product is **not** merely a text generator.

Target value proposition:

> Create a clinically useful physiotherapy referral quickly, while the interface quietly keeps the selected plan aligned with current reviewed evidence, shows uncertainty or disagreement honestly, and preserves clinician autonomy.

Value stack:

```text
speed
+ clinical structure
+ evidence transparency
+ evidence freshness/review state
+ flexible clinician override
+ high-quality referral output
```

---

# 3. First-slice rule

The first complete productization slice is:

```text
Knee Osteoarthritis only
```

Do not expand diagnoses until this vertical slice proves the reusable architecture:

```text
diagnosis/laterality
→ phenotype / function
→ evidence-aware defaults
→ context-driven suggestions
→ power-user options
→ live referral
→ evidence detail on demand
→ final review state
```

---

# 4. Relationship to existing CU-1

Existing CU-1 remains the structured clinical/runtime foundation and already provides route/state IDs, validation, safety/consistency semantics, formatting and ephemeral use.

Hard boundary:

```text
PRODUCTIZATION DOES NOT SILENTLY REWRITE FROZEN CU-1 CLINICAL TAXONOMY
```

Steps 2–3 confirmed that no broad taxonomy rewrite is required.

Explicit bounded seams remain:

```text
walking aid
→ canonical CU-1 ID exists
→ current Knee UI scope does not expose it

weight management
→ strong Knee-OA evidence when applicable
→ present in clinical profile prose
→ no dedicated selectable CU-1 machine ID

true locking / major mechanical ROM block
→ canonical CU-1 finding exists
→ current rule catalog does not derive a safety trigger from the finding itself
→ first Knee-OA product surface does not expose it until a bounded safety/reassessment mapping exists
```

These are explicit later design decisions, not hidden mutations.

---

# 5. Frozen product principles

## Minimal surface

Clinical/evidence complexity lives underneath. Routine use should feel direct and mobile-first.

## Smart default

The reviewed Knee-OA starting plan is:

```text
implicit individualized physiotherapy assessment / active rehabilitation
+ therapeutic exercise
+ progressive strengthening
+ education & self-management
```

Other components are contextual or power-user selected.

## Clinician authority

Suggestion != clinician selection. The product informs and proposes; it does not silently choose treatment.

Opening a fixed Knee-OA product screen does not itself assert the diagnosis. Final Copy requires the existing CU-1 formal-diagnosis semantics and explicit clinician assertion.

## Evidence-state integrity

Step 2 expanded the model to six states:

```text
recommended_or_supported
conditional_or_context_dependent
limited_or_insufficient_evidence
guideline_conflict_or_mixed
recommendation_against_routine_use
not_yet_assessed
```

Hard distinctions:

```text
INSUFFICIENT != AGAINST
CONFLICT != CONSENSUS
SOURCE YEAR != REVIEW DATE
BROAD RECOMMENDATION != ITEM-SPECIFIC STRONG RECOMMENDATION
```

## Evidence claim scope

Every source claim may distinguish:

```text
direct_item_recommendation
named_component_of_broader_recommendation
broader_recommendation_only
contextual_clinical_mapping
```

This prevents a broad recommendation for exercise from being falsely presented as a strong recommendation for every narrower technique.

## Deterministic live referral

Step 3 freezes a deterministic Greek composition layer:

```text
structured clinician-selected state
+ bounded product phenotype overlay
→ deterministic semantic projection
→ deterministic Greek referral
```

Routine output requires no LLM and no Generate button.

Hard Step-3 boundaries include:

```text
stiffness symptom != ROM restriction
generic weakness != objective weakness
functional limitation != automatic rehabilitation selection
selected product item must render, fail closed, or be explicitly semantically de-duplicated
manual edited prose != structured-state writeback
evidence UI metadata != copied referral prose
```

## Update governance

No autonomous literature-to-live-rule updates.

```text
surveillance
→ candidate change
→ source review
→ impact classification
→ clinician/product-owner approval
→ versioned contract
→ tests/review
→ separate runtime release
```

---

# 6. Privacy/product boundary

Preferred first commercial architecture remains data-minimizing:

```text
account/preferences may be persisted later
patient-identifiable referral draft need not be persisted
```

The existing ephemeral CU-1 behavior remains a desirable default unless a future explicit workflow requirement justifies protected persistence.

---

# 7. Future Clinical Cockpit architecture

Physio Referral should be commercially usable as a focused product while remaining architecturally compatible with a later modular Clinical Cockpit.

Candidate entitlements may later include:

```text
physio_referral
patient_education
clinical_calculators
osteoporosis_tools
reception
```

Do not build billing/auth/entitlements during the Knee-OA prototype unless separately authorized.

---

# 8. Commercial validation sequence

```text
complete one Knee-OA vertical slice
→ product-owner use
→ independent clinical / physiotherapy / UX / commercial review
→ bounded refinement
→ first external clinician pilot
→ first paying clinician
→ 5
→ 10
→ €250 MRR
→ €500 MRR
→ ~€1,000 MRR
```

The first paid clinician is more meaningful than speculative projections.

---

# 9. Independent review gate

After a functional Knee-OA vertical slice exists, request independent review across:

```text
clinical/evidence fidelity
physiotherapy usefulness/autonomy
UX/product simplicity
commercial willingness-to-pay / retention value
```

Review must explicitly ask:

> What should be removed?

The main product risk is not only missing features; it is burying a simple workflow under too many correct ideas.

---

# 10. Product sequence and current state

```text
STEP 1 — UX interaction contract                         COMPLETE / FROZEN
STEP 2 — Knee-OA evidence knowledge module              COMPLETE / FROZEN / DESIGN PASS
STEP 3 — dynamic referral/template contract             COMPLETE / FROZEN / DESIGN PASS
STEP 4 — evidence interaction/traceability contract     NEXT DESIGN STEP
STEP 5 — prototype                                      PENDING
STEP 6 — test/product-owner review                      PENDING
STEP 7 — independent multi-axis review                  PENDING
STEP 8 — bounded refinement                             PENDING
STEP 9 — external clinician/commercial pilot            PENDING
```

No runtime implementation, PR merge, deploy or production smoke is authorized merely by completion of Steps 1–3.