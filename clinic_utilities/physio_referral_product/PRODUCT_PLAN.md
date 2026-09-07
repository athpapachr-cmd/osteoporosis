# PRODUCT_PLAN.md — Physio Referral productization

> **STATUS:** PRODUCT-OWNER APPROVED DIRECTION — DESIGN ONLY.
> **Updated:** 2026-09-07 Asia/Nicosia.
> **Parent:** existing CU-1 Physiotherapy Referral v2.
> **First vertical slice:** Knee Osteoarthritis only.
> **Runtime implementation authority:** NONE in this design step.

---

# 1. Product objective

Evolve the existing CU-1 referral utility from a capable deterministic referral form into a small subscription clinical product that a clinician can reasonably value at approximately:

```text
€9.99 / month
or later equivalent annual pricing
```

Initial commercial ceiling/target is deliberately modest:

```text
~100 subscribers
→ ~€999 MRR
```

The first goal is not scale. It is to prove that clinicians repeatedly use and pay for an evidence-aware referral assistant.

---

# 2. Product positioning

The product is **not** merely a text generator.

Target value proposition:

> Create a clinically useful physiotherapy referral quickly, while the interface quietly keeps the selected plan aligned with current reviewed evidence, shows when evidence is weak or contrary, and preserves clinician autonomy.

Core value combines:

```text
speed
+ clinical structure
+ evidence transparency
+ current review state
+ flexible clinician override
+ high-quality referral output
```

---

# 3. First-slice rule — one diagnosis only

The first complete productization slice is:

```text
Knee Osteoarthritis
```

Do **not** add five or twenty diagnoses in parallel.

The Knee-OA slice must prove the complete reusable architecture:

```text
diagnosis
→ phenotype / impairments / function
→ evidence-aware defaults
→ clinician customization
→ evidence-positive suggestions
→ limited/against-evidence signals
→ power-user expansion
→ live dynamic referral
→ evidence detail on demand
→ final quality/safety check
```

Only after this vertical slice is accepted should the same architecture be generalized to additional conditions.

---

# 4. Relationship to existing CU-1

Existing CU-1 remains the clinical/runtime foundation and already provides:

- structured referral state;
- diagnosis-vs-finding separation;
- route validation;
- safety/consistency evaluation;
- short/detailed formatting;
- protected ephemeral operation;
- copy/print workflow;
- broad frozen physiotherapy taxonomy.

The productization track must reuse that work where valid rather than rebuild from the old standalone HTML.

Hard boundary:

```text
PRODUCTIZATION DOES NOT SILENTLY REWRITE FROZEN CU-1 CLINICAL TAXONOMY
```

If the Knee-OA evidence review demonstrates a true contradiction with the frozen CU-1 profile, that becomes an explicit maintenance/replan finding before runtime change.

---

# 5. Product principles

## 5.1 Clinical depth without visual complexity

Complex evidence and rule logic live underneath. The surface remains calm and simple.

## 5.2 Smart default, not blank form

Choosing Knee OA immediately creates a sensible reviewed starting plan. The user refines it rather than building from zero.

## 5.3 Clinician remains authoritative

The product may recommend, caution and explain. It does not lock a clinician into a guideline choice or silently make a treatment decision.

## 5.4 Evidence state must be explicit

At minimum the internal model distinguishes:

```text
recommended_or_supported
conditional_or_context_dependent
limited_or_insufficient_evidence
recommendation_against_routine_use
not_yet_assessed
```

`insufficient evidence` must never be presented as equivalent to `evidence against`.

## 5.5 Evidence currency is recurring subscription value

Each evidence-backed item should ultimately support:

```text
source / guideline
version or publication year
recommendation/rationale
strength/certainty when source provides it
reviewed_on
status/freshness
```

Guideline year and product review date are distinct. Example:

```text
NICE NG226 · 2022
Evidence reviewed · Sep 2026
```

The product must never imply “2026 guideline” merely because the product was reviewed in 2026.

## 5.6 Update governance

No autonomous literature-to-live-rule pipeline in the first product.

```text
literature surveillance
→ candidate change
→ clinical review
→ approval
→ versioned release
```

---

# 6. Privacy/product boundary

Preferred initial commercial architecture remains data-minimizing:

```text
account/preferences may be persisted later
patient-identifiable referral draft need not be persisted
```

The current CU-1 ephemeral principle is a desirable default for the first product unless a later explicit workflow requirement justifies protected patient-data storage.

---

# 7. Future Clinical Cockpit module architecture

Physio Referral should be deployable initially as a focused product but architected as one module of a later unified Clinical Cockpit.

Candidate entitlement model:

```text
physio_referral = true/false
patient_education = true/false
clinical_calculators = true/false
osteoporosis_tools = true/false
reception = true/false
```

Do not build the full entitlement/billing system during the Knee-OA prototype.

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

The first paid clinician is a stronger validation milestone than speculative large-user projections.

---

# 9. Independent review gate

After the complete Knee-OA vertical slice exists, but before diagnosis expansion or commercial release, request independent review across four axes:

```text
clinical/evidence fidelity
physiotherapy usefulness/autonomy
UX/product simplicity
commercial willingness-to-pay / retention value
```

Review must explicitly ask:

> What should be removed?

The primary product risk is not only missing features; it is destroying simplicity by adding too many correct features.

---

# 10. Current sequence

```text
STEP 1 — freeze UX interaction contract
STEP 2 — define Knee-OA evidence knowledge module
STEP 3 — define dynamic referral/template behavior
STEP 4 — define evidence interaction/traceability layer
STEP 5 — prototype
STEP 6 — test/review
STEP 7 — independent multi-axis review
STEP 8 — bounded refinement
STEP 9 — external clinician/commercial pilot
```

Current authorized work stops at **STEP 1 design freeze** unless the product owner explicitly continues.
