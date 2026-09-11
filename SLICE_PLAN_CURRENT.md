# SLICE_PLAN_CURRENT.md — Physio Referral Knee-OA Dynamic Referral / Template v1

> **STATUS:** DESIGN FROZEN / COMPLETE / CLOSED
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CU1-PRODUCT-KNEE-OA-TEMPLATE-V1-2026-09-11`.
> **Fresh `main`:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Frozen Step-2 parent:** `ab4b349223cd4c461837ab3125967a06d169a7e1`.
> **Branch:** `design/physio-referral-knee-oa-template-v1-2026-09-11`.
> **Reviewed substantive head:** `cd4a42b4582921df7eb64d6ff3fb7c718a141c3a`.
> **Review-artifact head:** `11f04d9ea316ba7fc30a03f4aaa13b60ea5c1e65`.
> **Vertical slice:** Knee Osteoarthritis only.
> **Runtime implementation authority:** NONE.
> **Merge/deploy authority:** NONE.
> **Writer lock:** NONE.

---

# 1. Problem solved by Step 3

The existing CU-1 formatter is deterministic but generic. Step 3 freezes a product-layer composition contract able to produce one coherent, concise Knee-OA referral that updates live from clinician-selected structured state without turning the handoff into a serialized checklist.

The design reuses CU-1 identities/validation/safety semantics and the frozen Step-2 evidence layer rather than rewriting the existing clinical taxonomy.

---

# 2. Frozen input/authority model

```text
ReferralDraftV1 clinician-selected state
+ frozen Step-2 evidence identities/states
+ bounded KneeOAProductPhenotypeV1
→ deterministic referral projection
```

The tiny product phenotype overlay contains only:

```text
stiffness_symptom
weakness_symptom_or_context
```

and remains ephemeral.

Hard distinctions:

```text
stiffness symptom != ROM restriction
generic weakness != objective weakness
output projection != structured-state mutation
```

---

# 3. Diagnosis and Copy authority

The fixed Knee-OA product flow does not autonomously diagnose Knee OA.

Copy-ready requires:

```text
formal_assertion_state = yes
laterality = right | left | bilateral
all selected product items within the supported Knee-OA scope
no inherited CU-1 formatter-blocking validation error
no unresolved inherited blocking/urgent safety state
```

Natural Greek laterality is frozen as:

```text
right      → δεξιού γόνατος
left       → αριστερού γόνατος
bilateral  → και των δύο γονάτων
```

Opening phrase:

```text
Παραπομπή για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση λόγω οστεοαρθρίτιδας {laterality_phrase}.
```

Preview may exist before Copy readiness; Copy remains separately gated.

---

# 4. Frozen referral structure

The standard referral is composed in this semantic order:

```text
indication / laterality
→ clinical picture when selected
→ functional impact when selected
→ active rehabilitation plan
→ non-redundant selected goals when present
→ adjunct sentence when explicitly selected
→ explicit restrictions when present
→ clinician note when present
```

Evidence-state labels, guideline names, source/year and citations remain clinician-facing UI metadata and do not enter copied referral prose by default.

---

# 5. Smart default / active-plan ownership

Frozen Step-2 visible default remains:

```text
therapeutic exercise
progressive strengthening
education & self-management
```

with individualized active physiotherapy implicit in the referral itself.

Default Greek plan phrase:

```text
θεραπευτική άσκηση, προοδευτική ενδυνάμωση και εκπαίδευση για αυτοδιαχείριση
```

If every visible plan item is removed, the implicit active-rehabilitation core renders as a fallback without silently reselecting removed items.

---

# 6. Selection, suggestion and contextual refinement

Hard rule:

```text
suggestion != selection != output
```

A functional limitation or finding may refine an already-selected intervention but may not create it.

Examples:

```text
quadriceps weakness + selected strengthening
→ emphasis on quadriceps strengthening

ROM restriction + selected mobility work
→ mobility directed to the documented ROM restriction

stairs + selected functional-task retraining
→ functional retraining for stairs
```

---

# 7. Power-user / de-duplication boundary

The `Περισσότερα` layer uses a bounded Knee-OA subset rather than the whole global Knee taxonomy.

Every selected product item must:

```text
render
OR fail closed
OR be covered by an explicit semantic de-duplication rule preserving its structured meaning
```

Reviewed semantic de-duplication includes:

```text
specific pain > generic pain prose
confirmed effusion > generic swelling prose
specific/objective weakness > generic weakness prose
active + passive ROM restriction → one combined phrase
generic plan-duplicating goals → prose-suppressed only, state retained
```

Restrictions and clinician notes have deterministic output ownership.

---

# 8. Safety integration finding

`true_locking_or_major_mechanical_rom_block` exists in CU-1 taxonomy, but the frozen current rule catalog does not infer an explicit safety concern from that finding itself.

Therefore:

```text
true locking
→ NOT exposed in first Knee-OA product surface
→ unsupported product selection fails closed
→ future exposure requires bounded safety/reassessment mapping
```

The product does not create a second hidden safety engine.

---

# 9. Other explicit seams

```text
walking aid
→ canonical CU-1 ID exists
→ template phrase exists
→ current Knee UI exposure remains separately gated

weight management
→ strong evidence when overweight/obesity applies
→ no dedicated current selectable CU-1 machine ID
→ no auto-insertion / hidden selection
```

---

# 10. Manual edit boundary

```text
auto_live referral
→ explicit Edit creates ephemeral text buffer
→ edited text does not reverse-write structured selections
```

A dirty manual buffer must not be silently overwritten. Exact edit-mode interaction remains prototype/Step-5 work.

---

# 11. Machine contracts / fixtures

Human design:

```text
clinic_utilities/physio_referral_product/KNEE_OA_TEMPLATE_DESIGN_V1.md
```

Machine contract:

```text
clinic_utilities/physio_referral_product/contracts/knee_oa_template_contract_v1.yaml
```

Primary fixtures:

```text
clinic_utilities/physio_referral_product/contracts/knee_oa_template_fixtures_v1.yaml
```

Power-user / fail-closed / Copy-readiness fixtures:

```text
clinic_utilities/physio_referral_product/contracts/knee_oa_template_edge_fixtures_v1.yaml
```

Validator:

```text
clinic_utilities/physio_referral_product/validate_knee_oa_template_contract_v1.py
```

---

# 12. Exact design review

```text
clinic_utilities/physio_referral_product/KNEE_OA_TEMPLATE_DESIGN_REVIEW_V1.md
```

Disposition:

```text
STEP 3 DESIGN                         PASS
MATERIAL OPEN FINDING                 NONE
HUMAN/MACHINE CONTRACT CONSISTENCY    PASS
STEP-2 EVIDENCE COMPATIBILITY         PASS
POWER-USER LOSS-OF-SELECTION GUARD    PASS
COPY-READINESS AUTHORITY              PASS
TRUE-LOCKING SAFETY SEAM              EXPLICITLY DEFERRED / FAIL-CLOSED
RUNTIME IMPLEMENTATION                NOT AUTHORIZED
MERGE / DEPLOY                        NOT AUTHORIZED
```

---

# 13. Verification evidence

Reviewed substantive gate:

```text
workflow: Physio Knee OA template design gate
run:      34561575795
head:     cd4a42b4582921df7eb64d6ff3fb7c718a141c3a
result:   SUCCESS
```

Review-artifact gate:

```text
workflow: Physio Knee OA template design gate
run:      34561638107
head:     11f04d9ea316ba7fc30a03f4aaa13b60ea5c1e65
result:   SUCCESS
```

Exact full-diff review from frozen Step-2 parent found `behind = 0` and no runtime/API/formatter/static UI/database leakage.

---

# 14. Out of scope / hold

```text
runtime implementation
visual prototype implementation
second diagnosis
billing/auth/entitlements
patient persistence
AI referral prose
autonomous evidence updating
PR/merge/deploy/production smoke
```

---

# 15. Exact next boundary

```text
STEP 4 — Evidence Interaction / Traceability Layer
```

Step 4 may design the evidence-state visual/interaction semantics required by the already-frozen UX and evidence contracts. It does not inherit runtime implementation authority from Step 3.
