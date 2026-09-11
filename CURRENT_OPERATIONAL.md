# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PHYSIO REFERRAL PRODUCTIZATION — STEP 3 KNEE-OA DYNAMIC REFERRAL/TEMPLATE DESIGN FROZEN / CLOSED
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh base `main`:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Frozen Step-2 parent head:** `ab4b349223cd4c461837ab3125967a06d169a7e1`.
> **Design branch:** `design/physio-referral-knee-oa-template-v1-2026-09-11`.
> **Closed slice:** `CU1-PRODUCT-KNEE-OA-TEMPLATE-V1-2026-09-11`.
> **Reviewed substantive head:** `cd4a42b4582921df7eb64d6ff3fb7c718a141c3a`.
> **Review-artifact head:** `11f04d9ea316ba7fc30a03f4aaa13b60ea5c1e65`.
> **ACTIVE DESIGN/CANONICAL WRITER:** NONE — Step 3 closed.
> **ACTIVE RUNTIME WRITER:** NONE.
> **Production config/secret authority:** NONE.
> **Patient-data mutation authority:** NONE.
> **PR/merge/deploy authority from this closeout:** NONE.

---

# 1. Preserved production state

Clinical Learning Hub L-1D remains `MERGED / DEPLOYED / PRODUCTION-SMOKE-VERIFIED / CLOSED` and was not modified by this Physio Referral design slice.

Existing CU-1 Physiotherapy Referral v2 remains the deployed production runtime. Step 3 changed no CU-1 runtime/API/formatter/static production UI/database file.

---

# 2. Step-3 closure artifacts

Human design:

```text
clinic_utilities/physio_referral_product/KNEE_OA_TEMPLATE_DESIGN_V1.md
```

Machine contract:

```text
clinic_utilities/physio_referral_product/contracts/knee_oa_template_contract_v1.yaml
```

Deterministic fixtures:

```text
clinic_utilities/physio_referral_product/contracts/knee_oa_template_fixtures_v1.yaml
clinic_utilities/physio_referral_product/contracts/knee_oa_template_edge_fixtures_v1.yaml
```

Exact design review:

```text
clinic_utilities/physio_referral_product/KNEE_OA_TEMPLATE_DESIGN_REVIEW_V1.md
DESIGN PASS / MATERIAL OPEN FINDING NONE
```

Machine evidence:

```text
Physio Knee OA template design gate
run 34561575795
head cd4a42b4582921df7eb64d6ff3fb7c718a141c3a
SUCCESS

Physio Knee OA template design gate
run 34561638107
head 11f04d9ea316ba7fc30a03f4aaa13b60ea5c1e65
SUCCESS
```

Closeout/status-only descendants do not replace `cd4a42b...` as the reviewed substantive head.

---

# 3. Frozen Step-3 decisions

## Deterministic live referral

```text
clinician-selected CU-1 state
+ bounded ephemeral Knee-OA phenotype overlay
→ deterministic semantic projection
→ deterministic Greek referral
```

Routine output requires no LLM and no Generate action.

## Diagnosis authority / Copy readiness

Opening the Knee-OA product flow does not diagnose Knee OA.

Copy requires:

```text
formal_assertion_state = yes
laterality = right | left | bilateral
selected product state within supported Knee-OA scope
no inherited CU-1 formatter-blocking validation error
no unresolved inherited blocking/urgent safety state
```

## Phenotype integrity

```text
stiffness symptom != ROM restriction
generic weakness != objective weakness
quadriceps weakness > objective weakness > generic weakness for output specificity
```

## Suggestion / plan integrity

```text
suggestion != selection != output
functional limitation != automatic rehabilitation selection
```

Only selected rehabilitation components enter the plan. Removing a default component removes its phrase.

## Power-user preservation

A selected product item must:

```text
render
OR fail closed
OR be covered by explicit semantic de-duplication that preserves structured state
```

Restrictions and clinician notes have deterministic output ownership.

## Evidence/referral boundary

Evidence colour/state, bubbles, source/year and bibliography remain clinician-facing UI information. They do not enter copied referral prose by default.

## Safety seam

`true_locking_or_major_mechanical_rom_block` remains in the global CU-1 taxonomy but is not exposed in the first Knee-OA product surface because the inherited rule catalog does not derive a safety concern from the finding itself. Future exposure requires a bounded safety/reassessment mapping.

## Manual editing

```text
auto_live derived referral
→ explicit Edit creates ephemeral text buffer
→ edited prose does not reverse-write structured selections
```

Exact edit-mode interaction remains later prototype work.

---

# 4. Scope/diff evidence

Exact full-diff review from frozen Step-2 parent showed only:

```text
root Step-3 canonicals
Step-3 human design/review
Step-3 machine contract
primary + edge synthetic fixtures
Step-3 validator/workflow
supporting Physio product closeout docs
```

No production runtime/API/formatter/static UI/database mutation was introduced.

---

# 5. Lifecycle

```text
STEP 1 UX CONTRACT                    FROZEN
STEP 2 KNEE-OA EVIDENCE DESIGN        FROZEN / DESIGN PASS
STEP 3 DYNAMIC REFERRAL DESIGN        FROZEN / DESIGN PASS
STEP 3 MATERIAL OPEN FINDING          NONE
STEP 3 RUNTIME IMPLEMENTATION         NOT AUTHORIZED
STEP 3 MERGED                         NO
STEP 3 DEPLOYED                       NO
ACTIVE WRITER                         NONE
```

---

# 6. Exact next boundary / HOLD

Exact next product-design step:

```text
STEP 4 — Evidence Interaction / Traceability Layer
```

Step 4 may design:

```text
colour + non-colour evidence-state cues
compact contextual bubbles
small `i` evidence sheet
source/year vs product-reviewed-on presentation
guideline-conflict presentation
suggestion evidence provenance
accessibility semantics
```

Current hold:

```text
NO runtime implementation
NO production UI rewrite
NO second diagnosis
NO billing/auth/entitlements
NO patient persistence
NO autonomous evidence updates
NO PR/merge/deploy/production smoke without separate authority
```
