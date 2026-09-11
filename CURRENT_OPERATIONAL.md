# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PHYSIO REFERRAL PRODUCTIZATION — STEP 3 KNEE-OA DYNAMIC REFERRAL/TEMPLATE DESIGN ACTIVE
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh base `main`:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Frozen Step-2 parent head:** `ab4b349223cd4c461837ab3125967a06d169a7e1`.
> **Active branch:** `design/physio-referral-knee-oa-template-v1-2026-09-11`.
> **Active slice:** `CU1-PRODUCT-KNEE-OA-TEMPLATE-V1-2026-09-11`.
> **ACTIVE DESIGN/CANONICAL WRITER:** this bounded Knee-OA dynamic referral/template design slice only.
> **ACTIVE RUNTIME WRITER:** NONE.
> **Production config/secret authority:** NONE.
> **Patient-data mutation authority:** NONE.
> **PR/merge/deploy authority:** NONE.

---

# 1. Frozen ancestry

Step 1 UX and Step 2 Knee-OA evidence design remain frozen and are inherited read-only by this slice.

Step-2 evidence state, claim-scope, default-plan and update-governance semantics must not be reopened merely to simplify text composition.

The existing deployed CU-1 runtime/API/formatter/static UI/database remain read-only during Step 3.

---

# 2. Step-3 objective

Define deterministic live Greek referral composition for the Knee-OA vertical slice:

```text
Knee-OA diagnosis + laterality
+ clinician-selected phenotype/findings
+ functional limitations
+ clinician-selected evidence-aware rehabilitation components
+ optional power-user selections
+ inherited restrictions/safety state
→ concise live referral text
```

The contract must support the frozen UX goal of immediate live text updates without a Generate button.

---

# 3. Key design constraints

```text
SUGGESTION != SELECTION != GENERATED PLAN
UNSELECTED evidence suggestions do not appear in the referral
EVIDENCE STATE != referral prose by default
MISSING != NORMAL / ABSENT
STIFFNESS SYMPTOM != ROM RESTRICTION
SUBJECTIVE/GENERIC WEAKNESS != OBJECTIVE WEAKNESS
FREE-TEXT EDIT != structured-state mutation
ADJUNCT != CORE ACTIVE REHABILITATION
```

The copied referral should remain useful to the physiotherapist rather than becoming a guideline report or an over-prescriptive treatment protocol.

---

# 4. Required Step-3 outputs

```text
human template/composition design
machine-readable template contract
product-local phenotype overlay only where current CU-1 cannot represent a frozen UX concept without false inference
phrase-group ordering and de-duplication rules
laterality grammar
context-triggered emphasis rules
adjunct wording rules
copy-readiness / safety boundary
manual-edit boundary
synthetic deterministic fixtures
contract validator + design gate
exact design review
```

---

# 5. Explicit hold

```text
NO CU-1 runtime mutation
NO production UI rewrite
NO change to frozen Step-2 evidence states/source positions
NO second diagnosis
NO patient persistence
NO billing/auth/entitlements
NO autonomous evidence updating
NO PR/merge/deploy/production smoke
```

---

# 6. Exact next action

```text
inspect existing CU-1 formatter/language/state seams
→ define Knee-OA product projection state
→ freeze phrase groups + ordering + suppression/de-duplication
→ freeze exact synthetic outputs
→ machine validate against existing CU-1 IDs and frozen Step-2 evidence contract
→ exact design review
→ close/release Step-3 writer if clean
```
