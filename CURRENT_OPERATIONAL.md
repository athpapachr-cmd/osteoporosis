# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-26 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified base main:** `9cdb6bb5db64a174573c385e90ba11485666fee2`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **CURRENT BODY-REGION DESIGN TARGET:** shoulder.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-shoulder-v1-design-2026-08-26` for shoulder CU-1 clinical/content design.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE in this repository.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED for CU-1; design only.
> **PR-1 TRANSCRIPT SLICE:** intentionally paused, preserved at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

This file is the sole owner of operational **NOW** for the active branch. Do not infer mutation authority from chat history.

---

# 1. Product boundary

```text
Clinical Excellence Core
→ reusable platform/workspace mechanics

Clinic Utilities / Clinical Operations
→ cross-module clinician workflow tools

Osteoporosis Module 01
→ osteoporosis-specific clinical standards/audit/workflows
```

CU-1 is a bounded cross-module Clinic Utility design detour. PR-1 remains paused intact, not abandoned.

---

# 2. Physiotherapy v2 architectural direction

```text
clinical problem
→ important findings
→ functional limitation
→ precautions/restrictions
→ goals
→ rehabilitation direction
→ structured ReferralDraft
→ short/detailed formatter
```

Hard rules:

```text
suggested finding != examined finding
selected goal != mandatory goal
condition profile != automatic diagnosis
not assessed != normal
adjunct technique != default primary treatment
clinician-entered diagnosis may be carried but not inferred
```

---

# 3. Frozen regional profiles

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
```

Frozen files:

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
clinic_utilities/physio_profiles/lumbar_v1_1.md
```

---

# 4. Shoulder — ACTIVE DESIGN CANDIDATE WORK

Authorized scope is clinical/content design only:

```text
primary pathway taxonomy
findings vs diagnosis separation
safety/reassessment semantics
functional limitations
condition-sensitive goals
active rehabilitation directions
adjunct visibility
generated wording
current evidence review
```

The target candidate file is:

```text
clinic_utilities/physio_profiles/shoulder_v1.md
```

It remains **DESIGN CANDIDATE / NOT FROZEN** until explicit product-owner approval.

---

# 5. Evidence framework

Use current high-quality shoulder guidance without silent hybridization. The 2025 rotator-cuff-tendinopathy CPG, the 2025 AAOS rotator cuff injury guideline, current frozen-shoulder guidance, current imaging appropriateness criteria and condition-specific evidence may inform production wording. Evidence-sensitive intervention wording must remain separable from stable structural design.

---

# 6. Repository-control housekeeping

Immediately before this branch was created, an accidental empty shoulder placeholder was created on `main` and removed in the next docs-only commit. `CURRENT_OPERATIONAL.md` on `main` records that housekeeping. No runtime content or frozen clinical profile content changed.

---

# 7. Exact next action

```text
1. create shoulder_v1.md design candidate
2. review current evidence
3. present strict clinical/structural review to product owner
4. revise based on real-workflow feedback
5. freeze/merge only after explicit approval
```

---

# 8. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
FREEZE or merge shoulder without product-owner approval
```

---

# 9. Handoff completeness

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
cervical = frozen v1.1
lumbar = frozen v1.1
shoulder = active design candidate work
canonical writer = docs/cu1-shoulder-v1-design-2026-08-26
runtime writer = none
runtime implementation = unauthorized
```
