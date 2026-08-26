# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-26 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified base main before this operational reconciliation:** `8649f9da02214751a5b5af8a0bacbe67e2a7a3a6`; this reconciliation commit advances `main` once written.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **NEXT BODY-REGION DESIGN TARGET:** shoulder.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE at this reconciliation point.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE in this repository.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED for CU-1; design only.
> **PR-1 TRANSCRIPT SLICE:** intentionally paused, preserved at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

This file is the sole owner of operational **NOW**. Do not infer mutation authority from chat history.

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

## Cervical — FROZEN v1.1

Authoritative file:

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
```

## Lumbar — FROZEN v1.1

Authoritative file:

```text
clinic_utilities/physio_profiles/lumbar_v1_1.md
```

Lumbar includes non-specific/mechanical LBP, radiating/radicular features, stenosis/neurogenic claudication and deep-gluteal/piriformis presentation. SI dysfunction is not a lumbar diagnosis. Acupuncture and dry needling remain optional adjuncts with evidence/competence caveats; routine lumbar traction and lumbar post-operative rehabilitation are excluded from the active MVP.

---

# 4. Current target — shoulder

The next authorized substantive action is shoulder **clinical/content design only** using the same method:

```text
primary pathway taxonomy
→ findings vs diagnosis separation
→ safety/reassessment semantics
→ functional limitations
→ condition-sensitive goals
→ active rehabilitation directions
→ adjunct visibility
→ generated wording
→ current evidence review
→ product-owner review/freeze
```

No shoulder runtime implementation is authorized.

---

# 5. Repository-control housekeeping recorded

During transition into shoulder design, an empty placeholder file was inadvertently created directly on `main` and immediately removed in the next docs-only commit.

```text
520c5ff7795011724fbd41728f5cf2f6703e5eda
→ accidental placeholder creation

8649f9da02214751a5b5af8a0bacbe67e2a7a3a6
→ placeholder removed
```

Net repository tree/content after the cleanup matches the pre-placeholder design state. No HTML/JS/CSS, runtime behavior, patient data, schemas or frozen cervical/lumbar profile content changed in those two commits.

---

# 6. RF / Secretary boundary

RF runtime mutation is outside CU-1 and remains governed by the separate Digital Secretary control plane. Do not create a competing RF data source or mutate Secretary/Calendar/Setmore/Zadarma from this slice.

---

# 7. Exact next action

```text
1. create a docs-only shoulder design branch from reconciled current main
2. claim that branch as the canonical writer for shoulder CU-1 design
3. create clinic_utilities/physio_profiles/shoulder_v1.md as DESIGN CANDIDATE
4. perform current high-quality evidence review
5. present the candidate to the product owner
6. do not freeze/merge shoulder until product-owner approval
```

---

# 8. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals without a frozen persistence decision
CREATE a second RF database/source of truth
COMMIT identifiable patient data
MODIFY Calendar/Setmore/Zadarma
RUN overlapping runtime writers
FREEZE shoulder without product-owner review
```

---

# 9. Handoff completeness

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
cervical = frozen v1.1
lumbar = frozen v1.1
next body region = shoulder
PR-1 transcript = paused + archived
runtime writer = none
runtime implementation = unauthorized
next action = shoulder design candidate on docs-only branch
```
