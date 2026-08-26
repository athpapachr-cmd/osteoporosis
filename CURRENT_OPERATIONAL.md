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
> **Frozen shoulder profile:** `clinic_utilities/physio_profiles/shoulder_v1_1.md`.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-shoulder-v1-design-2026-08-26` until shoulder freeze/handoff review is merged.
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

# 2. Physiotherapy v2 architecture

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
special test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
adjunct technique != default primary treatment
clinician-entered diagnosis may be carried but not inferred
```

---

# 3. Frozen regional profiles

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
```

Files:

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
clinic_utilities/physio_profiles/lumbar_v1_1.md
clinic_utilities/physio_profiles/shoulder_v1_1.md
```

---

# 4. Shoulder v1.1 freeze — current material state

Product-owner-approved shoulder design includes:

```text
RCRSP / rotator-cuff tendinopathy
confirmed full-thickness cuff tear — conservative rehab
calcific cuff tendinopathy
adhesive capsulitis
GH instability/dislocation rehabilitation
GH osteoarthritis
post-traumatic assessed shoulder injury
AC-joint pathology
sternoclavicular-joint pathology
postoperative shoulder rehabilitation
```

Important frozen shoulder semantics:

- `impingement syndrome` is not the preferred top-level diagnosis;
- long-head biceps tendinopathy is a common selectable secondary/coexisting diagnosis;
- AC-joint pathology can be a primary referral entity;
- sternoclavicular disease is diagnosis/context governed;
- suspected posterior SC dislocation or unexplained SC swelling/systemic concern triggers high-priority reassessment rather than routine physiotherapy reassurance;
- acute traumatic marked weakness/inability to elevate requires explicit reassessment semantics;
- acupuncture and dry needling remain optional adjuncts with competence/availability safeguards;
- ESWT is available only for calcific tendinopathy; prior barbotage is recorded as context and does not create an automatic sequence rule;
- postoperative shoulder is active and requires procedure/protocol/restriction context;
- shoulder-region fractures route to the shared fracture/post-immobilization profile rather than being duplicated inside shoulder.

No runtime behavior changed.

---

# 5. Evidence boundary

Stable architecture is frozen; technique/protocol claims remain evidence-sensitive and must be refreshed immediately before CU-2 production implementation.

Current shoulder evidence review included the 2025 rotator-cuff CPG, 2025 AAOS rotator-cuff guideline, acute-shoulder imaging appropriateness criteria, frozen-shoulder guidance, instability literature, calcific ESWT/lavage systematic reviews and sternoclavicular-joint literature.

---

# 6. Repository-control note

Before the shoulder branch was created, a placeholder file was accidentally created and immediately removed on `main`; the tree was restored before shoulder clinical content was written. The source-identity housekeeping commit was recorded before branch creation. No runtime or frozen clinical content was affected.

---

# 7. Exact next action

```text
1. exact branch-vs-main review of shoulder freeze
2. open/merge docs-only shoulder freeze PR if clean
3. clear canonical writer lock and record resulting main SHA
4. product owner selects next remaining regional profile
```

---

# 8. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
START next regional mutation before shoulder handoff is closed
```

---

# 9. Handoff completeness

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
cervical = frozen v1.1
lumbar = frozen v1.1
shoulder = frozen v1.1 on docs branch pending merge/handoff close
canonical writer = docs/cu1-shoulder-v1-design-2026-08-26
runtime writer = none
runtime implementation = unauthorized
```
