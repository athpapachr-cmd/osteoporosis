# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-26 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Current main:** `1c51b44fc3b89252224f78cfabc818155373e4c6` after PR #39 lumbar-freeze merge; this handoff commit will advance `main` once written.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **NEXT BODY-REGION DESIGN TARGET:** shoulder.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE in this repository.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED for CU-1; design only.
> **PR-1 TRANSCRIPT SLICE:** intentionally paused, preserved at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.
> **CALENDAR / DIGITAL SECRETARY:** paused for Clinical Excellence; Secretary has its own independent AC-2 writer lock.

This file is the sole owner of operational **NOW**. Do not infer mutation authority from chat history.

---

# 1. Product boundary

The active bounded detour integrates day-to-day clinic utilities into the broader Personal Clinical Excellence workspace.

```text
Clinical Excellence Core
→ reusable platform/workspace mechanics

Clinic Utilities / Clinical Operations
→ cross-module clinician workflow tools

Osteoporosis Module 01
→ osteoporosis-specific clinical standards/audit/workflows
```

The PR-1 transcript design is paused intact, not abandoned.

---

# 2. PR-1 state preserved

PR-1 Transcript Intake + Candidate Extraction v3 remains:

```text
DESIGNED
NOT IMPLEMENTED
NOT AUTHORIZED FOR IMPLEMENTATION
```

Archive:

```text
archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md
```

---

# 3. Physiotherapy v2 architectural direction

The standalone source was inspected read-only. Frozen v2 direction:

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
```

---

# 4. Cervical — FROZEN v1.1

Authoritative file:

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
```

Frozen pathways include non-specific/mechanical neck pain, radiating/radicular features, explicit formal cervicogenic headache when clinician-asserted, cervical/cervicogenic dizziness when clinician-asserted, and whiplash/post-traumatic neck pain.

Cervical post-operative rehabilitation is excluded from the active cervical MVP. Trigger-point/myofascial and referred shoulder-girdle findings remain directly selectable. Neurological status is component-level tri-state and no missing/unassessed state generates normal wording.

---

# 5. Lumbar — FROZEN v1.1

Authoritative file:

```text
clinic_utilities/physio_profiles/lumbar_v1_1.md
```

The prior `lumbar_v1.md` remains the historical design candidate.

Frozen primary lumbar pathways:

```text
non-specific / mechanical low-back pain
low-back pain with radiating leg symptoms / radicular features
lumbar spinal stenosis / neurogenic claudication
deep-gluteal / piriformis presentation
```

Frozen lumbar semantics:

- subjective radiating symptoms remain separate from objective motor/sensory/reflex deficits and formal radiculopathy diagnosis;
- motor/sensory/reflex are tri-state and preserve `not assessed != normal`;
- SLR/slump findings never create a diagnosis automatically;
- cauda-equina-type concerns produce high-priority clinician reassessment/disposition prompts without autonomous diagnosis;
- myofascial/trigger-point and referred buttock/leg findings are directly selectable modifiers;
- deep-gluteal syndrome or piriformis syndrome may be stated only when explicitly asserted by the clinician;
- `SI dysfunction` is not a lumbar diagnosis;
- SI-region/SIJ pathology is reserved for a future separate SI/pelvic profile;
- MRI may establish sacroiliitis/defined structural pathology but must not automatically establish a mechanically painful SI joint as the pain generator;
- acupuncture remains an optional clinician-selected adjunct with explicit NICE-vs-WHO evidence-framework transparency;
- dry needling remains an optional adjunct, with clinician-facing competence/availability caveat because correct technique depends materially on practitioner training;
- routine lumbar traction is excluded from the MVP;
- lumbar post-operative rehabilitation is excluded from the active lumbar MVP because it is not part of the product owner's current workflow;
- active rehabilitation, exercise, education and self-management remain the conceptual backbone.

Evidence-sensitive adjunct wording must be rechecked before CU-2 production implementation.

---

# 6. Evidence-framework note for lumbar needling

Current high-quality guidance is not uniform:

- NICE NG59 recommends against acupuncture for low-back pain with or without sciatica;
- WHO 2023 conditionally supports needling therapies, including acupuncture and dry needling, as part of broader care for chronic primary low-back pain, with low-certainty evidence and explicit practitioner-competence considerations.

Product rule:

```text
acupuncture/dry needling may be clinician-selected adjuncts
!=
universal guideline recommendation
!=
mandatory physiotherapy technique
```

---

# 7. RF source inspection — unchanged

The RF workflow was inspected read-only in `athpapachr-cmd/ortho-reception-backend-v2`.

Future target remains one authoritative RF workflow with lifecycle/history/reuse. RF runtime mutation is outside CU-1 and remains constrained by the separate Digital Secretary control plane.

---

# 8. Current writer / merge state

PR #39 was squash-merged into `main` as:

```text
1c51b44fc3b89252224f78cfabc818155373e4c6
```

It changed documentation/design only:

- added frozen `clinic_utilities/physio_profiles/lumbar_v1_1.md`;
- updated `SLICE_PLAN_CURRENT.md`, `CURRENT_OPERATIONAL.md` and `CLINIC_UTILITIES_PLAN.md`;
- advanced the next body-region target to shoulder.

No production HTML/JS/CSS or runtime behavior changed.

There is now **no active writer lock** in this repository.

---

# 9. Exact next action

```text
1. create/review shoulder profile design candidate
2. use the same strict taxonomy/findings/safety/goals/rehab/evidence method
3. obtain product-owner approval before freezing shoulder
```

Do not write runtime code.

---

# 10. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals without a frozen persistence decision
CREATE a second RF database/source of truth
COMMIT identifiable patient data
MODIFY Calendar/Setmore/Zadarma
RUN overlapping runtime writers
```

---

# 11. Handoff completeness

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
cervical = frozen v1.1
lumbar = frozen v1.1
next body region = shoulder
PR-1 transcript = paused + archived
canonical writer = none
runtime writer = none
runtime implementation = unauthorized
```