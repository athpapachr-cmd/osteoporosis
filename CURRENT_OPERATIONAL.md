# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-26 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Current main:** `482db59e2133f538cae985f60813f36d3d50f63e` after PR #38 cervical-freeze/lumbar-design merge; this operational handoff commit will advance `main` once written.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Current detailed design candidate:** `clinic_utilities/physio_profiles/lumbar_v1.md`.
> **CERVICAL FREEZE PR:** PR #38 squash-merged as `482db59e2133f538cae985f60813f36d3d50f63e`.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE in this repository.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED for CU-1; design only.
> **PR-1 TRANSCRIPT SLICE:** intentionally paused, preserved at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.
> **CALENDAR / DIGITAL SECRETARY:** paused for Clinical Excellence; Secretary has its own independent AC-2 writer lock.

This file is the sole owner of operational **NOW**. Do not infer current mutation authority from chat history or an old slice file.

---

# 1. Why the active slice changed

The product owner requested a bounded detour to integrate two existing day-to-day clinic utilities into the broader Personal Clinical Excellence workspace:

1. Physiotherapy referral text generator.
2. Radiofrequency treatment request / PDF workflow.

Permanent boundary:

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

# 2. Prior PR-1 state preserved

PR-1 Transcript Intake + Candidate Extraction v3 remains:

```text
DESIGNED
NOT IMPLEMENTED
NOT AUTHORIZED FOR IMPLEMENTATION
```

Frozen archive:

```text
archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md
```

---

# 3. Physiotherapy source inspection — completed read-only

Useful behavior to preserve:

- body-region condition groups;
- laterality/chronicity/session fields;
- clinical findings;
- goals;
- active vs adjunct interventions;
- short/detailed generated text;
- copy/print;
- local/no-server design;
- initial safety/consistency warnings;
- evidence/reference block.

Frozen v2 architectural direction:

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

# 4. Cervical profile — FROZEN v1.1

Product-owner review approved the final cervical design at:

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
```

The prior `cervical_v1.md` is retained as the historical candidate.

Frozen primary cervical pathways:

```text
non-specific / mechanical neck pain
neck pain with radiating upper-limb / radicular features
headache with cervical musculoskeletal features
  + explicit formal cervicogenic-headache clinician assertion
cervical/cervicogenic dizziness presentation
  + explicit clinician diagnosis assertion
whiplash / post-traumatic neck pain
```

Cervical post-operative rehabilitation is removed from the active cervical MVP because it is not part of the product owner's actual current referral workflow.

Important frozen semantics:

- formal cervicogenic headache can be output when explicitly asserted by the clinician;
- cervical/cervicogenic dizziness can be output when explicitly asserted by the clinician, but the software never infers cervical causation from dizziness + neck pain;
- trigger-point/myofascial findings and referred shoulder-girdle pain remain directly selectable and may appear in referral wording when actually selected;
- they remain findings/presentation modifiers by default rather than software-derived diagnoses;
- subjective radiating symptoms remain separate from objective motor/sensory/reflex deficits;
- motor/sensory/reflex states are tri-state and preserve `not assessed != normal`;
- no global `no neurological deficit` or default `no red flags` output exists;
- progressive neurological deficit / possible cord concern / trauma-instability / other material safety concern triggers explicit clinician-facing reassessment/disposition semantics;
- goals are context-sensitive and not globally preselected;
- active rehabilitation, exercise, education and self-management are the conceptual backbone;
- manual therapy, soft-tissue work, neurodynamics, selected traction, dry needling and acupuncture remain optional technique-level adjuncts under secondary visibility where appropriate.

Cervical technique-specific wording remains subject to one final evidence check immediately before CU-2 production implementation.

---

# 5. Lumbar profile — CURRENT DESIGN TARGET

A first lumbar design candidate now exists at:

```text
clinic_utilities/physio_profiles/lumbar_v1.md
```

Proposed primary pathways:

```text
non-specific / mechanical low-back pain
low-back pain with radiating leg symptoms / radicular features
lumbar spinal stenosis / neurogenic claudication pathway
```

The candidate deliberately keeps mobility restriction, load/postural aggravation, trunk deconditioning, myofascial/trigger-point findings and referred buttock/leg pain mainly as findings/modifiers rather than equivalent top-level diagnoses.

It inherits the cervical tri-state neurological model and adds explicit high-priority handling for new bladder/bowel/sexual-function change, perineal/saddle sensory change, progressive neurological deficit and other cauda-equina-type concerns.

Routine lumbar traction is not included as a default adjunct because NICE and WHO guidance recommend against routine traction. Needling/acupuncture requires explicit framework resolution because NICE and WHO recommendations differ.

Lumbar is **DESIGN CANDIDATE / NOT FROZEN**.

---

# 6. RF source inspection — completed read-only

The RF workflow was inspected in `athpapachr-cmd/ortho-reception-backend-v2`.

Existing useful seams include:

```text
/rf form
RF PDF templates: Medikey / DIROS / Thermedico
radiology PDF append
previous application lookup
rfa_applications PostgreSQL table
status field
patient/location/indication/consumable/VAS history
repeat-use support
```

Future target remains one authoritative RF workflow with lifecycle/history/reuse, not a second competing database.

RF runtime mutation remains blocked by the separate Digital Secretary AC-2 writer lock.

---

# 7. Current writer / merge state

PR #38 was squash-merged into `main` as:

```text
482db59e2133f538cae985f60813f36d3d50f63e
```

It changed documentation/design only:

- added frozen `clinic_utilities/physio_profiles/cervical_v1_1.md`;
- added candidate `clinic_utilities/physio_profiles/lumbar_v1.md`;
- updated the CU-1 slice/supporting plan/operational state.

No production HTML/JS/CSS or runtime behavior changed.

There is now **no active writer lock** in this repository.

---

# 8. Exact next action

```text
1. review clinic_utilities/physio_profiles/lumbar_v1.md with the product owner
2. resolve lumbar primary taxonomy and real-workflow fit
3. freeze findings-vs-diagnosis separation
4. freeze neurological / cauda-equina safety semantics
5. freeze functional-limit fields and goals
6. freeze active rehabilitation directions and adjunct visibility
7. resolve NICE-vs-WHO needling/acupuncture framework issue
8. freeze generated short/detailed wording
9. then proceed to shoulder profile design
```

Do not write runtime code.

---

# 9. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
MUTATE RF/Secretary runtime while AC-2 writer lock exists
AUTO-PERSIST physiotherapy referrals without a frozen persistence decision
CREATE a second RF database/source of truth
COMMIT identifiable patient data
MODIFY Calendar/Setmore/Zadarma
RUN overlapping runtime writers
```

---

# 10. Handoff completeness

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
cervical = frozen v1.1
current profile = lumbar_v1 design candidate
PR-1 transcript = paused + archived
physio source = inspected
RF source = inspected read-only
RF runtime = blocked by Secretary AC-2 lock
canonical writer = none
runtime writer = none
next substantive action = lumbar clinical/content review and freeze
```
