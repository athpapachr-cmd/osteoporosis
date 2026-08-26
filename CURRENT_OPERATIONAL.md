# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-26 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Current main:** `d790bf12c6342f7760151ad263eeb2e3a1de9c7e`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Current detailed design candidate:** `clinic_utilities/physio_profiles/cervical_v1.md`.
> **CERVICAL DESIGN PR:** PR #36 merged as `d790bf12c6342f7760151ad263eeb2e3a1de9c7e`.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED for CU-1; design only.
> **PR-1 TRANSCRIPT SLICE:** intentionally paused, preserved at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.
> **CALENDAR / DIGITAL SECRETARY:** paused for Clinical Excellence; Secretary has its own independent AC-2 writer lock.

This file is the sole owner of operational **NOW**. Do not infer current mutation authority from chat history or an old slice file.

---

# 1. Why the active slice changed

The product owner requested a small near-term detour to integrate two existing day-to-day clinic utilities into the future Clinical Excellence Cockpit:

1. Physiotherapy referral text generator.
2. Radiofrequency treatment request / PDF workflow.

This detour is part of the broader product goal — Clinical Excellence before, during and after clinic — and is not Osteoporosis Module 01 logic.

Permanent boundary:

```text
Clinical Excellence Core
→ reusable platform/workspace mechanics

Clinic Utilities / Clinical Operations
→ cross-module clinician workflow tools

Osteoporosis Module 01
→ osteoporosis-specific clinical standards/audit/workflows
```

The PR-1 transcript design is not abandoned. It is paused intact while CU-1 is reviewed.

---

# 2. Prior PR-1 state preserved

Before the detour, PR-1 Transcript Intake + Candidate Extraction had reached REPLAN-corrected design v3 and remained:

```text
DESIGNED
NOT IMPLEMENTED
NOT AUTHORIZED FOR IMPLEMENTATION
```

The complete v3 slice contract was archived unchanged at:

```text
archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md
```

When the detour ends, a fresh conversation can restore PR-1 as the active slice from that preserved contract after canonical review.

---

# 3. Physiotherapy source inspection — completed read-only

The supplied standalone physiotherapy referral HTML has been inspected.

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

Design gaps to correct:

- checkbox catalogue rather than clinically ordered referral flow;
- overly generic findings across conditions;
- globally preselected goals/interventions;
- insufficient condition-specific precautions/restrictions;
- direct phrase concatenation instead of a structured `ReferralDraft`;
- limited consistency/safety rules;
- incomplete common referral pathways;
- standalone styling rather than Clinical Excellence visual integration.

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
adjunct technique != default primary treatment
```

---

# 4. Cervical profile design candidate — created, not frozen

The first detailed body-region design candidate is now merged on `main`:

```text
clinic_utilities/physio_profiles/cervical_v1.md
```

It proposes:

```text
PRIMARY CLINICAL PROBLEM
+
MODIFIERS / FINDINGS
+
FUNCTIONAL IMPACT
+
SAFETY / PRECAUTIONS
```

Proposed primary cervical pathways:

```text
non-specific / mechanical neck pain
neck pain with radiating upper-limb / radicular features
cervicogenic-headache pathway
whiplash / post-traumatic neck pain
shared post-operative pathway
```

Important semantic decisions in the candidate:

- Spurling or radiating pain alone does not automatically become a definitive radiculopathy diagnosis;
- trigger points, referred shoulder-girdle pain, mobility restriction and ergonomic load are treated mainly as modifiers/findings rather than equivalent top-level diagnoses;
- neurological screen distinguishes `not assessed` from `normal`;
- objective deficits are represented separately from subjective radiating symptoms;
- progressive deficit / possible myelopathy / red-flag concerns produce explicit reassessment prompts;
- goals are context-sensitive rather than globally preselected;
- active rehabilitation remains the main direction and traction/dry needling/manual therapy/acupuncture are optional adjuncts where appropriate;
- short and detailed wording examples derive from the same conceptual structure.

This profile is **not yet product-owner approved or frozen**.

---

# 5. RF source inspection — completed read-only

The RF workflow was inspected in `athpapachr-cmd/ortho-reception-backend-v2`.

Current useful seams include:

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

RF runtime mutation is currently blocked by the separate Digital Secretary AC-2 writer lock.

---

# 6. Current merge / writer state

PR #36 was squash-merged into `main` as:

```text
d790bf12c6342f7760151ad263eeb2e3a1de9c7e
```

It:

- added `clinic_utilities/physio_profiles/cervical_v1.md`;
- updated `SLICE_PLAN_CURRENT.md` to point to the cervical design candidate;
- made cervical review/freeze the exact next step.

No production HTML/JS/CSS or runtime behavior changed.

There is now **no active writer lock** in this repository.

---

# 7. Exact next action

The next fresh conversation should:

```text
1. fresh-bootstrap current main + all six canonicals
2. read CLINIC_UTILITIES_PLAN.md
3. read clinic_utilities/physio_profiles/cervical_v1.md
4. critically review the cervical profile clinically and structurally
5. present recommended corrections/questions to the product owner
6. after approval, freeze cervical profile
7. then proceed to lumbar profile design
```

The cervical review should explicitly challenge:

```text
primary-problem taxonomy
findings/modifiers separation
neurological-screen detail
red-flag/reassessment semantics
functional-limit fields
goal suggestions
rehabilitation directions
visibility of adjunct techniques
short/detailed generated wording
```

Do not write runtime code.

---

# 8. Explicitly forbidden now

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

# 9. Handoff completeness

A fresh conversation should be able to recover from the repository alone:

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
current profile = cervical_v1 design candidate
PR-1 transcript = paused + archived, not lost
physio source = inspected
RF source = inspected read-only
RF runtime = blocked by Secretary AC-2 lock
runtime writer = none
next substantive action = review/freeze cervical profile, then lumbar
```
