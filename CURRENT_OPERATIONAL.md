# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-26 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh main at detour start:** `beb285a34751fb58baeb8be285025690c3ffc730`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **ACTIVE CANONICAL WRITER/LOCK:** current ChatGPT session — docs/design scope only on `docs/clinic-utilities-physio-v2-design`.
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

The product owner supplied the standalone physiotherapy referral HTML and it has now been inspected.

Observed useful capabilities:

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

Observed design gaps:

- checkbox catalogue rather than clinically ordered referral flow;
- overly generic findings across conditions;
- globally preselected goals/interventions;
- insufficient condition-specific precautions/restrictions;
- direct phrase concatenation instead of a structured `ReferralDraft`;
- limited consistency/safety rules;
- incomplete common referral pathways;
- standalone styling rather than Clinical Excellence visual integration.

Disposition:

```text
source is a useful MVP
→ preserve useful behavior
→ redesign clinical/content model before implementation
```

---

# 4. CU-1 frozen design direction

Target referral flow:

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

The detailed body-region taxonomy, `ReferralDraft`, consistency rules and output wording live in `CLINIC_UTILITIES_PLAN.md` and `SLICE_PLAN_CURRENT.md`.

No runtime code should be written until CU-1 is reviewed and the product owner explicitly authorizes implementation.

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

The existing RF database already stores structured request history and a status field, so future Cockpit tracking should extend/migrate one authoritative workflow rather than create a competing RF registry.

Target future lifecycle candidate:

```text
application_generated
→ submitted
→ pending_approval
→ approved_awaiting_procedure
→ performed
```

Historical reuse must create a **new request identity** and only clone reusable data after reconfirmation.

---

# 6. Digital Secretary lock constraint

The Secretary control plane currently reports an independent active writer for AC-2 in `athpapachr-cmd/ortho-reception-ops`.

Therefore:

```text
RF read-only inspection = allowed
RF runtime mutation in ortho-reception-backend-v2 = NOT ALLOWED NOW
```

Do not create a parallel Secretary branch or modify RF runtime there until a fresh Secretary bootstrap explicitly releases/replans that scope.

This constraint does not block CU-1 physiotherapy design because the physiotherapy source is a standalone file and CU-1 is design-only.

---

# 7. Current branch / mutation state

Current design branch:

```text
docs/clinic-utilities-physio-v2-design
```

Changes on this branch are documentation/design only:

- archive PR-1 v3 active-slice contract;
- add `CLINIC_UTILITIES_PLAN.md`;
- switch `SLICE_PLAN_CURRENT.md` to CU-1;
- update this operational handoff.

No application runtime behavior, patient persistence, Calendar, Digital Secretary or PDF generation has been changed.

---

# 8. Exact next action

Continue **CU-1 clinical-content design review** one body region at a time in this order:

```text
cervical spine
→ lumbar spine
→ shoulder
→ knee / hip
→ elbow
→ wrist / hand
→ ankle / foot
→ fracture / post-immobilization
→ muscle injury
→ post-operative / generalized deconditioning
```

For each region freeze:

```text
diagnoses/problems
→ key findings
→ functional limitations
→ precautions/restrictions
→ goals
→ rehabilitation directions
→ generated wording
```

At the end of CU-1:

```text
product-owner review
→ APPROVE / REPLAN
→ if approved, authorize CU-2 implementation
```

Do not start CU-2 runtime implementation without explicit approval.

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

A fresh conversation should now be able to recover from the repository alone:

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
PR-1 = paused + archived, not lost
physio source = inspected
RF source = inspected read-only
RF runtime = blocked by Secretary AC-2 lock
runtime writer = none
next action = body-region clinical-content design review
```
