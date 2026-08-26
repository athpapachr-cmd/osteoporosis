# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** ACTIVE PRE-CODE DESIGN — implementation not yet authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Current detailed profile:** `clinic_utilities/physio_profiles/cervical_v1.md` — design candidate, not yet frozen.
> **Prior active slice:** PR-1 Transcript Intake + Candidate Extraction v3, intentionally paused and archived at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

This slice is a bounded product-owner-approved detour. It does not cancel PR-1 and does not turn Clinic Utilities into a new clinical module.

---

# 1. Product outcome

Freeze the clinical/content and deterministic-generation contract for a substantially improved Physiotherapy Referral utility before any implementation.

Target workflow:

```text
Clinical problem
→ important findings
→ functional limitation
→ precautions/restrictions
→ goals
→ rehabilitation direction
→ structured ReferralDraft
→ short/detailed referral text
```

The future utility should improve referral quality and speed while preserving clinician judgment and physiotherapist autonomy.

---

# 2. Current source baseline

The supplied standalone HTML has been inspected read-only.

Useful behavior to preserve:

- body-region condition groups;
- optional laterality/chronicity/session count;
- clinical findings;
- goals;
- active vs adjunct interventions;
- short/detailed outputs;
- copy/print;
- local/no-server behavior;
- initial validation warnings;
- evidence/reference section.

Problems to correct:

- checkbox-catalogue flow rather than clinical referral flow;
- generic findings across unrelated conditions;
- globally preselected goals/interventions;
- insufficient condition-specific restrictions/precautions;
- direct phrase concatenation instead of a structured intermediate model;
- limited consistency/safety rules;
- incomplete common referral pathways;
- standalone visual identity rather than Clinical Excellence styling.

---

# 3. Frozen architectural direction

## 3.1 Structured intermediate object

Future implementation must build a deterministic `ReferralDraft` before formatting prose:

```text
ReferralDraft
  patient_context
  body_region
  primary_problem
  secondary_problems[]
  laterality
  chronicity
  key_findings[]
  functional_impairments[]
  precautions[]
  explicit_restrictions[]
  goals[]
  rehab_directions[]
  adjunct_options[]
  reassessment_criteria[]
  sessions_optional
  clinician_free_text_optional
```

Then:

```text
ReferralDraft
→ ShortReferralFormatter
→ DetailedReferralFormatter
```

No direct checkbox-to-prose monolith in v2.

## 3.2 Condition profiles

Each diagnosis/problem profile may define:

```text
condition_key
body_region
display_name
candidate_findings[]
candidate_functional_impairments[]
candidate_precautions[]
candidate_goals[]
candidate_rehab_directions[]
adjunct_options[]
consistency_rules[]
required_context_when_selected[]
```

Hard invariant:

```text
suggested != examined
suggested != selected
selected != mandatory
```

The UI may recommend relevant fields but must never silently claim that an examination finding occurred.

---

# 4. Initial condition-profile taxonomy to review/freeze

Detailed candidates live in `CLINIC_UTILITIES_PLAN.md` and include:

- cervical spine;
- lumbar spine;
- shoulder;
- elbow;
- wrist/hand;
- hip;
- knee;
- ankle/foot;
- muscle strain / myotendinous injury;
- fracture / post-immobilization;
- post-operative musculoskeletal rehabilitation;
- generalized deconditioning / balance / gait.

Important additions beyond the MVP include selected common pathways such as calcific/proximal-biceps shoulder presentations, instability/post-op shoulder, thumb CMC OA, carpal tunnel/post-op hand, greater trochanteric pain, THA/hip-fracture rehabilitation, ACL/TKA/patellar tendon, Achilles/post-fracture ankle and generalized balance/deconditioning.

These are **design candidates**, not automatically evidence-frozen production rules.

### 4.1 Cervical profile status

A first detailed cervical design candidate now exists at:

```text
clinic_utilities/physio_profiles/cervical_v1.md
```

It proposes:

- primary problem choices separated from modifiers/findings;
- mechanical/non-specific neck pain;
- radiating/radicular-feature pathway without overdiagnosing radiculopathy from one test;
- cervicogenic-headache pathway;
- whiplash/post-traumatic pathway;
- shared post-operative pathway;
- explicit neurological-screen semantics (`not assessed != normal`);
- functional-impact fields;
- safety/escalation prompts;
- context-sensitive goals;
- active rehabilitation as default direction with adjunct techniques optional;
- deterministic consistency rules;
- short/detailed wording examples.

This profile is **not yet frozen**. Product-owner review is required before moving to lumbar design.

---

# 5. Context-sensitive defaults

Remove the current global default that pain + ROM + strength + motor control + function and a fixed active-intervention bundle apply to every referral.

New rule:

```text
selected condition profile
→ suggest relevant goals / directions
→ clinician confirms or changes
→ confirmed values populate ReferralDraft
```

Examples:

- hand tendon disorder should not inherit cervical-style motor-control language;
- fracture/post-op rehabilitation must respect explicit loading/protocol restrictions;
- chronic OA may prioritize strength/function/activity rather than a universal ROM target;
- muscle injury should use criterion-based progressive loading rather than unsupported fixed timelines.

---

# 6. Safety / consistency engine v1

Initial candidate rules:

```text
fracture rehab + missing healing/weight-bearing context
→ warning

post-op + missing procedure/protocol/restrictions
→ warning

traction + no cervical radicular context
→ warning

dry needling + no myofascial/trigger-point context
→ soft warning

manual/passive techniques with no active rehabilitation direction
→ warning

gait training with no gait/function problem
→ soft warning

new/progressive neurological deficit
→ prominent medical reassessment warning

red-flag concern
→ require clinician acknowledgement before routine referral text is finalized
```

The engine prompts for consistency/safety; it does not autonomously prohibit clinician-selected treatment.

---

# 7. Output wording contract

Preferred referral structure:

```text
Clinical problem + important findings + functional impact.

Referral request + goals.

Rehabilitation direction + restrictions/precautions.

Optional reassessment / feedback criteria.
```

Rules:

- collaborative wording, not over-prescription of the physiotherapist;
- emphasize active rehabilitation, education/self-management and graded loading when appropriate;
- passive techniques remain adjunctive when selected;
- do not convert provocation tests into unsupported definitive diagnoses;
- do not state `no neurological deficit` or `no red flags` unless actually selected as assessed;
- preserve surgeon/healing restrictions explicitly;
- short and detailed outputs derive from the same `ReferralDraft`.

---

# 8. Persistence / patient linkage boundary

CU-1 does **not** freeze referral persistence yet.

Default first implementation direction:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Patient-aware prefill/history may follow only after the first generator works cleanly inside Clinical Excellence.

If persistence is later justified, reuse creates a **new referral draft** and never mutates historical referral truth.

---

# 9. Out of scope for CU-1

Do not yet:

- write production HTML/JS/CSS;
- integrate Cockpit navigation;
- add patient persistence/history;
- use AI to generate the referral;
- implement RF workflow changes;
- mutate `ortho-reception-backend-v2` while its AC-2 writer lock is active;
- alter Osteoporosis audit/PR-1 transcript code.

---

# 10. Design acceptance checklist

Before `IMPLEMENT` for CU-2, review/freeze:

```text
A. body-region taxonomy
B. condition profiles and naming
C. findings vs diagnosis separation
D. required context for fracture/post-op/high-risk pathways
E. goal/recovery-direction suggestions
F. safety/consistency warnings
G. ReferralDraft schema
H. short/detailed formatting rules
I. exact MVP fields to retain/remove/add
J. Clinical Excellence visual/navigation host
K. persistence decision for first implementation
L. evidence-review items that must be verified before production wording
```

---

# 11. Exact next action

The next fresh conversation should review and refine the detailed cervical design candidate first:

```text
clinic_utilities/physio_profiles/cervical_v1.md
```

It should challenge:

```text
primary problem taxonomy
→ findings/modifiers separation
→ neurological-screen semantics
→ safety prompts
→ goals
→ rehabilitation directions
→ adjunct-technique visibility
→ generated wording
```

After the product owner approves/freezes the cervical profile, continue in order:

```text
lumbar spine
→ shoulder
→ knee / hip
→ elbow
→ wrist / hand
→ ankle / foot
→ fracture / post-immobilization
→ muscle injury
→ post-operative / generalized deconditioning
```

Stop before runtime implementation and obtain explicit product-owner approval to move from CU-1 design to CU-2 implementation.
