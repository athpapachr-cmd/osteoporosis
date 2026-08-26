# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** ACTIVE PRE-CODE DESIGN — runtime implementation not authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **Next body-region design target:** shoulder.
> **Prior active slice:** PR-1 Transcript Intake + Candidate Extraction v3, intentionally paused and archived at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

CU-1 is a bounded product-owner-approved design detour. It does not cancel PR-1 and does not turn Clinic Utilities into a new clinical module.

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

The utility should improve referral quality and speed while preserving clinician judgment and physiotherapist autonomy.

---

# 2. Source baseline

The supplied standalone HTML was inspected read-only.

Useful behavior to preserve:

- body-region condition groups;
- optional laterality/chronicity/session count;
- clinical findings;
- goals;
- active vs adjunct interventions;
- short/detailed outputs;
- copy/print;
- local/no-server behavior;
- consistency warnings;
- evidence/reference section.

Problems to correct:

- checkbox-catalogue flow rather than clinical referral flow;
- generic findings across unrelated conditions;
- globally preselected goals/interventions;
- insufficient condition-specific restrictions/precautions;
- direct phrase concatenation instead of a structured intermediate model;
- limited consistency/safety rules;
- incomplete common pathways;
- standalone visual identity rather than Clinical Excellence styling.

---

# 3. Frozen architectural direction

## 3.1 Structured intermediate object

Future implementation must build a deterministic `ReferralDraft` before prose formatting:

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

No direct checkbox-to-prose monolith.

## 3.2 Condition profiles

A profile may define:

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

Hard invariants:

```text
suggested != examined
suggested != selected
selected != mandatory
symptom != objective deficit
provocation test != diagnosis
not assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

---

# 4. Body-region profile status

## 4.1 Cervical — FROZEN v1.1

Authoritative file:

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
```

Frozen pathways:

- non-specific/mechanical neck pain;
- neck pain with radiating upper-limb/radicular features;
- headache with cervical musculoskeletal features + optional formal cervicogenic-headache diagnosis;
- cervical/cervicogenic dizziness presentation + optional clinician diagnosis;
- whiplash/post-traumatic neck pain.

Key frozen semantics include component-level tri-state neurological status; no `not assessed → normal`; direct selection of myofascial/trigger-point and referred shoulder-girdle findings; active-first rehabilitation; adjunct techniques under secondary visibility; and no routine cervical post-operative pathway.

## 4.2 Lumbar — FROZEN v1.1

Authoritative file:

```text
clinic_utilities/physio_profiles/lumbar_v1_1.md
```

Frozen primary pathways:

```text
L1 non-specific / mechanical low-back pain
L2 low-back pain with radiating leg symptoms / radicular features
L3 lumbar spinal stenosis / neurogenic claudication
L4 deep-gluteal / piriformis presentation
```

Frozen lumbar decisions:

- radicular symptoms remain separate from objective motor/sensory/reflex deficits and from formal radiculopathy diagnosis;
- motor/sensory/reflex remain tri-state with `not assessed != normal`;
- SLR/slump findings never create a diagnosis automatically;
- cauda-equina-type concerns use high-priority clinician reassessment/disposition semantics without autonomous diagnosis;
- myofascial/trigger-point and referred buttock/leg findings are directly selectable modifiers;
- deep-gluteal/piriformis syndrome may be stated only when explicitly asserted by the clinician;
- `SI dysfunction` is not a lumbar diagnosis;
- SI-region/SIJ pathology is reserved for a future separate SI/pelvic profile;
- MRI may support sacroiliitis/defined structural pathology but must not automatically identify a mechanically painful SI joint as the pain generator;
- acupuncture remains an optional clinician-selected adjunct with explicit NICE-vs-WHO evidence-framework transparency;
- dry needling remains optional with an explicit competence/availability caveat;
- routine lumbar traction is excluded from the MVP;
- lumbar post-operative rehabilitation is excluded from the active lumbar MVP because it is not part of the product owner's current workflow;
- active rehabilitation, exercise, education and self-management remain the conceptual backbone.

Evidence-sensitive technique wording must be rechecked immediately before CU-2 implementation.

## 4.3 Next target — Shoulder

After lumbar freeze, CU-1 proceeds to shoulder design using the same method:

```text
primary pathway taxonomy
→ findings vs diagnosis separation
→ safety/reassessment semantics
→ functional limitations
→ context-sensitive goals
→ active rehabilitation directions
→ adjunct visibility
→ generated wording
→ evidence check
→ product-owner freeze
```

---

# 5. Context-sensitive defaults

Remove the global assumption that pain + ROM + strength + motor control + function and a fixed intervention bundle apply to every referral.

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms or changes
→ only confirmed values populate ReferralDraft
```

---

# 6. Safety / consistency engine

The engine provides clinician-facing consistency/safety prompts, not autonomous diagnostic decisions or treatment prohibitions.

Cross-cutting rules include:

```text
fracture rehab + missing healing/weight-bearing context
→ warning

post-op pathway + missing procedure/protocol/restrictions
→ warning

adjunct selected without active rehabilitation direction
→ warning

new/progressive objective neurological deficit
→ prominent medical reassessment prompt

material safety concern
→ require explicit clinician disposition before routine reassuring wording

not assessed neurological component
→ never generate normal wording
```

Region-specific rules belong in each frozen profile.

---

# 7. Output wording contract

Preferred structure:

```text
Clinical problem + important findings + functional impact.
Referral request + goals.
Rehabilitation direction + restrictions/precautions.
Optional reassessment/feedback criteria.
```

Rules:

- collaborative wording, not over-prescription of the physiotherapist;
- active rehabilitation, education/self-management and graded activity/loading where appropriate;
- passive/technique-level interventions remain adjunctive;
- no unsupported diagnosis from provocation tests or symptom combinations;
- no negative neurological/red-flag statements from missing or unassessed data;
- preserve explicit restrictions;
- short and detailed outputs derive from the same `ReferralDraft`.

---

# 8. Persistence / patient-linkage boundary

CU-1 does **not** freeze referral persistence yet.

Default first implementation direction:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

If persistence is later justified, reuse creates a new referral draft and never mutates historical referral truth.

---

# 9. Out of scope for CU-1

Do not yet:

- write production HTML/JS/CSS;
- integrate Clinical Excellence navigation;
- add patient persistence/history;
- use AI to generate the referral;
- implement RF workflow changes;
- alter Osteoporosis audit/PR-1 transcript runtime;
- create overlapping runtime writers.

---

# 10. Design acceptance checklist before CU-2

```text
A. body-region taxonomy
B. condition profiles and naming
C. findings vs diagnosis separation
D. high-risk/fracture/protocol context where relevant
E. goals and rehabilitation directions
F. safety/consistency warnings
G. ReferralDraft schema
H. short/detailed formatting rules
I. exact MVP fields
J. Clinical Excellence visual/navigation host
K. persistence decision for first implementation
L. final evidence check for production wording
```

---

# 11. Exact next action

Cervical v1.1 and lumbar v1.1 are frozen.

Next:

```text
1. design shoulder profile v1
2. clinically/structurally review it with the product owner
3. freeze shoulder after approval
4. continue through the remaining body-region sequence
```

Do not write runtime code. Explicit product-owner authorization is still required before any transition from CU-1 design to CU-2 implementation.