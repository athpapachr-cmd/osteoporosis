# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** ACTIVE PRE-CODE DESIGN — implementation not yet authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Current detailed profile under review:** `clinic_utilities/physio_profiles/lumbar_v1.md` — design candidate, not frozen.
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

Hard invariants:

```text
suggested != examined
suggested != selected
selected != mandatory
symptom != objective deficit
provocation test != diagnosis
not assessed != normal
adjunct != core rehabilitation
```

The UI may recommend relevant fields but must never silently claim that an examination finding occurred or infer a diagnosis.

---

# 4. Body-region profile sequence and status

Design sequence remains:

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
- post-operative musculoskeletal rehabilitation where relevant;
- generalized deconditioning / balance / gait.

These profiles are clinical-content designs, not automatically evidence-frozen production rules.

## 4.1 Cervical profile — FROZEN v1.1

The product owner approved and froze:

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
```

Frozen cervical pathways:

```text
non-specific / mechanical neck pain
neck pain with radiating upper-limb / radicular features
headache with cervical musculoskeletal features
  + explicit formal cervicogenic-headache clinician assertion
cervical/cervicogenic dizziness presentation
  + explicit clinician diagnosis assertion
whiplash / post-traumatic neck pain
```

Cervical post-operative rehabilitation is deliberately not part of the active cervical MVP because it is not part of the product owner's current clinical workflow; a shared post-operative pathway may remain available elsewhere in Clinic Utilities.

Frozen cervical semantic decisions:

- Spurling or radiating pain alone does not automatically become radiculopathy;
- trigger-point/myofascial findings and referred shoulder-girdle pain are directly selectable clinically useful findings/presentation modifiers, not automatically inferred diagnoses;
- formal cervicogenic headache may be stated only when explicitly asserted by the clinician;
- cervical/cervicogenic dizziness may be stated only when explicitly asserted by the clinician; the utility must not infer cervical causation from neck pain plus dizziness;
- neurological screen is component-level tri-state and preserves `not assessed != normal`;
- subjective radiating symptoms remain separate from objective motor/sensory/reflex deficits;
- progressive neurological deficit, possible cord/myelopathy concern, trauma/instability concern or other material safety concern produces clinician-facing reassessment/disposition prompts;
- there is no global `no neurological deficit` or default `no red flags` output;
- goals are context-sensitive and never globally preselected;
- active rehabilitation/education/self-management is the conceptual backbone;
- technique-level adjuncts are optional and live in a secondary expander;
- short/detailed wording derives only from confirmed `ReferralDraft` values.

Cervical evidence-sensitive technique wording must be rechecked immediately before CU-2 production implementation, especially while the APTA/JOSPT neck-pain CPG is under revision and because cervical-dizziness diagnostic/therapeutic evidence remains uncertain.

## 4.2 Lumbar profile — ACTIVE DESIGN CANDIDATE

Current design candidate:

```text
clinic_utilities/physio_profiles/lumbar_v1.md
```

It proposes three main pathways for product-owner review:

```text
non-specific / mechanical low-back pain
low-back pain with radiating leg symptoms / radicular features
lumbar spinal stenosis / neurogenic claudication pathway
```

Mobility restriction, load/postural aggravation, trunk deconditioning, myofascial/trigger-point findings and referred buttock/leg pain are treated primarily as findings/modifiers rather than equivalent top-level diagnoses.

The candidate inherits cervical safety semantics, adds explicit cauda-equina-type concern/disposition handling, and does not include routine lumbar traction as a default adjunct because major guidance recommends against it.

Lumbar profile remains **not frozen** pending product-owner review.

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

The engine provides consistency/safety prompts, not autonomous diagnostic decisions or treatment prohibitions.

Cross-cutting candidate rules include:

```text
fracture rehab + missing healing/weight-bearing context
→ warning

post-op + missing procedure/protocol/restrictions
→ warning

manual/passive adjunct selected without active rehabilitation direction
→ warning

gait training selected without gait/function problem
→ soft warning

new/progressive objective neurological deficit
→ prominent medical reassessment warning

material red-flag/safety concern
→ require explicit clinician disposition before routine reassuring wording

not assessed neurological component
→ never generate normal wording
```

Region-specific rules belong in each frozen profile.

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
- emphasize active rehabilitation, education/self-management and graded loading/activity when appropriate;
- passive/technique-level interventions remain adjunctive when selected;
- do not convert provocation tests into unsupported definitive diagnoses;
- do not generate negative neurological or red-flag statements from missing/unassessed data;
- preserve explicit surgeon/healing restrictions where relevant;
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

Cervical review/freeze is closed at v1.1.

Current exact next action:

```text
1. critically review clinic_utilities/physio_profiles/lumbar_v1.md
2. challenge primary-pathway taxonomy
3. challenge findings/modifiers separation
4. challenge neurological and cauda-equina safety semantics
5. challenge functional-limit fields and goals
6. challenge rehabilitation directions and adjunct visibility
7. resolve the NICE-vs-WHO needling/acupuncture framework question
8. review generated short/detailed wording
9. after product-owner approval, freeze lumbar profile
10. then proceed to shoulder
```

Stop before runtime implementation and obtain explicit product-owner approval to move from CU-1 design to CU-2 implementation.
