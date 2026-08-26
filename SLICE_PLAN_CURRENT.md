# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** ACTIVE PRE-CODE DESIGN — runtime implementation not authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **Frozen shoulder profile:** `clinic_utilities/physio_profiles/shoulder_v1_1.md`.
> **Next regional design selection:** continue remaining CU-1 body-region sequence; product-owner choice before the next freeze.
> **Prior active slice:** PR-1 Transcript Intake + Candidate Extraction v3, intentionally paused and archived at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

CU-1 is a bounded product-owner-approved design detour. It does not cancel PR-1 and does not turn Clinic Utilities into a new clinical module.

---

# 1. Product outcome

Freeze the clinical/content and deterministic-generation contract for a substantially improved Physiotherapy Referral utility before implementation.

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

The utility improves referral quality/speed while preserving clinician judgment and physiotherapist autonomy.

---

# 2. Frozen architecture

Future implementation must build a deterministic structured object before prose:

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

Hard invariants:

```text
suggested != examined
suggested != selected
selected != mandatory
symptom != diagnosis
objective deficit != subjective symptom
special/provocation test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

---

# 3. Frozen regional status

## 3.1 Cervical — FROZEN v1.1

Authoritative file:

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
```

Key frozen semantics: clinician-entered cervicogenic headache/cervical-dizziness diagnoses; strict neurological tri-state semantics; directly selectable myofascial/trigger-point and referred shoulder-girdle findings; active-first rehabilitation; no routine cervical post-operative pathway.

## 3.2 Lumbar — FROZEN v1.1

Authoritative file:

```text
clinic_utilities/physio_profiles/lumbar_v1_1.md
```

Frozen pathways include non-specific/mechanical LBP, radiating/radicular features, stenosis/neurogenic claudication and deep-gluteal/piriformis presentation. Lumbar preserves cauda-equina safety semantics, optional acupuncture/dry needling with evidence/competence caveats, no routine traction, no SI-dysfunction lumbar diagnosis and no active lumbar post-operative pathway.

## 3.3 Shoulder — FROZEN v1.1

Authoritative file:

```text
clinic_utilities/physio_profiles/shoulder_v1_1.md
```

Frozen primary pathways:

```text
S1 rotator-cuff-related shoulder pain / rotator-cuff tendinopathy
S2 confirmed full-thickness rotator-cuff tear — conservative rehabilitation
S3 calcific rotator-cuff tendinopathy
S4 adhesive capsulitis / frozen shoulder
S5 glenohumeral instability / dislocation rehabilitation
S6 glenohumeral osteoarthritis
S7 post-traumatic assessed shoulder injury
S8 acromioclavicular-joint disorder
S9 sternoclavicular-joint disorder
S10 postoperative shoulder rehabilitation
```

Frozen shoulder decisions:

- `impingement syndrome` is not the preferred top-level diagnosis;
- active and passive ROM remain distinct;
- special tests, painful arc and scapular findings remain findings, not diagnoses;
- pain-inhibited effort does not establish rotator-cuff tear;
- full-thickness tear requires established clinician/imaging context;
- acute trauma with marked new weakness/inability to elevate requires explicit reassessment semantics;
- adhesive capsulitis and instability are never inferred from isolated findings;
- long-head biceps tendinopathy is a common directly selectable secondary/coexisting diagnosis rather than a default primary pathway;
- AC-joint pathology is available as a primary pathway, including isolated load/weight-training presentations;
- sternoclavicular pathology is available as a primary pathway, with high-priority reassessment for suspected posterior dislocation or unexplained swelling/systemic concern;
- postoperative shoulder rehabilitation is part of the active shoulder MVP and requires procedure/protocol/restriction context;
- acupuncture and dry needling remain optional adjuncts with competence/availability safeguards;
- ESWT is a calcific-specific optional adjunct; prior barbotage/lavage is context and does not create an automatic sequencing rule;
- therapeutic ultrasound is not a standard rotator-cuff adjunct in the current profile;
- shoulder-region fractures route to a shared fracture/post-immobilization profile rather than duplicating fracture logic inside shoulder;
- active rehabilitation, education, graded loading/activity and physiotherapist autonomy remain the conceptual backbone.

Evidence-sensitive technique/protocol wording must be rechecked immediately before CU-2 implementation.

---

# 4. Shared fracture / post-immobilization boundary

A future shared profile will cover region-specific fracture rehabilitation, including proximal humerus, clavicle and scapula, and will require explicit healing/stability, immobilization, loading/ROM restrictions and orthopaedic instructions.

```text
fracture/post-immobilization
+ unresolved healing/loading context
→ warning
→ no unrestricted routine rehabilitation wording
```

---

# 5. Context-sensitive defaults

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms or changes
→ only confirmed values populate ReferralDraft
```

No global pain + ROM + strength + motor-control bundle.

---

# 6. Safety / consistency engine

The engine provides clinician-facing prompts, not autonomous diagnoses or treatment prohibitions.

Cross-region rules include:

```text
fracture/post-trauma + unresolved structural/healing context
→ warning/reassessment prompt

post-op pathway + missing procedure/protocol/restrictions
→ warning

adjunct selected without active rehabilitation direction
→ warning

new/progressive objective neurological deficit
→ prominent medical reassessment prompt

material safety concern + no clinician disposition
→ do not generate routine reassuring wording

not assessed neurological component
→ never generate normal wording
```

Region-specific rules live in each frozen profile.

---

# 7. Output wording contract

```text
Clinical problem + important findings + functional impact.
Referral request + goals.
Rehabilitation direction + restrictions/precautions.
Optional reassessment/feedback criteria.
```

Rules:

- collaborative wording, not over-prescription;
- active rehabilitation, education/self-management and graded activity/loading where appropriate;
- technique-level interventions remain adjunctive;
- no unsupported diagnosis from tests, symptoms or incidental imaging;
- no negative neurological/red-flag/structural statements from missing or unassessed data;
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
A. remaining body-region taxonomy
B. remaining condition profiles and naming
C. findings vs diagnosis separation
D. fracture/post-op/high-risk context
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

Cervical, lumbar and shoulder v1.1 are frozen.

```text
1. close shoulder docs-only freeze/handoff
2. product owner selects the next remaining regional profile
3. apply the same strict taxonomy/findings/safety/goals/rehab/evidence method
4. continue CU-1 design only
```

Do not write runtime code. Explicit product-owner authorization remains required before transition from CU-1 design to CU-2 implementation.
