# Lumbar Physiotherapy Referral Profile v1 — CU-1 design candidate

> **STATUS:** DESIGN CANDIDATE — product-owner review required before freeze.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful lumbar referral choices while preserving diagnosis-vs-finding separation, explicit neurological/safety semantics, active rehabilitation, and physiotherapist autonomy.
> **Prior body-region freeze:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.

---

# 1. Proposed lumbar architecture

```text
PRIMARY CLINICAL PATHWAY
+
ACTUAL FINDINGS / MODIFIERS
+
FUNCTIONAL IMPACT
+
SAFETY / REASSESSMENT CONTEXT
+
CONFIRMED GOALS
+
CONFIRMED REHABILITATION DIRECTIONS
```

Hard invariants inherited from the frozen cervical profile:

```text
suggested != examined
suggested != selected
symptom != objective deficit
provocation/neural-tension test != diagnosis
not assessed != normal
adjunct != core rehabilitation
```

---

# 2. Proposed primary lumbar pathways

## L1 — Non-specific / mechanical low-back pain

Structured key:

```text
nonspecific_low_back_pain
```

Display:

> Μη ειδική / μηχανικού τύπου οσφυαλγία

Use for axial/mechanical low-back pain without a more specific radicular, stenotic, fracture, traumatic or other defined pathway.

Mobility restriction, muscular tenderness, load sensitivity, trunk deconditioning and ergonomic/workload factors remain modifiers/findings rather than separate top-level diagnoses.

## L2 — Low-back pain with radiating leg symptoms / radicular features

Structured key:

```text
low_back_pain_with_radiating_leg_symptoms
```

Display:

> Οσφυαλγία με ακτινοβολία στο κάτω άκρο / ριζιτικού τύπου χαρακτηριστικά

Required semantic separation:

```text
subjective radiating leg pain / paresthesia / numbness
!=
objective motor/sensory/reflex deficit
!=
formal lumbar radiculopathy diagnosis
```

Optional clinician assertion:

```text
formal_lumbar_radiculopathy_diagnosis: yes / no / not_stated
```

Straight-leg raise, slump or other neural-tension findings must never cause the utility to assert lumbar radiculopathy automatically.

## L3 — Lumbar spinal stenosis / neurogenic claudication pathway

Structured key:

```text
lumbar_spinal_stenosis_neurogenic_claudication
```

Default display:

> Συμπτωματολογία συμβατή με οσφυϊκή στένωση / νευρογενή διαλείπουσα χωλότητα

Optional clinician assertion:

```text
formal_lumbar_spinal_stenosis_diagnosis: yes / no / not_stated
```

If a formal diagnosis is selected, generated wording may state it explicitly. Otherwise the formatter should describe the actual symptom/function pattern without manufacturing an imaging or stenosis diagnosis.

Useful context may include:

```text
walking/standing-limited leg symptoms
relief with sitting/flexion where reported
walking distance/tolerance
unilateral/bilateral symptoms
known imaging context optional
```

The utility must not infer stenosis from age plus walking limitation.

## Shared pathways not embedded in the lumbar MVP

The following remain available as separate/shared Clinic Utility pathways rather than ordinary lumbar primary diagnoses:

```text
vertebral/spinal fracture or post-immobilization
major post-traumatic spinal presentation
post-operative musculoskeletal rehabilitation
```

They require their own healing/protocol/restriction context and should not inherit unrestricted routine low-back wording.

---

# 3. Findings / modifiers — only when actually assessed or elicited

## 3.1 Pain distribution / symptom behaviour

```text
axial low-back pain
unilateral/bilateral buttock pain
referred non-radicular leg pain
radiating leg pain
paresthesia
numbness
movement/load-related aggravation
sitting intolerance
standing intolerance
walking intolerance
bending/lifting aggravation
recurrent/load-related presentation
night/sleep disturbance
```

Referred buttock/leg pain remains a symptom-distribution modifier unless the clinician selects a formal diagnosis separately.

## 3.2 Lumbar mobility / movement findings

```text
active ROM restricted
painful ROM
specific directional restriction optional
movement-direction preference/response if actually assessed
```

No ROM impairment is globally preselected.

## 3.3 Myofascial / muscular findings

```text
paraspinal tenderness
increased muscular tone
clinically active trigger points
gluteal/myofascial tenderness
trunk strength/endurance deficit where assessed
```

These remain findings/presentation modifiers rather than automatic separate diagnoses.

## 3.4 Neural/radicular provocation findings

Optional only when examined:

```text
straight-leg raise reproduces concordant symptoms
slump/neural-tension test reproduces concordant symptoms
other clinician-entered neural finding
```

A positive neural-tension test is not a standalone radiculopathy diagnosis.

---

# 4. Neurological-screen model

Subjective neural symptoms and objective neurological findings remain separate.

```text
subjective_neural_symptoms
  radiating_leg_pain
  paresthesia
  numbness
  distribution / laterality

objective_root_screen
  motor: normal / abnormal / not_assessed
  sensory: normal / abnormal / not_assessed
  reflexes: normal / abnormal / not_assessed

objective_details
  motor_detail_optional
  sensory_detail_optional
  reflex_detail_optional
```

Hard invariant:

```text
NOT_ASSESSED
→ must never generate NORMAL wording
```

No global `no neurological deficit` checkbox.

---

# 5. Safety / urgent reassessment semantics

The utility prompts the clinician; it does not diagnose cauda equina syndrome, malignancy, infection, fracture or other serious pathology.

## 5.1 High-priority neurological concerns

Candidate clinician-entered concerns:

```text
new_or_progressive_objective_motor_deficit
progressive_or_expanding_sensory_loss
new_bladder_dysfunction_concern
new_bowel_dysfunction_concern
new_sexual_function_change_concern
new_perineal_or_saddle_sensory_change
rapidly_progressive_bilateral_neurological_symptoms
other_cauda_equina_or_major_neurological_concern
```

Any selected cauda-equina-type concern should generate a high-priority clinician reassessment/urgent-pathway prompt rather than routine reassuring referral wording.

## 5.2 Other material safety concerns

```text
significant/relevant trauma or possible fracture
systemic/infectious/malignancy/inflammatory concern
severe/unremitting/progressive non-mechanical pain or other clinician red-flag concern
other material clinician concern
```

## 5.3 Safety state and clinician disposition

```text
safety_screen_status:
  not_assessed
  no_specific_concern_identified
  concern_present

clinician_disposition when concern present:
  reviewed_and_appropriate_to_proceed
  urgent_or_same_day_medical_assessment_arranged
  specialist_or_imaging_pathway_underway
  routine_physiotherapy_deferred
  other
```

No default `no red flags` generated sentence.

---

# 6. Functional limitations

Candidate fields:

```text
sitting tolerance
standing tolerance
walking tolerance/distance
sit-to-stand / transfers
bending
lifting/carrying
stairs where relevant
sleep
work tolerance / work absence
driving
exercise/sport
ADLs/self-care
patient-priority activity / free text
```

For L3 neurogenic-claudication presentations, walking/standing tolerance should be prominent but still clinician-confirmed.

---

# 7. Context-sensitive goal suggestions

Nothing is globally preselected.

## L1 — Non-specific/mechanical low-back pain

Possible suggestions:

- reduce symptom irritability;
- remain/return active;
- restore functional mobility where impaired;
- improve trunk/hip strength and endurance where relevant;
- improve tolerance of sitting/standing/walking/lifting as applicable;
- graded return to work/exercise/ADLs;
- improve self-management and recurrence-management capability.

## L2 — Radiating/radicular features

Possible suggestions:

- reduce back/leg symptom irritability;
- improve walking and activity tolerance;
- preserve/restore strength and function where appropriate;
- graded return to relevant activity;
- neural mobility/neurodynamic rehabilitation where indicated;
- monitor neurological status during rehabilitation.

Do not promise reversal of an objective neurological deficit.

## L3 — Lumbar stenosis / neurogenic claudication

Possible suggestions:

- improve walking/standing tolerance;
- improve lower-limb/trunk strength and conditioning where relevant;
- improve mobility/function according to the clinical presentation;
- graded physical activity and self-management;
- improve confidence and participation in daily activity.

---

# 8. Rehabilitation directions

## 8.1 Core active directions

```text
physiotherapy assessment and individualized active rehabilitation
structured therapeutic exercise / progressive activity
strength/endurance/conditioning where relevant
mobility exercise where restricted and clinically useful
graded loading and return to activity
education and self-management
home exercise programme where appropriate
work/activity/load adaptation where relevant
```

The utility should not prescribe one exercise school or a fixed stabilization programme to every patient.

## 8.2 Optional adjunct expander

Candidate adjuncts for design review:

```text
manual therapy / mobilization
soft-tissue techniques
neurodynamic techniques when neural/radiating context exists
dry needling / needling where clinically relevant and consistent with the chosen evidence framework
```

### Lumbar traction

Routine lumbar traction should **not** be offered as a standard adjunct in this profile. NICE recommends against traction for low-back pain with or without sciatica, and WHO recommends against routine traction for chronic primary low-back pain.

If the product owner has a specific exceptional clinical use case, it should require an explicit evidence/framework decision rather than appearing by default.

### Acupuncture / needling framework conflict

NICE NG59 recommends against acupuncture for low-back pain/sciatica, while the 2023 WHO chronic-primary-LBP guideline conditionally allows needling therapies in some contexts.

Therefore CU-1 must not create a silent hybrid rule. Before production freeze, choose whether:

```text
A. omit acupuncture from the lumbar MVP;
B. expose needling only under a named evidence-framework policy; or
C. retain generic clinician-selected needling without implying a guideline recommendation.
```

---

# 9. Deterministic consistency rules

```text
L2 selected
+ motor/sensory/reflex all not_assessed
→ prompt: consider documenting current neurological status

SLR/slump positive
+ no radiating/neural symptoms
→ soft warning: provocation finding alone does not establish radiculopathy

formal_lumbar_radiculopathy_diagnosis != yes
→ do not output definitive lumbar radiculopathy

L3 selected
+ formal stenosis diagnosis != yes
→ describe symptom pattern; do not assert imaging-confirmed stenosis

neurodynamic technique
+ no neural/radiating context
→ soft warning

any adjunct
+ no active/function-oriented rehabilitation direction
→ warning

new/progressive objective neurological deficit
→ high-priority medical reassessment prompt

new bladder/bowel/sexual dysfunction or perineal sensory change concern
→ high-priority urgent medical assessment prompt

material safety concern
+ no clinician disposition
→ do not generate routine reassuring wording

no selected safety concern
→ must not generate `no red flags`
```

---

# 10. Generated wording examples

## 10.1 Short — non-specific/mechanical low-back pain

> Μη ειδική / μηχανικού τύπου οσφυαλγία με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη ενεργητική αποκατάσταση με στόχο [selected goals], με έμφαση στη διατήρηση/επάνοδο στη δραστηριότητα, στην άσκηση, στην εκπαίδευση και στην αυτοδιαχείριση. [Selected additional directions if present.]

## 10.2 Radiating leg symptoms with incomplete neurological assessment

> Οσφυαλγία με ακτινοβολία/παραισθησίες προς το [side] κάτω άκρο, με [only documented findings]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένο ενεργητικό πρόγραμμα αποκατάστασης με προσαρμογή στην ερεθιστικότητα και τη λειτουργική εικόνα.

No negative neurological statement is generated from missing data.

## 10.3 Radiating symptoms with assessed normal motor and sensory findings

Only when `motor=normal` and `sensory=normal`:

> Κατά τον παρόντα έλεγχο δεν διαπιστώθηκε κινητικό ή αισθητικό έλλειμμα.

Reflexes are mentioned only if assessed.

## 10.4 Objective neurological deficit

> Οσφυαλγία με ακτινοβολία προς το [side] κάτω άκρο και [selected radicular features]. Κατά τον παρόντα έλεγχο καταγράφηκε [specific selected motor/sensory/reflex finding]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη αποκατάσταση με προσαρμογή στην κλινική εικόνα. Ιατρική επανεκτίμηση σε περίπτωση νέας ή περαιτέρω προοδευτικής νευρολογικής μεταβολής.

If the deficit is already progressive, the safety/disposition layer takes precedence.

## 10.5 Formal lumbar stenosis / neurogenic claudication

Only when the clinician has explicitly asserted the diagnosis:

> Οσφυϊκή στένωση με νευρογενή διαλείπουσα χωλότητα, με λειτουργικό περιορισμό κυρίως σε [walking/standing/other selected activities] και [selected findings]. Παρακαλώ για εξατομικευμένο πρόγραμμα αποκατάστασης με έμφαση στη βελτίωση της λειτουργικής ικανότητας, της βάδισης/αντοχής και της φυσικής δραστηριότητας, σύμφωνα με τη φυσιοθεραπευτική αξιολόγηση.

---

# 11. Current evidence boundary

Stable structural design supported across major guidance:

- remain active / self-management / exercise are central components of non-operative low-back care;
- manual therapy, where used, should not displace an active rehabilitation programme;
- routine lumbar traction should not be a default treatment recommendation;
- symptoms and provocation tests must not be converted automatically into radiculopathy;
- progressive neurological change and cauda-equina-type symptoms require explicit medical reassessment semantics;
- serious-pathology screening remains clinician judgment, not an automated diagnostic score.

Evidence/framework-sensitive items requiring explicit production review:

- exact exercise subtype recommendations by chronicity/presentation;
- manual therapy subgroup wording;
- dry needling/acupuncture policy because NICE and WHO differ;
- detailed non-surgical stenosis intervention wording;
- any newer CPG superseding NICE NG59, WHO 2023 or the 2021 PT/stenosis guidance before CU-2 production implementation.

---

# 12. Product-owner review questions before freeze

1. Keep the three main lumbar pathways proposed here: non-specific/mechanical LBP; radiating/radicular-feature LBP; lumbar stenosis/neurogenic claudication?
2. Do you commonly write physiotherapy for a formal diagnosis of lumbar spinal stenosis/neurogenic claudication, or should that remain secondary?
3. Do you want `myofascial/trigger-point dominant lumbar pain` and referred buttock pain directly selectable exactly as in the cervical profile?
4. Do you want a separate sacroiliac-region pathway, or should that be designed later as pelvis/hip rather than lumbar?
5. Should acupuncture/needling be omitted from the lumbar MVP because of the NICE/WHO framework conflict, while dry needling remains an optional clinician-selected myofascial adjunct?
6. Are lumbar post-operative referrals also outside your real workflow, as cervical post-operative referrals are?

No runtime implementation is authorized by this candidate.
