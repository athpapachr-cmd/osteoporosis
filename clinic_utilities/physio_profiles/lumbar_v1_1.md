# Lumbar Physiotherapy Referral Profile v1.1 — CU-1 FROZEN DESIGN

> **STATUS:** FROZEN CLINICAL/CONTENT DESIGN — product-owner approved 2026-08-26.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful lumbar referral choices while preserving diagnosis-vs-finding separation, explicit neurological/safety semantics, active rehabilitation, and physiotherapist autonomy.
> **Supersedes as active lumbar design:** `clinic_utilities/physio_profiles/lumbar_v1.md`.
> **Prior body-region freeze:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.

---

# 1. Core design contract

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

Hard invariants:

```text
suggested != examined
suggested != selected
symptom != objective deficit
provocation/neural-tension test != diagnosis
not assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

The utility structures a referral and checks internal consistency. It does not diagnose cauda equina syndrome, radiculopathy, spinal stenosis, sacroiliac-joint pain or deep-gluteal/piriformis pathology autonomously.

---

# 2. Primary lumbar pathways

## L1 — Non-specific / mechanical low-back pain

Structured key:

```text
nonspecific_low_back_pain
```

Display:

> Μη ειδική / μηχανικού τύπου οσφυαλγία

Use for axial/mechanical low-back pain without a more specific radicular, stenotic, fracture, inflammatory, traumatic or other defined pathway.

Mobility restriction, load sensitivity, trunk deconditioning, muscular tenderness, trigger points and work/postural aggravation remain findings/modifiers rather than top-level diagnoses.

## L2 — Low-back pain with radiating leg symptoms / radicular features

Structured key:

```text
low_back_pain_with_radiating_leg_symptoms
```

Display:

> Οσφυαλγία με ακτινοβολία στο κάτω άκρο / ριζιτικού τύπου χαρακτηριστικά

Mandatory semantic separation:

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

SLR, slump or another neural-tension finding must never cause the utility to assert radiculopathy automatically.

## L3 — Lumbar spinal stenosis / neurogenic claudication

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

If `yes`, generated wording may state the formal diagnosis. Otherwise describe only the actual symptom/function pattern.

Useful context:

```text
walking/standing-limited leg symptoms
walking distance/tolerance
relief with sitting/flexion where reported
unilateral/bilateral symptoms
known imaging context optional
```

Age plus walking limitation never creates an automatic stenosis diagnosis.

## L4 — Deep-gluteal / piriformis pathway

This pathway exists because piriformis/deep-gluteal presentations occur in the product owner's real referral workflow and can mimic lumbar referred/radicular pain.

Structured key:

```text
deep_gluteal_piriformis_presentation
```

Default display:

> Οπίσθιο γλουτιαίο άλγος / χαρακτηριστικά deep-gluteal ή απιοειδούς

Optional clinician assertion:

```text
formal_deep_gluteal_diagnosis:
  not_stated
  deep_gluteal_syndrome
  piriformis_syndrome
```

If explicitly selected, generated wording may state the clinician's formal diagnosis. Otherwise the utility describes the presentation only.

Useful findings/context may include:

```text
posterior/deep gluteal pain
difficulty or pain with prolonged sitting
sciatic-type symptoms without established lumbar root diagnosis
deep-gluteal/piriformis-region tenderness
provocation findings actually examined
hip/pelvic imaging context optional
```

The utility must not infer piriformis syndrome from buttock pain or tenderness alone.

---

# 3. Sacroiliac-region boundary — NOT a lumbar diagnosis

`SI dysfunction` is deliberately not part of the lumbar diagnosis taxonomy.

A future separate SI/pelvic profile may distinguish:

```text
sacroiliac-region pain / suspected SIJ-related pain
formal clinician diagnosis of SIJ-related pain
imaging-confirmed sacroiliitis or other defined structural/inflammatory SI pathology
```

Important evidence boundary:

- MRI is useful for detecting sacroiliitis/inflammatory or other structural pathology;
- MRI does not, by itself, reliably confirm that a mechanically painful SI joint is the pain generator;
- referred lumbar/gluteal pain must remain a competing clinical explanation unless the clinician has established a more specific diagnosis.

Therefore lumbar referral wording must not convert SI-region pain or an incidental SI imaging finding into `SI dysfunction`.

---

# 4. Findings / modifiers — only when actually assessed or elicited

## 4.1 Pain distribution / symptom behaviour

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

## 4.2 Lumbar mobility / movement findings

```text
active ROM restricted
painful ROM
specific directional restriction optional
movement-direction preference/response if actually assessed
```

## 4.3 Myofascial / muscular findings

Directly selectable:

```text
paraspinal tenderness
increased muscular tone
clinically active trigger points
gluteal/myofascial tenderness
trunk strength/endurance deficit where assessed
```

They remain findings/presentation modifiers by default and may appear in referral wording when actually selected.

## 4.4 Neural/radicular provocation findings

```text
straight-leg raise reproduces concordant symptoms
slump/neural-tension test reproduces concordant symptoms
other clinician-entered neural finding
```

A positive test is a finding, not a diagnosis.

---

# 5. Neurological-screen model

Subjective neural symptoms and objective findings remain separate.

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

There is no global `no neurological deficit` checkbox.

---

# 6. Safety / urgent reassessment semantics

The safety layer supports clinician consistency; it does not diagnose serious pathology.

## 6.1 High-priority neurological concerns

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

Any cauda-equina-type concern produces a high-priority medical/urgent-pathway prompt rather than routine reassuring referral wording.

## 6.2 Other material concerns

```text
significant/relevant trauma or possible fracture
systemic/infectious/malignancy/inflammatory concern
severe/unremitting/progressive non-mechanical pain or other clinician red-flag concern
other material clinician concern
```

## 6.3 Safety state and clinician disposition

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

No default `no red flags` sentence is generated.

---

# 7. Functional limitations

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

For L3, walking/standing tolerance is prominent but remains clinician-confirmed.

For L4, prolonged sitting and the patient's priority activity should be easy to capture.

---

# 8. Context-sensitive goals

Nothing is globally preselected.

## L1 — Non-specific/mechanical LBP

Possible suggestions:

- reduce symptom irritability;
- remain/return active;
- restore functional mobility where impaired;
- improve trunk/hip strength/endurance where relevant;
- improve tolerance of sitting/standing/walking/lifting as applicable;
- graded return to work/exercise/ADLs;
- improve self-management and recurrence-management capability.

## L2 — Radiating/radicular features

Possible suggestions:

- reduce back/leg symptom irritability;
- improve walking/activity tolerance;
- preserve/restore strength and function where appropriate;
- graded return to activity;
- neural mobility/neurodynamic rehabilitation where indicated;
- monitor neurological status during rehabilitation.

Do not promise reversal of an objective neurological deficit.

## L3 — Stenosis / neurogenic claudication

Possible suggestions:

- improve walking/standing tolerance;
- improve lower-limb/trunk strength and conditioning where relevant;
- improve mobility/function according to presentation;
- graded physical activity and self-management;
- improve participation in daily activity.

## L4 — Deep-gluteal / piriformis pathway

Possible suggestions:

- reduce posterior-gluteal/sciatic-type symptom irritability;
- improve sitting tolerance;
- improve hip/lumbopelvic strength, mobility or load tolerance where impaired;
- graded return to walking/exercise/work;
- improve self-management.

---

# 9. Rehabilitation directions

## 9.1 Core active directions

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

The utility must not prescribe one exercise school or a fixed stabilization programme globally.

## 9.2 Optional adjunct expander

All technique-level adjuncts are secondary and never preselected:

```text
manual therapy / mobilization
soft-tissue techniques
neurodynamic techniques when neural/radiating context exists
acupuncture
dry needling
```

### Acupuncture

Acupuncture remains available because it is part of the product owner's real referral practice.

Production wording must preserve evidence-framework transparency:

- NICE NG59 recommends against offering acupuncture for low-back pain with or without sciatica;
- WHO 2023 conditionally supports needling therapies, including acupuncture, as part of care for chronic primary low-back pain, with low-certainty evidence;
- therefore the utility must not describe acupuncture as universally guideline-recommended or mandatory.

When selected, referral wording should remain collaborative, for example:

> Acupuncture may be considered as an adjunct where clinically appropriate and where the treating professional has relevant competence.

### Dry needling

Dry needling also remains available as an optional adjunct, especially with selected myofascial/trigger-point findings.

The principal production safeguard is **provider competence**, not merely indication matching. WHO explicitly notes that health workers delivering needling therapies should have appropriate anatomical knowledge and competencies.

Therefore:

```text
dry needling selected
→ optional competence/availability note in clinician-facing UI
```

The referral should not imply that every physiotherapist should perform dry needling.

### Lumbar traction

Routine lumbar traction is not offered in the lumbar MVP. NICE recommends against traction for LBP with or without sciatica, and WHO recommends against routine traction for chronic primary LBP.

---

# 10. Deterministic consistency rules

```text
L2 selected
+ motor/sensory/reflex all not_assessed
→ prompt: consider documenting current neurological status

SLR/slump positive
+ no radiating/neural symptoms
→ soft warning: provocation alone does not establish radiculopathy

formal_lumbar_radiculopathy_diagnosis != yes
→ do not output definitive lumbar radiculopathy

L3 selected
+ formal stenosis diagnosis != yes
→ describe symptom pattern; do not assert formal stenosis

L4 selected
+ formal_deep_gluteal_diagnosis = not_stated
→ describe deep-gluteal/piriformis presentation only; do not assert syndrome

SI-region pain selected in free text
→ do not convert to SI dysfunction or SIJ pain diagnosis

neurodynamic technique selected
+ no neural/radiating context
→ soft warning

acupuncture or dry needling selected
+ no active/function-oriented rehabilitation direction
→ warning: adjunct should not replace active rehabilitation

dry needling selected
→ soft competence/availability reminder

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

# 11. Generated wording examples

## 11.1 Non-specific/mechanical LBP

> Μη ειδική / μηχανικού τύπου οσφυαλγία με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη ενεργητική αποκατάσταση με στόχο [selected goals], με έμφαση στη διατήρηση/επάνοδο στη δραστηριότητα, στην άσκηση, στην εκπαίδευση και στην αυτοδιαχείριση. [Selected additional directions/adjuncts if present.]

## 11.2 Radiating symptoms with incomplete neurological assessment

> Οσφυαλγία με ακτινοβολία/παραισθησίες προς το [side] κάτω άκρο, με [only documented findings]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένο ενεργητικό πρόγραμμα αποκατάστασης με προσαρμογή στην ερεθιστικότητα και τη λειτουργική εικόνα.

No negative neurological wording is generated from missing data.

## 11.3 Assessed normal motor and sensory findings

Only when `motor=normal` and `sensory=normal`:

> Κατά τον παρόντα έλεγχο δεν διαπιστώθηκε κινητικό ή αισθητικό έλλειμμα.

Reflexes are mentioned only if assessed.

## 11.4 Objective neurological deficit

> Οσφυαλγία με ακτινοβολία προς το [side] κάτω άκρο και [selected features]. Κατά τον παρόντα έλεγχο καταγράφηκε [specific motor/sensory/reflex finding]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη αποκατάσταση με προσαρμογή στην κλινική εικόνα. Ιατρική επανεκτίμηση σε περίπτωση νέας ή περαιτέρω προοδευτικής νευρολογικής μεταβολής.

If the deficit is already progressive, safety/disposition takes precedence.

## 11.5 Formal lumbar stenosis / neurogenic claudication

Only when formally asserted:

> Οσφυϊκή στένωση με νευρογενή διαλείπουσα χωλότητα, με λειτουργικό περιορισμό κυρίως σε [walking/standing/other selected activities] και [selected findings]. Παρακαλώ για εξατομικευμένο πρόγραμμα αποκατάστασης με έμφαση στη βελτίωση της λειτουργικής ικανότητας, της βάδισης/αντοχής και της φυσικής δραστηριότητας, σύμφωνα με τη φυσιοθεραπευτική αξιολόγηση.

## 11.6 Formal piriformis syndrome / deep-gluteal syndrome

Only when explicitly asserted by the clinician:

> [Σύνδρομο απιοειδούς / deep gluteal syndrome] με οπίσθιο γλουτιαίο άλγος, [selected sciatic-type or other findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη ενεργητική αποκατάσταση με στόχο τη βελτίωση της ανοχής στη φόρτιση/καθιστή θέση και την επάνοδο στις επιλεγμένες δραστηριότητες. Επικουρικές τεχνικές μπορούν να χρησιμοποιηθούν ανάλογα με την αξιολόγηση και την επάρκεια του θεραπευτή.

## 11.7 Acupuncture selected

When acupuncture is clinician-selected, it is appended as an optional adjunct, not as the core prescription:

> Δύναται να χρησιμοποιηθεί επικουρικά acupuncture όπου κρίνεται κλινικά κατάλληλο και υπάρχει σχετική εμπειρία/επάρκεια του θεραπευτή.

## 11.8 Dry needling selected

> Dry needling μπορεί να χρησιμοποιηθεί επικουρικά για τα επιλεγμένα μυοπεριτονιακά ευρήματα, εφόσον κρίνεται κατάλληλο και εφαρμόζεται από επαγγελματία με σχετική εκπαίδευση και επάρκεια.

---

# 12. Evidence-governance boundary

Stable structural decisions frozen in v1.1:

```text
presentation != diagnosis
subjective symptoms != objective deficit
not assessed != normal
SI dysfunction is not a lumbar diagnosis
piriformis/deep-gluteal diagnosis is clinician-entered, never inferred
active/function-oriented rehabilitation is the core direction
adjunct techniques remain optional
safety prompts support clinician consistency; they do not diagnose
```

Evidence-sensitive production wording must be rechecked immediately before CU-2 implementation.

Key evidence points reviewed for this freeze:

- NICE NG59 supports self-management/exercise, allows manual therapy only as part of an exercise-containing package, and recommends against traction and acupuncture for LBP/sciatica.
- WHO 2023 conditionally supports needling therapies including acupuncture and dry needling for chronic primary LBP, with low-certainty evidence, as part of a broader care package rather than isolated treatment; provider competence is explicitly important.
- Deep gluteal syndrome is a recognized broader non-discogenic sciatic-entrapment construct; piriformis syndrome may be carried when explicitly diagnosed but should not be inferred from buttock pain alone.
- Contemporary SIJ consensus guidance distinguishes MRI evidence of sacroiliitis/structural pathology from proof that an SI joint is the mechanical pain generator; imaging alone should not create an SIJ pain diagnosis.

---

# 13. Freeze decision

Product-owner approved on 2026-08-26:

- keep L1 non-specific/mechanical low-back pain;
- keep L2 radiating/radicular-feature pathway with strict neurological semantics;
- keep L3 lumbar stenosis/neurogenic claudication with optional formal clinician diagnosis;
- add L4 deep-gluteal/piriformis pathway, allowing explicit formal piriformis/deep-gluteal diagnosis but no software inference;
- keep lumbar/gluteal trigger-point and myofascial findings directly selectable;
- do not create `SI dysfunction` as a lumbar diagnosis;
- reserve SI-region/SIJ pathology for a separate future SI/pelvic profile, with explicit distinction between clinical pain attribution and imaging-confirmed sacroiliitis/structural pathology;
- keep acupuncture as an optional clinician-selected adjunct despite guideline divergence, with transparent evidence framing;
- keep dry needling as an optional adjunct with explicit provider-competence/availability caveat;
- exclude routine lumbar traction from the MVP;
- remove lumbar post-operative rehabilitation from the active lumbar MVP because it is not part of the product owner's current workflow;
- preserve active rehabilitation, exercise, education and self-management as the conceptual backbone;
- never generate `no neurological deficit` or `no red flags` from unassessed/missing data.

This file is the frozen lumbar clinical/content design for CU-1. Runtime implementation remains unauthorized.