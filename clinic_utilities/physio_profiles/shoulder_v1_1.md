# Shoulder Physiotherapy Referral Profile v1.1 — CU-1 FROZEN DESIGN

> **STATUS:** FROZEN CLINICAL/CONTENT DESIGN — product-owner approved 2026-08-26.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful shoulder referral choices while preserving diagnosis-vs-finding separation, explicit safety/reassessment semantics, active rehabilitation, and physiotherapist autonomy.
> **Supersedes as active shoulder design:** `clinic_utilities/physio_profiles/shoulder_v1.md`.
> **Prior frozen regional profiles:** `cervical_v1_1.md`, `lumbar_v1_1.md`.

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
symptom != diagnosis
special test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

The utility structures a referral and checks internal consistency. It must not diagnose rotator-cuff tear, instability, adhesive capsulitis, biceps disease, AC-joint disease, sternoclavicular disease or other pathology autonomously.

---

# 2. Frozen primary shoulder pathways

## S1 — Rotator-cuff-related shoulder pain / rotator-cuff tendinopathy

Structured key:

```text
rotator_cuff_related_shoulder_pain
```

Display:

> Πόνος ώμου σχετιζόμενος με το στροφικό πέταλο / τενοντοπάθεια στροφικού πετάλου

Preferred common non-traumatic/load-related cuff pathway.

Terminology rule:

```text
subacromial impingement syndrome
!= preferred top-level diagnosis
```

Painful arc, Hawkins/Neer-type provocation, Jobe/full-can, painful resisted abduction/external rotation, scapular findings or imaging abnormalities may contribute to the presentation but do not independently establish a definitive structural diagnosis.

Optional clinician assertions/context:

```text
formal_rotator_cuff_tendinopathy_diagnosis: yes / no / not_stated
known_partial_thickness_rotator_cuff_tear: yes / no / not_stated
```

## S2 — Confirmed full-thickness rotator-cuff tear — conservative rehabilitation

Structured key:

```text
confirmed_full_thickness_rotator_cuff_tear_nonoperative
```

Display:

> Επιβεβαιωμένη ρήξη πλήρους πάχους στροφικού πετάλου — συντηρητική αποκατάσταση

Required context:

```text
clinician/imaging-confirmed tear
atraumatic/degenerative vs traumatic onset if known
muscle/tendon involved if known
current management decision = nonoperative rehabilitation / conservative trial
explicit restrictions if any
```

The utility must not create this diagnosis from weakness, painful arc or special tests.

Safety boundary:

```text
acute trauma
+ new marked weakness / inability to actively elevate
+ unresolved acute cuff-tear concern
→ high-priority clinician reassessment/imaging/specialist-pathway prompt
```

Conservative physiotherapy may be appropriate for established tears, but generated wording must not claim that surgery is unnecessary or that structural progression cannot occur.

## S3 — Calcific rotator-cuff tendinopathy

Structured key:

```text
calcific_rotator_cuff_tendinopathy
```

Display:

> Ασβεστοποιός τενοντοπάθεια στροφικού πετάλου

Prefer clinician/imaging-established diagnosis.

Context:

```text
imaging-confirmed calcific deposit: yes / no / not_stated
location/tendon optional
acute highly irritable vs persistent/chronic presentation optional
prior barbotage/lavage: yes / no / not_stated
prior other treatment optional
```

ESWT remains an optional calcific-specific adjunct when clinically appropriate and available. Prior barbotage does not remove ESWT from the selectable options when clinically relevant, but the utility must not imply that sequential post-barbotage ESWT is automatically superior or required.

Therapeutic ultrasound is not offered as a standard evidence-supported adjunct for rotator-cuff tendinopathy in this profile.

## S4 — Adhesive capsulitis / frozen shoulder

Structured key:

```text
adhesive_capsulitis_frozen_shoulder
```

Default presentation wording when diagnosis is not formally asserted:

> Επώδυνος ώμος με σημαντικό περιορισμό ενεργητικού και παθητικού εύρους κίνησης

Optional clinician assertion:

```text
formal_adhesive_capsulitis_diagnosis: yes / no / not_stated
```

If `yes`, generated wording may state:

> Συμφυτική θυλακίτιδα / frozen shoulder

The utility does not infer adhesive capsulitis from pain plus one restricted movement or from imaging alone.

Useful modifiers:

```text
active ROM restriction
passive ROM restriction
external-rotation restriction
multi-directional/global restriction
pain/irritability
sleep/night pain
functional reach limitations
```

Rigid `freezing/frozen/thawing` staging is not required by the generator; irritability, actual ROM and function are more useful referral variables.

## S5 — Glenohumeral instability / dislocation rehabilitation

Structured key:

```text
glenohumeral_instability_dislocation
```

Display:

> Αστάθεια / εξάρθρημα γληνοβραχιόνιας άρθρωσης — αποκατάσταση

Required context:

```text
first-time vs recurrent
traumatic vs atraumatic
anterior / posterior / multidirectional / not_stated
reduction required if relevant
current structural/specialist assessment status
current restrictions if any
sport/work demands
```

Optional clinician assertion:

```text
formal_glenohumeral_instability_diagnosis: yes / no / not_stated
```

Apprehension/relocation findings, hyperlaxity or subjective instability are findings/context and do not independently establish the diagnosis.

## S6 — Glenohumeral osteoarthritis

Structured key:

```text
glenohumeral_osteoarthritis
```

Display:

> Οστεοαρθρίτιδα γληνοβραχιόνιας άρθρωσης

Use when clinician-established with compatible clinical/imaging context.

Rehabilitation wording focuses on pain/function, mobility where useful, strength and activity rather than structural reversal.

## S7 — Post-traumatic shoulder pain after assessed injury

Structured key:

```text
post_traumatic_shoulder_pain
```

Display:

> Μετατραυματικός πόνος ώμου / αποκατάσταση μετά από κάκωση

Use only after unresolved fracture/dislocation/acute repairable tendon injury has been addressed as clinically required and routine rehabilitation is considered appropriate.

Required context:

```text
date/phase of injury
known structural diagnosis if any
imaging/orthopaedic assessment context if relevant
current restrictions/precautions
```

The utility must not generate minimizing diagnoses such as `simple shoulder strain` automatically.

## S8 — Acromioclavicular-joint pathway

Structured key:

```text
acromioclavicular_joint_disorder
```

Display:

> Παθολογία / πόνος ακρωμιοκλειδικής άρθρωσης

Formal clinician-selected subtype:

```text
ac_joint_related_pain
ac_joint_osteoarthritis
ac_joint_sprain
other_ac_joint_diagnosis
```

This is a primary pathway because isolated AC-joint pathology is a real referral entity, including load/weight-training presentations.

Useful context/findings:

```text
focal AC-region pain/tenderness
cross-body adduction pain if examined
overhead/pressing/bench-press load intolerance
traumatic sprain grade/context if known
imaging OA/context optional
```

Cross-body adduction/local tenderness or imaging OA alone must not automatically create symptomatic AC-joint disease.

## S9 — Sternoclavicular-joint pathway

Structured key:

```text
sternoclavicular_joint_disorder
```

Display:

> Παθολογία / πόνος στερνοκλειδικής άρθρωσης

Formal clinician-selected subtype:

```text
sternoclavicular_joint_osteoarthritis
sternoclavicular_joint_inflammatory_arthritis
sternoclavicular_joint_anterior_dislocation_or_instability
sternoclavicular_joint_posterior_dislocation_history_after_appropriate_assessment
other_established_sternoclavicular_diagnosis
```

This pathway is deliberately diagnosis/context governed because SC-joint swelling and pain have a broader differential than ordinary shoulder pain.

Required context where relevant:

```text
traumatic vs atraumatic
pain vs swelling/deformity
anterior vs posterior dislocation if established
known RA/inflammatory disease context
infection/systemic concern addressed where relevant
imaging/specialist context where relevant
current restrictions
```

Hard safety boundary:

```text
suspected acute posterior SC dislocation
OR unexplained SC-joint swelling with systemic/infectious/malignancy concern
→ high-priority medical/specialist reassessment
→ do not generate routine physiotherapy reassurance
```

Known RA/inflammatory SC involvement or established SC osteoarthritis may be carried into referral wording when the clinician has already assessed the condition and considers rehabilitation appropriate.

## S10 — Post-operative shoulder rehabilitation

Structured key:

```text
postoperative_shoulder_rehabilitation
```

Display:

> Μετεγχειρητική αποκατάσταση ώμου

This pathway is part of the active shoulder MVP.

Required context:

```text
operation/procedure
operation date
surgeon/protocol when available
immobilization status
ROM restrictions
loading/strengthening restrictions
weight-bearing/use restrictions if relevant
other precautions
```

Examples may include rotator-cuff repair, stabilization/labral surgery, arthroplasty or other shoulder procedures. The generator must never invent a generic postoperative protocol.

---

# 3. Frozen secondary diagnoses / modifiers

## 3.1 Long-head-of-biceps-related pain/tendinopathy

Directly selectable as a common secondary diagnosis/modifier:

```text
long_head_biceps_tendinopathy_or_related_pain
```

Display:

> Πόνος / τενοντοπάθεια μακράς κεφαλής δικεφάλου

This is not a default primary pathway because in the product owner's real workflow it commonly coexists with another shoulder disorder.

Bicipital-groove tenderness, Speed/Yergason-type findings or anterior shoulder pain do not independently establish the diagnosis.

## 3.2 Scapular findings

```text
scapular dyskinesis / altered scapular movement observed
scapular control/endurance deficit
scapular muscle weakness where assessed
```

These remain findings/modifiers rather than diagnoses.

## 3.3 Myofascial findings

Directly selectable:

```text
upper-trapezius/periscapular tenderness
posterior-cuff tenderness
clinically active trigger points
myofascial pain presentation
```

They may appear in referral wording when actually selected.

---

# 4. Examination findings — selectable only when actually assessed

## 4.1 Pain / symptom behaviour

```text
anterolateral shoulder pain
lateral deltoid-region pain
posterior shoulder pain
AC-region pain
SC-region pain/swelling context
night pain / sleep disturbance
pain with overhead activity
pain with reaching away from body
pain with lifting/carrying
pain with pushing/pulling
pain with throwing/sport
```

## 4.2 Range of motion

```text
active ROM restricted
passive ROM restricted
painful active ROM
painful passive ROM
specific restriction optional
hand-behind-back/internal-rotation functional restriction
external-rotation restriction
```

Active and passive ROM remain separate.

## 4.3 Strength / load tolerance

```text
painful resisted abduction
painful resisted external rotation
objective abduction weakness
objective external-rotation weakness
objective internal-rotation/subscapular weakness
load intolerance without measured weakness
```

Pain-inhibited effort must not automatically become structural weakness/tear.

## 4.4 Special-test findings

Optional secondary expander only:

```text
painful arc
Hawkins-Kennedy/Neer-type provocation
Jobe/empty-can or full-can finding
external-rotation lag / drop-arm finding
belly-press / lift-off finding
Speed/Yergason-type finding
cross-body adduction finding
apprehension/relocation finding
other clinician-entered test
```

Special tests are findings, not diagnoses.

---

# 5. Cervical / neurological overlap

A full neurological screen is not required for every shoulder referral. If radiating arm symptoms, paresthesia, numbness, objective distal weakness or cervical source concern exists, allow:

```text
cervical_source_considered: yes / no / not_assessed
radiating_arm_pain
paresthesia
numbness
motor: normal / abnormal / not_assessed
sensory: normal / abnormal / not_assessed
reflexes: normal / abnormal / not_assessed
```

Hard invariant:

```text
not assessed != normal
```

If presentation is primarily cervical/radicular, use the frozen cervical profile.

---

# 6. Safety / reassessment semantics

## 6.1 High-priority shoulder concerns

```text
acute trauma with unresolved fracture concern
acute trauma with unresolved dislocation/instability concern
new marked weakness or inability to actively elevate after trauma
suspected acute full-thickness/massive cuff tear not yet appropriately assessed
new neurovascular deficit after shoulder trauma/dislocation
suspected acute posterior sternoclavicular dislocation
progressive unexplained weakness
other urgent structural concern
```

## 6.2 Other material concerns

```text
unexplained sternoclavicular swelling or mass
systemic/infectious/malignancy/inflammatory concern
severe unremitting/progressive non-mechanical pain
unexplained constitutional/systemic symptoms
cervical/neurological source concern requiring reassessment
other clinician concern
```

## 6.3 Safety state and clinician disposition

```text
safety_screen_status:
  not_assessed
  no_specific_concern_identified
  concern_present

clinician_disposition when concern present:
  reviewed_and_appropriate_to_proceed
  imaging/medical reassessment arranged
  orthopaedic/specialist pathway underway
  urgent/same-day assessment arranged
  routine physiotherapy deferred
  other
```

No default `no red flags`, `no tear`, `no instability` or `neurovascularly intact` wording is generated from missing data.

---

# 7. Functional limitations

```text
sleep / side-lying
reaching overhead
reaching away from body
hand behind back / dressing
washing/grooming hair
putting on coat / bra / clothing
lifting/carrying
pushing/pulling
pressing/weight-training
work/manual tasks
driving / seatbelt / steering
exercise/gym
throwing / overhead sport
swimming/racquet sport
ADLs/self-care
patient-priority activity / free text
```

---

# 8. Context-sensitive goals

Nothing is globally preselected.

Common goal families:

```text
reduce symptom irritability
restore/improve functional ROM where impaired
improve cuff/scapular strength/endurance where relevant
improve load tolerance
improve dynamic stability/proprioception where relevant
improve dressing/grooming/sleep/reaching/lifting as selected
graded return to work/gym/sport
improve self-management
```

Condition-specific cautions:

- full-thickness tear: do not promise tendon healing or reversal of structural tear;
- adhesive capsulitis: do not promise rapid full ROM restoration;
- instability/dislocation: respect current restrictions and use criterion-based progression;
- postoperative shoulder: surgeon/protocol restrictions outrank generic targets;
- SC-joint inflammatory/systemic disease: rehabilitation goals do not replace medical disease management.

---

# 9. Rehabilitation directions

## 9.1 Core active directions

```text
physiotherapy assessment and individualized active rehabilitation
therapeutic exercise
rotator-cuff/scapular strengthening and endurance where relevant
graded resistance/loading
mobility/ROM exercise where restricted and clinically appropriate
motor-control/proprioceptive/dynamic-stability work where relevant
graded activity/exposure
return-to-work/sport progression where relevant
education and self-management
home exercise programme where appropriate
activity/load/work modification where relevant
```

No single exercise type, scapular programme or stabilization protocol is globally mandatory.

## 9.2 Optional adjunct expander

```text
manual therapy / joint mobilization
soft-tissue techniques
taping where clinically appropriate
acupuncture
dry needling
ESWT — calcific pathway only
```

### Acupuncture

Remains a clinician-selectable adjunct and may be included in referral wording where clinically appropriate and practitioner competence exists. It must not displace active rehabilitation or be presented as mandatory.

### Dry needling

Remains optional, especially with selected myofascial/trigger-point findings. Competence/availability is a material safeguard; referral wording must not imply that every physiotherapist should perform it.

### ESWT for calcific tendinopathy

Available only in S3. Prior barbotage/lavage can be recorded as context and does not automatically exclude ESWT. The current evidence base supports both lavage and ESWT as recognized nonsurgical options but does not justify an automatic post-barbotage sequence rule.

### Therapeutic ultrasound

Not offered as a standard evidence-supported rotator-cuff adjunct in the profile.

---

# 10. Fracture boundary — routed to a separate shared profile

Shoulder-region fractures are clinically important but will not be fragmented across regional shoulder diagnoses.

The shoulder UI should expose a route such as:

```text
fracture / post-immobilization
→ shared fracture rehabilitation profile
```

Shoulder-region examples include:

```text
proximal humerus fracture
clavicle fracture
scapular fracture
other shoulder-girdle fracture
```

The future shared fracture profile must require:

```text
bone/site
fracture date/phase
treatment
healing/stability status if known
immobilization status
weight-bearing/use restrictions
ROM/loading restrictions
orthopaedic/surgeon instructions
```

Routine unrestricted shoulder rehabilitation wording must not be generated when healing/loading context is unresolved.

---

# 11. Deterministic consistency rules

```text
S1 + one positive provocation test only
→ do not output definitive structural diagnosis

known partial-thickness tear not explicitly entered
→ never infer tear from pain/weakness/tests

S2 + no confirmed full-thickness tear context
→ warning

acute trauma + marked new weakness/inability to elevate + no disposition
→ high-priority reassessment prompt

S3 + no established calcific diagnosis/imaging context
→ prompt before definitive calcific wording

ESWT + S3 not selected
→ warning

S4 + formal diagnosis != yes
→ use actual stiffness/ROM presentation wording only

S5 + instability context missing
→ prompt for first-time/recurrent/traumatic/direction context

S8 + isolated cross-body test/imaging OA only
→ do not auto-assert symptomatic AC-joint disease

S9 + suspected posterior SC dislocation
→ high-priority reassessment; no routine physio wording

S9 + unexplained swelling/systemic concern
→ medical reassessment prompt before routine referral

postoperative pathway + missing procedure/protocol/restrictions
→ warning

adjunct selected + no active rehabilitation direction
→ warning

dry needling selected
→ competence/availability reminder

radiating arm/paresthesia/neuro finding + cervical context not addressed
→ soft prompt

fracture route + healing/loading context missing
→ warning

material safety concern + no clinician disposition
→ do not generate routine reassuring wording

no selected safety concern
→ must not generate `no red flags`
```

---

# 12. Generated wording examples

## 12.1 RCRSP

> Πόνος ώμου σχετιζόμενος με το στροφικό πέταλο / τενοντοπάθεια στροφικού πετάλου, με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη ενεργητική αποκατάσταση με έμφαση στη σταδιακή φόρτιση, την ενδυνάμωση/αντοχή του ώμου και της ωμοπλάτης όπου ενδείκνυται, την εκπαίδευση και την επάνοδο στις επιλεγμένες δραστηριότητες. [Selected adjuncts if present.]

## 12.2 Full-thickness tear — conservative

> Επιβεβαιωμένη ρήξη πλήρους πάχους του [selected tendon if entered] στροφικού πετάλου, για συντηρητική αντιμετώπιση/αποκατάσταση, με [selected deficits]. Παρακαλώ για εξατομικευμένο πρόγραμμα φυσιοθεραπείας με στόχο τη βελτίωση της λειτουργίας, του διαθέσιμου ενεργητικού εύρους κίνησης και της μυϊκής ικανότητας, σύμφωνα με την κλινική εικόνα και τις τυχόν καταγεγραμμένες οδηγίες/περιορισμούς.

## 12.3 Calcific tendinopathy

> Ασβεστοποιός τενοντοπάθεια στροφικού πετάλου με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για ενεργητική φυσιοθεραπευτική αποκατάσταση με στόχο τη βελτίωση του πόνου, της λειτουργίας και της προοδευτικής φόρτισης. [If selected: ESWT may be considered adjunctively where clinically appropriate and available. Prior barbotage/lavage: documented if applicable.]

## 12.4 Adhesive capsulitis

Only when formally asserted:

> Συμφυτική θυλακίτιδα / frozen shoulder με σημαντικό περιορισμό [selected active/passive ROM findings], πόνο [if selected] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση προσαρμοσμένη στην ερεθιστικότητα, με στόχο τη σταδιακή βελτίωση της κινητικότητας και της λειτουργικότητας, ενεργητική συμμετοχή και αυτοδιαχείριση.

## 12.5 AC-joint pathology

> [Selected clinician-established AC-joint diagnosis] του [side] ώμου, με [selected findings] και λειτουργικό περιορισμό ιδιαίτερα σε [pressing/overhead/cross-body/other selected activities]. Παρακαλώ για εξατομικευμένη ενεργητική αποκατάσταση με προσαρμογή της φόρτισης και σταδιακή επάνοδο στις απαιτούμενες δραστηριότητες.

## 12.6 SC-joint established disorder

> [Selected clinician-established sternoclavicular diagnosis], με [selected pain/swelling/function findings]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και αποκατάσταση σύμφωνα με την καθορισμένη κλινική διάγνωση και τους τυχόν περιορισμούς. [Inflammatory/systemic treatment context if explicitly relevant.]

No routine referral wording is produced for suspected acute posterior dislocation or unresolved unexplained swelling/systemic concern.

## 12.7 Postoperative shoulder

> Μετεγχειρητική αποκατάσταση [selected procedure] του [side] ώμου, επέμβαση [date if entered]. Παρακαλώ για φυσιοθεραπευτική αποκατάσταση σύμφωνα με το διαθέσιμο χειρουργικό πρωτόκολλο/οδηγίες και τους καταγεγραμμένους περιορισμούς σε εύρος κίνησης, φόρτιση και ενδυνάμωση.

---

# 13. Evidence-governance boundary

Stable structural decisions frozen in v1.1:

```text
special test != diagnosis
impingement terminology != preferred top-level diagnosis
active ROM != passive ROM
pain-inhibited effort != proven tear
formal full-thickness tear requires established diagnosis
adhesive capsulitis is not inferred from generic stiffness
instability is not inferred from one apprehension test
LH biceps commonly lives as secondary/coexisting diagnosis
AC-joint pathology can be a primary referral entity
SC-joint disease requires stronger diagnosis/safety context
suspected posterior SC dislocation is not a routine physio pathway
postoperative shoulder is an active shoulder pathway
fractures route to a shared fracture/post-immobilization profile
active/function-oriented rehabilitation is the core
adjunct techniques remain optional
```

Current evidence anchors reviewed for this freeze include:

- 2025 JOSPT/APTA Rotator Cuff Tendinopathy CPG;
- 2025 AAOS Management of Rotator Cuff Injuries CPG;
- ACR Appropriateness Criteria Acute Shoulder Pain, revised 2024;
- 2025 clinical practice guidance for primary frozen shoulder;
- current instability/dislocation consensus literature;
- AAOS glenohumeral osteoarthritis guidance;
- 2024 systematic reviews comparing ESWT and ultrasound-guided lavage/barbotage for calcific tendinopathy;
- sternoclavicular-joint reviews describing OA, inflammatory disease, infection and dislocation, including the mediastinal risk of posterior SC dislocation.

Evidence-sensitive production wording to recheck before CU-2:

- exact exercise/loading dosage;
- manual therapy duration/role;
- acupuncture/dry-needling effect estimates;
- calcific ESWT/barbotage sequencing;
- full-thickness-tear surveillance wording;
- frozen-shoulder intervention dosage;
- postoperative protocol wording;
- instability return-to-sport criteria.

---

# 14. Freeze decision

Product-owner approved on 2026-08-26:

- keep RCRSP/tendinopathy as the common cuff pathway and avoid `impingement syndrome` as a top-level diagnosis;
- keep confirmed full-thickness tear as a separate conservative-rehabilitation pathway;
- keep calcific tendinopathy separate and retain optional ESWT, including after prior barbotage when clinically relevant without an automatic sequencing claim;
- keep adhesive capsulitis, instability/dislocation and glenohumeral OA pathways;
- keep post-traumatic assessed injury;
- add AC-joint pathology as a primary pathway;
- add sternoclavicular-joint pathology as a primary pathway with strict posterior-dislocation/systemic-swelling safety semantics;
- keep long-head biceps tendinopathy as a common selectable secondary/coexisting diagnosis rather than a default primary pathway;
- include postoperative shoulder rehabilitation in the active shoulder MVP with mandatory protocol/restriction context;
- retain acupuncture and dry needling as optional adjuncts, with competence/availability safeguards;
- route shoulder-region fractures to a separate shared fracture/post-immobilization profile rather than duplicating fracture logic inside shoulder;
- preserve active rehabilitation, education, graded loading/activity and physiotherapist autonomy as the conceptual backbone.

This file is the frozen shoulder clinical/content design for CU-1. Runtime implementation remains unauthorized.
