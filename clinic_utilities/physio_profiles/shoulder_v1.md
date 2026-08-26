# Shoulder Physiotherapy Referral Profile v1 — CU-1 DESIGN CANDIDATE

> **STATUS:** DESIGN CANDIDATE — product-owner review required before freeze.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful shoulder referral choices while preserving diagnosis-vs-finding separation, safety/reassessment semantics, active rehabilitation, and physiotherapist autonomy.
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

Hard invariants inherited from the frozen cervical/lumbar profiles:

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

The utility may structure the referral and check internal consistency. It must not diagnose a rotator-cuff tear, instability, adhesive capsulitis, biceps disorder, AC-joint disorder or other pathology autonomously.

---

# 2. Proposed primary shoulder pathways

## S1 — Rotator-cuff-related shoulder pain / rotator-cuff tendinopathy

Structured key:

```text
rotator_cuff_related_shoulder_pain
```

Display:

> Πόνος ώμου σχετιζόμενος με το στροφικό πέταλο / τενοντοπάθεια στροφικού πετάλου

Use as the main non-traumatic or load-related rotator-cuff pathway when a more specific major structural diagnosis is not required.

Important terminology rule:

```text
"subacromial impingement syndrome"
!= preferred top-level diagnosis
```

`Painful arc`, Hawkins-Kennedy/Neer-type provocation, Jobe/empty-can findings, painful resisted abduction/external rotation, scapular findings or imaging abnormalities may contribute to the clinical picture but must not independently create a definitive diagnosis.

The 2025 rotator-cuff-tendinopathy CPG uses a broad rotator-cuff-tendinopathy construct that includes common terms such as subacromial pain syndrome/bursopathy and may include partial-thickness tears within its nonsurgical scope. The utility should therefore avoid creating several overlapping pseudo-diagnoses for the same clinical presentation.

Optional clinician assertion/context:

```text
formal_rotator_cuff_tendinopathy_diagnosis: yes / no / not_stated
known_partial_thickness_rotator_cuff_tear: yes / no / not_stated
```

A known partial-thickness tear may be carried as an established finding/diagnosis when explicitly entered; the software does not infer it from weakness or provocative tests.

## S2 — Confirmed full-thickness rotator-cuff tear — conservative rehabilitation pathway

Structured key:

```text
confirmed_full_thickness_rotator_cuff_tear_nonoperative
```

Display:

> Επιβεβαιωμένη ρήξη πλήρους πάχους στροφικού πετάλου — συντηρητική αποκατάσταση

This pathway exists because established full-thickness tears have different prognosis/surveillance and possible surgical implications from ordinary rotator-cuff tendinopathy.

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
+ unresolved acute cuff tear concern
→ high-priority clinician reassessment/imaging/specialist-pathway prompt
```

Routine physiotherapy wording should not imply that an acute traumatic repairable tear has been fully evaluated when it has not.

## S3 — Calcific rotator-cuff tendinopathy

Structured key:

```text
calcific_rotator_cuff_tendinopathy
```

Display:

> Ασβεστοποιός τενοντοπάθεια στροφικού πετάλου

Prefer explicit clinician/imaging-established diagnosis.

Required/context fields may include:

```text
imaging-confirmed calcific deposit: yes / no / not_stated
location/tendon optional
acute highly irritable vs persistent/chronic presentation optional
prior conservative treatment optional
```

This is separate from S1 because evidence-sensitive adjunct options differ.

The physiotherapy utility may expose shockwave therapy as an optional calcific-specific adjunct where locally available/appropriate. It must not present therapeutic ultrasound as evidence-supported treatment for calcific rotator-cuff tendinopathy.

Calcific lavage/barbotage is not a physiotherapy technique in this generator; if refractory symptoms prompt consideration of lavage or another medical procedure, that belongs to clinician reassessment/management rather than the physiotherapist technique list.

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

The utility must not infer adhesive capsulitis from pain plus one restricted movement or from imaging alone.

Useful context:

```text
active ROM restriction
passive ROM restriction
external-rotation restriction
multi-directional/global restriction
pain/irritability
sleep/night pain
functional reach limitations
known diabetes/thyroid or other relevant context optional
```

The tool should not force a rigid clinical stage (`freezing/frozen/thawing`) because stage boundaries are not sufficiently reliable for automatic decision logic. Irritability and actual ROM/function are more useful modifiers.

## S5 — Glenohumeral instability / dislocation rehabilitation

Structured key:

```text
glenohumeral_instability_dislocation
```

Display:

> Αστάθεια / εξάρθρημα γληνοβραχιόνιας άρθρωσης — αποκατάσταση

Required context should support:

```text
first-time vs recurrent
traumatic vs atraumatic
anterior / posterior / multidirectional / not_stated
reduction required if relevant
current structural/specialist assessment status
current restrictions if any
sport/work demands
```

Optional clinician diagnosis/assertion:

```text
formal_glenohumeral_instability_diagnosis: yes / no / not_stated
```

Apprehension/relocation findings, hyperlaxity or a subjective sense of instability are findings/context and must not independently create a formal instability diagnosis.

First-time traumatic dislocation and recurrent instability should share the pathway but remain explicitly distinguishable because age, sport, recurrence history, bone loss and associated lesions may materially alter management.

## S6 — Glenohumeral osteoarthritis

Structured key:

```text
glenohumeral_osteoarthritis
```

Display:

> Οστεοαρθρίτιδα γληνοβραχιόνιας άρθρωσης

Use only when the clinician considers the diagnosis established, usually with compatible clinical/imaging context.

Useful context:

```text
pain/stiffness
active/passive ROM restriction
crepitus if clinically relevant
functional reach/lifting limitation
imaging context optional
```

Rehabilitation wording should focus on symptom/function, mobility where appropriate, strength and activity rather than promise structural reversal.

## S7 — Post-traumatic shoulder pain / rehabilitation after assessed injury

Structured key:

```text
post_traumatic_shoulder_pain
```

Display:

> Μετατραυματικός πόνος ώμου / αποκατάσταση μετά από κάκωση

Use only after the clinician considers routine rehabilitation appropriate and significant unresolved fracture/dislocation/acute repairable tendon injury has been addressed as clinically required.

Required context:

```text
date/phase of injury
known structural diagnosis if any
imaging/orthopaedic assessment context if relevant
current restrictions/precautions
```

This pathway must not generate the phrase `simple shoulder strain` or other minimizing diagnosis automatically.

---

# 3. Candidate secondary diagnoses / presentation modifiers — not top-level by default

## 3.1 Long-head-of-biceps-related pain/tendinopathy

Proposed default role:

```text
secondary_problem / clinician-entered diagnosis or modifier
```

Possible display:

> Πόνος / τενοντοπάθεια μακράς κεφαλής δικεφάλου

The 2025 rotator-cuff-tendinopathy CPG treats long-head-of-biceps tendinopathy within the broader rotator-cuff-related shoulder-pain clinical umbrella. Therefore it should not automatically create a competing top-level pathway unless the product owner's real workflow demonstrates that a dedicated biceps referral is commonly useful.

Speed/Yergason-type findings and bicipital-groove tenderness remain examination findings and do not establish the diagnosis by themselves.

## 3.2 Acromioclavicular-joint-related pain / arthropathy

Proposed role pending product-owner workflow confirmation:

```text
secondary_problem or dedicated pathway if commonly referred
```

Possible clinician-entered diagnoses:

```text
AC-joint-related pain
AC-joint osteoarthritis
AC-joint sprain after trauma
```

Cross-body adduction/local tenderness or imaging OA must not automatically become symptomatic AC-joint disease.

## 3.3 Scapular findings

Directly selectable findings/modifiers:

```text
scapular dyskinesis / altered scapular movement observed
scapular control/endurance deficit
scapular muscle weakness where assessed
```

These are not top-level diagnoses in the generator.

## 3.4 Myofascial findings

Directly selectable:

```text
upper-trapezius/periscapular tenderness
posterior-cuff tenderness
clinically active trigger points
myofascial pain presentation
```

They remain findings/presentation modifiers unless the clinician explicitly enters a formal diagnosis.

---

# 4. Examination findings — selectable only when actually assessed

## 4.1 Pain / symptom behaviour

```text
anterolateral shoulder pain
lateral deltoid-region pain
posterior shoulder pain
AC-region pain
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

Active vs passive ROM must remain separate because the distinction materially affects clinical interpretation.

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

Optional secondary expander only; tests are findings, not diagnoses:

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

Avoid making a long special-test checklist part of the default referral flow.

---

# 5. Cervical / neurological overlap

Shoulder pain may coexist with or be mimicked by cervical/neurological pathology.

The shoulder profile does not require a full neurological screen for every referral. If radiating arm symptoms, paresthesia, numbness, objective distal weakness or a cervical source concern is present, allow explicit crossover fields:

```text
cervical_source_considered: yes / no / not_assessed
radiating_arm_pain
paresthesia
numbness
motor: normal / abnormal / not_assessed
sensory: normal / abnormal / not_assessed
reflexes: normal / abnormal / not_assessed
```

Hard rule:

```text
not assessed != normal
```

If the presentation is primarily cervical/radicular, use the frozen cervical profile rather than forcing it into shoulder taxonomy.

---

# 6. Safety / reassessment semantics

The utility provides clinician-facing consistency prompts; it does not diagnose fracture, dislocation, acute rotator-cuff tear, infection, malignancy or neurovascular injury.

## 6.1 High-priority shoulder concerns

```text
acute trauma with unresolved fracture concern
acute trauma with unresolved dislocation/instability concern
new marked weakness or inability to actively elevate after trauma
suspected acute full-thickness/massive rotator-cuff tear not yet appropriately assessed
new neurovascular deficit after shoulder trauma/dislocation
progressive unexplained weakness
other urgent structural concern
```

## 6.2 Other material concerns

```text
systemic/infectious/malignancy/inflammatory concern
severe unremitting/progressive non-mechanical pain
unexplained constitutional/systemic symptoms relevant to shoulder pain
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
  routine physiotherapy deferred
  other
```

The utility does not choose disposition.

There is no default `no red flags`, `no tear`, `no instability` or `neurovascularly intact` generated sentence from missing information.

---

# 7. Functional limitations

Candidate fields:

```text
sleep / side-lying
reaching overhead
reaching away from body
hand behind back / dressing
washing or grooming hair
putting on coat / bra / clothing
lifting/carrying
pushing/pulling
work/manual tasks
driving / seatbelt / steering
exercise/gym
throwing / overhead sport
swimming/racquet sport where relevant
ADLs/self-care
patient-priority activity / free text
```

Functional choices should drive condition-sensitive goals; they are not merely descriptive text.

---

# 8. Context-sensitive goal suggestions

Nothing is globally preselected.

## S1 — Rotator-cuff-related shoulder pain

Possible suggestions:

- reduce symptom irritability;
- improve shoulder load tolerance;
- improve rotator-cuff/scapular strength and endurance where relevant;
- restore functional ROM where restricted;
- improve overhead/reaching/lifting tolerance;
- graded return to work/exercise/sport;
- improve self-management and recurrence/load-management capability.

## S2 — Confirmed full-thickness tear, nonoperative

Possible suggestions:

- optimize pain/function within the chosen conservative pathway;
- preserve/improve available active ROM;
- improve deltoid/scapular/remaining rotator-cuff capacity where appropriate;
- improve ADL/work tolerance;
- maintain function while monitoring deterioration or failure of conservative care when clinically relevant.

Do not promise tendon healing or reversal of tear.

## S3 — Calcific tendinopathy

Possible suggestions:

- reduce pain/irritability;
- restore functional ROM;
- improve cuff/scapular strength/load tolerance;
- graded return to activity;
- improve self-management.

## S4 — Adhesive capsulitis

Possible suggestions:

- reduce pain/irritability;
- improve tolerated active/passive ROM according to irritability;
- improve dressing/grooming/reaching/sleep function;
- maintain/improve shoulder/scapular strength as feasible;
- improve self-management and activity confidence.

Do not promise rapid full ROM restoration.

## S5 — Instability/dislocation

Possible suggestions:

- restore safe functional ROM according to current restrictions;
- improve rotator-cuff/scapular strength/endurance;
- improve proprioception/dynamic stability;
- graded exposure to apprehension-provoking positions when appropriate;
- criterion-based return to work/sport;
- improve confidence in shoulder use.

## S6 — Glenohumeral OA

Possible suggestions:

- reduce pain;
- maintain/improve functional mobility where feasible;
- improve strength/endurance;
- optimize ADLs, reaching and lifting tolerance;
- maintain physical activity/self-management.

## S7 — Post-traumatic assessed injury

Goals derive from the actual established injury, restrictions and functional deficit; no generic unrestricted ROM/strength target is assumed.

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

Technique-level adjuncts are secondary and never preselected:

```text
manual therapy / joint mobilization
soft-tissue techniques
taping where clinically appropriate
acupuncture
dry needling
calcific-specific shockwave therapy (S3 only)
```

### Manual therapy

May be selected as an adjunct, especially for short-term pain/mobility goals. It should not displace an active rehabilitation programme.

### Acupuncture

The 2025 rotator-cuff-tendinopathy CPG allows acupuncture in addition to active rehabilitation. It remains optional, not mandatory, and should be used only where clinically appropriate and practitioner competence exists.

### Dry needling

May remain a clinician-selected adjunct, particularly when myofascial/trigger-point findings are selected. Evidence is not strong enough to make it a routine shoulder default.

Consistent with the lumbar freeze, provider competence/availability is a material safeguard:

```text
dry needling selected
→ soft competence/availability reminder
```

### Shockwave therapy

```text
calcific rotator-cuff tendinopathy
→ may expose ESWT as an optional adjunct

non-calcific rotator-cuff tendinopathy
→ do not suggest routine ESWT
```

### Therapeutic ultrasound

Do not present therapeutic ultrasound as an evidence-supported treatment choice for rotator-cuff tendinopathy in this profile; the 2025 CPG recommends against it for both calcific and noncalcific rotator-cuff tendinopathy.

---

# 10. Condition-specific consistency rules

```text
S1 selected
+ only one positive impingement/provocation test
→ do not output a definitive structural diagnosis from that test

known partial-thickness tear not explicitly entered
→ never infer partial tear from pain/weakness/tests

S2 selected
+ no clinician/imaging-confirmed full-thickness tear context
→ warning: confirmed-tear pathway requires established diagnosis

acute trauma
+ marked new weakness / inability to elevate
+ no clinician disposition
→ high-priority reassessment prompt

S3 calcific selected
+ no established calcific diagnosis/imaging context
→ prompt to confirm diagnosis/context before definitive wording

ESWT selected
+ S3 not selected
→ warning: shockwave is not a routine noncalcific-cuff adjunct in the current evidence framework

therapeutic ultrasound requested
→ evidence warning; not offered as standard profile adjunct

S4 adhesive capsulitis selected
+ formal diagnosis != yes
→ use presentation wording based on actual active/passive ROM restriction; do not assert frozen shoulder

S5 instability selected
+ first-time/recurrent/traumatic context absent
→ prompt for clinically useful instability context

apprehension/relocation positive
+ no clinician instability diagnosis/context
→ do not infer instability diagnosis

post-traumatic pathway
+ unresolved fracture/dislocation/acute cuff-tear concern
→ safety prompt before routine referral wording

acupuncture/dry needling/manual therapy selected
+ no active/function-oriented rehabilitation direction
→ warning: adjunct should not replace active rehabilitation

dry needling selected
→ competence/availability reminder

radiating arm/paresthesia/neuro finding selected
+ cervical/neurological context not addressed
→ soft prompt to consider cervical/neuro overlap

material safety concern
+ no clinician disposition
→ do not generate routine reassuring wording

no selected safety concern
→ must not generate `no red flags`
```

---

# 11. Generated wording examples

## 11.1 Rotator-cuff-related shoulder pain

> Πόνος ώμου σχετιζόμενος με το στροφικό πέταλο / τενοντοπάθεια στροφικού πετάλου, με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη ενεργητική αποκατάσταση με έμφαση στη σταδιακή φόρτιση, την ενδυνάμωση/αντοχή του ώμου και της ωμοπλάτης όπου ενδείκνυται, την εκπαίδευση και την επάνοδο στις επιλεγμένες δραστηριότητες. [Selected adjuncts if present.]

The formatter should not automatically add the term `impingement syndrome`.

## 11.2 Confirmed full-thickness rotator-cuff tear — conservative pathway

> Επιβεβαιωμένη ρήξη πλήρους πάχους του [selected tendon if entered] στροφικού πετάλου, για συντηρητική αντιμετώπιση/αποκατάσταση, με [selected functional deficits]. Παρακαλώ για εξατομικευμένο πρόγραμμα φυσιοθεραπείας με στόχο τη βελτίωση της λειτουργίας, του διαθέσιμου ενεργητικού εύρους κίνησης και της μυϊκής ικανότητας σύμφωνα με την κλινική εικόνα και τις τυχόν καταγεγραμμένες οδηγίες/περιορισμούς.

No statement that surgery is unnecessary is generated.

## 11.3 Calcific rotator-cuff tendinopathy

> Ασβεστοποιός τενοντοπάθεια στροφικού πετάλου με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για ενεργητική φυσιοθεραπευτική αποκατάσταση με στόχο τη βελτίωση του πόνου, της λειτουργίας και της προοδευτικής φόρτισης. [If clinician-selected: ESWT may be considered as an adjunct where appropriate and available.]

## 11.4 Formal adhesive capsulitis

Only when `formal_adhesive_capsulitis_diagnosis=yes`:

> Συμφυτική θυλακίτιδα / frozen shoulder με σημαντικό περιορισμό [selected active/passive ROM findings], πόνο [if selected] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση προσαρμοσμένη στην ερεθιστικότητα, με στόχο τη σταδιακή βελτίωση της κινητικότητας και της λειτουργικότητας, ενεργητική συμμετοχή και αυτοδιαχείριση.

## 11.5 Shoulder stiffness without formal adhesive-capsulitis diagnosis

> Επώδυνος ώμος με περιορισμό ενεργητικού και παθητικού εύρους κίνησης [selected directions if entered] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη αποκατάσταση με βάση την κλινική εικόνα και την ερεθιστικότητα.

No unsupported frozen-shoulder diagnosis is added.

## 11.6 Instability/dislocation rehabilitation

> [Clinician-established instability/dislocation context] του [side] ώμου, [first-time/recurrent and direction if entered], με λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη αποκατάσταση με έμφαση στη δύναμη/αντοχή του στροφικού πετάλου και της ωμοπλάτης, στη δυναμική σταθερότητα/ιδιοδεκτικότητα και στη σταδιακή, κριτηριο-βασισμένη επάνοδο στις απαιτούμενες δραστηριότητες, με τήρηση των καταγεγραμμένων περιορισμών.

## 11.7 Glenohumeral OA

> Οστεοαρθρίτιδα γληνοβραχιόνιας άρθρωσης με [selected pain/ROM findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη ενεργητική αποκατάσταση με στόχο τη βελτίωση της λειτουργικότητας, της ανεκτής κινητικότητας και της μυϊκής ικανότητας, καθώς και την υποστήριξη της αυτοδιαχείρισης/φυσικής δραστηριότητας.

## 11.8 Post-traumatic assessed shoulder injury

> Μετατραυματικός πόνος/δυσλειτουργία του [side] ώμου μετά από [selected established injury/context], με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση σύμφωνα με την καταγεγραμμένη κλινική/απεικονιστική εικόνα και τους τυχόν περιορισμούς.

No negative fracture/tear/instability statement is generated unless specifically supported.

---

# 12. Evidence-governance boundary

Stable structural decisions proposed for shoulder:

```text
special test != diagnosis
painful arc/"impingement" finding != standalone structural diagnosis
active ROM != passive ROM
pain-inhibited effort != proven tear
formal full-thickness tear requires established diagnosis
adhesive capsulitis is never inferred from generic stiffness
instability is never inferred from one apprehension test
acute traumatic marked weakness requires explicit reassessment semantics
active/function-oriented rehabilitation is the core direction
adjunct techniques remain optional
```

Evidence-sensitive items to recheck immediately before CU-2 production implementation:

- exact exercise/loading recommendations and dosage by presentation;
- manual-therapy role/duration;
- acupuncture/dry-needling evidence;
- ESWT/calcific-specific recommendations;
- full-thickness-tear nonoperative/surgical surveillance wording;
- frozen-shoulder intervention wording;
- instability/dislocation return-to-sport and restriction guidance;
- any newer guideline superseding the cited frameworks.

Current evidence anchors reviewed for this candidate:

- 2025 JOSPT/APTA Rotator Cuff Tendinopathy CPG;
- 2025 AAOS Management of Rotator Cuff Injuries CPG;
- ACR Appropriateness Criteria Acute Shoulder Pain, revised 2024;
- 2025 Clinical Practice Guidelines for Diagnosis and Non-Surgical Treatment of Primary Frozen Shoulder;
- current international/ESSKA consensus literature for shoulder instability/dislocation, including 2026 age/time-specific traumatic anterior-instability consensus;
- AAOS Glenohumeral Osteoarthritis CPG remains the principal AAOS disease-specific guideline but is older and has limited evidence for exact nonsurgical PT prescriptions.

---

# 13. Product-owner review questions before freeze

1. Do the proposed primary pathways match your real referral practice: RCRSP/tendinopathy, confirmed full-thickness cuff tear for conservative rehab, calcific tendinopathy, adhesive capsulitis, instability/dislocation, glenohumeral OA and post-traumatic assessed shoulder injury?
2. Do you commonly refer **proximal long-head biceps tendinopathy** as a stand-alone diagnosis, or should it remain a selectable secondary problem/modifier under the rotator-cuff-related pathway?
3. Do you commonly refer **AC-joint pain / AC-joint osteoarthritis / AC sprain** often enough to justify a separate primary pathway?
4. Do you see enough **SLAP/labral lesions** in conservative physiotherapy to justify an explicit pathway, or should established labral pathology remain clinician free text/secondary diagnosis?
5. Do you see **post-operative shoulder patients** (rotator-cuff repair, stabilization, arthroplasty, etc.) in your practice? If not, keep shoulder post-op out of the active MVP as we did for cervical/lumbar.
6. Keep acupuncture and dry needling as optional shoulder adjuncts, with dry-needling competence/availability caveat as in lumbar?
7. Keep ESWT visible only for calcific rotator-cuff tendinopathy and exclude therapeutic ultrasound from the standard adjunct list?
8. Are there other common shoulder referrals in your practice that should be represented before freeze?

No runtime implementation is authorized by this candidate.
