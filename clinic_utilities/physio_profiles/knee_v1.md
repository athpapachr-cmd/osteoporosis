# Knee Physiotherapy Referral Profile v1 — CU-1 DESIGN CANDIDATE

> **STATUS:** DESIGN CANDIDATE — product-owner review required before freeze.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful knee referral choices while preserving diagnosis-vs-finding separation, meniscal/ligament/extensor-mechanism safety semantics, active rehabilitation, procedure-specific restrictions and physiotherapist autonomy.
> **Prior frozen regional profiles:** `cervical_v1_1.md`, `lumbar_v1_1.md`, `shoulder_v1_1.md`, `elbow_v1_1.md`, `wrist_hand_v1_1.md`.

---

# 1. Core design contract

```text
PRIMARY CLINICAL PATHWAY
+
ACTUAL FINDINGS / MODIFIERS
+
FUNCTIONAL IMPACT
+
SAFETY / STRUCTURAL / POSTOPERATIVE CONTEXT
+
CONFIRMED GOALS
+
CONFIRMED REHABILITATION DIRECTIONS
```

Inherited hard invariants:

```text
suggested != examined
suggested != selected
symptom != diagnosis
subjective giving-way != objectively demonstrated instability
pain-limited effort != structural weakness or tendon rupture
special/provocation test != diagnosis
MRI finding != automatically symptomatic diagnosis
not assessed != normal
brace/taping/orthosis != automatically mandatory
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

The utility may structure a referral and provide consistency prompts. It must not autonomously diagnose knee OA, meniscal tear, patellofemoral pain, patellar tendinopathy, ligament rupture, patellar instability, osteochondral injury or postoperative complication.

---

# 2. Proposed default primary knee pathways

## K1 — Knee osteoarthritis

Structured key:

```text
knee_osteoarthritis
```

Display:

> Οστεοαρθρίτιδα γόνατος

Possible clinician-entered compartment/context:

```text
medial_tibiofemoral
lateral_tibiofemoral
patellofemoral
multicompartment
not_stated
```

Useful findings/context:

```text
weight-bearing pain
start-up pain / stiffness
walking tolerance
stair limitation
sit-to-stand limitation
squat/kneeling limitation
knee flexion/extension restriction
quadriceps weakness if assessed
hip/lower-limb weakness if assessed
balance/neuromuscular deficit if assessed
varus/valgus alignment context if relevant
effusion if present
radiographic OA context if available
BMI/weight-management context when clinically relevant
```

Hard rules:

```text
radiographic OA alone != proof that all current knee symptoms arise from OA
radiographic severity != automatic functional severity
```

Core rehabilitation directions may include:

```text
education / self-management
individualized strengthening
low-impact aerobic activity
progressive functional exercise
neuromuscular / balance / coordination training when relevant
mobility/ROM exercise when restricted
graded walking/activity exposure
weight-management support/referral when relevant and clinician-selected
```

Current AAOS guidance strongly supports supervised, unsupervised and/or aquatic exercise and self-management/education, with moderate support for brace use and neuromuscular training in selected patients. No structural-reversal promise is generated.

Support options may include a knee brace, cane/walking aid or taping when clinically appropriate; these are not globally preselected.

## K2 — Degenerative meniscal lesion / tear — conservative rehabilitation

Structured key:

```text
degenerative_meniscal_lesion_conservative_rehabilitation
```

Display:

> Εκφυλιστική βλάβη / ρήξη μηνίσκου — συντηρητική αποκατάσταση

Use when the clinician has established or is carrying a degenerative meniscal diagnosis and a conservative pathway is appropriate.

Possible context:

```text
medial / lateral / both / not_stated
MRI-established tear if available
degenerative OA overlap
joint-line pain/tenderness if examined
clicking/catching if present
true locking: yes/no/not_stated
recurrent effusion if present
squat/twist/pivot provocation
```

Hard rules:

```text
MRI meniscal tear != automatically symptomatic pain generator
joint-line tenderness or McMurray/Thessaly finding != definitive structural diagnosis
clicking/catching != true locked knee
```

Evidence boundary:

- exercise-based physical therapy is a first-line treatment for degenerative meniscal lesions and has long-term randomized evidence showing noninferiority to arthroscopic partial meniscectomy for patient-reported function;
- the utility must not imply that a degenerative MRI tear automatically requires arthroscopy;
- true locking or another unresolved structural concern leaves this routine pathway.

Possible rehabilitation directions:

```text
quadriceps and lower-limb strengthening
progressive functional loading
ROM restoration if restricted
neuromuscular/balance control
movement/load modification
progressive squat/stair/walking tolerance
graded return to work/sport according to symptoms/function
```

## K3 — Acute isolated meniscal injury — assessed nonoperative pathway

Structured key:

```text
acute_isolated_meniscal_injury_nonoperative
```

Display:

> Οξεία τραυματική κάκωση μηνίσκου — συντηρητική αποκατάσταση μετά από αξιολόγηση

Use only when an acute meniscal injury has been clinically assessed and nonoperative rehabilitation is the current plan.

Required/important context:

```text
injury date / phase
medial / lateral / not_stated
tear morphology if established
displaced/displacing: yes/no/not_stated
ROM restriction / true locking
repairable-lesion or specialist context if known
weight-bearing restriction if any
brace restriction if any
associated ligament/chondral injury excluded or separately documented
```

Hard safety boundary:

```text
acute meniscal injury
+ displaced/displacing tear OR true locked knee / major ROM block OR repairable lesion requiring timely specialist decision
→ orthopaedic reassessment / early specialist pathway
→ no routine unrestricted rehabilitation wording
```

AAOS 2024 guidance supports a role for rehabilitation in selected acute non-displaced isolated tears but identifies displaced/displacing tears that restrict ROM as candidates for acute surgical intervention.

## K4 — Patellofemoral pain

Structured key:

```text
patellofemoral_pain
```

Default presentation wording without formal diagnosis:

> Πρόσθιος πόνος γόνατος με επιγονατιδομηριαία χαρακτηριστικά

Optional clinician assertion:

```text
formal_patellofemoral_pain_diagnosis: yes / no / not_stated
```

If `yes`:

> Επιγονατιδομηριαίος πόνος

Useful findings/context:

```text
peripatellar/retropatellar pain
pain with stairs
pain with squat
pain with running/jumping
pain with prolonged sitting
pain with kneeling if relevant
load-volume change
quadriceps weakness if assessed
hip strength/control deficit if assessed
movement/running pattern finding if assessed
foot/ankle contribution if assessed
patellar taping response if actually tested
```

Hard rules:

```text
anterior knee pain != automatically patellofemoral pain
patellar crepitus != chondromalacia diagnosis
patellofemoral cartilage MRI finding != automatically symptomatic pain generator
```

Core rehabilitation direction:

```text
education + knee-targeted exercise
± hip-targeted exercise according to assessment
load/activity modification and graded return
```

2024 best-practice evidence supports education and knee-targeted with or without hip-targeted exercise as the primary intervention. Supporting interventions such as taping, prefabricated foot orthoses, manual therapy or movement/running retraining should be selected only when the individual presentation justifies them.

## K5 — Patellar tendinopathy

Structured key:

```text
patellar_tendinopathy
```

Default wording without formal diagnosis:

> Πρόσθιος πόνος γόνατος / επιγονατιδικού τένοντα με χαρακτηριστικά load-related tendinopathy

Optional clinician assertion:

```text
formal_patellar_tendinopathy_diagnosis: yes / no / not_stated
```

Useful findings/context:

```text
inferior-pole / patellar-tendon pain
localized tendon tenderness
pain with jumping/hopping/running
pain with squat/decline squat if examined
sport/gym loading history
reactive vs chronic irritability context if clinician-entered
strength/load capacity
return-to-sport target
ultrasound/MRI context if available
```

Hard rules:

```text
anterior knee pain + tendon tenderness != automatic patellar tendinopathy
ultrasound/MRI tendon change != automatically symptomatic tendinopathy
```

Core direction:

```text
load monitoring / education
progressive tendon-loading exercise
progressive quadriceps and lower-limb strength
graded energy-storage / jumping-running return when relevant
```

Recent evidence supports progressive loading as first-line rehabilitation. The candidate does not freeze a single mandatory eccentric, isometric or heavy-slow-resistance protocol because comparative evidence does not establish one universally superior loading strategy.

ESWT is not proposed as a default adjunct in v1 because randomized evidence does not show clear superiority over sham when exercise is provided in both groups.

## K6 — Established knee ligament injury / instability — nonoperative rehabilitation

Structured key:

```text
knee_ligament_injury_instability_nonoperative
```

Required clinician-entered subtype:

```text
ACL
PCL
MCL
LCL
posterolateral_corner
combined_ligament_injury
other_established_ligament_injury
```

Display:

> Κάκωση / αστάθεια συνδέσμων γόνατος — συντηρητική αποκατάσταση

Required context where relevant:

```text
injury date / phase
partial vs complete if established
grade if established
objective instability if established
operative vs nonoperative decision
brace status
weight-bearing restriction
ROM restriction
sport/work demand
associated meniscus/chondral/neurovascular injury context
```

Hard rules:

```text
subjective giving-way != objective ligament instability
Lachman/drawer/valgus/varus/pivot finding != autonomous tear diagnosis
a single positive test != automatic injury grade
```

Possible rehabilitation directions:

```text
restore ROM according to restrictions
progressive strength
neuromuscular/proprioceptive control
progressive gait/loading
running/jumping/change-of-direction progression when relevant
criterion-based return to sport/work
```

ACL-specific return-to-sport decisions should not be generated from time alone. Functional testing may inform the decision but does not by itself guarantee safe return.

High-grade/combined instability, peroneal nerve deficit, vascular concern or unresolved repair/reconstruction indication requires specialist/reassessment semantics.

## K7 — Patellar instability / dislocation rehabilitation

Structured key:

```text
patellar_instability_dislocation_rehabilitation
```

Display:

> Αστάθεια / εξάρθρημα επιγονατίδας — αποκατάσταση

Context:

```text
first-time vs recurrent
traumatic vs low-energy/atraumatic context
reduction completed if acute dislocation
osteochondral injury assessed if relevant
MPFL injury/reconstruction context
anatomic recurrence-risk factors if established
brace status
weight-bearing/ROM restrictions
```

Hard rules:

```text
patellar apprehension != autonomous instability diagnosis
first-time dislocation != automatically routine PT without structural assessment
```

The ESSKA 2024 formal consensus supports individualized management and identifies physiotherapy as an essential component of operative and nonoperative treatment, while long-term benefit from bracing is unclear. Osteochondral injury or high recurrence-risk structural context can change management and requires specialist awareness.

## K8 — Post-traumatic knee pain / stiffness after assessed injury

Structured key:

```text
post_traumatic_knee_pain_stiffness
```

Display:

> Μετατραυματικός πόνος / δυσκαμψία γόνατος μετά από αξιολογημένη κάκωση

Use only after unresolved fracture, major ligament instability, displaced meniscal lesion, extensor-mechanism rupture, osteochondral injury or neurovascular concern has been addressed as required.

Required context:

```text
injury/date or phase
established structural diagnosis if any
imaging/orthopaedic context
immobilization history
current ROM/loading/weight-bearing restrictions
```

The utility never labels an unassessed knee trauma as a `simple sprain`.

## K9 — Postoperative knee rehabilitation — candidate active pathway

Structured key:

```text
postoperative_knee_rehabilitation
```

Display:

> Μετεγχειρητική αποκατάσταση γόνατος

Candidate procedure subtypes:

```text
total_knee_arthroplasty_TKA
unicompartmental_knee_arthroplasty
ACL_reconstruction
PCL_or_multiligament_reconstruction
meniscus_repair
partial_meniscectomy
MPFL_reconstruction_or_patellar_stabilization
cartilage_or_osteochondral_procedure
patellar_or_quadriceps_tendon_repair
other_knee_operation
```

Required context:

```text
procedure
operation date
surgeon/protocol where available
weight-bearing status
brace status
ROM restrictions
loading/strengthening restrictions
graft/repair-specific precautions
wound/infection context if relevant
return-to-work/sport target
```

Hard rule:

```text
procedure-specific protocol / surgeon restriction > generic knee rehabilitation default
```

No generic postoperative timeline is invented.

TKA-specific evidence note:

- APTA published a revised 2026 CPG supporting early mobilization, progressive strength/ROM/physical activity, motor-function training and early postoperative NMES in appropriate TKA patients;
- these recommendations are TKA-specific and must not leak into OA, meniscus or ligament pathways as universal defaults.

Product-owner confirmation is required before K9 becomes frozen as a default pathway.

---

# 3. Candidate secondary / rare / advanced entities

## 3.1 Quadriceps tendinopathy

Proposed role:

```text
quadriceps_tendinopathy → secondary/advanced unless product owner sees it frequently
```

Should remain distinct from acute quadriceps-tendon rupture.

## 3.2 Iliotibial band syndrome / lateral running-related knee pain

Proposed role:

```text
iliotibial_band_syndrome → candidate primary or secondary depending real workflow
```

Lateral knee pain alone does not establish ITB syndrome. Running/load context and clinician assessment are required.

## 3.3 Pes anserine region pain / bursitis / tendinopathy

Proposed role:

```text
pes_anserine_pain_or_established_bursitis_tendinopathy → secondary/context
```

Tenderness in the pes region does not establish bursitis. Acute erythematous/swollen or uncertain inflammatory/infectious presentation is not routine physiotherapy.

## 3.4 Baker / popliteal cyst

Medical/context only by default:

```text
known_Baker_or_popliteal_cyst_context
```

The generator does not infer a Baker cyst from posterior swelling and does not use it to dismiss DVT or other vascular pathology.

## 3.5 Prepatellar / infrapatellar bursitis

Medical/context or rare rehabilitation only after diagnosis and infection exclusion. Septic bursitis concern is a medical pathway.

## 3.6 Established osteochondral / chondral lesion or osteochondritis dissecans

Rare/advanced structural context:

```text
established_chondral_or_osteochondral_lesion
osteochondritis_dissecans
```

Imaging findings do not automatically establish symptom causality. Unstable lesion/loose body/mechanical block requires specialist context.

## 3.7 Meniscal root tear / complex structural meniscal lesion

Rare/advanced structural pathway rather than routine K2 degenerative meniscus. Root tears and other repair-relevant complex lesions require explicit orthopaedic context because they are not equivalent to common degenerative meniscal findings.

## 3.8 Hoffa fat-pad / synovial plica presentation

Not proposed as routine primary diagnosis. May be clinician-entered secondary context only if established. Anterior pain or imaging signal alone must not create these diagnoses.

## 3.9 Inflammatory / crystal knee context

Directly selectable only when established:

```text
known_inflammatory_arthritis_knee_involvement
known_gout_or_crystal_disease_context
```

Acute hot swollen monoarthritis remains a medical diagnostic issue rather than a routine physiotherapy referral.

## 3.10 Adolescent traction-apophysitis context

Possible future entries if the product owner sees adolescents:

```text
Osgood_Schlatter_disease
Sinding_Larsen_Johansson_syndrome
```

Not promoted in v1 without real-workflow need.

---

# 4. Examination findings — selectable only when actually assessed

## 4.1 Pain / symptom behaviour

```text
medial joint-line pain
lateral joint-line pain
anterior/peripatellar pain
patellar-tendon pain
quadriceps-tendon/suprapatellar pain
posterior/popliteal pain
pes-anserine-region pain
lateral femoral-condyle/ITB-region pain
diffuse knee pain
night pain
pain with walking
pain with stairs
pain with sit-to-stand
pain with squat
pain with kneeling
pain with running
pain with jumping/landing
pain with pivot/change of direction
pain with prolonged sitting
```

## 4.2 Mechanical symptoms

```text
clicking
catching
subjective giving-way
true locking / inability to fully move through expected ROM
recurrent instability episode
patellar subluxation/dislocation history
```

`clicking/catching` and `true locked knee` must remain distinct.

## 4.3 Swelling / effusion

```text
no swelling if actually assessed
mild/moderate/large effusion if assessed
recurrent activity-related effusion
acute hemarthrosis context
localized bursal swelling
posterior/popliteal swelling
```

## 4.4 Range of motion

```text
flexion restricted
extension restricted
extension lag
painful active ROM
painful passive ROM
fixed flexion contracture if present
```

Extension lag is a finding; after trauma it must not be treated as benign until extensor-mechanism integrity is appropriately assessed.

## 4.5 Strength / performance

```text
quadriceps weakness if assessed
hamstring weakness if assessed
hip abductor/extensor weakness if assessed
calf weakness if assessed
grip not applicable
single-leg squat deficit
sit-to-stand deficit
step-down deficit
balance/proprioception deficit
hop/performance deficit if assessed
running/landing/change-of-direction deficit if assessed
load intolerance without measured weakness
```

## 4.6 Special/provocation findings

Secondary expander only:

```text
joint-line tenderness
McMurray-type finding
Thessaly-type finding
Lachman finding
anterior-drawer finding
pivot-shift finding
posterior-drawer / posterior-sag finding
valgus-stress finding
varus-stress finding
patellar-apprehension finding
patellar-compression/grind-type finding
other clinician-entered test
```

Tests remain findings, not diagnoses.

---

# 5. Neurological / neurovascular model

Use when relevant to trauma, fibular-head/LCL/PLC injury, postoperative concern or lower-limb neurological overlap.

```text
motor: normal / abnormal / not_assessed
sensory: normal / abnormal / not_assessed
vascular_status: normal / abnormal / not_assessed
```

Possible specific context:

```text
common_peroneal_motor/sensory status
foot-drop concern
pulses/perfusion if clinically assessed
```

```text
not_assessed != normal
```

No global `neurovascularly intact` wording is generated from missing data.

---

# 6. Safety / reassessment semantics

## 6.1 High-priority structural concerns

```text
acute trauma with unresolved fracture concern
true locked knee / major mechanical ROM block
acute displaced/displacing meniscal tear concern
acute extensor-mechanism rupture concern
new inability to straight-leg raise after acute injury
acute patellar dislocation not appropriately assessed/reduced
major ligament instability / multiligament injury
new neurovascular deficit after trauma
large acute hemarthrosis with unresolved structural injury
osteochondral loose-body / unstable lesion concern
```

## 6.2 Medical / inflammatory / vascular concerns

```text
hot swollen knee / septic arthritis concern
systemic illness with acute knee swelling
wound/drainage/cellulitis or postoperative infection concern
acute calf swelling/tenderness with DVT concern
PE symptoms / cardiopulmonary concern
unexplained rapidly progressive swelling
acute crystal/inflammatory monoarthritis not yet established
```

## 6.3 Postoperative concerns

```text
missing procedure/protocol/restrictions
wound complication/infection concern
new disproportionate swelling/pain
DVT/PE concern
new neurovascular deficit
loss of expected extensor mechanism function
unexpected progressive ROM loss requiring surgical team feedback
```

Safety state:

```text
safety_screen_status:
  not_assessed
  no_specific_concern_identified
  concern_present
```

Disposition when concern present:

```text
reviewed_and_appropriate_to_proceed
imaging/medical reassessment arranged
orthopaedic/sports-medicine pathway underway
urgent/same-day assessment arranged
routine physiotherapy deferred
other
```

No default `no red flags`, `no fracture`, `stable ligaments`, `meniscus intact`, `no DVT`, `no infection` or `neurovascularly intact` wording is generated from missing information.

---

# 7. Functional limitations

```text
walking distance/tolerance
stairs up
stairs down
sit-to-stand
prolonged standing
squat
kneeling
floor transfer
car transfer
driving
running
jumping/landing
pivot/change of direction
sport-specific activity
gym / resistance training
manual work
carrying loads
work duties
ADLs/self-care
sleep disturbance
patient-priority activity / free text
```

---

# 8. Context-sensitive goals

Nothing is globally preselected.

Candidate goal families:

```text
reduce symptom irritability
restore safe knee ROM
improve quadriceps strength/capacity
improve lower-limb strength
improve gait/walking tolerance
improve stair and sit-to-stand function
improve balance/neuromuscular control
improve tendon load capacity
restore dynamic knee stability
restore patellar stability/control where relevant
progressive return to running/jumping/pivoting
criterion-based return to sport/work
restore function within surgical/structural restrictions
self-management and load adaptation
```

Condition cautions:

- OA: no structural reversal promise;
- degenerative meniscus: no promise that PT heals MRI morphology;
- acute meniscus/ligament: structural restriction outranks generic exercise progression;
- patellar tendinopathy: no universal loading protocol is mandatory;
- postoperative: surgeon/procedure-specific restrictions outrank generic defaults.

---

# 9. Rehabilitation directions and supports

## 9.1 Core active directions

```text
physiotherapy assessment and individualized active rehabilitation
education / self-management
activity/load modification
progressive strengthening
quadriceps strengthening
hip/lower-limb strengthening where relevant
ROM/mobility exercise where safe
neuromuscular / proprioceptive / balance training where relevant
gait retraining where relevant
movement/running retraining where relevant
progressive tendon loading where relevant
progressive functional loading
criterion-based return to work/gym/sport
home exercise programme where appropriate
```

## 9.2 Brace / taping / foot-support category

Possible condition-sensitive selections:

```text
knee OA brace where appropriate
patellofemoral taping
patellofemoral support/brace where appropriate
post-injury ligament brace according to plan
postoperative brace according to protocol
prefabricated foot orthosis for selected patellofemoral-pain presentation
walking aid / cane strategy where appropriate
```

Hard rule:

```text
brace/tape/orthosis suggested != automatically required
exact injury/surgical protocol > generic support suggestion
```

## 9.3 Optional adjunct expander — candidate

Possible items:

```text
manual therapy / joint mobilization where impairment-specific and appropriate
soft-tissue techniques where appropriate
taping where relevant
thermal strategy for selected OA symptoms
acupuncture — unresolved product-owner decision for knee OA only
dry needling — unresolved product-owner decision; not core treatment
ESWT — not proposed as default; unresolved only if product owner uses it specifically for patellar tendinopathy
NMES — procedure/context-specific, especially postoperative TKA; not a generic knee-OA modality
```

### Manual therapy

May support selected OA or patellofemoral presentations as part of a multimodal programme, but active exercise remains core. Guideline recommendations differ across frameworks, so manual therapy must not be framed as universally required or disease-modifying.

### Acupuncture

Knee-OA guidelines are not fully concordant: ACR conditionally recommends acupuncture, AAOS provides limited support, while NICE recommends against acupuncture/dry needling for OA. If the product owner uses acupuncture for knee OA, it should remain an explicitly evidence-sensitive optional adjunct rather than a core recommendation.

### Dry needling

Recent meta-analysis suggests possible short- to mid-term additive benefit with exercise for knee OA, but heterogeneity and long-term uncertainty remain. It should not be added automatically merely because newer evidence is positive.

### ESWT

Current patellar-tendinopathy evidence does not establish clear superiority over sham ESWT when both groups perform exercise. It is therefore omitted from default v1 unless the product owner's actual practice requires an evidence-sensitive option.

### NMES

Do not use as a generic OA electrotherapy recommendation. The 2026 APTA TKA CPG supports early postoperative quadriceps NMES in appropriate TKA patients; this belongs inside the TKA/postoperative context only.

---

# 10. Shared fracture / post-immobilization boundary

Knee-region fractures route to the shared fracture profile:

```text
patella fracture
distal femur fracture
proximal tibia / tibial plateau fracture
proximal fibula fracture
other knee-region fracture
```

Future shared required context:

```text
fracture site
date/phase
treatment
healing/stability status
immobilization/brace status
weight-bearing status
ROM restrictions
loading restrictions
orthopaedic instructions
```

```text
fracture route + unresolved healing/loading context
→ warning
→ no unrestricted routine rehabilitation wording
```

---

# 11. Deterministic consistency rules

```text
K1 + x-ray OA only
→ do not automatically attribute all symptoms to OA

K2 + MRI degenerative tear only
→ do not auto-assert symptomatic meniscal pain generator

K2 + true locked knee
→ leave routine degenerative-meniscus pathway; structural reassessment prompt

K3 + displaced/displacing tear or repairable-lesion concern
→ specialist prompt before generic rehabilitation

K4 + anterior knee pain only
→ do not infer patellofemoral pain automatically

K4 + cartilage/chondromalacia imaging wording only
→ do not convert to symptomatic PF diagnosis automatically

K5 + tendon imaging abnormality only
→ do not infer symptomatic patellar tendinopathy

K6 + subjective giving-way only
→ do not infer ligament rupture/instability

K6 + test finding only
→ do not infer exact ligament grade

K6 + multiligament/neurovascular concern
→ urgent/specialist reassessment semantics

K7 + acute first-time dislocation + unresolved osteochondral/structural assessment
→ no routine unrestricted PT wording

K8 + unresolved fracture/locked knee/extensor rupture/major instability
→ safety prompt

K9 postoperative + missing procedure/protocol/restrictions
→ warning

TKA-specific NMES suggestion outside postoperative/TKA context
→ invalid

acute loss of straight-leg raise + extensor-mechanism concern
→ structural reassessment prompt

posterior knee swelling + no established Baker cyst
→ do not infer Baker cyst or dismiss DVT

hot swollen knee + diagnostic uncertainty
→ medical reassessment; no routine reassuring PT wording

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurovascular component
→ never generate normal wording
```

---

# 12. Generated wording examples

## 12.1 Knee OA

> Οστεοαρθρίτιδα του [side] γόνατος με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση με έμφαση σε προοδευτική ενδυνάμωση, λειτουργική άσκηση, βελτίωση κινητικότητας/νευρομυϊκού ελέγχου όπου ενδείκνυται και εκπαίδευση αυτοδιαχείρισης. [Brace/taping/walking-aid support only if selected.]

## 12.2 Degenerative meniscal lesion

> Εκφυλιστική βλάβη / ρήξη [medial/lateral] μηνίσκου του [side] γόνατος, με [selected findings] και περιορισμό σε [selected activities], για συντηρητική αντιμετώπιση. Παρακαλώ για προοδευτική αποκατάσταση δύναμης, κινητικότητας, νευρομυϊκού ελέγχου και ανοχής στη λειτουργική φόρτιση. Δεν υπονοείται χειρουργική ένδειξη από την απεικονιστική ρήξη και μόνο.

## 12.3 Acute nonoperative meniscal injury

> Οξεία κάκωση [medial/lateral] μηνίσκου του [side] γόνατος μετά από αξιολόγηση, για συντηρητική αποκατάσταση σύμφωνα με τους καταγεγραμμένους περιορισμούς σε φόρτιση/κινητικότητα. Παρακαλώ για προοδευτική αποκατάσταση ROM, δύναμης και λειτουργίας, με επανεκτίμηση εάν εμφανιστεί/επιμένει πραγματικό κλείδωμα ή σημαντικός μηχανικός περιορισμός.

## 12.4 Patellofemoral pain

> Πρόσθιος/επιγονατιδομηριαίος πόνος του [side] γόνατος με [selected findings] και περιορισμό σε [stairs/squat/running/etc]. Παρακαλώ για εκπαίδευση και εξατομικευμένη άσκηση με έμφαση σε knee-targeted ± hip-targeted strengthening και σταδιακή επαναφορά στη φόρτιση. [Taping/foot orthosis/movement retraining only if selected.]

## 12.5 Patellar tendinopathy

> [Clinician-established patellar tendinopathy / patellar-tendon load-related presentation] του [side] γόνατος με [selected findings]. Παρακαλώ για διαχείριση φορτίου και προοδευτικό πρόγραμμα tendon loading/ενδυνάμωσης, με σταδιακή επαναφορά σε [jumping/running/sport] σύμφωνα με την ανοχή και τη λειτουργική πρόοδο.

## 12.6 Ligament injury

> Κάκωση [ACL/PCL/MCL/LCL/PLC] του [side] γόνατος, [established grade/stability if entered], για συντηρητική αποκατάσταση. Παρακαλώ για προστατευμένη και προοδευτική αποκατάσταση ROM, δύναμης, νευρομυϊκού ελέγχου και λειτουργικής σταθερότητας σύμφωνα με τους καταγεγραμμένους περιορισμούς, με criterion-based progression προς εργασία/άθληση όπου απαιτείται.

## 12.7 Postoperative knee

> Μετεγχειρητική αποκατάσταση μετά από [procedure] του [side] γόνατος, επέμβαση [date if entered]. Παρακαλώ για φυσιοθεραπεία σύμφωνα με το διαθέσιμο χειρουργικό πρωτόκολλο και τους καταγεγραμμένους περιορισμούς σε φόρτιση, brace, ROM, ενδυνάμωση και επιστροφή σε δραστηριότητα.

---

# 13. Evidence-governance boundary

Stable structural decisions proposed for Knee v1:

```text
exercise / education = core for knee OA
radiographic OA != automatic symptom generator
degenerative meniscal MRI tear != automatic symptom generator
exercise-based PT = first-line for degenerative meniscal lesions
true locked knee != routine degenerative-meniscus rehabilitation
acute displaced/displacing meniscal tear with ROM block = structural/specialist pathway
anterior knee pain != automatic PFP
PFP core = education + knee-targeted ± hip-targeted exercise
patellar tendinopathy core = progressive load-based rehabilitation
one loading style != mandatory universal patellar-tendon protocol
subjective giving-way != objective ligament instability
ligament special test != autonomous tear grade
patellar dislocation requires structural/osteochondral context
postoperative rehabilitation = exact procedure/protocol governed
TKA-specific interventions do not leak into generic OA pathways
possible Baker cyst != exclusion of DVT
fractures route to shared fracture/post-immobilization profile
```

Current evidence anchors reviewed for this candidate include:

- AAOS Management of Osteoarthritis of the Knee (Non-Arthroplasty), 3rd edition;
- 2019 ACR/Arthritis Foundation OA guideline and NICE OA guidance where framework differences matter;
- 2024 AAOS Acute Isolated Meniscal Pathology CPG;
- 2025 formal EU-US Meniscus Rehabilitation consensus (ESSKA-AOSSM-AASPT), operative and nonoperative parts;
- 5-year ESCAPE randomized trial of exercise PT vs arthroscopic partial meniscectomy for degenerative tears;
- 2024 patellofemoral-pain best-practice guide;
- 2024 Dutch multidisciplinary guideline for PFP and patellar tendinopathy;
- contemporary patellar-tendinopathy progressive-loading evidence including 2026 systematic-review updates;
- 2022 AAOS ACL CPG and Aspetar ACL-reconstruction rehabilitation CPG;
- ESSKA 2024 formal consensus on first-time patellar dislocation;
- APTA Physical Therapist Management of Total Knee Arthroplasty Revision 2026.

Evidence-sensitive details to refresh immediately before CU-2 implementation:

```text
OA brace/taping/manual-therapy wording across frameworks
PFP foot-orthosis/taping/movement-retraining selection
meniscus lesion-specific rehabilitation milestones
ACL/PCL/MCL/LCL criterion-based progression
patellar-instability brace/return-to-sport details
patellar-tendon loading progression
tendon-adjunct evidence including ESWT
TKA 2026 detailed intervention/dosage recommendations
procedure-specific postoperative restrictions
```

---

# 14. Product-owner decisions required before freeze

1. **Postoperative knee:** do you see enough TKA, ACL reconstruction, meniscal repair/meniscectomy, MPFL or other knee surgery to keep K9 as a default primary pathway?
2. **Degenerative vs acute meniscal tear:** I recommend keeping them as two separate pathways because the structural/surgical safety logic differs. Does that match your referrals?
3. **Ligaments:** do you want one consolidated ligament pathway with subtype ACL/PCL/MCL/LCL/PLC, or do ACL and/or MCL deserve their own top-level buttons because you see them frequently?
4. **Patellar instability/dislocation:** do you refer enough of these to keep it as a default primary pathway?
5. **Quadriceps tendinopathy:** default primary or rare/secondary? My default recommendation is rare/secondary unless you see it regularly.
6. **IT band syndrome:** do you see/refer it frequently enough for a default primary pathway?
7. **Pes anserine pain/bursitis/tendinopathy:** default primary, secondary/context, or omit from routine referral menu?
8. **Acupuncture for knee OA:** guidelines conflict. Do you use/refer for it? If yes, I recommend evidence-sensitive optional adjunct only.
9. **Dry needling:** retain as optional for selected knee-OA/myofascial contexts or omit from Knee v1?
10. **ESWT for patellar tendinopathy:** current evidence does not justify default use. Do you nevertheless use/refer for it enough to keep an evidence-sensitive optional item, or omit it?
11. **Taping / knee braces / foot orthoses:** I recommend keeping them as condition-sensitive supports rather than generic adjuncts. Agree?
12. **Adolescents:** do you see Osgood-Schlatter / Sinding-Larsen-Johansson often enough to include a pediatric/adolescent knee pathway?
13. Any common knee entity missing from your real referrals — e.g. Baker cyst after medical assessment, prepatellar bursitis, osteochondral lesion, popliteus/hamstring insertional pain, plica/fat-pad disorder, or something else?

This file remains a **DESIGN CANDIDATE / NOT FROZEN** until those real-workflow decisions are resolved. Runtime implementation remains unauthorized.
