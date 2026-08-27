# Knee Physiotherapy Referral Profile v1.1 — CU-1 FROZEN DESIGN

> **STATUS:** FROZEN CLINICAL/CONTENT DESIGN — product-owner approved 2026-08-27.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful knee referral choices that match the product owner's real workflow while preserving diagnosis-vs-finding separation, meniscal/ligament/extensor-mechanism safety semantics, pediatric/adolescent distinctions, active rehabilitation, procedure-specific restrictions and physiotherapist autonomy.
> **Supersedes as active knee design:** `clinic_utilities/physio_profiles/knee_v1.md`.
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
AGE / SKELETAL-MATURITY CONTEXT WHEN RELEVANT
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

The utility structures a referral and checks consistency. It must not autonomously diagnose knee OA, meniscal tear, patellofemoral pain, patellar/quadriceps tendinopathy, ACL/MCL or other ligament rupture, patellar instability, Hoffa/plica pathology, osteochondral injury, pediatric traction apophysitis or postoperative complication.

---

# 2. Frozen default primary knee pathways

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
radiographic OA alone != proof that all current symptoms arise from OA
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

Support options may include a knee brace, cane/walking aid or taping when clinically appropriate; these are condition-sensitive rather than globally preselected.

### Knee-OA acupuncture policy

The product owner does refer selected knee-OA patients for acupuncture.

Therefore:

```text
acupuncture_for_knee_OA = optional evidence-sensitive adjunct
```

It is never preselected, never a substitute for exercise/self-management and never presented as guideline-unanimous. Guideline positions differ materially, including permissive/limited-support frameworks and NICE guidance recommending against acupuncture for OA.

Dry needling is excluded from Knee v1.1 by product-owner decision.

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

- exercise-based physiotherapy is a first-line approach for degenerative meniscal lesions;
- long-term randomized evidence supports exercise-based PT as noninferior to arthroscopic partial meniscectomy for patient-reported function in common degenerative tears;
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
associated ligament/chondral injury context
```

Hard safety boundary:

```text
acute meniscal injury
+ displaced/displacing tear OR true locked knee / major ROM block OR repairable lesion requiring timely specialist decision
→ orthopaedic reassessment / early specialist pathway
→ no routine unrestricted rehabilitation wording
```

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

Supporting interventions such as taping, prefabricated foot orthoses, manual therapy or movement/running retraining are condition-sensitive and selected only when the individual presentation justifies them.

Adolescents with ordinary patellofemoral pain use this same pathway with age/skeletal-maturity context; they are not automatically moved into the growth-related pediatric subgroup.

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

No single mandatory eccentric, isometric or heavy-slow-resistance protocol is frozen.

ESWT is not a default clinician-selected adjunct. If a treating physiotherapist proposes ESWT, it may be recorded as therapist-proposed context, but the generator does not recommend it by default and does not claim superiority over exercise-based rehabilitation.

## K6 — Quadriceps tendinopathy

Structured key:

```text
quadriceps_tendinopathy
```

Display:

> Τενοντοπάθεια τετρακεφάλου

This is a default pathway because the product owner sees and refers it.

Useful context:

```text
superior-pole / quadriceps-tendon pain
localized tenderness
load-related pain with squat/jump/run/stairs
sport/gym loading history
strength/load-capacity deficit if assessed
ultrasound/MRI context if available
```

Hard rules:

```text
suprapatellar pain != automatic quadriceps tendinopathy
imaging tendon change != automatically symptomatic diagnosis
quadriceps tendinopathy != acute quadriceps-tendon rupture
```

Core rehabilitation is progressive load/capacity restoration with quadriceps and lower-limb strengthening and graded return to activity.

Acute traumatic loss of active extension/straight-leg raise or palpable extensor-mechanism defect leaves this pathway and triggers structural reassessment.

## K7 — ACL injury / instability rehabilitation

Structured key:

```text
ACL_injury_instability_rehabilitation
```

Display:

> Κάκωση / ανεπάρκεια πρόσθιου χιαστού (ACL) — αποκατάσταση

Possible pathway context:

```text
nonoperative_rehabilitation
preoperative_preparation
post_ACL_reconstruction
```

Required context where relevant:

```text
injury date / phase
partial vs complete if established
objective instability if established
operative vs nonoperative decision
graft/procedure if postoperative
brace status
weight-bearing status
ROM restrictions
associated meniscus/chondral injury
sport/work demand
```

Hard rules:

```text
subjective giving-way != objective ACL instability
Lachman/anterior-drawer/pivot-shift finding != autonomous ACL tear diagnosis
single functional test != automatic safe return to sport
time since injury/surgery alone != return-to-sport clearance
```

Core rehabilitation directions:

```text
restore ROM according to restrictions
progressive quadriceps/hamstring/lower-limb strength
neuromuscular/proprioceptive control
running progression when appropriate
plyometric/change-of-direction progression when appropriate
criterion-based functional progression
return-to-sport/work decision coordinated with structural/surgical context
```

Post-ACLR exact graft/procedure and surgeon restrictions outrank generic defaults.

## K8 — MCL injury rehabilitation

Structured key:

```text
MCL_injury_rehabilitation
```

Display:

> Κάκωση έσω πλαγίου συνδέσμου (MCL) — αποκατάσταση

This is a separate default pathway because the product owner sees it frequently.

Required context:

```text
injury date / phase
grade if established
partial/complete if established
isolated vs combined injury
valgus instability if objectively established
brace status
weight-bearing status
ROM restrictions
associated ACL/meniscus/other ligament injury
operative vs nonoperative plan
```

Hard rules:

```text
medial knee pain != MCL injury
valgus stress pain/laxity finding != autonomous tear grade
combined ligament injury != routine isolated-MCL pathway
```

Core rehabilitation may include protected progressive loading, ROM restoration, progressive strength, neuromuscular control and graded return according to stability and restrictions.

## K9 — Patellar instability / dislocation rehabilitation

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
skeletal_maturity when relevant
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

Physiotherapy is an important component of operative and nonoperative management, but structural/osteochondral context and recurrence risk must remain visible. Bracing is condition-sensitive and not represented as a universally beneficial long-term intervention.

## K10 — Iliotibial-band syndrome / lateral running-related knee pain

Structured key:

```text
iliotibial_band_syndrome
```

Default wording without formal diagnosis:

> Πλάγιος πόνος γόνατος με χαρακτηριστικά συνδρόμου λαγονοκνημιαίας ταινίας

Optional clinician assertion:

```text
formal_ITB_syndrome_diagnosis: yes / no / not_stated
```

This remains a default selectable pathway even though it is less common in the product owner's practice.

Useful context:

```text
running/cycling load
lateral femoral-condyle region pain
repetitive flexion-extension provocation
training-volume/surface/change context
hip/lower-limb strength/control findings if assessed
running biomechanics if assessed
```

Hard rule:

```text
lateral knee pain != automatically ITB syndrome
```

Rehabilitation emphasizes load modification, progressive lower-limb strength/capacity and activity/running progression according to findings.

## K11 — Pes anserine region pain / established tendinobursitis

Structured key:

```text
pes_anserine_region_pain_or_tendinobursitis
```

Default presentation wording without formal diagnosis:

> Πόνος περιοχής χήνειου ποδός / pes anserine region

Optional clinician assertion:

```text
formal_pes_anserine_bursitis_or_tendinopathy_diagnosis: yes / no / not_stated
```

This is a default pathway because the product owner sees and refers it frequently.

Useful context:

```text
inferomedial knee pain
localized tenderness
stair/walking load provocation
hamstring/adductor loading context
knee OA overlap
local swelling if present
```

Hard rules:

```text
pes-region tenderness != automatically bursitis
medial knee pain != automatically pes pathology
hot/erythematous/infectious-appearing swelling != routine PT pathway
```

Core rehabilitation may include load modification, progressive hamstring/adductor/lower-limb strength, mobility where relevant and functional progression.

## K12 — Post-traumatic knee pain / stiffness after assessed injury

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

## K13 — Postoperative knee rehabilitation

Structured key:

```text
postoperative_knee_rehabilitation
```

Display:

> Μετεγχειρητική αποκατάσταση γόνατος

This is an active pathway. In the product owner's workflow, **meniscus repair and partial meniscectomy are the most commonly seen postoperative knee referrals**.

Directly selectable procedure subtypes:

```text
meniscus_repair
partial_meniscectomy
ACL_reconstruction
MCL_repair_or_reconstruction
MPFL_reconstruction_or_patellar_stabilization
total_knee_arthroplasty_TKA
unicompartmental_knee_arthroplasty
PCL_or_multiligament_reconstruction
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

### Meniscus surgery distinction

```text
partial_meniscectomy
→ primarily criterion/milestone-based progression according to clinical recovery

meniscus_repair / reconstruction
→ time + criterion-based progression
→ lesion/repair-specific WB/ROM/loading restrictions preserved
```

No generic postoperative timeline is invented.

### TKA-specific boundary

TKA may use TKA-specific evidence-supported strategies such as early mobilization, progressive strength/ROM and, when appropriate, early quadriceps NMES. These interventions must not leak into generic OA, meniscus, PFP or ligament pathways as universal defaults.

---

# 3. Pediatric / adolescent knee category — UI grouping, not a diagnosis

The product owner sees children/adolescents and requested a dedicated category.

Frozen design:

```text
UI category: Παιδιά / Έφηβοι — γόνατο
```

This category is a **navigation/grouping layer**, not a single clinical pathway.

It contains the growth-related anterior-knee entities below and may also route the clinician to standard structural pathways with pediatric context.

## P-K1 — Osgood-Schlatter disease

Structured key:

```text
Osgood_Schlatter_disease
```

Display:

> Νόσος Osgood-Schlatter / αποφυσίτιδα κνημιαίου κυρτώματος

Use when clinician-established or carried as the working diagnosis.

Useful context:

```text
age / skeletal maturity
tibial-tubercle localized pain/tenderness
running/jumping/sport load
growth-spurt context if relevant
quadriceps/hamstring/calf flexibility or strength findings if assessed
functional limitation
```

Core approach:

```text
education
activity/load modification rather than automatic total rest
symptom-guided progressive strengthening/loading
mobility/flexibility work when an actual restriction is present
graded return to sport/activity
```

No rigid universal exercise protocol or fixed return-to-sport timeline is generated because treatment evidence remains limited/heterogeneous.

## P-K2 — Sinding-Larsen-Johansson syndrome

Structured key:

```text
Sinding_Larsen_Johansson_syndrome
```

Display:

> Σύνδρομο Sinding-Larsen-Johansson

Use when clinician-established or carried as the working diagnosis.

Useful context:

```text
age / skeletal maturity
inferior-patellar-pole localized pain/tenderness
running/jumping load
growth-related context
functional limitation
imaging context if available
```

Hard rules:

```text
inferior-pole pain in an adolescent != automatically SLJ
SLJ != patellar tendinopathy by default
```

Core approach is conservative, load/activity modification with progressive restoration of mobility, strength and sport tolerance; no rigid universal protocol is generated.

## Pediatric routing rule

Children/adolescents with the following do **not** use a generic pediatric diagnosis:

```text
patellofemoral pain → K4 + pediatric/adolescent context
meniscal injury → K2/K3/K13 as appropriate + pediatric context
ACL injury → K7 + pediatric context
MCL injury → K8 + pediatric context
patellar instability/dislocation → K9 + skeletal-maturity context
fracture → shared fracture profile + pediatric context
osteochondral/OCD lesion → advanced structural pathway
```

Optional age context:

```text
age_years_optional
skeletal_maturity: immature / mature / not_stated
```

---

# 4. Rare / advanced / secondary entities

## 4.1 PCL / LCL / posterolateral-corner / combined ligament injury

Not default top-level buttons in v1.1.

Possible keys:

```text
PCL_injury
LCL_injury
posterolateral_corner_injury
combined_multiligament_knee_injury
```

Established injuries require exact stability, neurovascular, operative/nonoperative and restriction context.

Combined/multiligament injury is never collapsed into routine ACL or MCL rehabilitation.

## 4.2 Distal hamstring insertional tendinopathy / posteromedial-posterolateral tendon pain

Directly selectable rare/secondary myotendinous entity because the product owner sees it occasionally.

Possible subtypes:

```text
pes_distal_hamstring_insertional_pathology
semimembranosus_or_medial_hamstring_insertional_pathology
biceps_femoris_distal_insertional_pathology
other_established_distal_hamstring_tendon_pathology
```

Pain location alone does not establish tendon diagnosis. Progressive load-based rehabilitation may be selected where clinically appropriate.

## 4.3 Hoffa fat-pad / synovial plica presentation

Rare clinician-entered secondary/advanced pathway/context because the product owner sees it occasionally.

Possible keys:

```text
established_Hoffa_fat_pad_pain_or_impingement
established_synovial_plica_syndrome
```

Hard rules:

```text
anterior knee pain != Hoffa/plica diagnosis
MRI fat-pad signal or visible plica != automatically symptomatic diagnosis
```

## 4.4 Baker / popliteal cyst

Medical/context only by product-owner workflow:

```text
known_Baker_or_popliteal_cyst_context
```

The generator does not infer a Baker cyst from posterior swelling and does not use it to dismiss DVT or other vascular pathology.

## 4.5 Prepatellar / infrapatellar bursitis

Medical/context only by product-owner workflow, not a routine physiotherapy primary pathway.

Septic bursitis concern is a medical pathway.

## 4.6 Established osteochondral / chondral lesion or osteochondritis dissecans

Rare/advanced structural context:

```text
established_chondral_or_osteochondral_lesion
osteochondritis_dissecans
```

Imaging findings do not automatically establish symptom causality. Unstable lesion/loose body/mechanical block requires specialist context.

In children/adolescents, skeletal maturity and lesion stability must remain explicit.

## 4.7 Meniscal root tear / complex structural meniscal lesion

Rare/advanced structural pathway rather than routine K2 degenerative meniscus.

Root tears and other repair-relevant complex lesions require explicit orthopaedic context because they are not equivalent to common degenerative meniscal findings.

## 4.8 Gastrocnemius strain / myotendinous injury

The product owner sees gastrocnemius sprain, but this is routed to the future shared **muscle / myotendinous injury profile**, not duplicated as a knee diagnosis.

Knee referral may carry gastrocnemius injury as secondary context until that shared profile is frozen.

## 4.9 Inflammatory / crystal knee context

Directly selectable only when established:

```text
known_inflammatory_arthritis_knee_involvement
known_gout_or_crystal_disease_context
```

Acute hot swollen monoarthritis remains a medical diagnostic issue rather than a routine physiotherapy referral.

---

# 5. Examination findings — selectable only when actually assessed

## 5.1 Pain / symptom behaviour

```text
medial joint-line pain
lateral joint-line pain
anterior/peripatellar pain
patellar-tendon pain
quadriceps-tendon/suprapatellar pain
posterior/popliteal pain
pes-anserine-region pain
lateral femoral-condyle/ITB-region pain
tibial-tubercle pain
inferior-patellar-pole pain
distal hamstring insertional pain
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

## 5.2 Mechanical symptoms

```text
clicking
catching
subjective giving-way
true locking / inability to move through expected ROM
recurrent instability episode
patellar subluxation/dislocation history
```

`clicking/catching` and `true locked knee` remain distinct.

## 5.3 Swelling / effusion

```text
no swelling if actually assessed
mild/moderate/large effusion if assessed
recurrent activity-related effusion
acute hemarthrosis context
localized bursal swelling
posterior/popliteal swelling
```

## 5.4 Range of motion

```text
flexion restricted
extension restricted
extension lag
painful active ROM
painful passive ROM
fixed flexion contracture if present
```

Extension lag is a finding; after trauma it must not be treated as benign until extensor-mechanism integrity is appropriately assessed.

## 5.5 Strength / performance

```text
quadriceps weakness if assessed
hamstring weakness if assessed
hip abductor/extensor weakness if assessed
calf weakness if assessed
single-leg squat deficit
sit-to-stand deficit
step-down deficit
balance/proprioception deficit
hop/performance deficit if assessed
running/landing/change-of-direction deficit if assessed
load intolerance without measured weakness
```

## 5.6 Special/provocation findings

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

# 6. Neurological / neurovascular model

Use when relevant to trauma, fibular-head/LCL/PLC injury, postoperative concern or lower-limb neurological overlap.

```text
motor: normal / abnormal / not_assessed
sensory: normal / abnormal / not_assessed
vascular_status: normal / abnormal / not_assessed
```

Possible specific context:

```text
common_peroneal motor/sensory status
foot-drop concern
pulses/perfusion if clinically assessed
```

```text
not_assessed != normal
```

No global `neurovascularly intact` wording is generated from missing data.

---

# 7. Safety / reassessment semantics

## 7.1 High-priority structural concerns

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

## 7.2 Medical / inflammatory / vascular concerns

```text
hot swollen knee / septic arthritis concern
systemic illness with acute knee swelling
wound/drainage/cellulitis or postoperative infection concern
acute calf swelling/tenderness with DVT concern
PE symptoms / cardiopulmonary concern
unexplained rapidly progressive swelling
acute crystal/inflammatory monoarthritis not yet established
```

## 7.3 Pediatric/adolescent concerns

```text
acute traumatic physeal/apophyseal fracture concern
persistent severe night/rest pain or systemic concern
inability to bear weight after trauma without adequate assessment
unstable osteochondral/OCD concern
acute extensor-mechanism injury that could mimic traction-apophysitis pain
```

Osgood/SLJ labels must not be used to explain away atypical or high-risk presentations.

## 7.4 Postoperative concerns

```text
missing procedure/protocol/restrictions
wound complication/infection concern
new disproportionate swelling/pain
DVT/PE concern
new neurovascular deficit
loss of expected extensor-mechanism function
unexpected progressive ROM loss requiring surgical-team feedback
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

# 8. Functional limitations

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
school PE / youth sport
work duties
ADLs/self-care
sleep disturbance
patient-priority activity / free text
```

---

# 9. Context-sensitive goals

Nothing is globally preselected.

Goal families:

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
age-appropriate return to school PE/sport
```

Condition cautions:

- OA: no structural reversal promise;
- degenerative meniscus: no promise that PT heals MRI morphology;
- acute meniscus/ligament: structural restriction outranks generic exercise progression;
- patellar/quadriceps tendinopathy: no universal loading protocol is mandatory;
- pediatric traction-apophysitis: no rigid timeline or requirement for total rest;
- postoperative: surgeon/procedure-specific restrictions outrank generic defaults.

---

# 10. Rehabilitation directions and supports

## 10.1 Core active directions

```text
physiotherapy assessment and individualized active rehabilitation
education / self-management
activity/load modification
progressive strengthening
quadriceps strengthening
hamstring/calf/hip/lower-limb strengthening where relevant
ROM/mobility exercise where safe
neuromuscular / proprioceptive / balance training where relevant
gait retraining where relevant
movement/running retraining where relevant
progressive tendon loading where relevant
progressive functional loading
criterion-based return to work/gym/sport
age-appropriate graded return to youth sport/school activity
home exercise programme where appropriate
```

## 10.2 Brace / taping / foot-support category

Frozen policy: these remain **condition-sensitive supports**, not generic adjuncts.

Possible selections:

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

## 10.3 Optional adjunct policy

Frozen selections:

```text
manual therapy / joint mobilization where impairment-specific and appropriate
soft-tissue techniques where appropriate
taping where relevant
thermal strategy for selected OA symptoms
acupuncture for selected knee OA → optional evidence-sensitive adjunct
```

Explicitly excluded as clinician-generated Knee v1.1 adjunct:

```text
dry needling
routine/default ESWT
```

ESWT may be documented as a treating-physiotherapist-proposed intervention in patellar tendinopathy, but is not suggested by the generator and is not represented as superior to exercise.

NMES is procedure/context-specific, especially postoperative TKA or other explicit postoperative indications, and not a generic OA modality.

---

# 11. Shared fracture / post-immobilization boundary

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
age/skeletal maturity when relevant
```

```text
fracture route + unresolved healing/loading context
→ warning
→ no unrestricted routine rehabilitation wording
```

---

# 12. Deterministic consistency rules

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

K5/K6 + tendon imaging abnormality only
→ do not infer symptomatic tendon diagnosis

K6 + acute loss of straight-leg raise
→ do not label quadriceps tendinopathy; extensor-mechanism reassessment prompt

K7 + subjective giving-way only
→ do not infer ACL rupture/instability

K7 + time since ACLR only
→ do not generate return-to-sport clearance

K8 + medial pain/valgus test only
→ do not infer exact MCL grade

K8 + combined instability
→ leave isolated-MCL routine pathway

K9 + acute first-time dislocation + unresolved osteochondral/structural assessment
→ no routine unrestricted PT wording

K10 + lateral pain only
→ do not infer ITB syndrome

K11 + pes tenderness only
→ do not infer bursitis

K12 + unresolved fracture/locked knee/extensor rupture/major instability
→ safety prompt

K13 postoperative + missing procedure/protocol/restrictions
→ warning

meniscectomy + repair protocol accidentally treated identically
→ warning

TKA-specific NMES suggestion outside postoperative/TKA context
→ invalid

posterior knee swelling + no established Baker cyst
→ do not infer Baker cyst or dismiss DVT

P-K1 tibial-tubercle pain only
→ do not autonomously diagnose Osgood-Schlatter

P-K2 inferior-patellar-pole pain only
→ do not autonomously diagnose SLJ

pediatric structural injury + generic pediatric category only
→ require the appropriate structural pathway

hot swollen knee + diagnostic uncertainty
→ medical reassessment; no routine reassuring PT wording

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurovascular component
→ never generate normal wording
```

---

# 13. Generated wording examples

## 13.1 Knee OA

> Οστεοαρθρίτιδα του [side] γόνατος με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση με έμφαση σε προοδευτική ενδυνάμωση, λειτουργική άσκηση, βελτίωση κινητικότητας/νευρομυϊκού ελέγχου όπου ενδείκνυται και εκπαίδευση αυτοδιαχείρισης. [Brace/taping/acupuncture only if explicitly selected.]

## 13.2 Degenerative meniscal lesion

> Εκφυλιστική βλάβη / ρήξη [medial/lateral] μηνίσκου του [side] γόνατος, με [selected findings] και περιορισμό σε [selected activities], για συντηρητική αντιμετώπιση. Παρακαλώ για προοδευτική αποκατάσταση δύναμης, κινητικότητας, νευρομυϊκού ελέγχου και ανοχής στη λειτουργική φόρτιση. Δεν υπονοείται χειρουργική ένδειξη από την απεικονιστική ρήξη και μόνο.

## 13.3 Acute nonoperative meniscal injury

> Οξεία κάκωση [medial/lateral] μηνίσκου του [side] γόνατος μετά από αξιολόγηση, για συντηρητική αποκατάσταση σύμφωνα με τους καταγεγραμμένους περιορισμούς σε φόρτιση/κινητικότητα. Παρακαλώ για προοδευτική αποκατάσταση ROM, δύναμης και λειτουργίας, με επανεκτίμηση εάν εμφανιστεί/επιμένει πραγματικό κλείδωμα ή σημαντικός μηχανικός περιορισμός.

## 13.4 ACL

> Κάκωση / ανεπάρκεια ACL του [side] γόνατος, [established structural/operative context]. Παρακαλώ για προοδευτική αποκατάσταση ROM, δύναμης, νευρομυϊκού ελέγχου και λειτουργικής σταθερότητας σύμφωνα με τους καταγεγραμμένους περιορισμούς, με criterion-based progression προς τρέξιμο/άθληση όπου απαιτείται. Η επιστροφή σε άθληση δεν καθορίζεται από τον χρόνο μόνο.

## 13.5 MCL

> Κάκωση MCL του [side] γόνατος, [grade/stability if established], για [conservative/postoperative] αποκατάσταση. Παρακαλώ για προστατευμένη και προοδευτική κινητοποίηση/φόρτιση, αποκατάσταση δύναμης και νευρομυϊκού ελέγχου σύμφωνα με τη σταθερότητα, το brace και τους καταγεγραμμένους περιορισμούς.

## 13.6 Patellar instability

> [First-time/recurrent] αστάθεια/εξάρθρημα επιγονατίδας του [side] γόνατος μετά από κατάλληλη δομική αξιολόγηση. Παρακαλώ για εξατομικευμένη αποκατάσταση δύναμης, νευρομυϊκού ελέγχου και σταδιακής επιστροφής στη λειτουργία σύμφωνα με το osteochondral/MPFL/anatomic-risk context και τυχόν brace/ROM restrictions.

## 13.7 Pes anserine

> Πόνος περιοχής χήνειου ποδός / [clinician-established pes anserine tendinobursitis] του [side] γόνατος με [selected findings]. Παρακαλώ για φυσιοθεραπευτική αποκατάσταση με προσαρμογή φορτίου, προοδευτική ενδυνάμωση και αποκατάσταση λειτουργικής ανοχής σύμφωνα με τα ευρήματα.

## 13.8 Meniscus repair postoperative

> Μετεγχειρητική αποκατάσταση μετά από συρραφή μηνίσκου του [side] γόνατος, επέμβαση [date]. Παρακαλώ για φυσιοθεραπεία σύμφωνα με το ακριβές πρωτόκολλο της συρραφής και τους καταγεγραμμένους περιορισμούς σε φόρτιση, ROM, brace και ενδυνάμωση, με time- και criterion-based progression.

## 13.9 Partial meniscectomy postoperative

> Μετεγχειρητική αποκατάσταση μετά από μερική μηνισκεκτομή του [side] γόνατος, επέμβαση [date]. Παρακαλώ για criterion-based αποκατάσταση κινητικότητας, δύναμης, βάδισης και λειτουργικής φόρτισης σύμφωνα με την κλινική πρόοδο και τυχόν ειδικές χειρουργικές οδηγίες.

## 13.10 Osgood-Schlatter

> Νόσος Osgood-Schlatter του [side] γόνατος σε [age/adolescent context], με [selected findings] και περιορισμό σε [sport/school activity]. Παρακαλώ για εκπαίδευση και εξατομικευμένη συντηρητική αποκατάσταση με προσαρμογή φορτίου, προοδευτική ενδυνάμωση/κινητικότητα όπου ενδείκνυται και σταδιακή επιστροφή σε δραστηριότητα χωρίς άκαμπτο προκαθορισμένο χρονοδιάγραμμα.

## 13.11 Sinding-Larsen-Johansson

> Σύνδρομο Sinding-Larsen-Johansson του [side] γόνατος σε [age/adolescent context], με [selected findings]. Παρακαλώ για συντηρητική αποκατάσταση με διαχείριση φόρτισης, προοδευτική βελτίωση κινητικότητας/δύναμης και σταδιακή επιστροφή σε αθλητική δραστηριότητα σύμφωνα με την ανοχή και την κλινική εξέλιξη.

---

# 14. Evidence-governance boundary

Stable structural decisions frozen in Knee v1.1:

```text
exercise / education = core for knee OA
radiographic OA != automatic symptom generator
acupuncture = optional evidence-sensitive knee-OA adjunct only
dry needling = excluded
degenerative meniscal MRI tear != automatic symptom generator
exercise-based PT = first-line for degenerative meniscal lesions
true locked knee != routine degenerative-meniscus rehabilitation
acute displaced/displacing meniscal tear with ROM block = structural/specialist pathway
meniscus repair != partial meniscectomy rehabilitation logic
anterior knee pain != automatic PFP
PFP core = education + knee-targeted ± hip-targeted exercise
patellar/quadriceps tendinopathy core = progressive load-based rehabilitation
routine ESWT = not a generator default
ACL and MCL = separate top-level pathways
subjective giving-way != objective ligament instability
ligament special test != autonomous tear grade
patellar dislocation requires structural/osteochondral context
ITB and pes-anserine pathways remain diagnosis-sensitive
postoperative rehabilitation = exact procedure/protocol governed
TKA-specific interventions do not leak into generic OA pathways
pediatric/adolescent category = UI grouping, not diagnosis
Osgood-Schlatter and SLJ = distinct growth-related pathways
pediatric meniscus/ACL/MCL/patellar instability use standard structural pathways + age/skeletal-maturity context
possible Baker cyst != exclusion of DVT
prepatellar bursitis = medical/context only in this workflow
Hoffa/plica = rare clinician-entered context only
fractures route to shared fracture/post-immobilization profile
gastrocnemius strain routes to shared muscle/myotendinous profile
```

Evidence anchors reviewed for this freeze include:

- AAOS Management of Osteoarthritis of the Knee (Non-Arthroplasty), 3rd edition;
- NICE OA guidance where framework differences matter, including its recommendation against acupuncture/dry needling;
- 2024 AAOS Acute Isolated Meniscal Pathology CPG;
- formal EU-US Meniscus Rehabilitation 2024 consensus (ESSKA-AOSSM-AASPT), operative and nonoperative parts;
- 5-year ESCAPE randomized trial of exercise PT vs arthroscopic partial meniscectomy for degenerative tears;
- 2024 patellofemoral-pain best-practice guide;
- contemporary progressive-loading evidence for patellar tendinopathy;
- AAOS ACL guidance and Aspetar ACL-reconstruction rehabilitation CPG/protocol;
- ESSKA 2024 formal consensus on first-time patellar dislocation;
- APTA Physical Therapist Management of Total Knee Arthroplasty Revision 2026;
- 2026 review of Osgood-Schlatter treatments plus prior conservative-treatment systematic review;
- 2024 scoping review of Sinding-Larsen-Johansson disease.

Evidence-sensitive details to refresh immediately before CU-2 implementation:

```text
OA brace/taping/manual-therapy/acupuncture wording across frameworks
PFP foot-orthosis/taping/movement-retraining selection
meniscus lesion/repair-specific rehabilitation milestones
ACL criterion-based progression and return-to-sport language
MCL grade-specific bracing/loading details
patellar-instability brace/return-to-sport details
patellar/quadriceps tendon loading progression
ITB/pes-anserine rehabilitation evidence
TKA 2026 detailed intervention/dosage recommendations
procedure-specific postoperative restrictions
Osgood/SLJ exercise and return-to-sport evidence
```

---

# 15. Product-owner decisions incorporated

Product-owner decisions on 2026-08-27:

- postoperative knee is active; meniscal repair and meniscectomy are seen most often;
- degenerative and acute traumatic meniscal pathways remain separate;
- ACL and MCL are separate top-level pathways;
- patellar instability/dislocation is a default pathway;
- quadriceps tendinopathy is a default pathway;
- ITB syndrome is a default selectable pathway despite being less common;
- pes-anserine region pathology is a default pathway because it is seen frequently;
- acupuncture is retained as an optional evidence-sensitive adjunct for selected knee OA;
- dry needling is excluded;
- ESWT is not a default clinician-generated intervention; therapist-proposed use may be documented in patellar tendinopathy;
- taping, knee braces and foot orthoses remain condition-sensitive supports;
- children/adolescents are seen and receive a dedicated pediatric/adolescent knee navigation category;
- Osgood-Schlatter and Sinding-Larsen-Johansson are directly selectable pediatric/adolescent growth-related pathways;
- Baker cyst and prepatellar bursitis are medical/context only because they are not routinely referred for physiotherapy;
- distal hamstring insertional pathology and Hoffa/plica are rare selectable secondary/advanced entities;
- gastrocnemius strain is seen but belongs in the future shared muscle/myotendinous injury profile.

This file is the frozen knee clinical/content design for CU-1. Runtime implementation remains unauthorized.